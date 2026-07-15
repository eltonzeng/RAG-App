"""Tests for the SSE streaming endpoint POST /ask/stream.

Mocks the pipeline so no live DB or LLM is touched; asserts the Server-Sent
Events frame sequence (meta → delta* → citations → done, or an error frame).
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from api.main import app
from api.models import Citation, MetadataFilters, QueryRewriteResult, ScoredChunk


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    """Parse an SSE response body into a list of (event, data) pairs."""
    frames = []
    for block in body.strip().split("\n\n"):
        event, data = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data = json.loads(line[len("data:") :].strip())
        if event is not None:
            frames.append((event, data))
    return frames


@pytest.fixture
def mock_pool():
    pool = MagicMock()
    conn = AsyncMock()
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=None)
    pool.acquire = MagicMock(return_value=conn)
    return pool


@pytest.fixture(autouse=True)
def inject_pool(mock_pool):
    app.state.pool = mock_pool


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _chunk() -> ScoredChunk:
    from api.models import Chunk

    return ScoredChunk(
        chunk=Chunk(
            id="c1",
            content="Revenue was $1.2B.",
            chunk_index=0,
            char_count=18,
            metadata={"source_filename": "aapl.pdf", "page_number": 45},
        ),
        score=0.9,
    )


async def _fake_deltas(*_args, **_kwargs):
    for piece in ["Revenue ", "was ", "$1.2B."]:
        yield piece


class TestAskStream:
    @pytest.mark.asyncio
    async def test_full_frame_sequence(self, client: AsyncClient) -> None:
        with (
            patch("api.routes.rewrite_query", new_callable=AsyncMock) as rw,
            patch("api.routes.hybrid_retrieve", new_callable=AsyncMock) as ret,
            patch("api.routes.rerank", new_callable=AsyncMock) as rr,
            patch("api.routes.generate_stream", _fake_deltas),
            patch("api.routes.extract_citations") as cites,
        ):
            rw.return_value = QueryRewriteResult(
                queries=["q1", "q2"], filters=MetadataFilters(ticker="AAPL")
            )
            ret.return_value = [_chunk()]
            rr.return_value = ([_chunk()], True, False)
            cites.return_value = [Citation(source="aapl.pdf", page=45, chunk_id="c1")]

            resp = await client.post("/ask/stream", json={"query": "revenue?"})

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        frames = _parse_sse(resp.text)
        events = [e for e, _ in frames]

        assert events[0] == "meta"
        assert events[-1] == "done"
        assert "citations" in events
        # Deltas appear in order and reconstruct the answer.
        deltas = [d["text"] for e, d in frames if e == "delta"]
        assert "".join(deltas) == "Revenue was $1.2B."
        # Meta carries the rewritten queries and applied filters.
        meta = frames[0][1]
        assert meta["rewritten_queries"] == ["q1", "q2"]
        assert meta["applied_filters"]["ticker"] == "AAPL"
        # Citations frame carries the citation.
        cite_frame = next(d for e, d in frames if e == "citations")
        assert cite_frame[0]["source"] == "aapl.pdf"
        # Done frame carries chunk counts.
        done = frames[-1][1]
        assert done["chunks_used"] == 1

    @pytest.mark.asyncio
    async def test_graceful_no_content_still_completes(self, client: AsyncClient) -> None:
        """When nothing is relevant, the real generate_stream yields the fallback
        message (no LLM call) and the stream still ends with a done frame."""
        with (
            patch("api.routes.rewrite_query", new_callable=AsyncMock) as rw,
            patch("api.routes.hybrid_retrieve", new_callable=AsyncMock) as ret,
            patch("api.routes.rerank", new_callable=AsyncMock) as rr,
        ):
            rw.return_value = QueryRewriteResult(queries=["q"], filters=MetadataFilters())
            ret.return_value = []
            rr.return_value = ([], False, False)  # not relevant

            resp = await client.post("/ask/stream", json={"query": "unknown"})

        frames = _parse_sse(resp.text)
        events = [e for e, _ in frames]
        assert events[-1] == "done"
        # Citations are empty when nothing was relevant.
        cite_frame = next(d for e, d in frames if e == "citations")
        assert cite_frame == []

    @pytest.mark.asyncio
    async def test_pipeline_error_emits_error_frame(self, client: AsyncClient) -> None:
        with (
            patch("api.routes.rewrite_query", new_callable=AsyncMock) as rw,
            patch("api.routes.hybrid_retrieve", new_callable=AsyncMock) as ret,
        ):
            rw.return_value = QueryRewriteResult(queries=["q"], filters=MetadataFilters())
            ret.side_effect = RuntimeError("db down")

            resp = await client.post("/ask/stream", json={"query": "revenue?"})

        frames = _parse_sse(resp.text)
        assert frames[-1][0] == "error"
        assert "detail" in frames[-1][1]
