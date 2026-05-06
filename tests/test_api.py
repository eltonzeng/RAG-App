"""Tests for the FastAPI routes.

Uses httpx.AsyncClient with mocked pipeline components to avoid
requiring live database or API connections.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from api.main import app
from api.models import Chunk, Citation, ScoredChunk


@pytest.fixture
def mock_pool():
    """Create a mock asyncpg pool for injection into app.state."""
    pool = MagicMock()
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value={"cnt": 42})
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=None)
    pool.acquire = MagicMock(return_value=conn)
    return pool


@pytest.fixture(autouse=True)
def inject_pool(mock_pool):
    """Inject the mock pool into app.state before each test."""
    app.state.pool = mock_pool


@pytest.fixture
def sample_scored_chunk() -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            id="test-chunk-id",
            content="The company reported revenue of $1.2 billion in fiscal year 2023.",
            chunk_index=0,
            char_count=65,
            metadata={"source_filename": "apple_10k.pdf", "page_number": 45},
        ),
        score=0.87,
    )


@pytest_asyncio.fixture
async def async_client():
    """Create an AsyncClient for testing without running the server."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


class TestHealthEndpoint:
    """Tests for GET /health."""

    @pytest.mark.asyncio
    async def test_health_ok(self, async_client: AsyncClient, mock_pool) -> None:
        """Health endpoint should return 200 with connected status."""
        response = await async_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["database_connected"] is True
        assert data["chunk_count"] == 42
        assert "embedding_model" in data
        assert "generation_model" in data

    @pytest.mark.asyncio
    async def test_health_db_failure_returns_degraded(
        self, async_client: AsyncClient, mock_pool
    ) -> None:
        """If DB query fails, status should be 'degraded'."""
        mock_pool.acquire.return_value.__aenter__.side_effect = Exception("DB down")
        response = await async_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["database_connected"] is False


class TestIngestEndpoint:
    """Tests for POST /ingest."""

    @pytest.mark.asyncio
    async def test_ingest_no_sources_returns_400(self, async_client: AsyncClient) -> None:
        """Ingest with no files or URLs should return 400."""
        response = await async_client.post("/ingest", json={})
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_ingest_invalid_strategy_returns_400(self, async_client: AsyncClient) -> None:
        """Ingest with invalid chunk strategy should return 400."""
        response = await async_client.post(
            "/ingest",
            json={"file_paths": ["/tmp/test.pdf"], "chunk_strategy": "invalid"},
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_ingest_file_not_found_returns_400(self, async_client: AsyncClient) -> None:
        """Ingest with missing file should return 400."""
        response = await async_client.post(
            "/ingest",
            json={"file_paths": ["/nonexistent/file.pdf"]},
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_ingest_success(
        self, async_client: AsyncClient, mock_pool, sample_scored_chunk
    ) -> None:
        """Successful ingest should return 200 with document/chunk counts."""
        from api.models import Chunk, Document

        with (
            patch("api.routes.load_pdf") as mock_load,
            patch("api.routes.chunk_recursive") as mock_chunk,
            patch("api.routes.embed_and_store", new_callable=AsyncMock) as mock_embed,
        ):
            mock_load.return_value = [
                Document(
                    content="SEC filing content",
                    metadata={"source_filename": "test.pdf", "page_number": 1},
                )
            ]
            mock_chunk.return_value = [sample_scored_chunk.chunk]
            mock_embed.return_value = 1

            response = await async_client.post(
                "/ingest",
                json={"file_paths": ["/fake/test.pdf"], "chunk_strategy": "recursive"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["documents_loaded"] == 1
        assert data["chunks_created"] == 1
        assert data["chunks_embedded"] == 1


class TestAskEndpoint:
    """Tests for POST /ask."""

    @pytest.mark.asyncio
    async def test_ask_returns_answer_and_citations(
        self, async_client: AsyncClient, sample_scored_chunk
    ) -> None:
        """Successful ask should return answer with citations."""
        with (
            patch("api.routes.retrieve", new_callable=AsyncMock) as mock_retrieve,
            patch("api.routes.rerank") as mock_rerank,
            patch("api.routes.generate", new_callable=AsyncMock) as mock_generate,
        ):
            mock_retrieve.return_value = [sample_scored_chunk]
            mock_rerank.return_value = ([sample_scored_chunk], True)
            mock_generate.return_value = (
                "Revenue was $1.2 billion.",
                [Citation(source="apple_10k.pdf", page=45, chunk_id="test-chunk-id")],
            )

            response = await async_client.post(
                "/ask",
                json={"query": "What was the revenue?"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Revenue was $1.2 billion."
        assert len(data["citations"]) == 1
        assert data["citations"][0]["source"] == "apple_10k.pdf"
        assert data["chunks_retrieved"] == 1
        assert data["chunks_used"] == 1
        assert data["latency_ms"] > 0

    @pytest.mark.asyncio
    async def test_ask_no_relevant_content(self, async_client: AsyncClient) -> None:
        """Ask with no relevant content should return graceful message."""
        from generation.prompts import NO_RELEVANT_CONTENT_RESPONSE

        with (
            patch("api.routes.retrieve", new_callable=AsyncMock) as mock_retrieve,
            patch("api.routes.rerank") as mock_rerank,
            patch("api.routes.generate", new_callable=AsyncMock) as mock_generate,
        ):
            mock_retrieve.return_value = []
            mock_rerank.return_value = ([], False)
            mock_generate.return_value = (NO_RELEVANT_CONTENT_RESPONSE, [])

            response = await async_client.post(
                "/ask",
                json={"query": "Completely unrelated question"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "don't have enough information" in data["answer"]

    @pytest.mark.asyncio
    async def test_ask_retrieval_failure_returns_503(self, async_client: AsyncClient) -> None:
        """If retrieval fails, /ask should return 503."""
        with patch("api.routes.retrieve", new_callable=AsyncMock) as mock_retrieve:
            mock_retrieve.side_effect = Exception("DB unavailable")
            response = await async_client.post(
                "/ask",
                json={"query": "What is the revenue?"},
            )

        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_ask_empty_query_returns_422(self, async_client: AsyncClient) -> None:
        """Empty query should fail Pydantic validation and return 422."""
        response = await async_client.post("/ask", json={"query": ""})
        assert response.status_code == 422
