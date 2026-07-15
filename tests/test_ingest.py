"""Tests for ingest dedup and metadata extraction.

External calls (OpenAI, database) are mocked so no live services are needed.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.models import Chunk, Document
from ingest.embedder import (
    _content_hash,
    _dedupe_chunks,
    _source_element,
    embed_and_store,
)
from ingest.metadata import extract_filing_metadata


def make_chunk(content: str, **metadata) -> Chunk:
    """Build a Chunk with the given content and source metadata."""
    return Chunk(
        id="unused",
        content=content,
        chunk_index=metadata.get("chunk_index", 0),
        char_count=len(content),
        metadata={"source_filename": "test.pdf", **metadata},
    )


class _FakePool:
    """Minimal asyncpg-pool stand-in whose acquire() yields a shared conn."""

    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        conn = self._conn

        class _CM:
            async def __aenter__(self):
                return conn

            async def __aexit__(self, *exc):
                return False

        return _CM()


def _embeddings_client(dim: int = 1536) -> MagicMock:
    """Fake AsyncOpenAI whose create() returns one embedding per input text."""
    client = MagicMock()

    async def _create(model, input):
        data = [SimpleNamespace(embedding=[0.0] * dim) for _ in input]
        return SimpleNamespace(data=data)

    client.embeddings.create = AsyncMock(side_effect=_create)
    return client


class TestContentHash:
    def test_whitespace_normalized(self) -> None:
        """Trivial whitespace differences hash identically."""
        assert _content_hash("Total  revenue\n was $5B") == _content_hash("Total revenue was $5B")

    def test_distinct_content_differs(self) -> None:
        assert _content_hash("revenue") != _content_hash("expenses")


class TestSourceHelpers:
    def test_source_element_omits_missing(self) -> None:
        """Only present metadata keys appear in the source element."""
        chunk = make_chunk("x", page_number=3, ticker="AAPL")
        elem = _source_element(chunk)
        assert elem == {
            "source_filename": "test.pdf",
            "page_number": 3,
            "chunk_index": 0,
            "ticker": "AAPL",
        }
        assert "quarter" not in elem

    def test_dedupe_merges_sources(self) -> None:
        """Identical content across pages collapses to one entry, two sources."""
        chunks = [
            make_chunk("shared boilerplate", page_number=1, chunk_index=0),
            make_chunk("shared boilerplate", page_number=2, chunk_index=0),
            make_chunk("unique text", page_number=3, chunk_index=1),
        ]
        by_hash = _dedupe_chunks(chunks)
        assert len(by_hash) == 2
        shared = by_hash[_content_hash("shared boilerplate")]
        assert len(shared["sources"]) == 2


class TestEmbedAndStore:
    @pytest.mark.asyncio
    async def test_all_new_content_embedded(self) -> None:
        """When nothing exists, all unique chunks are embedded and inserted."""
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])  # no existing hashes
        pool = _FakePool(conn)
        chunks = [make_chunk("alpha", page_number=1), make_chunk("beta", page_number=2)]

        with patch("ingest.embedder.get_openai_client", return_value=_embeddings_client()):
            embedded, skipped = await embed_and_store(chunks, pool)

        assert (embedded, skipped) == (2, 0)
        conn.executemany.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_existing_content_skipped_not_reembedded(self) -> None:
        """Content already stored is skipped; only new content is embedded."""
        existing_hash = _content_hash("alpha")
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[{"content_hash": existing_hash}])
        pool = _FakePool(conn)
        chunks = [make_chunk("alpha", page_number=1), make_chunk("beta", page_number=2)]

        client = _embeddings_client()
        with patch("ingest.embedder.get_openai_client", return_value=client):
            embedded, skipped = await embed_and_store(chunks, pool)

        assert (embedded, skipped) == (1, 1)
        # Only the new chunk ("beta") should have been embedded.
        client.embeddings.create.assert_awaited_once()
        _, kwargs = client.embeddings.create.call_args
        assert kwargs["input"] == ["beta"]
        # Existing chunk gets a provenance-append UPDATE (no insert for it).
        conn.execute.assert_awaited()

    @pytest.mark.asyncio
    async def test_within_batch_duplicates_embed_once(self) -> None:
        """Duplicate content in one ingest is embedded a single time."""
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        pool = _FakePool(conn)
        chunks = [
            make_chunk("same text", page_number=1),
            make_chunk("same text", page_number=2),
        ]

        client = _embeddings_client()
        with patch("ingest.embedder.get_openai_client", return_value=client):
            embedded, skipped = await embed_and_store(chunks, pool)

        assert (embedded, skipped) == (1, 0)
        _, kwargs = client.embeddings.create.call_args
        assert kwargs["input"] == ["same text"]

    @pytest.mark.asyncio
    async def test_empty_returns_zero(self) -> None:
        conn = AsyncMock()
        pool = _FakePool(conn)
        assert await embed_and_store([], pool) == (0, 0)


class TestMetadataExtraction:
    def test_extracts_from_filename(self) -> None:
        doc = Document(
            content="Some filing body text.",
            metadata={"source_filename": "AAPL_10-K_2023.pdf"},
        )
        meta = extract_filing_metadata(doc)
        assert meta["ticker"] == "AAPL"
        assert meta["form_type"] == "10-K"
        assert meta["fiscal_year"] == 2023

    def test_extracts_quarter_and_form_from_content(self) -> None:
        doc = Document(
            content="FORM 10-Q\nQuarterly report for the third quarter of fiscal year 2022.",
            metadata={"source_filename": "filing.pdf"},
        )
        meta = extract_filing_metadata(doc)
        assert meta["form_type"] == "10-Q"
        assert meta["quarter"] == 3
        assert meta["fiscal_year"] == 2022

    def test_no_confident_match_returns_empty(self) -> None:
        doc = Document(content="hello world", metadata={"source_filename": "notes.txt"})
        assert extract_filing_metadata(doc) == {}
