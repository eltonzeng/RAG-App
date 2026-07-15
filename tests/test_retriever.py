"""Tests for retrieval/retriever.py and retrieval/reranker.py.

Uses mocks to avoid requiring live database or API connections.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.models import Chunk, ScoredChunk
from retrieval.reranker import (
    FALLBACK_SIMILARITY_THRESHOLD,
    RELEVANCE_THRESHOLD,
    rerank,
)
from retrieval.retriever import _row_to_scored_chunk, _rrf_fuse


def make_scored_chunk(chunk_id: str, content: str, score: float) -> ScoredChunk:
    """Helper to create a ScoredChunk for testing."""
    return ScoredChunk(
        chunk=Chunk(
            id=chunk_id,
            content=content,
            chunk_index=0,
            char_count=len(content),
            metadata={"source_filename": "test.pdf", "page_number": 1},
        ),
        score=score,
    )


class TestReranker:
    """Tests for the Cohere reranker with mocked API responses."""

    @pytest.mark.asyncio
    async def test_rerank_returns_top_n(self) -> None:
        """Should return at most top_n chunks."""
        chunks = [
            make_scored_chunk(f"id-{i}", f"Content about SEC filings {i}", 0.5 + i * 0.01)
            for i in range(10)
        ]

        mock_result = MagicMock()
        mock_result.results = [
            MagicMock(index=i, relevance_score=0.9 - i * 0.05)
            for i in range(5)
        ]

        with patch("retrieval.reranker.cohere.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.rerank = AsyncMock(return_value=mock_result)
            mock_client_cls.return_value = mock_client

            reranked, is_relevant = await rerank("SEC filings query", chunks, top_n=5)

        assert len(reranked) <= 5
        assert is_relevant is True

    @pytest.mark.asyncio
    async def test_rerank_below_threshold_flags_irrelevant(self) -> None:
        """If all scores below threshold, should return is_relevant=False."""
        chunks = [
            make_scored_chunk("id-1", "Unrelated content about cooking recipes", 0.1),
        ]

        mock_result = MagicMock()
        mock_result.results = [MagicMock(index=0, relevance_score=0.05)]

        with patch("retrieval.reranker.cohere.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.rerank = AsyncMock(return_value=mock_result)
            mock_client_cls.return_value = mock_client

            reranked, is_relevant = await rerank(
                "SEC annual report revenue", chunks, top_n=1
            )

        assert is_relevant is False

    @pytest.mark.asyncio
    async def test_rerank_above_threshold_flags_relevant(self) -> None:
        """If any score >= threshold, should return is_relevant=True."""
        chunks = [make_scored_chunk("id-1", "10-K annual report revenue disclosure", 0.8)]

        mock_result = MagicMock()
        mock_result.results = [MagicMock(index=0, relevance_score=RELEVANCE_THRESHOLD)]

        with patch("retrieval.reranker.cohere.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.rerank = AsyncMock(return_value=mock_result)
            mock_client_cls.return_value = mock_client

            reranked, is_relevant = await rerank("annual report", chunks, top_n=1)

        assert is_relevant is True

    @pytest.mark.asyncio
    async def test_rerank_empty_input(self) -> None:
        """Empty input should return empty list and False."""
        reranked, is_relevant = await rerank("any query", [], top_n=5)
        assert reranked == []
        assert is_relevant is False

    @pytest.mark.asyncio
    async def test_rerank_fallback_on_cohere_failure(self) -> None:
        """When Cohere fails, should fall back to similarity-sorted top-N."""
        chunks = [
            make_scored_chunk("id-1", "High relevance content", 0.9),
            make_scored_chunk("id-2", "Medium relevance content", 0.5),
            make_scored_chunk("id-3", "Low relevance content", 0.2),
        ]

        with patch("retrieval.reranker.cohere.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.rerank = AsyncMock(side_effect=Exception("Cohere API unavailable"))
            mock_client_cls.return_value = mock_client

            reranked, is_relevant = await rerank("query", chunks, top_n=2)

        assert len(reranked) == 2
        # Should be sorted by similarity score descending
        assert reranked[0].chunk.id == "id-1"
        assert reranked[1].chunk.id == "id-2"
        # Fallback uses the cosine-calibrated threshold; top score 0.9 clears it.
        assert is_relevant is True
        assert FALLBACK_SIMILARITY_THRESHOLD > RELEVANCE_THRESHOLD

    @pytest.mark.asyncio
    async def test_rerank_fallback_below_threshold(self) -> None:
        """Fallback should also correctly flag irrelevant content."""
        # Score sits above the old Cohere threshold (0.3) but below the
        # cosine-calibrated fallback threshold — the regression this guards against.
        chunks = [make_scored_chunk("id-1", "Irrelevant text", 0.4)]

        with patch("retrieval.reranker.cohere.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.rerank = AsyncMock(side_effect=Exception("Cohere unavailable"))
            mock_client_cls.return_value = mock_client

            reranked, is_relevant = await rerank("unrelated query", chunks, top_n=1)

        assert is_relevant is False


def _row(chunk_id: str, similarity: float, sources: list[dict]) -> dict:
    """Build a fake DB row (asyncpg Record is dict-accessible in these tests)."""
    return {
        "id": chunk_id,
        "content": f"content for {chunk_id}",
        "sources": json.dumps(sources),
        "similarity": similarity,
    }


class TestRowConversion:
    """Tests for _row_to_scored_chunk."""

    def test_carries_sources_and_derives_first_source(self) -> None:
        sources = [
            {"source_filename": "aapl_10k.pdf", "page_number": 5, "chunk_index": 2},
            {"source_filename": "aapl_10q.pdf", "page_number": 1},
        ]
        sc = _row_to_scored_chunk(_row("c1", 0.83, sources))

        assert sc.chunk.id == "c1"
        assert sc.score == 0.83
        assert sc.chunk.metadata["sources"] == sources
        # Back-compat fields derive from the first source.
        assert sc.chunk.metadata["source_filename"] == "aapl_10k.pdf"
        assert sc.chunk.metadata["page_number"] == 5
        assert sc.chunk.chunk_index == 2

    def test_handles_empty_sources(self) -> None:
        sc = _row_to_scored_chunk(_row("c2", 0.5, []))
        assert sc.chunk.metadata["source_filename"] == "unknown"
        assert sc.chunk.metadata["page_number"] is None


class _FakeAcquire:
    """Async context manager mimicking asyncpg pool.acquire()."""

    def __init__(self, conn) -> None:
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc) -> bool:
        return False


class _FakePool:
    """Minimal asyncpg.Pool stand-in that hands out a fixed connection."""

    def __init__(self, conn) -> None:
        self._conn = conn

    def acquire(self) -> _FakeAcquire:
        return _FakeAcquire(self._conn)


def _patch_embeddings():
    """Patch AsyncOpenAI in the retriever to return one dummy embedding."""
    mock_client = MagicMock()
    emb_resp = MagicMock()
    emb_resp.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    mock_client.embeddings.create = AsyncMock(return_value=emb_resp)
    patcher = patch("retrieval.retriever.AsyncOpenAI", return_value=mock_client)
    return patcher


class TestHybridRetrieveFallback:
    """The zero-result fallback: a filter matching nothing retries unfiltered."""

    @pytest.mark.asyncio
    async def test_bad_filter_falls_back_to_unfiltered(self) -> None:
        from retrieval import retriever

        row = _row("c1", 0.8, [{"source_filename": "aaoi.pdf", "page_number": 5}])

        # Filtered branches match nothing (e.g. hallucinated ticker); unfiltered
        # find the chunk.
        async def fake_semantic(conn, embedding, fj, limit):
            return [] if fj is not None else [row]

        async def fake_bm25(conn, query, embedding, fj, limit):
            return []

        pool = _FakePool(AsyncMock())
        with _patch_embeddings(), \
                patch.object(retriever, "_search_semantic", side_effect=fake_semantic), \
                patch.object(retriever, "_search_bm25", side_effect=fake_bm25):
            result = await retriever.hybrid_retrieve(
                ["q"], {"ticker": "AOEO"}, pool, top_k=10
            )

        assert [sc.chunk.id for sc in result] == ["c1"]

    @pytest.mark.asyncio
    async def test_no_fallback_when_filter_matches(self) -> None:
        from retrieval import retriever

        row = _row("c1", 0.8, [{"source_filename": "mu.pdf", "page_number": 5}])
        seen_fj: list = []

        async def fake_semantic(conn, embedding, fj, limit):
            seen_fj.append(fj)
            return [row]

        async def fake_bm25(conn, query, embedding, fj, limit):
            return []

        pool = _FakePool(AsyncMock())
        with _patch_embeddings(), \
                patch.object(retriever, "_search_semantic", side_effect=fake_semantic), \
                patch.object(retriever, "_search_bm25", side_effect=fake_bm25):
            result = await retriever.hybrid_retrieve(
                ["q"], {"ticker": "MU"}, pool, top_k=10
            )

        assert [sc.chunk.id for sc in result] == ["c1"]
        # The filter matched, so retrieval never retried unfiltered.
        assert all(fj is not None for fj in seen_fj)


class TestRRFFusion:
    """Tests for _rrf_fuse."""

    def _chunk_map(self, ids_scores: dict[str, float]) -> dict[str, ScoredChunk]:
        return {cid: make_scored_chunk(cid, cid, score) for cid, score in ids_scores.items()}

    def test_consensus_across_lists_ranks_first(self) -> None:
        """A chunk ranked highly in both lists beats list-specific top hits."""
        chunk_map = self._chunk_map({"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6})
        semantic = ["a", "b", "c"]
        bm25 = ["b", "d", "a"]

        fused = _rrf_fuse([semantic, bm25], chunk_map, top_k=4)
        # 'b' is rank1+rank0, 'a' is rank0+rank2 — 'b' should edge ahead overall.
        assert fused[0].chunk.id == "b"
        assert {sc.chunk.id for sc in fused} == {"a", "b", "c", "d"}

    def test_respects_top_k(self) -> None:
        chunk_map = self._chunk_map({"a": 0.9, "b": 0.8, "c": 0.7})
        fused = _rrf_fuse([["a", "b", "c"]], chunk_map, top_k=2)
        assert len(fused) == 2

    def test_preserves_cosine_score(self) -> None:
        """Fused chunks keep their cosine similarity as score (for reranking)."""
        chunk_map = self._chunk_map({"a": 0.42})
        fused = _rrf_fuse([["a"]], chunk_map, top_k=1)
        assert fused[0].score == 0.42

    def test_empty_lists_return_empty(self) -> None:
        assert _rrf_fuse([], {}, top_k=5) == []
