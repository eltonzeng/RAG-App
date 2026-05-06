"""Tests for retrieval/retriever.py and retrieval/reranker.py.

Uses mocks to avoid requiring live database or API connections.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.models import Chunk, ScoredChunk
from retrieval.reranker import RELEVANCE_THRESHOLD, rerank


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

    def test_rerank_returns_top_n(self) -> None:
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

        with patch("retrieval.reranker.cohere.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.rerank.return_value = mock_result
            mock_client_cls.return_value = mock_client

            reranked, is_relevant = rerank("SEC filings query", chunks, top_n=5)

        assert len(reranked) <= 5
        assert is_relevant is True

    def test_rerank_below_threshold_flags_irrelevant(self) -> None:
        """If all scores below threshold, should return is_relevant=False."""
        chunks = [
            make_scored_chunk("id-1", "Unrelated content about cooking recipes", 0.1),
        ]

        mock_result = MagicMock()
        mock_result.results = [MagicMock(index=0, relevance_score=0.05)]

        with patch("retrieval.reranker.cohere.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.rerank.return_value = mock_result
            mock_client_cls.return_value = mock_client

            reranked, is_relevant = rerank("SEC annual report revenue", chunks, top_n=1)

        assert is_relevant is False

    def test_rerank_above_threshold_flags_relevant(self) -> None:
        """If any score >= threshold, should return is_relevant=True."""
        chunks = [make_scored_chunk("id-1", "10-K annual report revenue disclosure", 0.8)]

        mock_result = MagicMock()
        mock_result.results = [MagicMock(index=0, relevance_score=RELEVANCE_THRESHOLD)]

        with patch("retrieval.reranker.cohere.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.rerank.return_value = mock_result
            mock_client_cls.return_value = mock_client

            reranked, is_relevant = rerank("annual report", chunks, top_n=1)

        assert is_relevant is True

    def test_rerank_empty_input(self) -> None:
        """Empty input should return empty list and False."""
        reranked, is_relevant = rerank("any query", [], top_n=5)
        assert reranked == []
        assert is_relevant is False

    def test_rerank_fallback_on_cohere_failure(self) -> None:
        """When Cohere fails, should fall back to similarity-sorted top-N."""
        chunks = [
            make_scored_chunk("id-1", "High relevance content", 0.9),
            make_scored_chunk("id-2", "Medium relevance content", 0.5),
            make_scored_chunk("id-3", "Low relevance content", 0.2),
        ]

        with patch("retrieval.reranker.cohere.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.rerank.side_effect = Exception("Cohere API unavailable")
            mock_client_cls.return_value = mock_client

            reranked, is_relevant = rerank("query", chunks, top_n=2)

        assert len(reranked) == 2
        # Should be sorted by similarity score descending
        assert reranked[0].chunk.id == "id-1"
        assert reranked[1].chunk.id == "id-2"
        assert is_relevant is True  # top score 0.9 >= 0.3

    def test_rerank_fallback_below_threshold(self) -> None:
        """Fallback should also correctly flag irrelevant content."""
        chunks = [make_scored_chunk("id-1", "Irrelevant text", 0.1)]

        with patch("retrieval.reranker.cohere.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.rerank.side_effect = Exception("Cohere unavailable")
            mock_client_cls.return_value = mock_client

            reranked, is_relevant = rerank("unrelated query", chunks, top_n=1)

        assert is_relevant is False
