"""Cohere reranking for retrieved chunks.

Reranks candidate chunks using the Cohere Rerank API, applying a
relevance threshold to detect "no relevant content" cases. Falls back
to top-N similarity results if Cohere is unavailable.
"""

import logging
import time

import cohere

from api.models import ScoredChunk

logger = logging.getLogger(__name__)

RERANK_MODEL = "rerank-english-v3.0"
RELEVANCE_THRESHOLD = 0.3  # Below this, content is considered not relevant


def rerank(
    query: str,
    scored_chunks: list[ScoredChunk],
    top_n: int = 5,
) -> tuple[list[ScoredChunk], bool]:
    """Rerank retrieved chunks using the Cohere Rerank API.

    If all reranked scores fall below RELEVANCE_THRESHOLD, the second
    return value signals that no relevant content was found. Falls back
    to returning top-N similarity-scored chunks if Cohere fails.

    Args:
        query: The user's natural language question.
        scored_chunks: Candidate chunks from the retriever.
        top_n: Number of chunks to return after reranking.

    Returns:
        Tuple of:
          - List of reranked ScoredChunk objects (length <= top_n)
          - Boolean: True if content is relevant, False if below threshold
    """
    if not scored_chunks:
        logger.warning("rerank called with empty chunk list")
        return [], False

    start_time = time.perf_counter()
    documents = [sc.chunk.content for sc in scored_chunks]

    try:
        client = cohere.Client()
        response = client.rerank(
            model=RERANK_MODEL,
            query=query,
            documents=documents,
            top_n=top_n,
        )

        reranked: list[ScoredChunk] = []
        for result in response.results:
            original = scored_chunks[result.index]
            reranked.append(ScoredChunk(
                chunk=original.chunk,
                score=result.relevance_score,
            ))

        elapsed = time.perf_counter() - start_time
        scores = [sc.score for sc in reranked]
        logger.info(
            "Reranked %d → %d chunks in %.3fs — scores: %s",
            len(scored_chunks), len(reranked), elapsed,
            [f"{s:.4f}" for s in scores],
        )

        # Log chunk IDs for traceability
        for sc in reranked:
            logger.debug("Reranked chunk %s score=%.4f", sc.chunk.id, sc.score)

        max_score = max(scores) if scores else 0.0
        is_relevant = max_score >= RELEVANCE_THRESHOLD

        if not is_relevant:
            logger.info(
                "Max rerank score %.4f below threshold %.2f — flagging as not relevant",
                max_score, RELEVANCE_THRESHOLD,
            )

        return reranked, is_relevant

    except Exception as e:
        logger.warning(
            "Cohere rerank failed, falling back to similarity top-%d: %s", top_n, e
        )
        # Fallback: return top-N by similarity score
        fallback = sorted(scored_chunks, key=lambda sc: sc.score, reverse=True)[:top_n]
        max_score = max(sc.score for sc in fallback) if fallback else 0.0
        is_relevant = max_score >= RELEVANCE_THRESHOLD
        return fallback, is_relevant
