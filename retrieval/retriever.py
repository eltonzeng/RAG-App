"""Vector similarity retrieval from pgvector.

Embeds the query using OpenAI text-embedding-3-small and performs
cosine similarity search against the chunks table.
"""

import logging
import time

import asyncpg
from openai import AsyncOpenAI

from api.models import Chunk, ScoredChunk

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"


async def retrieve(
    query: str,
    db_pool: asyncpg.Pool,
    top_k: int = 20,
) -> list[ScoredChunk]:
    """Embed a query and retrieve the most similar chunks via cosine similarity.

    Args:
        query: The user's natural language question.
        db_pool: asyncpg connection pool to the PostgreSQL database.
        top_k: Number of top results to return.

    Returns:
        List of ScoredChunk objects sorted by descending similarity score.

    Raises:
        Exception: If embedding or database query fails.
    """
    start_time = time.perf_counter()
    client = AsyncOpenAI()

    try:
        response = await client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[query],
        )
        query_embedding = response.data[0].embedding
    except Exception as e:
        logger.error("Failed to embed query: %s", e)
        raise

    embed_time = time.perf_counter()
    logger.debug("Query embedded in %.3fs", embed_time - start_time)

    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    id::text,
                    source_filename,
                    page_number,
                    line_range,
                    chunk_index,
                    char_count,
                    content,
                    1 - (embedding <=> $1::vector) AS similarity
                FROM chunks
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """,
                query_embedding,
                top_k,
            )
    except Exception as e:
        logger.error("Database similarity search failed: %s", e)
        raise

    results: list[ScoredChunk] = []
    for row in rows:
        chunk = Chunk(
            id=row["id"],
            content=row["content"],
            chunk_index=row["chunk_index"],
            char_count=row["char_count"],
            metadata={
                "source_filename": row["source_filename"],
                "page_number": row["page_number"],
                "line_range": row["line_range"],
            },
        )
        results.append(ScoredChunk(chunk=chunk, score=float(row["similarity"])))

    total_time = time.perf_counter() - start_time
    logger.info(
        "Retrieved %d chunks for query (top_k=%d) in %.3fs — top score: %.4f",
        len(results), top_k, total_time,
        results[0].score if results else 0.0,
    )
    return results
