"""Embedding and vector store persistence for text chunks.

Batches chunks, calls OpenAI text-embedding-3-small, and upserts
results into the pgvector chunks table. Includes retry logic for
OpenAI rate limits.
"""

import logging
import time
import uuid
from datetime import datetime, timezone

import asyncpg
from openai import AsyncOpenAI, RateLimitError

from api.models import Chunk

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
BATCH_SIZE = 100
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds, doubles each retry


async def _embed_batch(client: AsyncOpenAI, texts: list[str]) -> list[list[float]]:
    """Call OpenAI embeddings API for a batch of texts with retry logic.

    Args:
        client: Async OpenAI client instance.
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors (one per input text).

    Raises:
        RuntimeError: If all retries are exhausted.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except RateLimitError as e:
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning(
                "OpenAI rate limit hit (attempt %d/%d). Retrying in %.1fs: %s",
                attempt + 1, MAX_RETRIES, delay, e,
            )
            time.sleep(delay)
        except Exception as e:
            logger.error("OpenAI embedding error on attempt %d: %s", attempt + 1, e)
            raise

    raise RuntimeError(f"OpenAI embedding failed after {MAX_RETRIES} retries")


async def embed_and_store(chunks: list[Chunk], db_pool: asyncpg.Pool) -> int:
    """Embed chunks and upsert them into the pgvector chunks table.

    Processes chunks in batches of BATCH_SIZE. Each chunk is assigned a
    UUID and stored with its embedding, metadata, and ingestion timestamp.

    Args:
        chunks: List of Chunk objects to embed and store.
        db_pool: asyncpg connection pool to the PostgreSQL database.

    Returns:
        Number of chunks successfully embedded and stored.

    Raises:
        Exception: If database insert fails (logged before re-raising).
    """
    if not chunks:
        logger.warning("embed_and_store called with empty chunk list")
        return 0

    client = AsyncOpenAI()
    total_stored = 0
    start_time = time.perf_counter()

    logger.info("Starting embedding of %d chunks in batches of %d", len(chunks), BATCH_SIZE)

    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]
        texts = [chunk.content for chunk in batch]

        try:
            embeddings = await _embed_batch(client, texts)
        except Exception as e:
            logger.error(
                "Failed to embed batch %d-%d: %s",
                batch_start, batch_start + len(batch), e,
            )
            raise

        records = []
        now = datetime.now(timezone.utc)
        for chunk, embedding in zip(batch, embeddings):
            records.append((
                uuid.UUID(chunk.id),
                chunk.metadata.get("source_filename", "unknown"),
                chunk.metadata.get("page_number"),
                chunk.metadata.get("line_range"),
                chunk.chunk_index,
                chunk.char_count,
                chunk.content,
                embedding,
                now,
            ))

        try:
            async with db_pool.acquire() as conn:
                await conn.executemany(
                    """
                    INSERT INTO chunks
                        (id, source_filename, page_number, line_range, chunk_index,
                         char_count, content, embedding, ingested_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8::vector, $9)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        ingested_at = EXCLUDED.ingested_at
                    """,
                    records,
                )
            total_stored += len(batch)
            logger.info(
                "Stored batch %d-%d (%d chunks)",
                batch_start, batch_start + len(batch), len(batch),
            )
        except Exception as e:
            logger.error("Database insert failed for batch %d-%d: %s", batch_start, batch_start + len(batch), e)
            raise

    elapsed = time.perf_counter() - start_time
    logger.info(
        "Embedding complete: %d chunks stored in %.2fs (%.1f chunks/s)",
        total_stored, elapsed, total_stored / elapsed if elapsed > 0 else 0,
    )
    return total_stored
