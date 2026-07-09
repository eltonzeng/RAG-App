"""Embedding, dedup, and vector store persistence for text chunks.

Deduplicates chunks by a content hash so re-ingesting the same (or shared)
text never re-pays for embeddings: each unique piece of content is embedded and
stored once, and every filing/page it appears in is recorded as an element of
the chunk's ``sources`` JSONB array. New content is embedded via OpenAI
text-embedding-3-small and upserted into the pgvector/ParadeDB chunks table.
"""

import asyncio
import hashlib
import json
import logging
import re
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

# Keys copied from a chunk's metadata into its `sources` provenance element.
# (chunk_index is a Chunk attribute, added separately.)
_SOURCE_KEYS = (
    "source_filename",
    "page_number",
    "ticker",
    "fiscal_year",
    "quarter",
    "form_type",
)

# Union two JSONB source arrays and drop duplicate elements. Used by both the
# ON CONFLICT insert path and the update-only path so provenance accumulates
# without repeats.
_MERGE_SOURCES_SQL = """
    SELECT COALESCE(jsonb_agg(DISTINCT e), '[]'::jsonb)
    FROM jsonb_array_elements({left} || {right}) AS e
"""


def _content_hash(text: str) -> str:
    """Compute a stable dedup hash for chunk content.

    Normalizes surrounding and internal whitespace before hashing so that
    trivially different copies of the same text collapse together. Must stay in
    sync with the backfill in db/migrations/001_hybrid.sql.

    Args:
        text: The chunk content.

    Returns:
        Hex-encoded sha256 digest of the normalized content.
    """
    normalized = re.sub(r"\s+", " ", text.strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _source_element(chunk: Chunk) -> dict:
    """Build a provenance element for a chunk's ``sources`` array.

    Args:
        chunk: The chunk whose metadata carries source/filing fields.

    Returns:
        Dict of the present source keys (missing/None values omitted so JSONB
        containment filters stay clean).
    """
    element: dict = {"chunk_index": chunk.chunk_index}
    for key in _SOURCE_KEYS:
        value = chunk.metadata.get(key)
        if value is not None:
            element[key] = value
    return element


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
            await asyncio.sleep(delay)
        except Exception as e:
            logger.error("OpenAI embedding error on attempt %d: %s", attempt + 1, e)
            raise

    raise RuntimeError(f"OpenAI embedding failed after {MAX_RETRIES} retries")


def _dedupe_chunks(chunks: list[Chunk]) -> dict[str, dict]:
    """Collapse chunks by content hash, merging their provenance.

    Args:
        chunks: Chunks to deduplicate (may contain repeated content across
            pages/filings within a single ingest).

    Returns:
        Ordered mapping of content_hash → {"content": str, "sources": list[dict]}
        with duplicate source elements removed.
    """
    by_hash: dict[str, dict] = {}
    for chunk in chunks:
        digest = _content_hash(chunk.content)
        element = _source_element(chunk)
        entry = by_hash.get(digest)
        if entry is None:
            by_hash[digest] = {"content": chunk.content, "sources": [element]}
        elif element and element not in entry["sources"]:
            entry["sources"].append(element)
    return by_hash


async def embed_and_store(chunks: list[Chunk], db_pool: asyncpg.Pool) -> tuple[int, int]:
    """Embed new chunks and upsert them into the chunks table, deduping by hash.

    Content already present in the table is not re-embedded; instead the new
    filing/page provenance is appended to the existing row's ``sources`` array.
    New content is embedded in batches of BATCH_SIZE and inserted.

    Args:
        chunks: List of Chunk objects to embed and store.
        db_pool: asyncpg connection pool to the PostgreSQL database.

    Returns:
        Tuple of (chunks_embedded, chunks_skipped) where chunks_embedded is the
        number of newly embedded unique chunks and chunks_skipped is the number
        of unique chunks whose content already existed (provenance appended).

    Raises:
        Exception: If embedding or a database write fails (logged before re-raise).
    """
    if not chunks:
        logger.warning("embed_and_store called with empty chunk list")
        return 0, 0

    by_hash = _dedupe_chunks(chunks)
    all_hashes = list(by_hash.keys())
    start_time = time.perf_counter()

    # Which content is already stored? Those only need a provenance append.
    try:
        async with db_pool.acquire() as conn:
            existing_rows = await conn.fetch(
                "SELECT content_hash FROM chunks WHERE content_hash = ANY($1::text[])",
                all_hashes,
            )
    except Exception as e:
        logger.error("Failed to look up existing content hashes: %s", e)
        raise

    existing_hashes = {row["content_hash"] for row in existing_rows}
    new_hashes = [h for h in all_hashes if h not in existing_hashes]

    logger.info(
        "Ingesting %d chunks → %d unique (%d new, %d duplicate)",
        len(chunks), len(all_hashes), len(new_hashes), len(existing_hashes),
    )

    client = AsyncOpenAI()

    # 1) Append provenance for content that already exists — no embedding cost.
    if existing_hashes:
        try:
            async with db_pool.acquire() as conn:
                for digest in existing_hashes:
                    sources_json = json.dumps(by_hash[digest]["sources"])
                    await conn.execute(
                        f"""
                        UPDATE chunks
                        SET sources = ({_MERGE_SOURCES_SQL.format(
                            left="sources", right="$2::jsonb")})
                        WHERE content_hash = $1
                        """,
                        digest,
                        sources_json,
                    )
        except Exception as e:
            logger.error("Failed to append provenance for existing chunks: %s", e)
            raise

    # 2) Embed and insert new content in batches.
    now = datetime.now(timezone.utc)
    for batch_start in range(0, len(new_hashes), BATCH_SIZE):
        batch_hashes = new_hashes[batch_start : batch_start + BATCH_SIZE]
        texts = [by_hash[h]["content"] for h in batch_hashes]

        try:
            embeddings = await _embed_batch(client, texts)
        except Exception as e:
            logger.error(
                "Failed to embed batch %d-%d: %s",
                batch_start, batch_start + len(batch_hashes), e,
            )
            raise

        records = []
        for digest, embedding in zip(batch_hashes, embeddings):
            entry = by_hash[digest]
            records.append((
                uuid.uuid4(),
                digest,
                entry["content"],
                embedding,
                json.dumps(entry["sources"]),
                now,
            ))

        try:
            async with db_pool.acquire() as conn:
                # ON CONFLICT guards against a concurrent insert of the same
                # content: fall back to merging provenance rather than erroring.
                await conn.executemany(
                    f"""
                    INSERT INTO chunks
                        (id, content_hash, content, embedding, sources, ingested_at)
                    VALUES ($1, $2, $3, $4::vector, $5::jsonb, $6)
                    ON CONFLICT (content_hash) DO UPDATE SET
                        sources = ({_MERGE_SOURCES_SQL.format(
                            left="chunks.sources", right="EXCLUDED.sources")})
                    """,
                    records,
                )
            logger.info(
                "Stored batch %d-%d (%d new chunks)",
                batch_start, batch_start + len(batch_hashes), len(batch_hashes),
            )
        except Exception as e:
            logger.error(
                "Database insert failed for batch %d-%d: %s",
                batch_start, batch_start + len(batch_hashes), e,
            )
            raise

    elapsed = time.perf_counter() - start_time
    embedded = len(new_hashes)
    skipped = len(existing_hashes)
    logger.info(
        "Embedding complete: %d new chunks embedded, %d skipped (duplicate) in %.2fs",
        embedded, skipped, elapsed,
    )
    return embedded, skipped
