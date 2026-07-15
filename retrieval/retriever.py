"""Hybrid retrieval from ParadeDB (pgvector + pg_search).

Runs, for each rewritten query variant, a semantic branch (pgvector cosine
similarity) and a lexical branch (ParadeDB BM25), optionally constrained by
JSONB metadata filters. All ranked lists are fused with Reciprocal Rank Fusion
(RRF) into a single ordering.

Each returned ScoredChunk carries the chunk's best cosine similarity as its
``score`` (so the reranker's cosine-calibrated fallback threshold stays valid);
the RRF score only determines ordering, not the stored score.
"""

import asyncio
import json
import logging
import time

import asyncpg

from api.models import Chunk, ScoredChunk
from core.clients import get_openai_client
from core.config import get_settings

logger = logging.getLogger(__name__)

RRF_K = 60  # RRF damping constant; standard default.
# Per-branch candidate pool. Widened beyond top_k so fusion has material to work
# with before the final trim.
BRANCH_LIMIT_MULTIPLIER = 2


def _row_to_scored_chunk(row: asyncpg.Record) -> ScoredChunk:
    """Convert a DB row into a ScoredChunk carrying its sources provenance.

    Args:
        row: A row with id, content, sources (JSONB text), and similarity.

    Returns:
        ScoredChunk whose metadata holds the full ``sources`` list plus the
        first source's filename/page for back-compatible citation code.
    """
    sources = row["sources"]
    if isinstance(sources, str):
        sources = json.loads(sources)
    first = sources[0] if sources else {}
    chunk = Chunk(
        id=row["id"],
        content=row["content"],
        chunk_index=first.get("chunk_index", 0),
        char_count=len(row["content"]),
        metadata={
            "sources": sources,
            "source_filename": first.get("source_filename", "unknown"),
            "page_number": first.get("page_number"),
        },
    )
    return ScoredChunk(chunk=chunk, score=float(row["similarity"]))


async def _search_semantic(
    conn: asyncpg.Connection,
    embedding: list[float],
    filter_json: str | None,
    limit: int,
) -> list[asyncpg.Record]:
    """Cosine-similarity search, optionally filtered by JSONB containment."""
    if filter_json is None:
        return await conn.fetch(
            """
            SELECT id::text, content, sources,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM chunks
            ORDER BY embedding <=> $1::vector
            LIMIT $2
            """,
            embedding,
            limit,
        )
    return await conn.fetch(
        """
        SELECT id::text, content, sources,
               1 - (embedding <=> $1::vector) AS similarity
        FROM chunks
        WHERE sources @> $2::jsonb
        ORDER BY embedding <=> $1::vector
        LIMIT $3
        """,
        embedding,
        filter_json,
        limit,
    )


async def _search_bm25(
    conn: asyncpg.Connection,
    query: str,
    embedding: list[float],
    filter_json: str | None,
    limit: int,
) -> list[asyncpg.Record]:
    """BM25 lexical search via ParadeDB, optionally filtered by JSONB containment.

    Cosine similarity is computed alongside so BM25-only hits still carry a
    similarity score for downstream reranking.
    """
    if filter_json is None:
        return await conn.fetch(
            """
            SELECT id::text, content, sources,
                   1 - (embedding <=> $2::vector) AS similarity
            FROM chunks
            WHERE id @@@ paradedb.match('content', $1)
            ORDER BY paradedb.score(id) DESC
            LIMIT $3
            """,
            query,
            embedding,
            limit,
        )
    return await conn.fetch(
        """
        SELECT id::text, content, sources,
               1 - (embedding <=> $2::vector) AS similarity
        FROM chunks
        WHERE id @@@ paradedb.match('content', $1)
          AND sources @> $3::jsonb
        ORDER BY paradedb.score(id) DESC
        LIMIT $4
        """,
        query,
        embedding,
        filter_json,
        limit,
    )


def _rrf_fuse(
    ranked_lists: list[list[str]],
    chunk_map: dict[str, ScoredChunk],
    top_k: int,
) -> list[ScoredChunk]:
    """Fuse multiple ranked ID lists via Reciprocal Rank Fusion.

    Args:
        ranked_lists: Each inner list is chunk IDs ordered best-first for one
            branch/variant.
        chunk_map: Maps chunk ID → ScoredChunk (score = best cosine similarity).
        top_k: Maximum number of fused results to return.

    Returns:
        Up to top_k ScoredChunks ordered by descending RRF score. Each retains
        its cosine-similarity score for reranking.
    """
    rrf_scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)

    ordered_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
    return [chunk_map[cid] for cid in ordered_ids[:top_k]]


async def hybrid_retrieve(
    queries: list[str],
    filters: dict,
    db_pool: asyncpg.Pool,
    top_k: int = 20,
) -> list[ScoredChunk]:
    """Hybrid (semantic + BM25) multi-query retrieval with RRF fusion.

    Args:
        queries: One or more query variants (from the rewrite step). Each is run
            through both a semantic and a BM25 branch.
        filters: Metadata filters (e.g. {"ticker": "AAPL", "fiscal_year": 2023}).
            Applied as JSONB containment against each chunk's sources array.
            Empty dict means no filtering.
        db_pool: asyncpg connection pool.
        top_k: Number of fused chunks to return.

    Returns:
        Up to top_k ScoredChunk objects ordered by RRF score, each carrying its
        best cosine similarity as ``score``.

    Raises:
        Exception: If embedding or the database queries fail.
    """
    if not queries:
        logger.warning("hybrid_retrieve called with no queries")
        return []

    start_time = time.perf_counter()
    filter_json = json.dumps([filters]) if filters else None

    # Embed every query variant in a single OpenAI call.
    client = get_openai_client()
    try:
        response = await client.embeddings.create(
            model=get_settings().embedding_model, input=queries
        )
        embeddings = [item.embedding for item in response.data]
    except Exception as e:
        logger.error("Failed to embed query variants: %s", e)
        raise

    branch_limit = max(top_k * BRANCH_LIMIT_MULTIPLIER, top_k)

    async def _fuse(fj: str | None) -> tuple[list[ScoredChunk], int]:
        """Run both branches for every query under filter `fj` and fuse via RRF.

        Returns the fused ScoredChunks and the number of unique chunks seen.
        """

        async def _run_variant(query: str, embedding: list[float]) -> list[list[asyncpg.Record]]:
            async with db_pool.acquire() as conn:
                semantic = await _search_semantic(conn, embedding, fj, branch_limit)
                bm25 = await _search_bm25(conn, query, embedding, fj, branch_limit)
            return [semantic, bm25]

        per_variant = await asyncio.gather(
            *(_run_variant(q, emb) for q, emb in zip(queries, embeddings, strict=True))
        )

        # Collect every branch's ranked ID list and a unified chunk map (keeping
        # the highest cosine similarity seen for each chunk).
        chunk_map: dict[str, ScoredChunk] = {}
        ranked_lists: list[list[str]] = []
        for branches in per_variant:
            for rows in branches:
                ranked_ids: list[str] = []
                for row in rows:
                    sc = _row_to_scored_chunk(row)
                    ranked_ids.append(sc.chunk.id)
                    existing = chunk_map.get(sc.chunk.id)
                    if existing is None or sc.score > existing.score:
                        chunk_map[sc.chunk.id] = sc
                ranked_lists.append(ranked_ids)
        return _rrf_fuse(ranked_lists, chunk_map, top_k), len(chunk_map)

    try:
        fused, unique = await _fuse(filter_json)
    except Exception as e:
        logger.error("Hybrid retrieval database queries failed: %s", e)
        raise

    # Zero-result fallback. An over-restrictive or wrong metadata filter — e.g. a
    # ticker hallucinated by the rewrite step (AAOI → "AOEO") — can match no
    # chunks via JSONB containment, zeroing out retrieval. Rather than return
    # nothing, retry unfiltered so the user still gets an answer (degraded, and
    # logged) instead of a false "no relevant content".
    used_fallback = False
    if not fused and filter_json is not None:
        logger.warning(
            "Filtered retrieval returned 0 results (filters=%s); retrying unfiltered",
            filters,
        )
        try:
            fused, unique = await _fuse(None)
            used_fallback = True
        except Exception as e:
            logger.error("Unfiltered fallback retrieval failed: %s", e)
            raise

    elapsed = time.perf_counter() - start_time
    logger.info(
        "Hybrid retrieve: %d variants × 2 branches → %d unique → %d fused "
        "(top_k=%d, filters=%s, fallback=%s) in %.3fs — top cosine: %.4f",
        len(queries),
        unique,
        len(fused),
        top_k,
        filters or "none",
        used_fallback,
        elapsed,
        fused[0].score if fused else 0.0,
    )
    return fused
