"""Retrieval ablation variants.

Each variant toggles pipeline stages and runs through the *real* retriever
internals (no reimplementation), so the comparison is honest. This is the grid
that shows what each feature — BM25, hybrid fusion, multi-query rewrite, rerank,
metadata filters — actually earns.

Design notes:
- "single-query" variants use the original question verbatim and (optionally) the
  dataset's own filters, keeping them deterministic and LLM-free.
- "multiquery" variants invoke the Haiku rewrite (generation.query_rewriter) to
  produce query variants and extracted filters.
"""

import json
from dataclasses import dataclass

import asyncpg
from openai import AsyncOpenAI

from api.models import ScoredChunk
from generation.query_rewriter import rewrite_query
from retrieval.reranker import rerank
from retrieval.retriever import (
    BRANCH_LIMIT_MULTIPLIER,
    EMBEDDING_MODEL,
    _row_to_scored_chunk,
    _rrf_fuse,
    _search_bm25,
    _search_semantic,
)


@dataclass(frozen=True)
class Variant:
    """A named retrieval configuration.

    Attributes:
        name: Identifier used in the report.
        use_semantic: Include the pgvector cosine branch.
        use_bm25: Include the ParadeDB BM25 branch.
        use_rewrite: Expand the query into variants via the Haiku rewrite (also
            sources extracted filters when use_filters is set).
        use_rerank: Apply the Cohere reranker to the fused candidates.
        use_filters: Constrain retrieval with metadata filters (dataset-provided
            for single-query variants; LLM-extracted when use_rewrite is set).
    """

    name: str
    use_semantic: bool = True
    use_bm25: bool = True
    use_rewrite: bool = False
    use_rerank: bool = False
    use_filters: bool = False


# The ablation grid. Order matters for the report (least → most capable).
VARIANTS: dict[str, Variant] = {
    "semantic_only": Variant("semantic_only", use_bm25=False),
    "bm25_only": Variant("bm25_only", use_semantic=False),
    "hybrid": Variant("hybrid"),
    "hybrid_filters": Variant("hybrid_filters", use_filters=True),
    "hybrid_multiquery": Variant("hybrid_multiquery", use_rewrite=True, use_filters=True),
    "hybrid_multiquery_rerank": Variant(
        "hybrid_multiquery_rerank", use_rewrite=True, use_rerank=True, use_filters=True
    ),
}


def _collect(rows, chunk_map: dict[str, ScoredChunk]) -> list[str]:
    """Convert DB rows to a ranked id list, updating the shared chunk map.

    Keeps the highest cosine similarity seen for each chunk (mirrors
    hybrid_retrieve so ScoredChunk.score stays cosine-calibrated).

    Args:
        rows: Rows returned by a search branch (best-first).
        chunk_map: Shared id → ScoredChunk map, updated in place.

    Returns:
        Ranked list of chunk ids for this branch.
    """
    ranked_ids: list[str] = []
    for row in rows:
        sc = _row_to_scored_chunk(row)
        ranked_ids.append(sc.chunk.id)
        existing = chunk_map.get(sc.chunk.id)
        if existing is None or sc.score > existing.score:
            chunk_map[sc.chunk.id] = sc
    return ranked_ids


async def run_variant(
    variant: Variant,
    question: str,
    dataset_filters: dict,
    db_pool: asyncpg.Pool,
    top_k: int = 20,
    top_n: int = 5,
) -> list[ScoredChunk]:
    """Run one retrieval variant for a single question.

    Args:
        variant: The configuration to run.
        question: The raw user question.
        dataset_filters: Filters attached to the dataset row (used when
            use_filters is set and use_rewrite is not).
        db_pool: asyncpg pool.
        top_k: Candidates to fuse/return before reranking.
        top_n: Chunks to keep if reranking.

    Returns:
        Ranked list of ScoredChunk (RRF order, or rerank order if enabled).
    """
    if variant.use_rewrite:
        rewrite = await rewrite_query(question)
        queries = rewrite.queries
        filters = rewrite.filters.as_containment() if variant.use_filters else {}
    else:
        queries = [question]
        filters = dict(dataset_filters) if variant.use_filters else {}

    filter_json = json.dumps([filters]) if filters else None

    client = AsyncOpenAI()
    response = await client.embeddings.create(model=EMBEDDING_MODEL, input=queries)
    embeddings = [item.embedding for item in response.data]

    branch_limit = max(top_k * BRANCH_LIMIT_MULTIPLIER, top_k)
    chunk_map: dict[str, ScoredChunk] = {}
    ranked_lists: list[list[str]] = []

    async with db_pool.acquire() as conn:
        for query, embedding in zip(queries, embeddings):
            if variant.use_semantic:
                rows = await _search_semantic(conn, embedding, filter_json, branch_limit)
                ranked_lists.append(_collect(rows, chunk_map))
            if variant.use_bm25:
                rows = await _search_bm25(conn, query, embedding, filter_json, branch_limit)
                ranked_lists.append(_collect(rows, chunk_map))

    fused = _rrf_fuse(ranked_lists, chunk_map, top_k)

    if variant.use_rerank:
        fused, _ = await rerank(question, fused, top_n=top_n)

    return fused
