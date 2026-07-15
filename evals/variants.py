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

from api.models import ScoredChunk
from core.clients import get_openai_client
from core.config import get_settings
from generation.query_rewriter import rewrite_query
from retrieval.reranker import rerank
from retrieval.retriever import (
    BRANCH_LIMIT_MULTIPLIER,
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
) -> list[ScoredChunk]:
    """Run one retrieval variant for a single question.

    Args:
        variant: The configuration to run.
        question: The raw user question.
        dataset_filters: Filters attached to the dataset row (used when
            use_filters is set and use_rewrite is not).
        db_pool: asyncpg pool.
        top_k: Candidates to fuse/return (and to rerank, when enabled).

    Returns:
        Ranked list of up to top_k ScoredChunk (RRF order, or rerank order if
        enabled). Reranking reorders the full top_k candidate set rather than
        truncating it, so rank metrics (recall@k, nDCG@k) stay comparable across
        variants — a rerank cutoff below k would otherwise cap those metrics.
    """
    if variant.use_rewrite:
        rewrite = await rewrite_query(question)
        queries = rewrite.queries
        filters = rewrite.filters.as_containment() if variant.use_filters else {}
    else:
        queries = [question]
        filters = dict(dataset_filters) if variant.use_filters else {}

    client = get_openai_client()
    response = await client.embeddings.create(model=get_settings().embedding_model, input=queries)
    embeddings = [item.embedding for item in response.data]

    branch_limit = max(top_k * BRANCH_LIMIT_MULTIPLIER, top_k)

    async def _fuse(fj: str | None) -> list[ScoredChunk]:
        """Run the enabled branches under filter `fj` and fuse via RRF."""
        chunk_map: dict[str, ScoredChunk] = {}
        ranked_lists: list[list[str]] = []
        async with db_pool.acquire() as conn:
            for query, embedding in zip(queries, embeddings, strict=True):
                if variant.use_semantic:
                    rows = await _search_semantic(conn, embedding, fj, branch_limit)
                    ranked_lists.append(_collect(rows, chunk_map))
                if variant.use_bm25:
                    rows = await _search_bm25(conn, query, embedding, fj, branch_limit)
                    ranked_lists.append(_collect(rows, chunk_map))
        return _rrf_fuse(ranked_lists, chunk_map, top_k)

    filter_json = json.dumps([filters]) if filters else None
    fused = await _fuse(filter_json)

    # Mirror hybrid_retrieve's zero-result fallback: if a filter (e.g. an
    # LLM-hallucinated ticker) matches nothing, retry unfiltered so the variant
    # reflects production behavior rather than scoring a hard zero.
    if not fused and filter_json is not None:
        fused = await _fuse(None)

    if variant.use_rerank and fused:
        # Rerank the entire fused set (top_n = len) so the returned list is the
        # full top_k reordered, not truncated to a production top_n. This keeps
        # recall@k / nDCG@k comparable with the non-rerank variants.
        fused, _ = await rerank(question, fused, top_n=len(fused))

    return fused
