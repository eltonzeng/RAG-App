"""Retrieval evaluation: run each ablation variant over the gold dataset.

Produces mean recall@{5,10}, MRR, and nDCG@10 per variant — the table that
quantifies what each pipeline stage contributes. No generation cost.
"""

import json
import logging
from pathlib import Path

import asyncpg

from evals import metrics
from evals.variants import VARIANTS, Variant, run_variant

logger = logging.getLogger(__name__)

DATASET_PATH = Path(__file__).parent / "datasets" / "retrieval_qa.jsonl"
RECALL_KS = (5, 10)
NDCG_K = 10
TOP_K = 20


def load_dataset(path: Path = DATASET_PATH) -> list[dict]:
    """Load the JSONL gold dataset.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of dataset rows (dicts). Blank lines are skipped.

    Raises:
        FileNotFoundError: If the dataset file is missing.
    """
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _to_retrieved(scored_chunks) -> list[tuple[str, set]]:
    """Convert ScoredChunks into the metrics' (identity, source_keys) form.

    Uses the chunk id as the content identity — it is the dedup key in the DB and
    stable within a run, so duplicate collapsing behaves correctly.

    Args:
        scored_chunks: Ranked ScoredChunk list, best-first.

    Returns:
        Ranked list of (content_identity, source_keys).
    """
    retrieved = []
    for sc in scored_chunks:
        keys = metrics.source_keys(sc.chunk.metadata.get("sources", []))
        retrieved.append((sc.chunk.id, keys))
    return retrieved


def _blank_scores() -> dict[str, float]:
    """Return a zeroed metrics accumulator."""
    scores = {f"recall@{k}": 0.0 for k in RECALL_KS}
    scores[f"ndcg@{NDCG_K}"] = 0.0
    scores["mrr"] = 0.0
    scores[f"hit_rate@{RECALL_KS[-1]}"] = 0.0
    return scores


async def evaluate_variant(
    variant: Variant,
    dataset: list[dict],
    db_pool: asyncpg.Pool,
    top_k: int = TOP_K,
) -> dict[str, float]:
    """Run a variant over the dataset and return mean metrics.

    Args:
        variant: The retrieval configuration.
        dataset: Loaded dataset rows.
        db_pool: asyncpg pool.
        top_k: Retrieval cutoff.

    Returns:
        Dict of metric name → mean value across queries.
    """
    totals = _blank_scores()
    n = len(dataset)
    if n == 0:
        return totals

    for row in dataset:
        gold = metrics.gold_keys(row.get("gold", []))
        scored = await run_variant(
            variant, row["question"], row.get("filters", {}), db_pool, top_k=top_k
        )
        retrieved = _to_retrieved(scored)
        for k in RECALL_KS:
            totals[f"recall@{k}"] += metrics.recall_at_k(retrieved, gold, k)
        totals[f"ndcg@{NDCG_K}"] += metrics.ndcg_at_k(retrieved, gold, NDCG_K)
        totals["mrr"] += metrics.mrr(retrieved, gold)
        totals[f"hit_rate@{RECALL_KS[-1]}"] += metrics.hit_rate_at_k(
            retrieved, gold, RECALL_KS[-1]
        )

    return {name: value / n for name, value in totals.items()}


async def run_retrieval_eval(
    db_pool: asyncpg.Pool,
    variant_names: list[str] | None = None,
    dataset_path: Path = DATASET_PATH,
    limit: int | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluate the requested variants over the dataset.

    Args:
        db_pool: asyncpg pool.
        variant_names: Subset of VARIANTS to run; None runs all.
        dataset_path: Path to the gold dataset.
        limit: If set, evaluate only the first N dataset rows (smoke testing).

    Returns:
        Mapping of variant name → metrics dict.
    """
    dataset = load_dataset(dataset_path)
    if limit is not None:
        dataset = dataset[:limit]
    logger.info("Loaded %d gold questions from %s", len(dataset), dataset_path)

    names = variant_names or list(VARIANTS)
    results: dict[str, dict[str, float]] = {}
    for name in names:
        variant = VARIANTS[name]
        logger.info("Evaluating variant '%s'", name)
        results[name] = await evaluate_variant(variant, dataset, db_pool)
    return results
