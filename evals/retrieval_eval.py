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
) -> tuple[dict[str, float], list[dict]]:
    """Run a variant over the dataset, returning mean metrics and per-question rows.

    Args:
        variant: The retrieval configuration.
        dataset: Loaded dataset rows.
        db_pool: asyncpg pool.
        top_k: Retrieval cutoff.

    Returns:
        Tuple of (mean metrics dict, per-question rows). Each row carries the
        question id and its individual metric values.
    """
    totals = _blank_scores()
    rows: list[dict] = []
    n = len(dataset)
    if n == 0:
        return totals, rows

    for row in dataset:
        gold = metrics.gold_keys(row.get("gold", []))
        scored = await run_variant(
            variant, row["question"], row.get("filters", {}), db_pool, top_k=top_k
        )
        retrieved = _to_retrieved(scored)
        per_q: dict[str, object] = {"id": row.get("id"), "question": row["question"]}
        for k in RECALL_KS:
            value = metrics.recall_at_k(retrieved, gold, k)
            totals[f"recall@{k}"] += value
            per_q[f"recall@{k}"] = value
        ndcg = metrics.ndcg_at_k(retrieved, gold, NDCG_K)
        mrr = metrics.mrr(retrieved, gold)
        hit = metrics.hit_rate_at_k(retrieved, gold, RECALL_KS[-1])
        totals[f"ndcg@{NDCG_K}"] += ndcg
        totals["mrr"] += mrr
        totals[f"hit_rate@{RECALL_KS[-1]}"] += hit
        per_q[f"ndcg@{NDCG_K}"] = ndcg
        per_q["mrr"] = mrr
        per_q[f"hit_rate@{RECALL_KS[-1]}"] = hit
        rows.append(per_q)

    means = {name: value / n for name, value in totals.items()}
    return means, rows


async def run_retrieval_eval(
    db_pool: asyncpg.Pool,
    variant_names: list[str] | None = None,
    dataset_path: Path = DATASET_PATH,
    limit: int | None = None,
) -> dict:
    """Evaluate the requested variants over the dataset.

    Args:
        db_pool: asyncpg pool.
        variant_names: Subset of VARIANTS to run; None runs all.
        dataset_path: Path to the gold dataset.
        limit: If set, evaluate only the first N dataset rows (smoke testing).

    Returns:
        Dict with "aggregates" (variant → mean metrics), "rows" (variant →
        per-question rows), and "caveats".
    """
    dataset = load_dataset(dataset_path)
    if limit is not None:
        dataset = dataset[:limit]
    logger.info("Loaded %d gold questions from %s", len(dataset), dataset_path)

    names = variant_names or list(VARIANTS)
    aggregates: dict[str, dict[str, float]] = {}
    rows: dict[str, list[dict]] = {}
    for name in names:
        variant = VARIANTS[name]
        logger.info("Evaluating variant '%s'", name)
        aggregates[name], rows[name] = await evaluate_variant(variant, dataset, db_pool)

    return {"aggregates": aggregates, "rows": rows, "caveats": []}
