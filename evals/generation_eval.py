"""Generation evaluation: run the full pipeline, then judge each answer.

Exercises the same path as POST /ask — rewrite → hybrid retrieve → rerank →
generate — and grades the answer with the judge model. Returns aggregate mean
judge scores plus per-question rows (including the judge rationale and whether
rerank fell back), so a reader can see the "why" behind each score and spot
Cohere rate-limit artifacts.
"""

import asyncio
import logging
from pathlib import Path

import asyncpg

from evals.judge import judge_answer
from evals.retrieval_eval import DATASET_PATH, load_dataset
from generation.generator import generate
from generation.query_rewriter import rewrite_query
from retrieval.reranker import rerank
from retrieval.retriever import hybrid_retrieve

logger = logging.getLogger(__name__)

TOP_K = 20
TOP_N = 5


async def _answer_and_judge(
    row: dict,
    db_pool: asyncpg.Pool,
    semaphore: asyncio.Semaphore,
    pace: float = 0.0,
) -> dict:
    """Run the pipeline for one question and judge the answer.

    Args:
        row: Dataset row with at least a "question" (and optional "id").
        db_pool: asyncpg pool.
        semaphore: Bounds concurrent full-pipeline runs.
        pace: Seconds to wait before starting (serialized under the semaphore);
            used to stay under the Cohere trial-key rate limit.

    Returns:
        A per-question result row: id, question, the four judge scores, grounded,
        rationale, error, and rerank_fallback.
    """
    async with semaphore:
        if pace:
            await asyncio.sleep(pace)
        question = row["question"]
        rewrite = await rewrite_query(question)
        scored = await hybrid_retrieve(
            rewrite.queries, rewrite.filters.as_containment(), db_pool, top_k=TOP_K
        )
        reranked, is_relevant, rerank_fallback = await rerank(question, scored, top_n=TOP_N)
        answer, citations = await generate(
            question, reranked, is_relevant, rewrite.filters.as_containment()
        )
        verdict = await judge_answer(question, answer, reranked, citations)
        logger.info(
            "Judged '%s' — faithful=%d cite=%d rel=%d grounded=%s fallback=%s",
            question[:60],
            verdict.faithfulness,
            verdict.citation_accuracy,
            verdict.answer_relevance,
            verdict.grounded,
            rerank_fallback,
        )
        return {
            "id": row.get("id"),
            "question": question,
            "faithfulness": verdict.faithfulness,
            "citation_accuracy": verdict.citation_accuracy,
            "answer_relevance": verdict.answer_relevance,
            "grounded": verdict.grounded,
            "rationale": verdict.rationale,
            "error": verdict.error,
            "rerank_fallback": rerank_fallback,
        }


def _empty_aggregates() -> dict[str, float]:
    """Zeroed aggregate metrics for an empty dataset."""
    return {
        "faithfulness": 0.0,
        "citation_accuracy": 0.0,
        "answer_relevance": 0.0,
        "groundedness_rate": 0.0,
        "judge_error_rate": 0.0,
        "n": 0.0,
    }


def _aggregate(rows: list[dict]) -> dict[str, float]:
    """Mean judge scores and pass-rates across per-question rows."""
    n = len(rows)
    return {
        "faithfulness": sum(r["faithfulness"] for r in rows) / n,
        "citation_accuracy": sum(r["citation_accuracy"] for r in rows) / n,
        "answer_relevance": sum(r["answer_relevance"] for r in rows) / n,
        "groundedness_rate": sum(1 for r in rows if r["grounded"]) / n,
        "judge_error_rate": sum(1 for r in rows if r["error"]) / n,
        "n": float(n),
    }


def _caveats(rows: list[dict]) -> list[str]:
    """Auto-generated caveats flagging artifacts that distort the scores."""
    caveats: list[str] = []
    fallbacks = sum(1 for r in rows if r["rerank_fallback"])
    if fallbacks:
        caveats.append(
            f"{fallbacks}/{len(rows)} questions used the similarity rerank fallback "
            "(Cohere unavailable/rate-limited); their scores understate quality."
        )
    errors = sum(1 for r in rows if r["error"])
    if errors:
        caveats.append(f"{errors}/{len(rows)} questions had a judge error (sentinel 1/1/1 scores).")
    return caveats


async def run_generation_eval(
    db_pool: asyncpg.Pool,
    dataset_path: Path = DATASET_PATH,
    concurrency: int = 3,
    limit: int | None = None,
    pace: float = 0.0,
) -> dict:
    """Run the generation suite over the dataset.

    Args:
        db_pool: asyncpg pool.
        dataset_path: Path to the gold dataset.
        concurrency: Max concurrent pipeline+judge runs.
        limit: If set, evaluate only the first N dataset rows (smoke testing;
            keeps judge costs bounded while iterating).
        pace: Seconds to wait before each question (use with concurrency=1 to
            stay under the Cohere trial-key limit).

    Returns:
        Dict with "aggregates" (mean judge scores + rates), "rows" (per-question
        detail incl. rationale and rerank_fallback), and "caveats".
    """
    dataset = load_dataset(dataset_path)
    if limit is not None:
        dataset = dataset[:limit]
    logger.info("Running generation eval over %d questions", len(dataset))
    if not dataset:
        return {"aggregates": _empty_aggregates(), "rows": [], "caveats": []}

    semaphore = asyncio.Semaphore(concurrency)
    rows = await asyncio.gather(
        *(_answer_and_judge(row, db_pool, semaphore, pace=pace) for row in dataset)
    )

    return {
        "aggregates": _aggregate(rows),
        "rows": rows,
        "caveats": _caveats(rows),
    }
