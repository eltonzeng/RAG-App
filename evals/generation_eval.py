"""Generation evaluation: run the full pipeline, then judge each answer.

Exercises the same path as POST /ask — rewrite → hybrid retrieve → rerank →
generate — and grades the answer with the Opus 4.8 judge. Aggregates mean judge
scores plus a groundedness pass-rate.
"""

import asyncio
import logging
from pathlib import Path

import asyncpg

from evals.judge import JudgeVerdict, judge_answer
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
) -> JudgeVerdict:
    """Run the pipeline for one question and judge the answer.

    Args:
        row: Dataset row with at least a "question".
        db_pool: asyncpg pool.
        semaphore: Bounds concurrent full-pipeline runs.

    Returns:
        The judge's verdict for this question.
    """
    async with semaphore:
        question = row["question"]
        rewrite = await rewrite_query(question)
        scored = await hybrid_retrieve(
            rewrite.queries, rewrite.filters.as_containment(), db_pool, top_k=TOP_K
        )
        reranked, is_relevant = await rerank(question, scored, top_n=TOP_N)
        answer, citations = await generate(question, reranked, is_relevant)
        verdict = await judge_answer(question, answer, reranked, citations)
        logger.info(
            "Judged '%s' — faithful=%d cite=%d rel=%d grounded=%s",
            question[:60], verdict.faithfulness, verdict.citation_accuracy,
            verdict.answer_relevance, verdict.grounded,
        )
        return verdict


async def run_generation_eval(
    db_pool: asyncpg.Pool,
    dataset_path: Path = DATASET_PATH,
    concurrency: int = 3,
) -> dict[str, float]:
    """Run the generation suite over the dataset.

    Args:
        db_pool: asyncpg pool.
        dataset_path: Path to the gold dataset.
        concurrency: Max concurrent pipeline+judge runs.

    Returns:
        Aggregate metrics: mean faithfulness / citation_accuracy /
        answer_relevance, groundedness pass-rate, and judge_error_rate.
    """
    dataset = load_dataset(dataset_path)
    logger.info("Running generation eval over %d questions", len(dataset))
    if not dataset:
        return {
            "faithfulness": 0.0,
            "citation_accuracy": 0.0,
            "answer_relevance": 0.0,
            "groundedness_rate": 0.0,
            "judge_error_rate": 0.0,
            "n": 0.0,
        }

    semaphore = asyncio.Semaphore(concurrency)
    verdicts = await asyncio.gather(
        *(_answer_and_judge(row, db_pool, semaphore) for row in dataset)
    )

    n = len(verdicts)
    return {
        "faithfulness": sum(v.faithfulness for v in verdicts) / n,
        "citation_accuracy": sum(v.citation_accuracy for v in verdicts) / n,
        "answer_relevance": sum(v.answer_relevance for v in verdicts) / n,
        "groundedness_rate": sum(1 for v in verdicts if v.grounded) / n,
        "judge_error_rate": sum(1 for v in verdicts if v.error) / n,
        "n": float(n),
    }
