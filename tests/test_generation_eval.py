"""Tests for evals/generation_eval.py — fully mocked pipeline + judge.

Verifies per-question rows (with rationale + rerank_fallback) are persisted,
aggregates are computed, caveats are generated, and --pace is honored.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.models import MetadataFilters, QueryRewriteResult
from evals.generation_eval import run_generation_eval
from evals.judge import JudgeVerdict


def _verdict(**kw) -> JudgeVerdict:
    base = dict(
        faithfulness=5,
        citation_accuracy=4,
        answer_relevance=5,
        grounded=True,
        rationale="well supported",
        error=False,
    )
    base.update(kw)
    return JudgeVerdict(**base)


def _patch_pipeline(rerank_fallback: bool = False, verdict: JudgeVerdict | None = None):
    """Patch the whole generation pipeline in the eval namespace."""
    verdict = verdict or _verdict()
    return (
        patch(
            "evals.generation_eval.rewrite_query",
            new=AsyncMock(
                return_value=QueryRewriteResult(queries=["q"], filters=MetadataFilters())
            ),
        ),
        patch("evals.generation_eval.hybrid_retrieve", new=AsyncMock(return_value=[])),
        patch(
            "evals.generation_eval.rerank",
            new=AsyncMock(return_value=([], True, rerank_fallback)),
        ),
        patch("evals.generation_eval.generate", new=AsyncMock(return_value=("answer", []))),
        patch("evals.generation_eval.judge_answer", new=AsyncMock(return_value=verdict)),
        patch(
            "evals.generation_eval.load_dataset",
            return_value=[{"id": "q1", "question": "What is revenue?"}],
        ),
    )


class TestRunGenerationEval:
    @pytest.mark.asyncio
    async def test_persists_rows_with_rationale(self) -> None:
        patchers = _patch_pipeline()
        with patchers[0], patchers[1], patchers[2], patchers[3], patchers[4], patchers[5]:
            out = await run_generation_eval(MagicMock(), concurrency=1)

        assert out["aggregates"]["faithfulness"] == 5
        assert out["aggregates"]["n"] == 1
        row = out["rows"][0]
        assert row["id"] == "q1"
        assert row["rationale"] == "well supported"
        assert row["rerank_fallback"] is False

    @pytest.mark.asyncio
    async def test_fallback_generates_caveat(self) -> None:
        patchers = _patch_pipeline(rerank_fallback=True)
        with patchers[0], patchers[1], patchers[2], patchers[3], patchers[4], patchers[5]:
            out = await run_generation_eval(MagicMock(), concurrency=1)

        assert out["rows"][0]["rerank_fallback"] is True
        assert any("rerank fallback" in c for c in out["caveats"])

    @pytest.mark.asyncio
    async def test_judge_error_generates_caveat(self) -> None:
        patchers = _patch_pipeline(verdict=_verdict(error=True, faithfulness=1))
        with patchers[0], patchers[1], patchers[2], patchers[3], patchers[4], patchers[5]:
            out = await run_generation_eval(MagicMock(), concurrency=1)

        assert out["aggregates"]["judge_error_rate"] == 1.0
        assert any("judge error" in c for c in out["caveats"])

    @pytest.mark.asyncio
    async def test_pace_is_awaited(self) -> None:
        patchers = _patch_pipeline()
        with (
            patchers[0],
            patchers[1],
            patchers[2],
            patchers[3],
            patchers[4],
            patchers[5],
            patch("evals.generation_eval.asyncio.sleep", new=AsyncMock()) as sleep,
        ):
            await run_generation_eval(MagicMock(), concurrency=1, pace=8.0)

        sleep.assert_awaited_with(8.0)

    @pytest.mark.asyncio
    async def test_empty_dataset(self) -> None:
        with patch("evals.generation_eval.load_dataset", return_value=[]):
            out = await run_generation_eval(MagicMock())
        assert out["rows"] == []
        assert out["aggregates"]["n"] == 0
