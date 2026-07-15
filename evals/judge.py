"""LLM-as-judge for generation quality (Claude Sonnet 5).

Grades a generated answer against the exact context it was given, scoring
faithfulness, citation accuracy, and answer relevance. Uses a forced tool call
for reliable structured output (mirrors generation.query_rewriter and stays
compatible with the pinned anthropic SDK).

The judge is a newer model than the sonnet-4 generator it grades, keeping
verdicts independent of judge==generator self-consistency bias.
"""

import logging

from pydantic import BaseModel, Field

from core.clients import get_anthropic_client
from core.config import get_settings
from evals.prompts import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_TEMPLATE,
    format_citations,
    format_context,
)

logger = logging.getLogger(__name__)

MAX_TOKENS = 1024

_VERDICT_TOOL = {
    "name": "record_verdict",
    "description": "Record the evaluation scores for the generated answer.",
    "input_schema": {
        "type": "object",
        "properties": {
            "faithfulness": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "Is every claim supported by the context? 1-5.",
            },
            "citation_accuracy": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "Do citations point to supporting excerpts? 1-5.",
            },
            "answer_relevance": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "Does the answer address the question? 1-5.",
            },
            "grounded": {
                "type": "boolean",
                "description": "True only if the answer makes no unsupported claims.",
            },
            "rationale": {
                "type": "string",
                "description": "One to three sentences justifying the scores.",
            },
        },
        "required": [
            "faithfulness",
            "citation_accuracy",
            "answer_relevance",
            "grounded",
            "rationale",
        ],
    },
}


class JudgeVerdict(BaseModel):
    """Structured judge output for one answer.

    Attributes:
        faithfulness: 1-5, claims supported by context.
        citation_accuracy: 1-5, citations match supporting excerpts.
        answer_relevance: 1-5, answer addresses the question.
        grounded: True if no unsupported factual claims.
        rationale: Short justification.
        error: True if the judge call failed and this is a sentinel verdict.
    """

    faithfulness: int = Field(ge=1, le=5)
    citation_accuracy: int = Field(ge=1, le=5)
    answer_relevance: int = Field(ge=1, le=5)
    grounded: bool
    rationale: str
    error: bool = False


def _sentinel(reason: str) -> JudgeVerdict:
    """Build a failed-judgement verdict (lowest scores, flagged as error)."""
    return JudgeVerdict(
        faithfulness=1,
        citation_accuracy=1,
        answer_relevance=1,
        grounded=False,
        rationale=f"Judge unavailable: {reason}",
        error=True,
    )


async def judge_answer(
    question: str,
    answer: str,
    context_chunks: list,
    citations: list,
) -> JudgeVerdict:
    """Grade a generated answer against its retrieval context.

    Args:
        question: The user's question.
        answer: The generated answer text.
        context_chunks: ScoredChunks passed to the generator.
        citations: Citation objects the generator produced.

    Returns:
        A JudgeVerdict. On any failure, a sentinel low-score verdict with
        ``error=True`` (so a broken judge is visible, never silently a pass).
    """
    user_message = JUDGE_USER_TEMPLATE.format(
        question=question,
        context=format_context(context_chunks),
        answer=answer,
        citations=format_citations(citations),
    )

    client = get_anthropic_client()
    try:
        # The raw-dict tool schema + tool_choice are valid at runtime but don't
        # match the SDK's strict TypedDict overloads for messages.create.
        response = await client.messages.create(  # type: ignore[call-overload]
            model=get_settings().judge_model,
            max_tokens=MAX_TOKENS,
            system=JUDGE_SYSTEM_PROMPT,
            tools=[_VERDICT_TOOL],
            tool_choice={"type": "tool", "name": "record_verdict"},
            messages=[{"role": "user", "content": user_message}],
        )
    except Exception as e:
        logger.warning("Judge call failed: %s", e)
        return _sentinel(str(e))

    tool_input = next(
        (block.input for block in response.content if block.type == "tool_use"),
        None,
    )
    if not tool_input:
        logger.warning("Judge returned no tool_use block")
        return _sentinel("no verdict returned")

    try:
        return JudgeVerdict.model_validate(tool_input)
    except Exception as e:
        logger.warning("Failed to parse judge verdict: %s", e)
        return _sentinel(str(e))
