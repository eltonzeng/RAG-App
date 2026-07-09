"""Prompt constants for the LLM-as-judge.

Kept as module-level constants per the project convention (never hardcode prompts
inline). The judge grades a generated answer against the exact context chunks the
generator was given, so it can check faithfulness and citation accuracy.
"""

JUDGE_SYSTEM_PROMPT = """You are a strict evaluator of a retrieval-augmented \
generation system for SEC filings. You grade a generated answer against ONLY the \
context excerpts that were provided to the generator — not outside knowledge.

Score each dimension from 1 (worst) to 5 (best):

1. faithfulness — Is every factual claim in the answer directly supported by the \
provided context? Penalize any statement, figure, or inference not grounded in the \
excerpts. A graceful "I don't have enough information" answer, when the context is \
genuinely irrelevant, is faithful (score 5).

2. citation_accuracy — Do the answer's citations (source filename and page) point \
to context excerpts that actually contain the cited claim? Penalize missing \
citations for factual claims and citations that don't match the supporting excerpt. \
If the answer makes no factual claims requiring citation, score 5.

3. answer_relevance — Does the answer actually address the user's question (as \
opposed to being faithful but off-topic)?

Also set `grounded` to true only if the answer makes no unsupported factual claims \
(i.e. faithfulness is effectively 5). Give a one- to three-sentence rationale citing \
the specific excerpt numbers you relied on.

Be discriminating: reserve 5s for genuinely flawless output, and do not inflate \
scores. You must call the `record_verdict` tool with your scores."""

JUDGE_USER_TEMPLATE = """## User question
{question}

## Context excerpts provided to the generator
{context}

## Generated answer
{answer}

## Citations produced by the generator
{citations}

Grade the answer against the context excerpts above."""


def format_context(scored_chunks: list) -> str:
    """Render retrieved chunks as numbered excerpts for the judge.

    Args:
        scored_chunks: The ScoredChunk list passed to the generator.

    Returns:
        Numbered excerpt block with source labels.
    """
    parts: list[str] = []
    for i, sc in enumerate(scored_chunks, start=1):
        sources = sc.chunk.metadata.get("sources") or [{
            "source_filename": sc.chunk.metadata.get("source_filename", "Unknown"),
            "page_number": sc.chunk.metadata.get("page_number"),
        }]
        labels = []
        for src in sources:
            name = src.get("source_filename", "Unknown")
            page = src.get("page_number")
            labels.append(f"{name}, page {page}" if page else name)
        parts.append(f"[Excerpt {i} — {'; '.join(labels)}]\n{sc.chunk.content}")
    return "\n\n".join(parts) if parts else "(no context was retrieved)"


def format_citations(citations: list) -> str:
    """Render the generator's citations as a readable list for the judge.

    Args:
        citations: List of Citation objects.

    Returns:
        Bulleted citation list, or a placeholder when empty.
    """
    if not citations:
        return "(no citations produced)"
    lines = []
    for c in citations:
        page = f", page {c.page}" if c.page is not None else ""
        lines.append(f"- {c.source}{page}")
    return "\n".join(lines)
