"""Answer generation using Anthropic Claude with citation extraction.

Builds a prompt from reranked chunks, streams the response from Claude,
and extracts structured citations from chunk metadata.
"""

import logging
import re
import time

import anthropic

from api.models import Citation, ScoredChunk
from generation.prompts import (
    NO_RELEVANT_CONTENT_RESPONSE,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    build_context_block,
)

logger = logging.getLogger(__name__)

GENERATION_MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 1024


def _extract_citations(scored_chunks: list[ScoredChunk]) -> list[Citation]:
    """Build citation objects from the chunks that were passed to the generator.

    Args:
        scored_chunks: The reranked chunks used as context for generation.

    Returns:
        Deduplicated list of Citation objects derived from chunk metadata.
    """
    seen: set[str] = set()
    citations: list[Citation] = []

    for sc in scored_chunks:
        chunk = sc.chunk
        # A deduplicated chunk may appear in several filings; emit one citation
        # per distinct (source, page). Fall back to top-level metadata if the
        # sources array is absent (e.g. legacy/mocked chunks).
        sources = chunk.metadata.get("sources") or [{
            "source_filename": chunk.metadata.get("source_filename", "Unknown"),
            "page_number": chunk.metadata.get("page_number"),
        }]

        for src in sources:
            source = src.get("source_filename", "Unknown")
            page = src.get("page_number")
            key = f"{source}:{page}"
            if key not in seen:
                seen.add(key)
                citations.append(Citation(source=source, page=page, chunk_id=chunk.id))

    return citations


async def generate(
    query: str,
    scored_chunks: list[ScoredChunk],
    is_relevant: bool,
) -> tuple[str, list[Citation]]:
    """Generate an answer from context chunks using Claude.

    If no relevant content was found (is_relevant=False), returns a
    graceful "not enough information" response without calling the API.

    Streams the Anthropic response and collects it into a single string
    for structured API return. SSE streaming is a planned future enhancement.

    Args:
        query: The user's natural language question.
        scored_chunks: Reranked chunks to use as context.
        is_relevant: Whether any chunk exceeded the relevance threshold.

    Returns:
        Tuple of (answer_text, citations).

    Raises:
        anthropic.APIError: If the Anthropic API call fails.
    """
    if not is_relevant or not scored_chunks:
        logger.info("No relevant content found — returning graceful fallback response")
        return NO_RELEVANT_CONTENT_RESPONSE, []

    start_time = time.perf_counter()
    context_block = build_context_block(scored_chunks)
    user_message = USER_PROMPT_TEMPLATE.format(
        context=context_block,
        query=query,
    )

    client = anthropic.AsyncAnthropic()
    answer_parts: list[str] = []

    try:
        async with client.messages.stream(
            model=GENERATION_MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            async for text in stream.text_stream:
                answer_parts.append(text)

        answer = "".join(answer_parts)
        elapsed = time.perf_counter() - start_time

        logger.info(
            "Generation complete: %d chars in %.2fs using %d context chunks",
            len(answer), elapsed, len(scored_chunks),
        )

    except anthropic.APIError as e:
        logger.error("Anthropic API error during generation: %s", e)
        raise

    citations = _extract_citations(scored_chunks)
    return answer, citations
