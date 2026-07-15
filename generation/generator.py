"""Answer generation using Anthropic Claude with citation extraction.

Builds a prompt from reranked chunks, streams the response from Claude,
and extracts structured citations from chunk metadata.
"""

import logging
import time
from collections.abc import AsyncIterator

import anthropic

from api.models import Citation, ScoredChunk
from core.clients import get_anthropic_client
from core.config import get_settings
from generation.prompts import (
    NO_RELEVANT_CONTENT_RESPONSE,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    build_context_block,
    relevant_sources,
)

logger = logging.getLogger(__name__)

MAX_TOKENS = 1024


def extract_citations(
    scored_chunks: list[ScoredChunk],
    filters: dict | None = None,
) -> list[Citation]:
    """Build citation objects from the chunks that were passed to the generator.

    Args:
        scored_chunks: The reranked chunks used as context for generation.
        filters: Metadata filters applied at retrieval time. When set, a
            deduplicated chunk's sources are narrowed to those matching the
            filter so we never cite a filing the user filtered out.

    Returns:
        Deduplicated list of Citation objects derived from chunk metadata.
    """
    seen: set[str] = set()
    citations: list[Citation] = []

    for sc in scored_chunks:
        chunk = sc.chunk
        # A deduplicated chunk may appear in several filings; emit one citation
        # per distinct (source, page), narrowed to the applied filter. Fall back
        # to top-level metadata if the sources array is absent (e.g. legacy/
        # mocked chunks).
        raw_sources = chunk.metadata.get("sources")
        if raw_sources:
            sources = relevant_sources(raw_sources, filters)
        else:
            sources = [
                {
                    "source_filename": chunk.metadata.get("source_filename", "Unknown"),
                    "page_number": chunk.metadata.get("page_number"),
                }
            ]

        for src in sources:
            source = src.get("source_filename", "Unknown")
            page = src.get("page_number")
            key = f"{source}:{page}"
            if key not in seen:
                seen.add(key)
                citations.append(Citation(source=source, page=page, chunk_id=chunk.id))

    return citations


async def generate_stream(
    query: str,
    scored_chunks: list[ScoredChunk],
    is_relevant: bool,
    filters: dict | None = None,
) -> AsyncIterator[str]:
    """Stream the answer text token-by-token from Claude.

    Yields incremental text deltas as they arrive. If no relevant content was
    found (is_relevant=False), yields the graceful "not enough information"
    message as a single delta and makes no API call. This is the single source
    of generation logic; ``generate`` collects it for the non-streaming path.

    Args:
        query: The user's natural language question.
        scored_chunks: Reranked chunks to use as context.
        is_relevant: Whether any chunk exceeded the relevance threshold.
        filters: Metadata filters applied at retrieval time (narrow the context
            labels to the matching filings).

    Yields:
        Text deltas of the answer, in order.

    Raises:
        anthropic.APIError: If the Anthropic API call fails.
    """
    if not is_relevant or not scored_chunks:
        logger.info("No relevant content found — returning graceful fallback response")
        yield NO_RELEVANT_CONTENT_RESPONSE
        return

    start_time = time.perf_counter()
    context_block = build_context_block(scored_chunks, filters)
    user_message = USER_PROMPT_TEMPLATE.format(context=context_block, query=query)

    client = get_anthropic_client()
    char_count = 0

    try:
        async with client.messages.stream(
            model=get_settings().generation_model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            async for text in stream.text_stream:
                char_count += len(text)
                yield text
    except anthropic.APIError as e:
        logger.error("Anthropic API error during generation: %s", e)
        raise

    elapsed = time.perf_counter() - start_time
    logger.info(
        "Generation complete: %d chars in %.2fs using %d context chunks",
        char_count,
        elapsed,
        len(scored_chunks),
    )


async def generate(
    query: str,
    scored_chunks: list[ScoredChunk],
    is_relevant: bool,
    filters: dict | None = None,
) -> tuple[str, list[Citation]]:
    """Generate an answer and citations (non-streaming collector).

    Collects ``generate_stream`` into a single string for the JSON /ask contract.

    Args:
        query: The user's natural language question.
        scored_chunks: Reranked chunks to use as context.
        is_relevant: Whether any chunk exceeded the relevance threshold.
        filters: Metadata filters applied at retrieval time. When set, each
            deduplicated chunk's sources are narrowed to the matching filings for
            both the context labels and the emitted citations.

    Returns:
        Tuple of (answer_text, citations). Citations are empty when no relevant
        content was found.

    Raises:
        anthropic.APIError: If the Anthropic API call fails.
    """
    parts = [text async for text in generate_stream(query, scored_chunks, is_relevant, filters)]
    answer = "".join(parts)
    citations = extract_citations(scored_chunks, filters) if (is_relevant and scored_chunks) else []
    return answer, citations
