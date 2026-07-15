"""Query rewriting and metadata extraction via a single Claude Haiku call.

Turns the user's raw question into 3-4 search-query variants (for multi-query
RRF retrieval) and a set of structured metadata filters (ticker, fiscal year,
quarter, form type). Uses a forced tool call so the model returns reliable
structured JSON.

Degrades gracefully: any failure (API error, malformed output) falls back to
searching the original query with no filters, so /ask never fails because of the
rewrite step.
"""

import logging

from api.models import MetadataFilters, QueryRewriteResult
from core.clients import get_anthropic_client
from core.config import get_settings
from generation.prompts import QUERY_REWRITE_SYSTEM_PROMPT, QUERY_REWRITE_TOOL

logger = logging.getLogger(__name__)

MAX_TOKENS = 512
MAX_QUERIES = 4


def _fallback(query: str) -> QueryRewriteResult:
    """Build the graceful-degradation result: original query, no filters."""
    return QueryRewriteResult(queries=[query], filters=MetadataFilters())


async def rewrite_query(query: str) -> QueryRewriteResult:
    """Rewrite a user query into search variants + metadata filters.

    Args:
        query: The user's natural language question.

    Returns:
        QueryRewriteResult with 1-4 query variants and extracted MetadataFilters.
        On any failure, returns the original query with empty filters (logged as
        a warning) so retrieval can still proceed.
    """
    client = get_anthropic_client()

    try:
        # The raw-dict tool schema + tool_choice are valid at runtime but don't
        # match the SDK's strict TypedDict overloads for messages.create.
        response = await client.messages.create(  # type: ignore[call-overload]
            model=get_settings().query_rewrite_model,
            max_tokens=MAX_TOKENS,
            system=QUERY_REWRITE_SYSTEM_PROMPT,
            tools=[QUERY_REWRITE_TOOL],
            tool_choice={"type": "tool", "name": "search_plan"},
            messages=[{"role": "user", "content": query}],
        )
    except Exception as e:
        logger.warning("Query rewrite call failed, using original query: %s", e)
        return _fallback(query)

    tool_input = next(
        (block.input for block in response.content if block.type == "tool_use"),
        None,
    )
    if not tool_input:
        logger.warning("Query rewrite returned no tool_use block, using original query")
        return _fallback(query)

    try:
        raw_queries = tool_input.get("queries") or []
        queries = [q.strip() for q in raw_queries if isinstance(q, str) and q.strip()]
        queries = queries[:MAX_QUERIES] or [query]

        raw_filters = tool_input.get("filters") or {}
        filters = MetadataFilters.model_validate(raw_filters)
        if filters.ticker:
            filters.ticker = filters.ticker.upper()
        if filters.form_type:
            filters.form_type = filters.form_type.upper()
    except Exception as e:
        logger.warning("Failed to parse query rewrite output, using original query: %s", e)
        return _fallback(query)

    logger.info(
        "Rewrote query into %d variants — filters: %s",
        len(queries),
        filters.as_containment() or "none",
    )
    return QueryRewriteResult(queries=queries, filters=filters)
