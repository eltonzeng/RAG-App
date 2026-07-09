"""Tests for generation/query_rewriter.py.

Mocks the Anthropic client so no live API call is made.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from generation.query_rewriter import rewrite_query


def _tool_use_response(tool_input: dict) -> SimpleNamespace:
    """Build a fake Anthropic response containing a single tool_use block."""
    block = SimpleNamespace(type="tool_use", name="search_plan", input=tool_input)
    return SimpleNamespace(content=[block])


def _patched_client(create_mock: AsyncMock):
    """Patch anthropic.AsyncAnthropic to return a client with a mocked create."""
    client = SimpleNamespace(messages=SimpleNamespace(create=create_mock))
    return patch(
        "generation.query_rewriter.anthropic.AsyncAnthropic",
        return_value=client,
    )


class TestRewriteQuery:
    @pytest.mark.asyncio
    async def test_parses_queries_and_filters(self) -> None:
        """A well-formed tool call yields variants + normalized filters."""
        create = AsyncMock(return_value=_tool_use_response({
            "queries": ["apple revenue 2023", "AAPL total net sales fiscal 2023"],
            "filters": {"ticker": "aapl", "fiscal_year": 2023, "quarter": None,
                        "form_type": "10-k"},
        }))
        with _patched_client(create):
            result = await rewrite_query("What was Apple's revenue in 2023?")

        assert result.queries == ["apple revenue 2023", "AAPL total net sales fiscal 2023"]
        assert result.filters.ticker == "AAPL"          # uppercased
        assert result.filters.form_type == "10-K"       # uppercased
        assert result.filters.fiscal_year == 2023
        assert result.filters.quarter is None
        assert result.filters.as_containment() == {
            "ticker": "AAPL", "fiscal_year": 2023, "form_type": "10-K",
        }

    @pytest.mark.asyncio
    async def test_caps_queries_at_four(self) -> None:
        """More than four variants are trimmed to four."""
        create = AsyncMock(return_value=_tool_use_response({
            "queries": ["q1", "q2", "q3", "q4", "q5", "q6"],
            "filters": {"ticker": None, "fiscal_year": None, "quarter": None,
                        "form_type": None},
        }))
        with _patched_client(create):
            result = await rewrite_query("some question")

        assert result.queries == ["q1", "q2", "q3", "q4"]
        assert result.filters.as_containment() == {}

    @pytest.mark.asyncio
    async def test_falls_back_on_api_error(self) -> None:
        """Any API failure falls back to the original query, no filters."""
        create = AsyncMock(side_effect=RuntimeError("network down"))
        with _patched_client(create):
            result = await rewrite_query("original question")

        assert result.queries == ["original question"]
        assert result.filters.as_containment() == {}

    @pytest.mark.asyncio
    async def test_falls_back_when_no_tool_use(self) -> None:
        """A response without a tool_use block falls back to the original query."""
        text_only = SimpleNamespace(content=[SimpleNamespace(type="text", text="hi")])
        create = AsyncMock(return_value=text_only)
        with _patched_client(create):
            result = await rewrite_query("original question")

        assert result.queries == ["original question"]
        assert result.filters.as_containment() == {}

    @pytest.mark.asyncio
    async def test_empty_queries_falls_back_to_original(self) -> None:
        """An empty queries array falls back to the original query string."""
        create = AsyncMock(return_value=_tool_use_response({
            "queries": [],
            "filters": {"ticker": None, "fiscal_year": None, "quarter": None,
                        "form_type": None},
        }))
        with _patched_client(create):
            result = await rewrite_query("original question")

        assert result.queries == ["original question"]
