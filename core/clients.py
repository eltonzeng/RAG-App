"""Shared singleton API clients with enforced timeouts and bounded retries.

Modules must obtain external clients from these accessors instead of
constructing their own. This guarantees:

- every request carries a deadline (a stalled provider connection can no
  longer hang a pipeline run indefinitely),
- retry budgets are bounded and configured in exactly one place,
- httpx connection pools are created once and reused (per-call construction
  leaks pools and re-pays TCP/TLS setup on every request),
- tests patch one accessor rather than SDK constructors in many modules.

Accessors are lazy (``lru_cache``) so importing this module never requires
API keys — construction happens on first use.
"""

from functools import lru_cache

import anthropic
import cohere
from openai import AsyncOpenAI

from core.config import get_settings


@lru_cache
def get_openai_client() -> AsyncOpenAI:
    """Return the shared AsyncOpenAI client (embeddings).

    Returns:
        AsyncOpenAI configured with the settings timeout and retry budget.
    """
    settings = get_settings()
    return AsyncOpenAI(
        timeout=settings.openai_timeout_s,
        max_retries=settings.openai_max_retries,
    )


@lru_cache
def get_anthropic_client() -> anthropic.AsyncAnthropic:
    """Return the shared AsyncAnthropic client (generation, rewrite, judge).

    Returns:
        AsyncAnthropic configured with the settings timeout and retry budget.
    """
    settings = get_settings()
    return anthropic.AsyncAnthropic(
        timeout=settings.anthropic_timeout_s,
        max_retries=settings.anthropic_max_retries,
    )


@lru_cache
def get_cohere_client() -> cohere.AsyncClient:
    """Return the shared Cohere AsyncClient (reranking).

    Returns:
        cohere.AsyncClient configured with the settings timeout.
    """
    settings = get_settings()
    return cohere.AsyncClient(timeout=settings.cohere_timeout_s)
