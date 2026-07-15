"""Shared pytest fixtures.

Keeps the suite hermetic: fake API keys and a dummy DATABASE_URL are injected
into the environment so nothing ever reaches a live provider or database, and
the cached settings/client singletons are reset around every test so an
env override in one test can't leak into the next.
"""

from collections.abc import Iterator

import pytest

from core import clients, config


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Set fake secrets/config and clear cached singletons for each test."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("COHERE_API_KEY", "test-cohere")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:5432/testdb")

    config.get_settings.cache_clear()
    clients.get_openai_client.cache_clear()
    clients.get_anthropic_client.cache_clear()
    clients.get_cohere_client.cache_clear()
    yield
    config.get_settings.cache_clear()
    clients.get_openai_client.cache_clear()
    clients.get_anthropic_client.cache_clear()
    clients.get_cohere_client.cache_clear()
