"""Tests for core.config — proves the official run is a pure env change.

If these pass, swapping generation/judge models (or any knob) for the final
metric run requires only environment variables, never a code edit.
"""

import pytest

from core.config import Settings, get_settings


def test_defaults_are_dev_models() -> None:
    """Out of the box, the cost-conscious dev models are selected."""
    settings = Settings()
    assert settings.generation_model == "claude-sonnet-4-6"
    assert settings.judge_model == "claude-sonnet-5"
    assert settings.embedding_model == "text-embedding-3-small"


def test_env_overrides_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """The official-run overrides take effect from the environment alone."""
    monkeypatch.setenv("GENERATION_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("JUDGE_MODEL", "claude-opus-4-8")
    get_settings.cache_clear()

    settings = get_settings()
    assert settings.generation_model == "claude-sonnet-5"
    assert settings.judge_model == "claude-opus-4-8"


def test_env_overrides_reliability_knobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Timeouts and retry budgets are env-tunable with correct types."""
    monkeypatch.setenv("OPENAI_TIMEOUT_S", "12.5")
    monkeypatch.setenv("ANTHROPIC_MAX_RETRIES", "5")
    get_settings.cache_clear()

    settings = get_settings()
    assert settings.openai_timeout_s == 12.5
    assert settings.anthropic_max_retries == 5


def test_get_settings_is_cached() -> None:
    """get_settings returns the same instance until the cache is cleared."""
    assert get_settings() is get_settings()


def test_cors_origin_list_splits_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A comma-separated CORS_ORIGINS env value parses into a clean list."""
    monkeypatch.setenv("CORS_ORIGINS", "http://a.com, http://b.com ,")
    get_settings.cache_clear()

    assert get_settings().cors_origin_list == ["http://a.com", "http://b.com"]
