"""Centralized application settings (pydantic-settings).

Every tunable — model names, client timeouts/retries, thresholds, service
endpoints — lives here and is overridable via environment variables (or .env).
This is what makes an eval run against different models a pure configuration
change: e.g. ``GENERATION_MODEL=claude-sonnet-5 JUDGE_MODEL=claude-opus-4-8``.

Settings are read lazily through ``get_settings()`` (not at import time) so
test environments can inject fake values and CI can run without any keys.
"""

from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration, sourced from the environment and .env.

    Attributes:
        embedding_model: OpenAI embedding model for ingest and retrieval.
        generation_model: Claude model that answers user questions.
        query_rewrite_model: Claude model for multi-query rewrite + filters.
        rerank_model: Cohere rerank model.
        judge_model: Claude model grading answers in the generation eval.
        openai_timeout_s: Per-request timeout for OpenAI calls (seconds).
        openai_max_retries: SDK-level retry budget for OpenAI calls.
        anthropic_timeout_s: Per-request timeout for Anthropic calls (seconds).
        anthropic_max_retries: SDK-level retry budget for Anthropic calls.
        cohere_timeout_s: Per-request timeout for Cohere rerank calls (seconds).
        relevance_threshold: Min Cohere rerank score to treat content as
            relevant; below it the API returns a graceful "no info" message.
        pdf_extract_tables: When True, detect tables during PDF ingest and
            render them as Markdown to preserve alignment for numeric queries.
        database_url: asyncpg DSN for the ParadeDB/pgvector store.
        rag_api_key: When set, /ask and /ingest require a matching X-API-Key
            header; when unset (default), the API is open for local dev.
        cors_origins: Comma-separated list of allowed CORS origins.
        log_format: "text" for local dev, "json" for structured (Docker) logs.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Models
    embedding_model: str = "text-embedding-3-small"
    generation_model: str = "claude-sonnet-4-6"
    query_rewrite_model: str = "claude-haiku-4-5-20251001"
    rerank_model: str = "rerank-english-v3.0"
    judge_model: str = "claude-sonnet-5"

    # Client reliability (every external call gets a deadline and a bounded
    # retry budget — an unresponsive provider must never hang the pipeline)
    openai_timeout_s: float = 30.0
    openai_max_retries: int = 3
    anthropic_timeout_s: float = 60.0
    anthropic_max_retries: int = 2
    cohere_timeout_s: float = 15.0

    # Pipeline behavior
    relevance_threshold: float = 0.3
    # Detect tables in PDFs and render them as Markdown (preserves column/row
    # alignment for numeric queries). Disable for plain-text-only extraction.
    pdf_extract_tables: bool = True

    # Services
    database_url: str = "postgresql://raguser:ragpass@localhost:5434/ragdb"

    # API surface
    rag_api_key: str | None = None
    cors_origins: str = "http://localhost:8501"
    log_format: str = "text"

    @property
    def cors_origin_list(self) -> list[str]:
        """CORS origins as a list (env value is comma-separated)."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide Settings singleton.

    Loads .env into the process environment first so SDK clients that read
    their API keys directly from ``os.environ`` (OpenAI/Anthropic/Cohere)
    see them regardless of which entrypoint (API, UI, evals) started up.

    Returns:
        The cached Settings instance. Tests can call
        ``get_settings.cache_clear()`` after mutating the environment.
    """
    load_dotenv()
    return Settings()
