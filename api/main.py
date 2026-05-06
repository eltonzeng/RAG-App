"""FastAPI application entry point.

Manages the asyncpg connection pool via lifespan and registers all routes.
The pool is stored on app.state so routes can access it via request.app.state.pool.
"""

import logging
from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI

from api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# These are imported lazily in routes to avoid circular deps,
# but we expose them here as single source of truth.
EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "claude-sonnet-4-20250514"


def _get_database_url() -> str:
    """Read DATABASE_URL from environment, raising clearly if missing.

    Returns:
        The database connection string.

    Raises:
        RuntimeError: If DATABASE_URL is not set.
    """
    import os
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    return url


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage asyncpg pool lifecycle tied to FastAPI app startup/shutdown.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control to the application while the pool is active.
    """
    from dotenv import load_dotenv
    load_dotenv()

    db_url = _get_database_url()

    # asyncpg requires the scheme to be postgresql:// not postgres://
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    logger.info("Creating database connection pool")
    pool = await asyncpg.create_pool(
        dsn=db_url,
        min_size=2,
        max_size=10,
        command_timeout=30,
        init=_init_connection,
    )
    app.state.pool = pool
    logger.info("Database pool created successfully")

    yield

    logger.info("Closing database connection pool")
    await pool.close()
    logger.info("Database pool closed")


async def _init_connection(conn: asyncpg.Connection) -> None:
    """Register pgvector codec on each new connection.

    Args:
        conn: The newly created asyncpg connection.
    """
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await conn.set_type_codec(
        "vector",
        encoder=lambda v: str(v),
        decoder=lambda v: [float(x) for x in v.strip("[]").split(",")],
        schema="pg_catalog",
        format="text",
    )


app = FastAPI(
    title="SEC Filings RAG API",
    description="Retrieval-augmented generation over SEC EDGAR filings",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
