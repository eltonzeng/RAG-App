"""Database pool for offline eval runs.

Reuses api.main.create_pool so the harness registers the same pgvector codec and
reads DATABASE_URL the same way the API does.
"""

import asyncpg

from api.main import create_pool


async def open_pool() -> asyncpg.Pool:
    """Open an asyncpg pool sized for a short-lived eval run.

    Returns:
        An initialized asyncpg connection pool (caller is responsible for close).
    """
    return await create_pool(min_size=1, max_size=5)
