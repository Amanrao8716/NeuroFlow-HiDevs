
"""Manage the asynchronous PostgreSQL connection pool."""

from contextlib import asynccontextmanager
from typing import AsyncIterator
from uuid import UUID

import asyncpg

from config import settings


_pool: asyncpg.Pool | None = None


async def open_pool() -> None:
    """Create the shared connection pool at startup."""
    global _pool

    if _pool is None:
        _pool = await asyncpg.create_pool(
            host=settings.postgres_host,
            port=settings.postgres_port,
            database=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
            min_size=2,
            max_size=10,
            command_timeout=30,
        )


async def close_pool() -> None:
    """Close the shared connection pool at shutdown."""
    global _pool

    if _pool is not None:
        await _pool.close()
        _pool = None


def get_pool() -> asyncpg.Pool:
    """Return the initialized pool."""
    if _pool is None:
        raise RuntimeError("PostgreSQL pool has not been initialized.")
    return _pool


@asynccontextmanager
async def connection() -> AsyncIterator[asyncpg.Connection]:
    """Acquire and release a database connection."""
    async with get_pool().acquire() as conn:
        yield conn


@asynccontextmanager
async def pipeline_connection(
    pipeline_id: UUID,
) -> AsyncIterator[asyncpg.Connection]:
    """Use a transaction-scoped pipeline context for RLS."""
    async with get_pool().acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "SELECT set_config('app.pipeline_id', $1, true)",
                str(pipeline_id),
            )
            yield conn
