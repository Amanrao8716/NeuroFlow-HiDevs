
"""Database connectivity and health checks."""

import asyncpg

from db.pool import get_pool


async def check_postgres() -> dict[str, str]:
    """Check PostgreSQL connectivity."""
    try:
        async with get_pool().acquire() as conn:
            await conn.fetchval("SELECT 1")

        return {"status": "healthy"}

    except (asyncpg.PostgresError, OSError, RuntimeError) as exc:
        return {
            "status": "unhealthy",
            "error": str(exc),
        }
