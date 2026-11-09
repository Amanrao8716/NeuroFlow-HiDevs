"""Daily data retention job for automated cleanup of old pipeline data.

Deletes:
  - Completed pipeline_runs older than ``retention_run_days`` (default 90)
    that have no associated evaluation record.
  - Evaluation records older than ``retention_eval_days`` (default 180).
  - Chunks belonging to documents with status 'archived'.

All operations run inside transactions, batch-delete in groups of 1000,
and acquire a Redis distributed lock to prevent overlap across replicas.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

try:
    from config import settings
except ImportError:
    from backend.config import settings

logger = logging.getLogger("neuroflow.tasks.retention")

BATCH_SIZE = 1000
LOCK_KEY = "neuroflow:lock:retention"
LOCK_TTL_SECONDS = 3600  # 1 hour max


async def _acquire_lock(redis_client: object) -> bool:
    """Acquire a distributed Redis lock via SET NX EX."""
    if redis_client is None:
        # No Redis — allow execution (single-instance mode).
        return True
    try:
        acquired = await redis_client.set(  # type: ignore[attr-defined]
            LOCK_KEY,
            datetime.now(UTC).isoformat(),
            nx=True,
            ex=LOCK_TTL_SECONDS,
        )
        return bool(acquired)
    except Exception:
        logger.warning("Redis lock acquisition failed; proceeding without lock.")
        return True


async def _release_lock(redis_client: object) -> None:
    """Release the distributed retention lock."""
    if redis_client is None:
        return
    try:
        await redis_client.delete(LOCK_KEY)  # type: ignore[attr-defined]
    except Exception:
        logger.warning("Failed to release Redis retention lock.")


async def _execute_in_tx(conn: object, query: str, *args: object) -> str:
    """Execute a query wrapped in an async transaction if supported."""
    tx_fn = getattr(conn, "transaction", None)
    if tx_fn is not None and callable(tx_fn):
        tx = tx_fn()
        if hasattr(tx, "__aenter__"):
            async with tx:
                return str(await conn.execute(query, *args))  # type: ignore[attr-defined]
    return str(await conn.execute(query, *args))  # type: ignore[attr-defined]


async def _delete_old_runs(conn: object, cutoff: datetime) -> int:
    """Delete completed pipeline_runs with no evaluation, older than cutoff."""
    total_deleted = 0
    while True:
        result = await _execute_in_tx(
            conn,
            """
            DELETE FROM pipeline_runs
            WHERE id IN (
                SELECT r.id
                FROM pipeline_runs r
                LEFT JOIN evaluations e ON e.run_id = r.id
                WHERE r.status = 'completed'
                  AND r.created_at < $1
                  AND e.id IS NULL
                LIMIT $2
            )
            """,
            cutoff,
            BATCH_SIZE,
        )
        count = int(result.split()[-1]) if result and " " in result else 0
        total_deleted += count
        if count < BATCH_SIZE:
            break
    return total_deleted


async def _delete_old_evaluations(conn: object, cutoff: datetime) -> int:
    """Delete evaluation records older than cutoff."""
    total_deleted = 0
    while True:
        result = await _execute_in_tx(
            conn,
            """
            DELETE FROM evaluations
            WHERE id IN (
                SELECT id FROM evaluations
                WHERE evaluated_at < $1
                LIMIT $2
            )
            """,
            cutoff,
            BATCH_SIZE,
        )
        count = int(result.split()[-1]) if result and " " in result else 0
        total_deleted += count
        if count < BATCH_SIZE:
            break
    return total_deleted


async def _delete_archived_chunks(conn: object) -> int:
    """Delete chunks belonging to documents with status='archived'."""
    total_deleted = 0
    while True:
        result = await _execute_in_tx(
            conn,
            """
            DELETE FROM chunks
            WHERE id IN (
                SELECT c.id
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE d.status = 'archived'
                LIMIT $1
            )
            """,
            BATCH_SIZE,
        )
        count = int(result.split()[-1]) if result and " " in result else 0
        total_deleted += count
        if count < BATCH_SIZE:
            break
    return total_deleted


async def run_retention(
    conn: object = None,
    redis_client: object = None,
    *,
    run_days: int | None = None,
    eval_days: int | None = None,
) -> dict[str, int]:
    """Execute the full retention sweep.

    Args:
        conn: An asyncpg connection or pool. If None, uses ``connection()`` from ``db.pool``.
        redis_client: Optional Redis client for distributed locking.
        run_days: Override retention_run_days (default from settings).
        eval_days: Override retention_eval_days (default from settings).

    Returns:
        Dict with counts: ``runs_deleted``, ``evaluations_deleted``, ``chunks_deleted``.
    """
    run_days = run_days if run_days is not None else getattr(settings, "retention_run_days", 90)
    eval_days = (
        eval_days if eval_days is not None else getattr(settings, "retention_eval_days", 180)
    )

    if not await _acquire_lock(redis_client):
        logger.info("Retention job skipped — another instance holds the lock.")
        return {"runs_deleted": 0, "evaluations_deleted": 0, "chunks_deleted": 0}

    async def _execute_with_conn(c: object) -> dict[str, int]:
        now = datetime.now(UTC)
        run_cutoff = now - timedelta(days=run_days)
        eval_cutoff = now - timedelta(days=eval_days)

        logger.info(
            "Starting retention sweep: runs older than %d days, evals older than %d days",
            run_days,
            eval_days,
        )

        runs_del = await _delete_old_runs(c, run_cutoff)
        evals_del = await _delete_old_evaluations(c, eval_cutoff)
        chunks_del = await _delete_archived_chunks(c)

        logger.info(
            "Retention complete: %d runs, %d evaluations, %d chunks deleted",
            runs_del,
            evals_del,
            chunks_del,
        )
        return {
            "runs_deleted": runs_del,
            "evaluations_deleted": evals_del,
            "chunks_deleted": chunks_del,
        }

    try:
        if conn is not None:
            return await _execute_with_conn(conn)
        else:
            from db.pool import connection

            async with connection() as active_conn:
                return await _execute_with_conn(active_conn)
    finally:
        await _release_lock(redis_client)


async def retention_worker_task(ctx: dict[str, object]) -> dict[str, int]:
    """ARQ worker task entry point."""
    redis_client = ctx.get("redis")
    return await run_retention(redis_client=redis_client)
