"""Unit tests for automated data retention job."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from backend.tasks.retention import (
    BATCH_SIZE,
    _acquire_lock,
    _delete_archived_chunks,
    _delete_old_evaluations,
    _delete_old_runs,
    _execute_in_tx,
    _release_lock,
    run_retention,
)


def make_mock_conn():
    """Create a mock connection with async transaction support."""
    conn = AsyncMock()
    tx = MagicMock()
    tx.__aenter__ = AsyncMock(return_value=None)
    tx.__aexit__ = AsyncMock(return_value=None)
    conn.transaction = MagicMock(return_value=tx)
    return conn


@pytest.mark.asyncio
async def test_acquire_and_release_lock():
    """Verify Redis distributed lock behavior."""
    redis = AsyncMock()
    redis.set.return_value = True

    acquired = await _acquire_lock(redis)
    assert acquired is True
    redis.set.assert_called_once()
    assert redis.set.call_args[1].get("nx") is True

    await _release_lock(redis)
    redis.delete.assert_called_once()


@pytest.mark.asyncio
async def test_acquire_lock_failure():
    """Verify lock acquisition returns False when lock is held."""
    redis = AsyncMock()
    redis.set.return_value = False

    acquired = await _acquire_lock(redis)
    assert acquired is False


@pytest.mark.asyncio
async def test_acquire_lock_none_redis():
    """Verify fallback when Redis is None (single-instance)."""
    assert await _acquire_lock(None) is True
    await _release_lock(None)


@pytest.mark.asyncio
async def test_execute_in_tx_with_and_without_tx():
    """Verify _execute_in_tx handles connections with and without transaction method."""
    conn_with_tx = make_mock_conn()
    conn_with_tx.execute.return_value = "DELETE 1"
    res1 = await _execute_in_tx(conn_with_tx, "DELETE FROM foo")
    assert res1 == "DELETE 1"
    conn_with_tx.transaction.assert_called_once()

    conn_no_tx = AsyncMock()
    del conn_no_tx.transaction
    conn_no_tx.execute.return_value = "DELETE 2"
    res2 = await _execute_in_tx(conn_no_tx, "DELETE FROM bar")
    assert res2 == "DELETE 2"


@pytest.mark.asyncio
async def test_delete_old_runs_batches():
    """Verify batch deletion of old pipeline runs until fewer than BATCH_SIZE."""
    conn = make_mock_conn()
    # First batch deletes BATCH_SIZE (1000), second deletes 50 (last batch)
    conn.execute.side_effect = [f"DELETE {BATCH_SIZE}", "DELETE 50"]

    cutoff = datetime.now(UTC) - timedelta(days=90)
    total = await _delete_old_runs(conn, cutoff)

    assert total == BATCH_SIZE + 50
    assert conn.execute.call_count == 2


@pytest.mark.asyncio
async def test_retention_safety_sql_constraints():
    """Verify SQL predicates strictly enforce retention safety rules."""
    conn = make_mock_conn()
    conn.execute.return_value = "DELETE 0"

    cutoff_runs = datetime.now(UTC) - timedelta(days=90)
    await _delete_old_runs(conn, cutoff_runs)
    run_sql = conn.execute.call_args[0][0]

    # Completed runs only (non-completed retained)
    assert "r.status = 'completed'" in run_sql
    # Cutoff check (recent runs retained)
    assert "r.created_at < $1" in run_sql
    # No evaluation check (runs with evaluations retained)
    assert "e.id IS NULL" in run_sql

    conn.execute.reset_mock()
    cutoff_evals = datetime.now(UTC) - timedelta(days=180)
    await _delete_old_evaluations(conn, cutoff_evals)
    eval_sql = conn.execute.call_args[0][0]
    # Cutoff check (recent evaluations retained)
    assert "evaluated_at < $1" in eval_sql

    conn.execute.reset_mock()
    await _delete_archived_chunks(conn)
    chunk_sql = conn.execute.call_args[0][0]
    # Chunks deleted ONLY when document status is 'archived'
    assert "d.status = 'archived'" in chunk_sql
    assert "p.status = 'archived'" not in chunk_sql


@pytest.mark.asyncio
async def test_delete_old_evaluations_batches():
    """Verify batch deletion of old evaluation records."""
    conn = make_mock_conn()
    conn.execute.side_effect = ["DELETE 25"]

    cutoff = datetime.now(UTC) - timedelta(days=180)
    total = await _delete_old_evaluations(conn, cutoff)

    assert total == 25
    assert conn.execute.call_count == 1


@pytest.mark.asyncio
async def test_delete_archived_chunks():
    """Verify deletion of chunks belonging to archived documents/pipelines."""
    conn = make_mock_conn()
    conn.execute.side_effect = ["DELETE 120"]

    total = await _delete_archived_chunks(conn)
    assert total == 120
    assert conn.execute.call_count == 1


@pytest.mark.asyncio
async def test_run_retention_orchestration():
    """Verify full retention run when lock is acquired."""
    conn = make_mock_conn()
    conn.execute.side_effect = [
        "DELETE 10",   # runs
        "DELETE 5",    # evals
        "DELETE 20",   # chunks
    ]

    redis = AsyncMock()
    redis.set.return_value = True

    result = await run_retention(
        conn=conn,
        redis_client=redis,
        run_days=90,
        eval_days=180,
    )

    assert result == {
        "runs_deleted": 10,
        "evaluations_deleted": 5,
        "chunks_deleted": 20,
    }
    redis.delete.assert_called_once()


@pytest.mark.asyncio
async def test_run_retention_skips_when_locked():
    """Verify retention skips execution when lock cannot be acquired."""
    conn = make_mock_conn()
    redis = AsyncMock()
    redis.set.return_value = False

    result = await run_retention(
        conn=conn,
        redis_client=redis,
    )

    assert result == {
        "runs_deleted": 0,
        "evaluations_deleted": 0,
        "chunks_deleted": 0,
    }
    conn.execute.assert_not_called()
