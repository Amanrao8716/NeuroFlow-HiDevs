"""FastAPI endpoints for Task 7 automated evaluation and human rating calibration."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncGenerator
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

try:
    from db.pool import connection, pipeline_connection
except ImportError:
    connection = None
    pipeline_connection = None

from backend.security.auth import require_scope

router = APIRouter(tags=["evaluations"], dependencies=[Depends(require_scope("query"))])
logger = logging.getLogger("neuroflow.api.evaluations")


class RatingRequest(BaseModel):
    """Payload for human rating submission."""

    rating: int = Field(..., ge=1, le=5, description="Human evaluation rating from 1 to 5")
    pipeline_id: UUID | None = Field(
        default=None, description="Optional tenant pipeline ID for row-level security"
    )


class RatingResponse(BaseModel):
    """Response containing human rating, model overall score, and calibration requirement."""

    run_id: str
    user_rating: int
    normalized_rating: float
    overall_score: float | None
    calibration_needed: bool
    discrepancy: float | None
    faithfulness: float | None = None
    answer_relevance: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    message: str


class EvaluationResponse(BaseModel):
    """Full evaluation record for a pipeline run."""

    id: str
    run_id: str
    faithfulness: float | None
    answer_relevance: float | None
    context_precision: float | None
    context_recall: float | None
    overall_score: float | None
    judge_model: str | None
    user_rating: int | None
    calibration_needed: bool = False


async def _resolve_pipeline_id(run_id: UUID, provided_pid: UUID | None) -> UUID:
    """Resolve pipeline_id for the given run_id from the database if not provided."""
    if provided_pid is not None:
        return provided_pid

    if connection is None:
        raise HTTPException(
            status_code=503,
            detail="Database connection pool is not available.",
        )

    async with connection() as conn:
        row = await conn.fetchrow(
            "SELECT pipeline_id FROM pipeline_runs WHERE id = $1",
            run_id,
        )
        if not row:
            raise HTTPException(
                status_code=404,
                detail=f"Pipeline run {run_id} not found.",
            )
        return row["pipeline_id"]


@router.patch("/runs/{run_id}/rating", response_model=RatingResponse)
async def patch_run_rating(
    run_id: UUID,
    payload: RatingRequest,
    req: Request,
):
    """Record a human rating (1-5) and check if judge calibration is needed.

    Calibration is required when abs(overall_score - (rating / 5.0)) > 0.3.
    """
    if pipeline_connection is None:
        raise HTTPException(status_code=503, detail="Database connection pool is not available.")

    pipeline_id = await _resolve_pipeline_id(run_id, payload.pipeline_id)

    normalized_rating = round(payload.rating / 5.0, 4)

    async with pipeline_connection(pipeline_id) as conn:
        # Check if evaluation record already exists for this run
        eval_row = await conn.fetchrow(
            """
            SELECT id, faithfulness, answer_relevance, context_precision, context_recall,
                   overall_score, judge_model, user_rating
            FROM evaluations
            WHERE run_id = $1
            ORDER BY evaluated_at DESC
            LIMIT 1
            """,
            run_id,
        )

        overall_score: float | None = None
        faithfulness: float | None = None
        answer_relevance: float | None = None
        context_precision: float | None = None
        context_recall: float | None = None

        if eval_row:
            overall_score = eval_row["overall_score"]
            faithfulness = eval_row["faithfulness"]
            answer_relevance = eval_row["answer_relevance"]
            context_precision = eval_row["context_precision"]
            context_recall = eval_row["context_recall"]

            await conn.execute(
                """
                UPDATE evaluations
                SET user_rating = $1
                WHERE id = $2
                """,
                payload.rating,
                eval_row["id"],
            )
        else:
            # If no eval record yet, insert placeholder with user rating
            await conn.execute(
                """
                INSERT INTO evaluations (run_id, user_rating, evaluated_at)
                VALUES ($1, $2, NOW())
                """,
                run_id,
                payload.rating,
            )

    discrepancy = (
        round(abs(overall_score - normalized_rating), 4)
        if overall_score is not None
        else None
    )
    calibration_needed = bool(discrepancy is not None and discrepancy > 0.3)

    msg = (
        f"Rating {payload.rating} recorded. "
        + ("Calibration recommended due to score divergence > 0.3." if calibration_needed else "Score is well calibrated.")
    )

    return RatingResponse(
        run_id=str(run_id),
        user_rating=payload.rating,
        normalized_rating=normalized_rating,
        overall_score=overall_score,
        calibration_needed=calibration_needed,
        discrepancy=discrepancy,
        faithfulness=faithfulness,
        answer_relevance=answer_relevance,
        context_precision=context_precision,
        context_recall=context_recall,
        message=msg,
    )


@router.get("/runs/{run_id}/evaluation", response_model=EvaluationResponse)
async def get_run_evaluation(
    run_id: UUID,
    pipeline_id: UUID | None = Query(default=None),
):
    """Retrieve the evaluation record for a pipeline run."""
    if pipeline_connection is None:
        raise HTTPException(status_code=503, detail="Database connection pool is not available.")

    p_id = await _resolve_pipeline_id(run_id, pipeline_id)

    async with pipeline_connection(p_id) as conn:
        row = await conn.fetchrow(
            """
            SELECT id, run_id, faithfulness, answer_relevance, context_precision,
                   context_recall, overall_score, judge_model, user_rating
            FROM evaluations
            WHERE run_id = $1
            ORDER BY evaluated_at DESC
            LIMIT 1
            """,
            run_id,
        )

        if not row:
            raise HTTPException(status_code=404, detail=f"No evaluation found for run {run_id}.")

        user_rating = row["user_rating"]
        overall_score = row["overall_score"]
        calibration_needed = False
        if user_rating is not None and overall_score is not None:
            calibration_needed = abs(overall_score - (user_rating / 5.0)) > 0.3

        return EvaluationResponse(
            id=str(row["id"]),
            run_id=str(row["run_id"]),
            faithfulness=row["faithfulness"],
            answer_relevance=row["answer_relevance"],
            context_precision=row["context_precision"],
            context_recall=row["context_recall"],
            overall_score=overall_score,
            judge_model=row["judge_model"],
            user_rating=user_rating,
            calibration_needed=calibration_needed,
        )


@router.get("/evaluations/stream")
async def stream_evaluations(req: Request):
    """Server-Sent Events streaming newly persisted evaluation events from Redis pub/sub."""
    redis_client = getattr(req.app.state, "redis", None)
    if redis_client is None:
        raise HTTPException(
            status_code=503,
            detail="Redis pub/sub is currently unavailable for evaluation streaming.",
        )

    async def event_generator() -> AsyncGenerator[dict[str, Any], None]:
        pubsub = redis_client.pubsub()
        try:
            await pubsub.subscribe("evaluations:new")
            logger.info("Client subscribed to Redis channel 'evaluations:new'")
            while True:
                if await req.is_disconnected():
                    logger.info("Client disconnected from /evaluations/stream")
                    break

                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if msg and msg.get("type") == "message":
                    raw_data = msg.get("data")
                    if isinstance(raw_data, bytes):
                        raw_data = raw_data.decode("utf-8")
                    yield {
                        "event": "evaluation",
                        "data": raw_data,
                    }
                else:
                    await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            logger.info("SSE evaluation stream cancelled by client")
            raise
        except Exception as exc:
            logger.error("Error in evaluation stream: %s", exc)
            yield {
                "event": "error",
                "data": json.dumps({"error": str(exc)}),
            }
        finally:
            try:
                await pubsub.unsubscribe("evaluations:new")
                await pubsub.close()
                logger.info("Cleaned up Redis pubsub subscription for 'evaluations:new'")
            except Exception as close_err:
                logger.debug("Error closing pubsub: %s", close_err)

    return EventSourceResponse(
        event_generator(),
        ping=15,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/evaluations")
async def list_evaluations(
    pipeline_id: UUID | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
):
    """Retrieve recent evaluation records with pipeline and run details for initial feed load."""
    if connection is None:
        raise HTTPException(status_code=503, detail="Database connection pool unavailable.")

    async with connection() as conn:
        filter_clause = "WHERE r.pipeline_id = $1" if pipeline_id else ""
        query_sql = f"""
            SELECT e.id as eval_id, e.run_id, e.faithfulness, e.answer_relevance,
                   e.context_precision, e.context_recall, e.overall_score,
                   e.judge_model, e.user_rating, e.evaluated_at,
                   r.pipeline_id, p.name as pipeline_name,
                   r.query, r.generation as answer, r.retrieved_chunk_ids
            FROM evaluations e
            JOIN pipeline_runs r ON r.id = e.run_id
            JOIN pipelines p ON p.id = r.pipeline_id
            {filter_clause}
            ORDER BY e.evaluated_at DESC
            LIMIT {'$2' if pipeline_id else '$1'}
        """
        params = [pipeline_id, limit] if pipeline_id else [limit]
        rows = await conn.fetch(query_sql, *params)

        results = []
        for r in rows:
            results.append({
                "eval_id": str(r["eval_id"]),
                "run_id": str(r["run_id"]),
                "pipeline_id": str(r["pipeline_id"]),
                "pipeline_name": r["pipeline_name"],
                "query": r["query"],
                "answer": r["answer"],
                "retrieved_chunk_ids": [str(cid) for cid in (r["retrieved_chunk_ids"] or [])],
                "faithfulness": r["faithfulness"],
                "answer_relevance": r["answer_relevance"],
                "context_precision": r["context_precision"],
                "context_recall": r["context_recall"],
                "overall_score": r["overall_score"],
                "judge_model": r["judge_model"],
                "user_rating": r["user_rating"],
                "evaluated_at": r["evaluated_at"].isoformat() if r["evaluated_at"] else None,
            })
        return results
