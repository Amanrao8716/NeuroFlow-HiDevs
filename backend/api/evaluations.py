"""FastAPI endpoints for Task 7 automated evaluation and human rating calibration."""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

try:
    from db.pool import connection, pipeline_connection
except ImportError:
    connection = None
    pipeline_connection = None

router = APIRouter(tags=["evaluations"])
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
