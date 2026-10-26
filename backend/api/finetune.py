"""REST API endpoints for Fine-Tuning Pipeline management, preview, and tracking."""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from config import settings
from db.pool import connection
from models.finetune import (
    CreateFinetuneJobRequest,
    FinetuneJobListResponse,
    FinetuneJobResponse,
    TrainingDataPreviewItem,
    TrainingDataPreviewResponse,
)
from pipelines.finetuning.extractor import TrainingDataExtractor
from pipelines.finetuning.job_manager import FinetuneJobManager

logger = logging.getLogger("neuroflow.api.finetune")

from backend.security.auth import require_scope

router = APIRouter(prefix="/finetune", tags=["finetuning"], dependencies=[Depends(require_scope("admin"))])


def _format_mlflow_url(run_id: str | None) -> str | None:
    if not run_id:
        return None
    base_uri = settings.mlflow_tracking_uri.rstrip("/")
    return f"{base_uri}/#/experiments/0/runs/{run_id}"


@router.post("/jobs", response_model=FinetuneJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_finetune_job(payload: CreateFinetuneJobRequest):
    """Trigger training data extraction, validation, MLflow tracking setup, and asynchronous fine-tuning job submission."""
    manager = FinetuneJobManager()

    try:
        result = await manager.create_and_submit_job(
            base_model=payload.base_model,
            pipeline_id=payload.pipeline_id,
            min_quality_score=payload.min_quality_score,
            training_format=payload.format,
        )
    except Exception as exc:
        logger.exception("Failed to initiate fine-tuning job: %s", exc)
        safe_msg = str(exc)
        if "API key" in safe_msg or "Bearer" in safe_msg:
            safe_msg = "Authentication error during provider submission."
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fine-tuning submission failed: {safe_msg}",
        )

    # Fetch full record from database
    jid = UUID(result["job_id"])
    async with connection() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, provider_job_id, base_model, status, training_pair_count,
                   mlflow_run_id, fine_tuned_model, metrics, trained_tokens,
                   error_message, extraction_report, pipeline_id, created_at, completed_at
            FROM finetune_jobs
            WHERE id = $1
            """,
            jid,
        )

    if not row:
        raise HTTPException(status_code=500, detail="Job record creation failed.")

    report_data = row["extraction_report"]
    report_dict = json.loads(report_data) if isinstance(report_data, str) else report_data
    metrics_data = row["metrics"]
    metrics_dict = json.loads(metrics_data) if isinstance(metrics_data, str) else metrics_data

    mlflow_id = row["mlflow_run_id"]

    return FinetuneJobResponse(
        id=str(row["id"]),
        provider_job_id=row["provider_job_id"],
        base_model=row["base_model"],
        status=row["status"],
        training_pair_count=row["training_pair_count"],
        mlflow_run_id=mlflow_id,
        mlflow_run_url=_format_mlflow_url(mlflow_id),
        fine_tuned_model=row["fine_tuned_model"],
        metrics=metrics_dict,
        trained_tokens=row["trained_tokens"],
        error_message=row["error_message"],
        extraction_report=report_dict,
        pipeline_id=str(row["pipeline_id"]) if row["pipeline_id"] else None,
        created_at=row["created_at"],
        completed_at=row["completed_at"],
    )


@router.get("/jobs", response_model=FinetuneJobListResponse)
async def list_finetune_jobs(
    pipeline_id: UUID | None = Query(default=None, description="Optional pipeline ID filter"),
    limit: int = Query(default=50, ge=1, le=100),
):
    """List fine-tuning jobs with status, base model, timestamps, provider job ID, and MLflow run ID."""
    query = """
        SELECT id, provider_job_id, base_model, status, training_pair_count,
               mlflow_run_id, fine_tuned_model, metrics, trained_tokens,
               error_message, extraction_report, pipeline_id, created_at, completed_at
        FROM finetune_jobs
    """
    params: list[Any] = []
    if pipeline_id is not None:
        query += " WHERE pipeline_id = $1"
        params.append(pipeline_id)

    query += f" ORDER BY created_at DESC LIMIT {limit}"

    async with connection() as conn:
        rows = await conn.fetch(query, *params)
        total = len(rows)

    jobs: list[FinetuneJobResponse] = []
    for r in rows:
        report_data = r["extraction_report"]
        report_dict = json.loads(report_data) if isinstance(report_data, str) else report_data
        metrics_data = r["metrics"]
        metrics_dict = json.loads(metrics_data) if isinstance(metrics_data, str) else metrics_data
        mlflow_id = r["mlflow_run_id"]

        jobs.append(
            FinetuneJobResponse(
                id=str(r["id"]),
                provider_job_id=r["provider_job_id"],
                base_model=r["base_model"],
                status=r["status"],
                training_pair_count=r["training_pair_count"],
                mlflow_run_id=mlflow_id,
                mlflow_run_url=_format_mlflow_url(mlflow_id),
                fine_tuned_model=r["fine_tuned_model"],
                metrics=metrics_dict,
                trained_tokens=r["trained_tokens"],
                error_message=r["error_message"],
                extraction_report=report_dict,
                pipeline_id=str(r["pipeline_id"]) if r["pipeline_id"] else None,
                created_at=r["created_at"],
                completed_at=r["completed_at"],
            )
        )

    return FinetuneJobListResponse(jobs=jobs, total=total)


@router.get("/jobs/{job_id}", response_model=FinetuneJobResponse)
async def get_finetune_job_detail(job_id: UUID):
    """Get detailed status including training metrics, extraction summary, provider model ID, and MLflow run URL."""
    async with connection() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, provider_job_id, base_model, status, training_pair_count,
                   mlflow_run_id, fine_tuned_model, metrics, trained_tokens,
                   error_message, extraction_report, pipeline_id, created_at, completed_at
            FROM finetune_jobs
            WHERE id = $1
            """,
            job_id,
        )

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Fine-tuning job {job_id} not found.",
        )

    report_data = row["extraction_report"]
    report_dict = json.loads(report_data) if isinstance(report_data, str) else report_data
    metrics_data = row["metrics"]
    metrics_dict = json.loads(metrics_data) if isinstance(metrics_data, str) else metrics_data
    mlflow_id = row["mlflow_run_id"]

    return FinetuneJobResponse(
        id=str(row["id"]),
        provider_job_id=row["provider_job_id"],
        base_model=row["base_model"],
        status=row["status"],
        training_pair_count=row["training_pair_count"],
        mlflow_run_id=mlflow_id,
        mlflow_run_url=_format_mlflow_url(mlflow_id),
        fine_tuned_model=row["fine_tuned_model"],
        metrics=metrics_dict,
        trained_tokens=row["trained_tokens"],
        error_message=row["error_message"],
        extraction_report=report_dict,
        pipeline_id=str(row["pipeline_id"]) if row["pipeline_id"] else None,
        created_at=row["created_at"],
        completed_at=row["completed_at"],
    )


@router.get("/training-data/preview", response_model=TrainingDataPreviewResponse)
async def preview_training_data(
    pipeline_id: UUID | None = Query(default=None, description="Optional pipeline ID filter"),
    min_quality_score: float = Query(default=0.82, ge=0.0, le=1.0),
    limit: int = Query(default=5, ge=1, le=20),
):
    """Return up to 5 eligible example training pairs without creating a job, writing files, or mutating state."""
    extractor = TrainingDataExtractor(min_quality_score=min_quality_score)
    items, total_eligible = await extractor.preview_eligible_pairs(
        pipeline_id=pipeline_id,
        limit=limit,
    )

    preview_items = [
        TrainingDataPreviewItem(
            id=item["id"],
            run_id=item["run_id"],
            user_message=item["user_message"],
            assistant_message=item["assistant_message"],
            quality_score=item["quality_score"],
            faithfulness=item["faithfulness"],
            user_rating=item["user_rating"],
            token_count=item["token_count"],
            citations_count=item["citations_count"],
            has_valid_citations=item["has_valid_citations"],
        )
        for item in items
    ]

    return TrainingDataPreviewResponse(
        items=preview_items,
        total_eligible=total_eligible,
    )
