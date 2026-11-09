"""Fine-Tuning Job Manager coordinating extraction, MLflow tracking, and OpenAI submission."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from arq.connections import RedisSettings, create_pool
from openai import AsyncOpenAI

from config import settings
from db.pool import connection
from pipelines.finetuning.extractor import TrainingDataExtractor
from pipelines.finetuning.tracker import MLflowTracker

logger = logging.getLogger("neuroflow.finetuning.job_manager")

OPENAI_MINIMUM_EXAMPLES = 10


class FinetuneJobManager:
    """Orchestrates end-to-end fine-tuning lifecycle: extraction, tracking, submission, and polling."""

    def __init__(
        self,
        extractor: TrainingDataExtractor | None = None,
        tracker: MLflowTracker | None = None,
        openai_client: AsyncOpenAI | None = None,
        redis_settings: RedisSettings | None = None,
    ):
        self.extractor = extractor or TrainingDataExtractor()
        self.tracker = tracker or MLflowTracker()
        self.openai_client = openai_client or AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
        )
        self.redis_settings = redis_settings or RedisSettings(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password,
            database=0,
        )

    async def schedule_polling_task(self, job_id: UUID, defer_seconds: int = 60) -> None:
        """Enqueue status polling job in ARQ worker queue."""
        try:
            pool = await create_pool(self.redis_settings)
            try:
                await pool.enqueue_job(
                    "poll_finetune_job",
                    str(job_id),
                    _defer_by=defer_seconds,
                )
                logger.info("Enqueued ARQ polling task for job %s deferred by %ds", job_id, defer_seconds)
            finally:
                await pool.aclose()
        except Exception as exc:
            logger.warning("Failed to enqueue ARQ polling task for job %s: %s", job_id, exc)

    async def create_and_submit_job(
        self,
        base_model: str = "gpt-4o-mini",
        pipeline_id: UUID | None = None,
        min_quality_score: float = 0.82,
        training_format: str = "sft",
    ) -> dict[str, Any]:
        """Execute steps 1-10: DB record creation, extraction, validation, MLflow, upload, submission, and ARQ schedule."""
        job_id = uuid4()
        now = datetime.now(timezone.utc)

        # 1. Create a pending fine-tuning job record in PostgreSQL
        async with connection() as conn:
            await conn.execute(
                """
                INSERT INTO finetune_jobs (
                    id, base_model, status, pipeline_id, created_at
                )
                VALUES ($1, $2, 'pending', $3, $4)
                """,
                job_id,
                base_model,
                pipeline_id,
                now,
            )

        logger.info("Created pending fine-tuning job %s with base model %s", job_id, base_model)

        # 2. Extract and validate training data
        self.extractor.min_quality_score = min_quality_score
        try:
            file_path, report, accepted_pairs = await self.extractor.extract_for_job(
                job_id=job_id,
                pipeline_id=pipeline_id,
            )
        except Exception as exc:
            err_msg = f"Data extraction failed: {exc}"
            logger.exception(err_msg)
            async with connection() as conn:
                await conn.execute(
                    """
                    UPDATE finetune_jobs
                    SET status = 'failed', error_message = $1
                    WHERE id = $2
                    """,
                    err_msg,
                    job_id,
                )
            raise RuntimeError(err_msg) from exc

        accepted_count = report["accepted_count"]

        # 3. Reject or clearly report insufficient valid training examples (OpenAI requires >= 10)
        if accepted_count < OPENAI_MINIMUM_EXAMPLES:
            reason = (
                f"Insufficient training examples: {accepted_count} valid examples extracted, "
                f"but a minimum of {OPENAI_MINIMUM_EXAMPLES} is required by the provider."
            )
            logger.warning("Job %s rejected: %s", job_id, reason)

            async with connection() as conn:
                await conn.execute(
                    """
                    UPDATE finetune_jobs
                    SET status = 'failed',
                        error_message = $1,
                        extraction_report = $2::jsonb,
                        training_pair_count = $3
                    WHERE id = $4
                    """,
                    reason,
                    json.dumps(report),
                    accepted_count,
                    job_id,
                )

            return {
                "job_id": str(job_id),
                "status": "failed",
                "base_model": base_model,
                "training_pair_count": accepted_count,
                "error_message": reason,
                "extraction_report": report,
            }

        # 4. Create an MLflow run and log the training data artifact
        mlflow_run_id = None
        mlflow_run_url = None
        try:
            mlflow_run_id, mlflow_run_url = self.tracker.create_run(
                job_id=job_id,
                base_model=base_model,
                training_pair_count=accepted_count,
                average_quality_score=report["average_quality_score"],
                date_range=report["date_range"],
                training_file_path=file_path,
            )
        except Exception as exc:
            logger.warning("MLflow tracking encountered error for job %s: %s", job_id, exc)
            # Record error in report but allow submission or capture failure
            report["mlflow_error"] = str(exc)

        # 5. Upload JSONL using AsyncOpenAI with purpose 'fine-tune'
        provider_file_id = None
        provider_job_id = None
        try:
            with open(file_path, "rb") as f:
                upload_res = await self.openai_client.files.create(
                    file=f,
                    purpose="fine-tune",
                )
            provider_file_id = upload_res.id
            logger.info("Uploaded training file for job %s: %s", job_id, provider_file_id)

            # 6. Submit the fine-tuning job with the selected base model
            ft_res = await self.openai_client.fine_tuning.jobs.create(
                training_file=provider_file_id,
                model=base_model,
            )
            provider_job_id = ft_res.id
            logger.info("Submitted OpenAI fine-tuning job %s -> provider ID %s", job_id, provider_job_id)

        except Exception as exc:
            err_msg = f"Provider submission failed: {exc}"
            logger.exception(err_msg)

            # Update job as failed without consuming pairs
            async with connection() as conn:
                await conn.execute(
                    """
                    UPDATE finetune_jobs
                    SET status = 'failed',
                        error_message = $1,
                        mlflow_run_id = $2,
                        extraction_report = $3::jsonb,
                        training_pair_count = $4
                    WHERE id = $5
                    """,
                    err_msg[:500],
                    mlflow_run_id,
                    json.dumps(report),
                    accepted_count,
                    job_id,
                )

            if mlflow_run_id:
                try:
                    self.tracker.log_completion_metrics(mlflow_run_id, status="FAILED")
                except Exception:
                    pass

            raise RuntimeError(err_msg) from exc

        # 7. Safely associate training pairs with the job in a transaction after successful submission
        pair_ids = [p.id for p in accepted_pairs]
        marked_count = await self.extractor.mark_pairs_included_in_job(job_id, pair_ids)
        logger.info("Marked %d training pairs as consumed by job %s", marked_count, job_id)

        # 8. Update DB with provider job ID, running status, and MLflow run ID
        async with connection() as conn:
            await conn.execute(
                """
                UPDATE finetune_jobs
                SET provider_job_id = $1,
                    status = 'running',
                    mlflow_run_id = $2,
                    training_pair_count = $3,
                    extraction_report = $4::jsonb
                WHERE id = $5
                """,
                provider_job_id,
                mlflow_run_id,
                accepted_count,
                json.dumps(report),
                job_id,
            )

        # 9. Schedule status polling through ARQ infrastructure at 60-second intervals
        await self.schedule_polling_task(job_id, defer_seconds=60)

        return {
            "job_id": str(job_id),
            "status": "running",
            "base_model": base_model,
            "provider_job_id": provider_job_id,
            "training_pair_count": accepted_count,
            "mlflow_run_id": mlflow_run_id,
            "mlflow_run_url": mlflow_run_url,
            "extraction_report": report,
        }
