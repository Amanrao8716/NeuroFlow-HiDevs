
"""Background tasks for document ingestion."""

import json
import logging
import time
from typing import Any
from uuid import UUID

from opentelemetry import trace

from db.pool import connection, pipeline_connection
from pipelines.ingestion.csv_extractor import extract_csv
from pipelines.ingestion.docx_extractor import extract_docx
from pipelines.ingestion.image_extractor import extract_image
from pipelines.ingestion.pdf_extractor import extract_pdf
from pipelines.ingestion.url_extractor import extract_url
from pipelines.ingestion.chunker import chunk_pages
from pipelines.finetuning.tracker import MLflowTracker
from providers.client import NeuroFlowClient

logger = logging.getLogger("neuroflow.ingestion")
tracer = trace.get_tracer("neuroflow.ingestion")


async def startup(ctx):
    """Initialize resources used by arq workers."""
    from db.pool import open_pool

    await open_pool()
    logger.info("Ingestion worker started.")


async def shutdown(ctx):
    """Release worker resources."""
    from db.pool import close_pool

    await close_pool()
    logger.info("Ingestion worker stopped.")


async def process_document(ctx, document_id: str, pipeline_id: str):
    """Extract, chunk, embed, and persist one document."""
    started = time.perf_counter()
    doc_id = UUID(document_id)
    tenant_id = UUID(pipeline_id)
    redis = ctx["redis"]
    client = NeuroFlowClient(redis)
    page_count = 0
    embedding_calls = 0

    with tracer.start_as_current_span("ingestion.process") as span:
        span.set_attribute("document_id", document_id)
        span.set_attribute("pipeline_id", pipeline_id)

        try:
            async with pipeline_connection(tenant_id) as conn:
                document = await conn.fetchrow(
                    """
                    SELECT id, filename, source_type, source_bytes, metadata
                    FROM documents
                    WHERE id = $1 AND pipeline_id = $2
                    FOR UPDATE
                    """,
                    doc_id,
                    tenant_id,
                )

                if document is None:
                    raise ValueError("Document not found for this pipeline.")

                await conn.execute(
                    "UPDATE documents SET status = 'processing' WHERE id = $1",
                    doc_id,
                )

                source_type = document["source_type"]
                source_bytes = document["source_bytes"]
                raw_metadata = document["metadata"]
                metadata = (
                    json.loads(raw_metadata)
                    if isinstance(raw_metadata, str)
                    else raw_metadata or {}
                )

            span.set_attribute("source_type", source_type)

            if source_type == "url":
                pages = await extract_url(metadata["url"])
            elif source_type == "pdf":
                pages = extract_pdf(source_bytes)
                for page in pages:
                    page.metadata["page_count"] = len(
                        {p.page_number for p in pages if p.content_type == "text"}
                    )
            elif source_type == "docx":
                pages = extract_docx(source_bytes)
            elif source_type == "csv":
                pages = extract_csv(source_bytes)
            elif source_type == "image":
                pages = await extract_image(source_bytes, client)
            else:
                raise ValueError(f"Unsupported source type: {source_type}")

            if not pages:
                raise ValueError("No content could be extracted.")

            page_count = len(pages)
            span.set_attribute("page_count", page_count)

            chunks = await chunk_pages(
                pages,
                embed=client.embed,
                strategy="auto",
            )

            if not chunks:
                raise ValueError("No chunks were produced.")

            # Embed in batches to avoid oversized provider requests.
            embeddings = []
            batch_size = 64
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start:start + batch_size]
                vectors = await client.embed(
                    [chunk["content"] for chunk in batch]
                )
                embedding_calls += 1

                if len(vectors) != len(batch):
                    raise ValueError("Embedding count does not match chunk count.")

                embeddings.extend(vectors)

            async with pipeline_connection(tenant_id) as conn:
                for chunk, vector in zip(chunks, embeddings):
                    vector_value = "[" + ",".join(str(value) for value in vector) + "]"

                    await conn.execute(
                        """
                        INSERT INTO chunks
                            (document_id, content, embedding, chunk_index,
                             token_count, metadata)
                        VALUES ($1, $2, $3::vector, $4, $5, $6::jsonb)
                        ON CONFLICT (document_id, chunk_index) DO NOTHING
                        """,
                        doc_id,
                        chunk["content"],
                        vector_value,
                        chunk["chunk_index"],
                        chunk["token_count"],
                        json.dumps(chunk["metadata"]),
                    )

                await conn.execute(
                    """
                    UPDATE documents
                    SET status = 'completed',
                        chunk_count = $2,
                        source_bytes = NULL,
                        metadata = metadata || $3::jsonb
                    WHERE id = $1
                    """,
                    doc_id,
                    len(chunks),
                    json.dumps({
                        "page_count": page_count,
                        "chunking_strategy": chunks[0]["metadata"].get("strategy"),
                    }),
                )

            span.set_attribute("chunk_count", len(chunks))
            span.set_attribute("embedding_calls", embedding_calls)

            duration_ms = round((time.perf_counter() - started) * 1000)
            logger.info(json.dumps({
                "event": "ingestion_complete",
                "document_id": document_id,
                "duration_ms": duration_ms,
                "chunks": len(chunks),
                "tokens": sum(chunk["token_count"] for chunk in chunks),
            }))

        except Exception as exc:
            logger.exception("Ingestion failed for document %s", document_id)

            async with pipeline_connection(tenant_id) as conn:
                await conn.execute(
                    """
                    UPDATE documents
                    SET status = 'failed',
                        metadata = metadata || $2::jsonb
                    WHERE id = $1
                    """,
                    doc_id,
                    json.dumps({"error": str(exc)[:500]}),
                )

            raise


async def evaluate_pipeline_run(ctx, run_id: str, pipeline_id: str):
    """Asynchronously evaluate a completed RAG pipeline run using LLM-as-a-judge."""
    from evaluation.judge import EvaluationJudge

    redis = ctx.get("redis")
    client = NeuroFlowClient(redis) if redis else None
    judge = EvaluationJudge(llm_client=client)

    with tracer.start_as_current_span("evaluation.process") as span:
        span.set_attribute("run_id", run_id)
        span.set_attribute("pipeline_id", pipeline_id)
        try:
            report = await judge.evaluate_run_from_db(
                run_id=run_id,
                pipeline_id=pipeline_id,
            )
            span.set_attribute("overall_score", report.overall_score)
            span.set_attribute("judge_model", report.judge_model)
            logger.info(
                "Evaluation complete for run %s: overall_score=%.4f (candidate=%s)",
                run_id,
                report.overall_score,
                report.is_training_candidate,
            )
            return report.to_dict()
        except Exception as exc:
            logger.exception("Evaluation failed for run %s: %s", run_id, exc)
            raise


async def poll_finetune_job(ctx, job_id: str, openai_client: Any = None):
    """Poll asynchronous provider fine-tuning job status and handle terminal states."""
    import os
    from openai import AsyncOpenAI

    jid = UUID(job_id)
    redis = ctx.get("redis")

    with tracer.start_as_current_span("finetune.poll") as span:
        span.set_attribute("job_id", job_id)

        # 1. Fetch job record
        async with connection() as conn:
            job_row = await conn.fetchrow(
                """
                SELECT id, provider_job_id, status, mlflow_run_id, fine_tuned_model, base_model
                FROM finetune_jobs
                WHERE id = $1
                """,
                jid,
            )

        if not job_row:
            logger.warning("Finetune job %s not found in DB.", job_id)
            return {"status": "not_found"}

        current_status = job_row["status"]
        if current_status in ("succeeded", "failed", "cancelled"):
            logger.info("Job %s is already in terminal state %s.", job_id, current_status)
            return {"status": current_status}

        provider_job_id = job_row["provider_job_id"]
        if not provider_job_id:
            logger.warning("Job %s has no provider_job_id.", job_id)
            return {"status": current_status}

        client = openai_client or AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
        )

        try:
            remote_job = await client.fine_tuning.jobs.retrieve(provider_job_id)
        except Exception as exc:
            logger.warning("Transient error polling provider job %s: %s", provider_job_id, exc)
            from arq.connections import create_pool, RedisSettings
            from config import settings
            pool = await create_pool(RedisSettings(
                host=settings.redis_host,
                port=settings.redis_port,
                password=settings.redis_password,
                database=0,
            ))
            try:
                await pool.enqueue_job("poll_finetune_job", job_id, _defer_by=60)
            finally:
                await pool.aclose()
            return {"status": "retry_deferred", "error": str(exc)}

        remote_status = getattr(remote_job, "status", "running")
        logger.info("Polled job %s -> remote status: %s", job_id, remote_status)

        if remote_status == "succeeded":
            ft_model = getattr(remote_job, "fine_tuned_model", None) or f"ft:{job_row['base_model']}:{job_id}"
            trained_tokens = getattr(remote_job, "trained_tokens", None)
            metrics = {}
            if hasattr(remote_job, "training_metrics"):
                metrics = dict(getattr(remote_job, "training_metrics") or {})

            # 1. Update DB
            async with connection() as conn:
                await conn.execute(
                    """
                    UPDATE finetune_jobs
                    SET status = 'succeeded',
                        fine_tuned_model = $1,
                        trained_tokens = $2,
                        metrics = $3::jsonb,
                        completed_at = NOW()
                    WHERE id = $4
                    """,
                    ft_model,
                    trained_tokens,
                    json.dumps(metrics),
                    jid,
                )

            # 2. Log metrics to MLflow
            mlflow_id = job_row["mlflow_run_id"]
            if mlflow_id:
                try:
                    tracker = MLflowTracker()
                    tracker.log_completion_metrics(
                        run_id=mlflow_id,
                        metrics=metrics,
                        trained_tokens=trained_tokens,
                        status="FINISHED",
                    )
                except Exception as mf_err:
                    logger.warning("Failed to log MLflow completion for job %s: %s", job_id, mf_err)

            # 3. Register model in Redis ModelRouter registry (avoid duplicate registration)
            if redis:
                existing_entry = await redis.hget("router:models", ft_model)
                if not existing_entry:
                    model_entry = {
                        "name": ft_model,
                        "role": "general",
                        "task_type": "rag_qa",
                        "fine_tuned": True,
                        "context_window": 128000,
                        "estimated_cost_per_call": 0.002,
                    }
                    await redis.hset("router:models", ft_model, json.dumps(model_entry))
                    logger.info("Registered fine-tuned model %s in router:models", ft_model)

            return {"status": "succeeded", "fine_tuned_model": ft_model}

        elif remote_status in ("failed", "cancelled"):
            err_obj = getattr(remote_job, "error", None)
            err_msg = str(getattr(err_obj, "message", err_obj) or f"Fine-tuning job {remote_status}")[:500]

            async with connection() as conn:
                await conn.execute(
                    """
                    UPDATE finetune_jobs
                    SET status = $1,
                        error_message = $2,
                        completed_at = NOW()
                    WHERE id = $3
                    """,
                    remote_status,
                    err_msg,
                    jid,
                )

            mlflow_id = job_row["mlflow_run_id"]
            if mlflow_id:
                try:
                    tracker = MLflowTracker()
                    tracker.log_completion_metrics(
                        run_id=mlflow_id,
                        status="FAILED" if remote_status == "failed" else "KILLED",
                    )
                except Exception:
                    pass

            return {"status": remote_status, "error": err_msg}

        else:
            from arq.connections import create_pool, RedisSettings
            from config import settings
            pool = await create_pool(RedisSettings(
                host=settings.redis_host,
                port=settings.redis_port,
                password=settings.redis_password,
                database=0,
            ))
            try:
                await pool.enqueue_job("poll_finetune_job", job_id, _defer_by=60)
            finally:
                await pool.aclose()

            return {"status": remote_status}
