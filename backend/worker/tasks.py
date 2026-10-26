
"""Background tasks for document ingestion."""

from datetime import datetime, timezone
import asyncio
import json
import logging
import time
from typing import Any
from uuid import UUID

import httpx
from opentelemetry import trace
from prometheus_client import start_http_server
from redis.exceptions import RedisError

from config import settings
from db.pool import connection, pipeline_connection
from observability.metrics import (
    EVALUATION_JOBS,
    INGESTION_DOCUMENTS,
    QUEUE_DEPTH,
)
from observability.tracing import configure_tracing, extract_trace_context
from resilience.circuit_breaker import CircuitBreakerRegistry
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
    configure_tracing(
        f"{settings.app_name}-worker",
        settings.otel_exporter_otlp_endpoint,
        settings.app_env,
    )
    ctx["circuit_breakers"] = CircuitBreakerRegistry(ctx["redis"])
    ctx["metrics_server"], ctx["metrics_thread"] = start_http_server(8001)
    ctx["evaluation_consumer_task"] = asyncio.create_task(
        _evaluation_queue_consumer(ctx)
    )
    ctx["evaluation_metrics_task"] = asyncio.create_task(
        _evaluation_metrics_refresher()
    )
    logger.info("Ingestion worker started.")


async def shutdown(ctx):
    """Release worker resources."""
    from db.pool import close_pool

    consumer = ctx.get("evaluation_consumer_task")
    if consumer:
        consumer.cancel()
        try:
            await consumer
        except asyncio.CancelledError:
            pass
    metrics_task = ctx.get("evaluation_metrics_task")
    if metrics_task:
        metrics_task.cancel()
        try:
            await metrics_task
        except asyncio.CancelledError:
            pass
    metrics_server = ctx.get("metrics_server")
    if metrics_server:
        metrics_server.shutdown()
        metrics_server.server_close()
    await close_pool()
    logger.info("Ingestion worker stopped.")


async def _evaluation_queue_consumer(ctx):
    """Consume legacy evaluation-list messages and run each evaluation once."""
    redis = ctx["redis"]
    queue_key = "neuroflow:queue:evaluations"
    try:
        QUEUE_DEPTH.labels(queue="evaluation").set(await redis.llen(queue_key))
    except (RedisError, OSError, TimeoutError):
        logger.exception("Could not read the evaluation queue depth.")
    while True:
        try:
            item = await redis.blpop(queue_key, timeout=1)
        except (RedisError, OSError, TimeoutError):
            logger.exception("Evaluation queue read failed; retrying shortly.")
            await asyncio.sleep(1)
            continue
        if item is None:
            continue
        _, raw_message = item
        payload = None
        try:
            payload = json.loads(raw_message)
            if not isinstance(payload, dict):
                raise ValueError("Evaluation queue message must be a JSON object.")
            await evaluate_pipeline_run(
                ctx,
                payload["run_id"],
                payload["pipeline_id"],
                trace_context=payload.get("trace_context"),
            )
        except asyncio.CancelledError:
            await redis.lpush(queue_key, raw_message)
            raise
        except Exception:
            if payload is None:
                logger.exception("Discarding malformed evaluation queue message.")
            else:
                attempt = int(payload.get("attempt", 0)) + 1
                if attempt < 3:
                    payload["attempt"] = attempt
                    await redis.rpush(queue_key, json.dumps(payload))
                    logger.exception(
                        "Evaluation job failed; retry %s/2 queued.",
                        attempt,
                    )
                    await asyncio.sleep(min(2**attempt, 30))
                else:
                    logger.exception("Evaluation job failed after %s attempts.", attempt)
        finally:
            try:
                QUEUE_DEPTH.labels(queue="evaluation").set(
                    await redis.llen(queue_key)
                )
            except (RedisError, OSError, TimeoutError):
                logger.exception("Could not refresh the evaluation queue depth.")


async def _evaluation_metrics_refresher():
    """Keep rolling evaluation gauges current for pipelines seen by this worker."""
    from observability.quality import (
        known_pipeline_ids,
        refresh_rolling_evaluation_metrics,
    )

    while True:
        for pipeline_id in known_pipeline_ids():
            try:
                async with pipeline_connection(UUID(pipeline_id)) as conn:
                    await refresh_rolling_evaluation_metrics(conn, pipeline_id)
            except Exception:
                logger.exception(
                    "Could not refresh persisted evaluation metrics for pipeline %s.",
                    pipeline_id,
                )
        await asyncio.sleep(60)


async def process_document(ctx, document_id: str, pipeline_id: str):
    """Extract, chunk, embed, and persist one document."""
    started = time.perf_counter()
    doc_id = UUID(document_id)
    tenant_id = UUID(pipeline_id)
    redis = ctx["redis"]
    client = NeuroFlowClient(
        redis,
        circuit_breakers=ctx.get("circuit_breakers"),
    )
    page_count = 0
    embedding_calls = 0
    QUEUE_DEPTH.labels(queue="ingestion").set(
        await redis.llen("queue:ingest")
    )

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

            with tracer.start_as_current_span("ingestion.extract"):
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

            with tracer.start_as_current_span("ingestion.chunk"):
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
                with tracer.start_as_current_span("ingestion.embed") as embed_span:
                    embed_span.set_attribute("embedding.batch_size", len(batch))
                    vectors = await client.embed(
                        [chunk["content"] for chunk in batch]
                    )
                embedding_calls += 1

                if len(vectors) != len(batch):
                    raise ValueError("Embedding count does not match chunk count.")

                embeddings.extend(vectors)

            async with pipeline_connection(tenant_id) as conn:
                with tracer.start_as_current_span("ingestion.write_db"):
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
            INGESTION_DOCUMENTS.labels(status="succeeded").inc()
            QUEUE_DEPTH.labels(queue="ingestion").set(
                await redis.llen("queue:ingest")
            )

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

            INGESTION_DOCUMENTS.labels(status="failed").inc()
            QUEUE_DEPTH.labels(queue="ingestion").set(
                await redis.llen("queue:ingest")
            )
            raise


async def evaluate_pipeline_run(
    ctx,
    run_id: str,
    pipeline_id: str,
    trace_context: dict[str, str] | None = None,
):
    """Asynchronously evaluate a completed RAG pipeline run using LLM-as-a-judge."""
    from evaluation.judge import EvaluationJudge

    redis = ctx.get("redis")
    client = (
        NeuroFlowClient(
            redis,
            circuit_breakers=ctx.get("circuit_breakers"),
        )
        if redis
        else None
    )
    judge = EvaluationJudge(llm_client=client)

    with tracer.start_as_current_span(
        "evaluation.process",
        context=extract_trace_context(trace_context),
    ) as span:
        span.set_attribute("run_id", run_id)
        span.set_attribute("pipeline_id", pipeline_id)
        try:
            report = await judge.evaluate_run_from_db(
                run_id=run_id,
                pipeline_id=pipeline_id,
            )
            async with pipeline_connection(UUID(pipeline_id)) as conn:
                from observability.quality import (
                    detect_quality_anomaly,
                    remember_pipeline_for_refresh,
                    refresh_rolling_evaluation_metrics,
                )

                remember_pipeline_for_refresh(pipeline_id)
                await refresh_rolling_evaluation_metrics(conn, pipeline_id)
                baseline_rows = await conn.fetch(
                    """
                    SELECT e.overall_score
                    FROM evaluations e
                    JOIN pipeline_runs r ON r.id = e.run_id
                    WHERE r.pipeline_id = $1
                      AND r.id <> $2
                      AND e.evaluated_at >= NOW() - INTERVAL '7 days'
                      AND e.overall_score IS NOT NULL
                    ORDER BY e.evaluated_at DESC
                    LIMIT 100
                    """,
                    UUID(pipeline_id),
                    UUID(run_id),
                )

            anomaly = detect_quality_anomaly(
                [float(row["overall_score"]) for row in baseline_rows],
                report.overall_score,
            )
            if anomaly:
                await _submit_quality_anomaly(
                    run_id=run_id,
                    pipeline_id=pipeline_id,
                    latest_score=anomaly.latest_score,
                    baseline_mean=anomaly.baseline_mean,
                    baseline_stddev=anomaly.baseline_stddev,
                )
            EVALUATION_JOBS.labels(status="succeeded").inc()
            span.set_attribute("overall_score", report.overall_score)
            span.set_attribute("judge_model", report.judge_model)
            logger.info(
                "Evaluation complete for run %s: overall_score=%.4f (candidate=%s)",
                run_id,
                report.overall_score,
                report.is_training_candidate,
            )

            # Publish evaluation event to Redis pub/sub channel 'evaluations:new'
            eval_dict = report.to_dict()
            eval_dict["evaluated_at"] = datetime.now(timezone.utc).isoformat()
            if redis:
                try:
                    # Enrich with query and pipeline name if database is reachable
                    if connection:
                        try:
                            async with connection() as conn:
                                p_row = await conn.fetchrow(
                                    """
                                    SELECT p.name, r.query, r.generation, r.retrieved_chunk_ids
                                    FROM pipelines p
                                    JOIN pipeline_runs r ON r.pipeline_id = p.id
                                    WHERE r.id = $1
                                    """,
                                    UUID(run_id),
                                )
                                if p_row:
                                    eval_dict["pipeline_name"] = p_row["name"]
                                    eval_dict["query"] = p_row["query"]
                                    eval_dict["answer"] = p_row["generation"]
                                    eval_dict["retrieved_chunk_ids"] = [
                                        str(cid) for cid in (p_row["retrieved_chunk_ids"] or [])
                                    ]
                        except Exception as enrich_err:
                            logger.debug("Could not enrich eval payload: %s", enrich_err)

                    await redis.publish("evaluations:new", json.dumps(eval_dict))
                    logger.info("Published evaluation for run %s to Redis channel evaluations:new", run_id)
                except Exception as pub_err:
                    logger.warning("Failed to publish evaluation to evaluations:new: %s", pub_err)

            return eval_dict
        except Exception as exc:
            EVALUATION_JOBS.labels(status="failed").inc()
            logger.exception("Evaluation failed for run %s: %s", run_id, exc)
            raise


async def _submit_quality_anomaly(
    *,
    run_id: str,
    pipeline_id: str,
    latest_score: float,
    baseline_mean: float,
    baseline_stddev: float,
) -> None:
    """Submit a confirmed anomaly through the authenticated internal API."""
    if not settings.observability_service_token:
        logger.error(
            "Quality anomaly for run %s was not persisted: "
            "OBSERVABILITY_SERVICE_TOKEN is not configured.",
            run_id,
        )
        return
    try:
        async with httpx.AsyncClient(
            base_url=settings.neuroflow_api_url.rstrip("/"),
            timeout=10,
        ) as client:
            response = await client.post(
                f"/pipelines/{pipeline_id}/suggestions",
                json={
                    "run_id": run_id,
                    "latest_score": latest_score,
                    "baseline_mean": baseline_mean,
                    "baseline_stddev": baseline_stddev,
                },
                headers={
                    "Authorization": (
                        f"Bearer {settings.observability_service_token}"
                    )
                },
            )
            response.raise_for_status()
    except httpx.HTTPError:
        logger.exception(
            "Failed to persist quality anomaly for run %s.",
            run_id,
        )


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
