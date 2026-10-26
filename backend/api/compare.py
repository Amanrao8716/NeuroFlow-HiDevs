"""A/B Pipeline Comparison API endpoint.

Executes two configured RAG pipelines concurrently via asyncio.gather,
capturing individual timings, chunks used, evaluation status, and errors side-by-side.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from observability.metrics import QUERY_LATENCY, QUERY_TOTAL
from pipelines.generation.generator import GenerationPipeline

try:
    from db.pool import connection, pipeline_connection
except ImportError:
    connection = None
    pipeline_connection = None

from backend.security.auth import require_scope
from backend.security.prompt_injection import classify_query, detect_prompt_injection
from backend.security.secrets import redact_secrets
from backend.security.validators import sanitize_text

router = APIRouter(tags=["compare"], dependencies=[Depends(require_scope("query"))])
logger = logging.getLogger("neuroflow.api.compare")

_pipeline_instance: GenerationPipeline | None = None


def get_generation_pipeline(request: Request | None = None) -> GenerationPipeline:
    """Obtain or initialize the GenerationPipeline singleton."""
    global _pipeline_instance
    redis_client = getattr(request.app.state, "redis", None) if request and hasattr(request, "app") else None
    circuit_breakers = (
        getattr(request.app.state, "circuit_breakers", None)
        if request and hasattr(request, "app")
        else None
    )
    if _pipeline_instance is None:
        _pipeline_instance = GenerationPipeline(redis_client=redis_client)
        if _pipeline_instance.client:
            _pipeline_instance.client.circuit_breakers = circuit_breakers
    elif _pipeline_instance.client is None and redis_client is not None:
        from providers.client import NeuroFlowClient
        _pipeline_instance.redis = redis_client
        _pipeline_instance.client = NeuroFlowClient(
            redis_client,
            circuit_breakers=circuit_breakers,
        )
    return _pipeline_instance


class ComparePipelinesRequest(BaseModel):
    """Request payload for comparing two pipelines."""

    query: str = Field(..., min_length=1, max_length=5000, description="Query to execute against both pipelines")
    pipeline_a_id: UUID = Field(..., description="First pipeline ID")
    pipeline_b_id: UUID = Field(..., description="Second pipeline ID")
    pipeline_a_version: int | None = Field(default=None, description="Optional version for pipeline A")
    pipeline_b_version: int | None = Field(default=None, description="Optional version for pipeline B")


class PipelineComparisonResult(BaseModel):
    """Execution results for one side of an A/B pipeline comparison."""

    pipeline_id: UUID
    version: int | None = None
    run_id: str | None = None
    answer: str | None = None
    retrieval_latency_ms: float | None = None
    total_latency_ms: float | None = None
    chunks_used: int | None = None
    evaluation_score: float | None = None
    error: str | None = None


class ComparePipelinesResponse(BaseModel):
    """Side-by-side comparison response for two pipelines."""

    query: str
    pipeline_a: PipelineComparisonResult
    pipeline_b: PipelineComparisonResult


async def _check_pipeline_exists_and_active(pipeline_id: UUID) -> dict[str, Any]:
    """Verify that a pipeline exists and is not archived.

    Raises:
        HTTPException: 404 if pipeline is missing, 400 if archived.
    """
    if pipeline_connection is None:
        return {"id": pipeline_id, "status": "active", "version": 1}

    async with pipeline_connection(pipeline_id) as conn:
        row = await conn.fetchrow(
            """
            SELECT id, name, status, version
            FROM pipelines
            WHERE id = $1
            """,
            pipeline_id,
        )
        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pipeline {pipeline_id} not found",
            )
        if row["status"] == "archived":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Pipeline {pipeline_id} is archived and cannot be executed",
            )
        return dict(row)


async def _execute_single_pipeline(
    generator: GenerationPipeline,
    query: str,
    pipeline_id: UUID,
    version: int | None,
) -> PipelineComparisonResult:
    """Execute a single pipeline safely, capturing latency, answer, and any execution error."""
    start_time = time.perf_counter()
    try:
        gen_result = await generator.generate(
            query=query,
            pipeline_id=pipeline_id,
            pipeline_version=version,
        )
        QUERY_TOTAL.labels(status=gen_result.status).inc()
        QUERY_LATENCY.observe(time.perf_counter() - start_time)

        total_latency = float(gen_result.latency_ms) if gen_result.latency_ms is not None else round((time.perf_counter() - start_time) * 1000.0, 2)
        retrieval_lat = float(gen_result.retrieval_latency_ms) if gen_result.retrieval_latency_ms is not None else None

        # Fetch evaluation score if completed synchronously or previously recorded
        eval_score: float | None = None
        if gen_result.run_id and pipeline_connection is not None:
            try:
                async with pipeline_connection(pipeline_id) as conn:
                    eval_row = await conn.fetchrow(
                        """
                        SELECT overall_score
                        FROM evaluations
                        WHERE run_id = $1
                        ORDER BY created_at DESC
                        LIMIT 1
                        """,
                        UUID(gen_result.run_id),
                    )
                    if eval_row and eval_row["overall_score"] is not None:
                        eval_score = float(eval_row["overall_score"])
            except Exception as e:
                logger.debug("Could not fetch evaluation score for run %s: %s", gen_result.run_id, e)

        if gen_result.status == "failed":
            return PipelineComparisonResult(
                pipeline_id=pipeline_id,
                version=gen_result.pipeline_version,
                run_id=gen_result.run_id,
                answer=None,
                retrieval_latency_ms=retrieval_lat,
                total_latency_ms=total_latency,
                chunks_used=gen_result.chunks_used,
                evaluation_score=eval_score,
                error=gen_result.error_message or "Pipeline execution failed",
            )

        return PipelineComparisonResult(
            pipeline_id=pipeline_id,
            version=gen_result.pipeline_version,
            run_id=gen_result.run_id,
            answer=gen_result.generation,
            retrieval_latency_ms=retrieval_lat,
            total_latency_ms=total_latency,
            chunks_used=gen_result.chunks_used,
            evaluation_score=eval_score,
            error=None,
        )
    except Exception as exc:
        total_latency = round((time.perf_counter() - start_time) * 1000.0, 2)
        QUERY_TOTAL.labels(status="failed").inc()
        QUERY_LATENCY.observe(time.perf_counter() - start_time)
        logger.exception("Pipeline %s execution raised an exception: %s", pipeline_id, exc)
        return PipelineComparisonResult(
            pipeline_id=pipeline_id,
            version=version,
            run_id=None,
            answer=None,
            retrieval_latency_ms=None,
            total_latency_ms=total_latency,
            chunks_used=None,
            evaluation_score=None,
            error=str(exc),
        )


@router.post(
    "/pipelines/compare",
    response_model=ComparePipelinesResponse,
    summary="Compare two RAG pipelines side-by-side concurrently",
)
async def compare_pipelines(
    payload: ComparePipelinesRequest,
    request: Request,
) -> ComparePipelinesResponse:
    """Concurrently execute two pipelines for A/B comparison.

    - Validates both pipelines exist and are not archived.
    - Prevents identical pipeline comparison when versions are identical or unspecified.
    - Executes both pipelines concurrently via asyncio.gather.
    - Captures independent errors, retrieval/total latencies, chunks, and evaluation scores.
    """
    # 1. Handle identical pipeline IDs
    if payload.pipeline_a_id == payload.pipeline_b_id:
        if payload.pipeline_a_version is None or payload.pipeline_a_version == payload.pipeline_b_version:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot compare identical pipeline IDs. pipeline_a_id and pipeline_b_id must be distinct, or specify different versions.",
            )

    # 2. Check existence and active status
    await asyncio.gather(
        _check_pipeline_exists_and_active(payload.pipeline_a_id),
        _check_pipeline_exists_and_active(payload.pipeline_b_id),
    )

    # 3. Obtain generator instance
    generator = get_generation_pipeline(request)
    query_text = sanitize_text(payload.query)
    query_text, _ = redact_secrets(query_text)
    detection = detect_prompt_injection(query_text)
    if detection.detected:
        logger.warning("Comparison query matched prompt injection signal %s", detection.pattern)
    try:
        if await classify_query(getattr(generator, "client", None), query_text):
            return JSONResponse(status_code=400, content={"error": "query_rejected", "reason": "potential_prompt_injection"})
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Prompt injection classifier unavailable (%s); failing closed.", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Query security classification unavailable.") from exc

    # 4. Execute both pipelines concurrently using asyncio.gather
    result_a, result_b = await asyncio.gather(
        _execute_single_pipeline(
            generator=generator,
            query=query_text,
            pipeline_id=payload.pipeline_a_id,
            version=payload.pipeline_a_version,
        ),
        _execute_single_pipeline(
            generator=generator,
            query=query_text,
            pipeline_id=payload.pipeline_b_id,
            version=payload.pipeline_b_version,
        ),
    )

    return ComparePipelinesResponse(
        query=query_text,
        pipeline_a=result_a,
        pipeline_b=result_b,
    )
