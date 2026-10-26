"""FastAPI endpoint for Task 6 RAG generation, progressive streaming SSE, and citation tracking."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
import logging
import time
from typing import Any, AsyncGenerator
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from observability.metrics import QUERY_LATENCY, QUERY_TOTAL
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from pipelines.generation.generator import GenerationPipeline, GenerationResult
from backend.security.auth import require_scope
from backend.security.prompt_injection import classify_query, detect_prompt_injection
from backend.security.validators import sanitize_text
from backend.security.secrets import redact_secrets

router = APIRouter(tags=["generation"], dependencies=[Depends(require_scope("query"))])
logger = logging.getLogger("neuroflow.api.query")

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


class QueryRequest(BaseModel):
    """Request model for POST /query endpoint."""

    query: str = Field(..., min_length=1, max_length=5000, description="User search query")
    pipeline_id: UUID = Field(..., description="Tenant pipeline ID for data isolation")
    stream: bool = Field(default=False, description="Whether to stream response tokens via SSE")
    top_k: int = Field(default=10, ge=1, le=50, description="Top K retrieval candidates")
    max_context_tokens: int = Field(
        default=4000, ge=100, le=20000, description="Token budget for context"
    )
    rerank: bool = Field(default=True, description="Whether to apply reranking")
    pipeline_version: int | None = Field(
        default=None, ge=1, description="Specific pipeline version to execute"
    )


class CitationItem(BaseModel):
    """Structured citation returned to API client."""

    reference: str
    source_number: int
    chunk_id: str | None
    document_name: str | None
    page_number: int | None
    content_preview: str | None
    is_valid: bool
    error: str | None = None


class QueryResponse(BaseModel):
    """Synchronous JSON response for POST /query (stream=false)."""

    run_id: str
    query: str
    query_type: str
    answer: str
    citations: list[CitationItem]
    sources: list[dict[str, Any]]
    chunks_used: int
    input_tokens: int
    output_tokens: int
    latency_ms: int
    model_used: str
    status: str


class StreamInitResponse(BaseModel):
    """Response returned when stream=true, pointing to the SSE stream URL."""

    run_id: str
    stream_url: str
    status: str = "running"


# In-memory registry fallback for streaming tasks (backed by Redis when available)
_active_streams: dict[str, dict[str, Any]] = {}


@router.post("/query")
async def post_query(request: QueryRequest, req: Request):
    """Execute RAG generation pipeline.

    If stream=false, returns complete JSON response with citations and metadata.
    If stream=true, creates run ID and initiates stream channel for GET /query/{run_id}/stream.
    """
    cleaned_query = sanitize_text(request.query)
    if not cleaned_query.strip():
        raise HTTPException(status_code=422, detail="Query must contain text.")
    cleaned_query, secret_types = redact_secrets(cleaned_query)
    for pattern_type in sorted(set(secret_types)):
        logger.warning("Sensitive value redacted from query (%s).", pattern_type)
    injection = detect_prompt_injection(cleaned_query)
    if injection.detected:
        logger.warning("User query matched prompt injection signal %s", injection.pattern)
    request.query = cleaned_query
    pipeline = get_generation_pipeline(req)
    try:
        if await classify_query(getattr(pipeline, "client", None), cleaned_query):
            return JSONResponse(status_code=400, content={"error": "query_rejected", "reason": "potential_prompt_injection"})
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Prompt injection classifier unavailable (%s); failing closed.", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Query security classification unavailable.") from exc
    run_id = uuid4()

    if not request.stream:
        started = time.perf_counter()
        try:
            result: GenerationResult = await pipeline.generate(
                query=request.query,
                pipeline_id=request.pipeline_id,
                run_id=run_id,
                top_k=request.top_k,
                max_context_tokens=request.max_context_tokens,
                rerank=request.rerank,
                pipeline_version=request.pipeline_version,
            )
            QUERY_TOTAL.labels(status=result.status).inc()
            QUERY_LATENCY.observe(time.perf_counter() - started)
            return QueryResponse(
                run_id=result.run_id,
                query=result.query,
                query_type=result.query_type,
                answer=result.generation,
                citations=[
                    CitationItem(**c.to_dict()) for c in result.citations.all_citations
                ],
                sources=result.sources,
                chunks_used=result.chunks_used,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                latency_ms=result.latency_ms,
                model_used=result.model_used,
                status=result.status,
            )
        except Exception as exc:
            QUERY_TOTAL.labels(status="failed").inc()
            QUERY_LATENCY.observe(time.perf_counter() - started)
            logger.exception("Query generation failed: %s", exc)
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(exc)}") from exc

    # Stream=True: register run context and return stream URL
    run_id_str = str(run_id)
    _active_streams[run_id_str] = {
        "query": request.query,
        "pipeline_id": request.pipeline_id,
        "top_k": request.top_k,
        "max_context_tokens": request.max_context_tokens,
        "rerank": request.rerank,
    }

    return StreamInitResponse(
        run_id=run_id_str,
        stream_url=f"/query/{run_id_str}/stream",
        status="running",
    )


@router.get("/query/{run_id}/stream")
async def get_query_stream(run_id: UUID, req: Request):
    """Server-Sent Events endpoint streaming retrieval and token events."""
    run_id_str = str(run_id)
    pipeline = get_generation_pipeline(req)

    stream_info = _active_streams.get(run_id_str)
    if not stream_info:
        # Check if run exists in DB
        raise HTTPException(status_code=404, detail=f"Active stream for run {run_id} not found.")

    query = stream_info["query"]
    pipeline_id = stream_info["pipeline_id"]
    top_k = stream_info["top_k"]
    max_context_tokens = stream_info["max_context_tokens"]
    rerank = stream_info["rerank"]

    async def event_generator() -> AsyncGenerator[dict[str, Any], None]:
        started = time.perf_counter()
        query_status = "completed"
        try:
            async for event in pipeline.stream_events(
                query=query,
                pipeline_id=pipeline_id,
                run_id=run_id,
                top_k=top_k,
                max_context_tokens=max_context_tokens,
                rerank=rerank,
            ):
                if event.get("type") == "error":
                    query_status = "failed"
                # Yield SSE data as JSON string
                yield {
                    "event": "message",
                    "data": json.dumps(event),
                }
        except asyncio.CancelledError:
            query_status = "cancelled"
            raise
        except Exception:
            query_status = "failed"
            raise
        finally:
            QUERY_TOTAL.labels(status=query_status).inc()
            QUERY_LATENCY.observe(time.perf_counter() - started)
            # Clean up active stream registration
            _active_streams.pop(run_id_str, None)

    # ping=15 sets a keepalive SSE comment ping every 15 seconds
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
