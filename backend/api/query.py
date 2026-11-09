"""FastAPI endpoint for Task 6 RAG generation, progressive streaming SSE, and citation tracking."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from observability.metrics import QUERY_LATENCY, QUERY_TOTAL
from pydantic import BaseModel, ConfigDict, Field
from sse_starlette.sse import EventSourceResponse

from backend.security.auth import require_scope
from backend.security.prompt_injection import classify_query, detect_prompt_injection
from backend.security.secrets import redact_secrets
from backend.security.validators import sanitize_text
from pipelines.generation.generator import GenerationPipeline, GenerationResult

router = APIRouter(tags=["Query"], dependencies=[Depends(require_scope("query"))])
logger = logging.getLogger("neuroflow.api.query")

_pipeline_instance: GenerationPipeline | None = None


def get_generation_pipeline(request: Request | None = None) -> GenerationPipeline:
    """Obtain or initialize the GenerationPipeline singleton."""
    global _pipeline_instance
    redis_client = (
        getattr(request.app.state, "redis", None) if request and hasattr(request, "app") else None
    )
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

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "How does reciprocal rank fusion merge dense and sparse results?",
                "pipeline_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "stream": False,
                "top_k": 10,
                "max_context_tokens": 4000,
                "rerank": True,
                "pipeline_version": 1,
            }
        }
    )

    query: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="User search query",
        examples=["How does reciprocal rank fusion merge dense and sparse results?"],
    )
    pipeline_id: UUID = Field(
        ...,
        description="Tenant pipeline ID for data isolation",
        examples=["3fa85f64-5717-4562-b3fc-2c963f66afa6"],
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream response tokens via SSE",
        examples=[False],
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Top K retrieval candidates",
        examples=[10],
    )
    max_context_tokens: int = Field(
        default=4000,
        ge=100,
        le=20000,
        description="Token budget for context",
        examples=[4000],
    )
    rerank: bool = Field(
        default=True,
        description="Whether to apply reranking",
        examples=[True],
    )
    pipeline_version: int | None = Field(
        default=None,
        ge=1,
        description="Specific pipeline version to execute",
        examples=[1],
    )


class CitationItem(BaseModel):
    """Structured citation returned to API client."""

    reference: str = Field(
        ...,
        description="Raw citation reference string from the generated answer",
        examples=["[1]"],
    )
    source_number: int = Field(
        ...,
        description="Sequential citation number, 1-based",
        examples=[1],
    )
    chunk_id: str | None = Field(
        None,
        description="UUID of the retrieved chunk backing this citation",
        examples=["a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d"],
    )
    document_name: str | None = Field(
        None,
        description="Filename of the source document",
        examples=["rag_architecture.pdf"],
    )
    page_number: int | None = Field(
        None,
        description="Page number in the source document, if available",
        examples=[4],
    )
    content_preview: str | None = Field(
        None,
        description="Short excerpt from the source chunk",
        examples=["Reciprocal rank fusion assigns weighted scores..."],
    )
    is_valid: bool = Field(
        ...,
        description="Whether the citation resolves to a real retrieved chunk",
        examples=[True],
    )
    error: str | None = Field(
        None,
        description="Validation error if citation is invalid",
        examples=[None],
    )


class QueryResponse(BaseModel):
    """Synchronous JSON response for POST /query (stream=false)."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "run_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "query": "How does RRF work?",
                "query_type": "technical",
                "answer": "RRF combines ranks from multiple retrievers using reciprocal rank scoring [1].",
                "citations": [
                    {
                        "reference": "[1]",
                        "source_number": 1,
                        "chunk_id": "a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d",
                        "document_name": "rag_architecture.pdf",
                        "page_number": 4,
                        "content_preview": "Reciprocal rank fusion assigns weighted scores...",
                        "is_valid": True,
                        "error": None,
                    }
                ],
                "sources": [],
                "chunks_used": 1,
                "input_tokens": 350,
                "output_tokens": 45,
                "latency_ms": 420,
                "model_used": "gpt-4o-mini",
                "status": "completed",
            }
        }
    )

    run_id: str = Field(..., description="Unique ID for this query pipeline run", examples=["3fa85f64-5717-4562-b3fc-2c963f66afa6"])
    query: str = Field(..., description="Original user query", examples=["How does RRF work?"])
    query_type: str = Field(..., description="Classified intent (e.g. technical, factual, comparison)", examples=["technical"])
    answer: str = Field(..., description="Generated answer with reference markers", examples=["RRF combines ranks..."])
    citations: list[CitationItem] = Field(default_factory=list, description="Extracted and verified citations")
    sources: list[dict[str, Any]] = Field(default_factory=list, description="Retrieved source chunks metadata")
    chunks_used: int = Field(..., description="Count of retrieved chunks assembled into context", examples=[1])
    input_tokens: int = Field(..., description="Prompt token count", examples=[350])
    output_tokens: int = Field(..., description="Completion token count", examples=[45])
    latency_ms: int = Field(..., description="Total execution latency in milliseconds", examples=[420])
    model_used: str = Field(..., description="Model identifier used for generation", examples=["gpt-4o-mini"])
    status: str = Field(..., description="Execution status ('completed' or 'failed')", examples=["completed"])


class StreamInitResponse(BaseModel):
    """Response returned when stream=true, pointing to the SSE stream URL."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "run_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "stream_url": "/query/3fa85f64-5717-4562-b3fc-2c963f66afa6/stream",
                "status": "running",
            }
        }
    )

    run_id: str = Field(..., description="UUID assigned to this streaming execution run", examples=["3fa85f64-5717-4562-b3fc-2c963f66afa6"])
    stream_url: str = Field(..., description="Relative URL to subscribe for Server-Sent Events", examples=["/query/3fa85f64-5717-4562-b3fc-2c963f66afa6/stream"])
    status: str = Field(default="running", description="Initial streaming status", examples=["running"])


# In-memory registry fallback for streaming tasks (backed by Redis when available)
_active_streams: dict[str, dict[str, Any]] = {}


@router.post(
    "/query",
    summary="Execute RAG query",
    description=(
        "Run a full retrieval-augmented generation pipeline. "
        "Set stream=false (default) for JSON response, stream=true for SSE.\n\n"
        "**Requires scope**: `query`"
    ),
    response_description="QueryResponse (stream=false) or StreamInitResponse (stream=true).",
)
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
            return JSONResponse(
                status_code=400,
                content={"error": "query_rejected", "reason": "potential_prompt_injection"},
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning(
            "Prompt injection classifier unavailable (%s); failing closed.", type(exc).__name__
        )
        raise HTTPException(
            status_code=503, detail="Query security classification unavailable."
        ) from exc
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
                citations=[CitationItem(**c.to_dict()) for c in result.citations.all_citations],
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


@router.get(
    "/query/{run_id}/stream",
    summary="Stream query results via SSE",
    description=(
        "Connect after POST /query with stream=true. Returns text/event-stream of generation events.\n\n"
        "**Requires scope**: `query`"
    ),
    response_description="SSE stream of generation events.",
)
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
