"""FastAPI endpoint for Task 6 RAG generation, progressive streaming SSE, and citation tracking."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
import logging
from typing import Any, AsyncGenerator
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from pipelines.generation.generator import GenerationPipeline, GenerationResult

router = APIRouter(tags=["generation"])
logger = logging.getLogger("neuroflow.api.query")

_pipeline_instance: GenerationPipeline | None = None


def get_generation_pipeline(request: Request | None = None) -> GenerationPipeline:
    """Obtain or initialize the GenerationPipeline singleton."""
    global _pipeline_instance
    redis_client = getattr(request.app.state, "redis", None) if request and hasattr(request, "app") else None
    if _pipeline_instance is None:
        _pipeline_instance = GenerationPipeline(redis_client=redis_client)
    elif _pipeline_instance.client is None and redis_client is not None:
        from providers.client import NeuroFlowClient
        _pipeline_instance.redis = redis_client
        _pipeline_instance.client = NeuroFlowClient(redis_client)
    return _pipeline_instance


class QueryRequest(BaseModel):
    """Request model for POST /query endpoint."""

    query: str = Field(..., min_length=1, max_length=20000, description="User search query")
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
    pipeline = get_generation_pipeline(req)
    run_id = uuid4()

    if not request.stream:
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
        try:
            async for event in pipeline.stream_events(
                query=query,
                pipeline_id=pipeline_id,
                run_id=run_id,
                top_k=top_k,
                max_context_tokens=max_context_tokens,
                rerank=rerank,
            ):
                # Yield SSE data as JSON string
                yield {
                    "event": "message",
                    "data": json.dumps(event),
                }
        finally:
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
