"""FastAPI endpoint for hybrid retrieval, reranking, and context assembly."""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from pipelines.retrieval import (
    ContextAssembler,
    CrossEncoderReranker,
    HybridRetriever,
    QueryProcessor,
    reciprocal_rank_fusion,
)

router = APIRouter(tags=["retrieval"])
logger = logging.getLogger("neuroflow.api.retrieve")

_processor = QueryProcessor()
_retriever = HybridRetriever(query_processor=_processor)
_reranker = CrossEncoderReranker()
_assembler = ContextAssembler()


class RetrieveRequest(BaseModel):
    """Retrieval query request body."""

    query: str = Field(..., min_length=1, max_length=20000, description="User search query")
    pipeline_id: UUID = Field(..., description="Tenant pipeline ID for data isolation")
    filters: dict[str, Any] | None = Field(default=None, description="Optional metadata filters")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results to return")
    max_context_tokens: int = Field(
        default=4000, ge=100, le=20000, description="Token budget for assembled context"
    )
    rerank: bool = Field(default=True, description="Whether to apply cross-encoder reranking")


class ChunkResponse(BaseModel):
    """Metadata and content for a single retrieved chunk."""

    chunk_id: str
    document_id: str
    content: str
    score: float
    rank: int | None
    source_type: str
    filename: str | None
    page_number: int | None
    metadata: dict[str, Any]


class RetrieveResponse(BaseModel):
    """Full hybrid retrieval and context assembly response."""

    query: str
    query_type: str
    expanded_queries: list[str]
    filters_applied: dict[str, Any]
    context: str
    chunks_used: int
    total_tokens: int
    sources: list[dict[str, Any]]
    chunks: list[ChunkResponse]


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_endpoint(request: RetrieveRequest):
    """Execute hybrid retrieval, reciprocal rank fusion, optional reranking, and context assembly."""
    try:
        # 1. Query Processing
        processed = await _processor.process(
            request.query, explicit_filters=request.filters
        )

        # 2. Parallel Multi-Method Retrieval (Dense, Sparse, Metadata)
        dense_res, sparse_res, meta_res = await _retriever.retrieve_parallel(
            query=processed,
            k=request.top_k * 2,
            pipeline_id=request.pipeline_id,
            filters=processed.filters,
        )

        # 3. Reciprocal Rank Fusion
        result_lists = [dense_res, sparse_res]
        if meta_res:
            result_lists.append(meta_res)
        fused = reciprocal_rank_fusion(result_lists)

        # 4. Optional Cross-Encoder Reranking
        if request.rerank and fused:
            try:
                candidates = await _reranker.rerank(processed.query, fused, top_n=40)
            except Exception as rerank_err:
                logger.warning("Reranking failed, using RRF candidates: %s", rerank_err)
                candidates = fused
        else:
            candidates = fused

        top_candidates = candidates[: request.top_k]

        # 5. Context Assembly with Token Budget
        assembled = _assembler.assemble(
            top_candidates, max_tokens=request.max_context_tokens
        )

        chunk_responses = [
            ChunkResponse(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                content=c.content,
                score=round(c.score, 6),
                rank=c.rank,
                source_type=c.source_type,
                filename=c.filename,
                page_number=c.page_number,
                metadata=c.metadata,
            )
            for c in top_candidates
        ]

        return RetrieveResponse(
            query=processed.query,
            query_type=processed.query_type,
            expanded_queries=processed.expanded_queries,
            filters_applied=processed.filters,
            context=assembled.context,
            chunks_used=assembled.chunks_used,
            total_tokens=assembled.total_tokens,
            sources=assembled.sources,
            chunks=chunk_responses,
        )

    except Exception as exc:
        logger.exception("Retrieval pipeline failed for query '%s'", request.query)
        raise HTTPException(
            status_code=500, detail=f"Retrieval failed: {str(exc)}"
        ) from exc
