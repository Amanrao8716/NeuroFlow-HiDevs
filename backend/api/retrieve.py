"""FastAPI endpoint for hybrid retrieval, reranking, and context assembly."""

from __future__ import annotations

import logging
import os
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.security.auth import require_scope
from backend.security.prompt_injection import classify_query, detect_prompt_injection
from backend.security.secrets import redact_secrets
from backend.security.validators import sanitize_text
from pipelines.retrieval import (
    ContextAssembler,
    CrossEncoderReranker,
    HybridRetriever,
    QueryProcessor,
    reciprocal_rank_fusion,
)

router = APIRouter(tags=["Query"], dependencies=[Depends(require_scope("query"))])
logger = logging.getLogger("neuroflow.api.retrieve")

_processor = QueryProcessor()
_retriever = HybridRetriever(query_processor=_processor)
_reranker = CrossEncoderReranker()
_assembler = ContextAssembler()


class RetrieveRequest(BaseModel):
    """Retrieval query request body."""

    query: str = Field(..., min_length=1, max_length=5000, description="User search query")
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


class RetrievalRankingResponse(BaseModel):
    """Actual chunk ID rankings from each retrieval stage for isolated benchmarking."""

    dense_only: list[str]
    sparse_only: list[str]
    hybrid_rrf: list[str]
    hybrid_reranked: list[str]


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    summary="Execute hybrid retrieval",
    description=(
        "Run hybrid dense+sparse retrieval, RRF fusion, reranking, and context assembly. "
        "**Requires scope**: `query`"
    ),
    response_description="Retrieved chunks with scores, assembled context, and retrieval metadata.",
)
async def retrieve_endpoint(request: RetrieveRequest, req: Request) -> RetrieveResponse:
    """Execute hybrid retrieval, reciprocal rank fusion, and context assembly."""
    try:
        # 1. Query Processing
        query_text = sanitize_text(request.query)
        if not query_text.strip():
            raise HTTPException(status_code=422, detail="Query must contain text.")
        query_text, secret_types = redact_secrets(query_text)
        for pattern_type in sorted(set(secret_types)):
            logger.warning("Sensitive value redacted from query (%s).", pattern_type)
        detection = detect_prompt_injection(query_text)
        if detection.detected:
            logger.warning("User query matched prompt injection signal %s", detection.pattern)
        from api.query import get_generation_pipeline

        try:
            if await classify_query(
                getattr(get_generation_pipeline(req), "client", None), query_text
            ):
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
        processed = await _processor.process(query_text, explicit_filters=request.filters)

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
        assembled = _assembler.assemble(top_candidates, max_tokens=request.max_context_tokens)

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
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(exc)}") from exc


@router.post("/__test__/retrieval-rankings", response_model=RetrievalRankingResponse)
async def retrieval_rankings_for_test(
    request: RetrieveRequest,
    req: Request,
    _admin: object = Depends(require_scope("admin")),
) -> RetrievalRankingResponse:
    """Return actual retrieval rankings only when an isolated test deployment opts in."""
    if os.getenv("NEUROFLOW_TEST_BENCHMARK_ENDPOINT") != "true":
        raise HTTPException(status_code=404, detail="Not found.")
    if os.getenv("HF_HUB_OFFLINE") != "1" or os.getenv("TRANSFORMERS_OFFLINE") != "1":
        raise HTTPException(
            status_code=503,
            detail=(
                "Benchmark model loading must be offline; "
                "configure HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1."
            ),
        )

    query_text = sanitize_text(request.query)
    if not query_text.strip():
        raise HTTPException(status_code=422, detail="Query must contain text.")
    query_text, secret_types = redact_secrets(query_text)
    for pattern_type in sorted(set(secret_types)):
        logger.warning("Sensitive value redacted from benchmark query (%s).", pattern_type)
    detection = detect_prompt_injection(query_text)
    if detection.detected:
        raise HTTPException(status_code=400, detail="Benchmark query rejected by security policy.")
    from api.query import get_generation_pipeline

    try:
        if await classify_query(getattr(get_generation_pipeline(req), "client", None), query_text):
            raise HTTPException(
                status_code=400, detail="Benchmark query rejected by security policy."
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Benchmark query classifier unavailable (%s).", type(exc).__name__)
        raise HTTPException(
            status_code=503, detail="Query security classification unavailable."
        ) from exc

    processed = await _processor.process(query_text, explicit_filters=request.filters)
    dense, sparse, _metadata = await _retriever.retrieve_parallel(
        query=processed,
        k=request.top_k,
        pipeline_id=request.pipeline_id,
        filters=processed.filters,
    )
    fused = reciprocal_rank_fusion([dense, sparse])
    reranked = await _reranker.rerank(processed.query, fused, top_n=40) if fused else []
    return RetrievalRankingResponse(
        dense_only=[item.chunk_id for item in dense],
        sparse_only=[item.chunk_id for item in sparse],
        hybrid_rrf=[item.chunk_id for item in fused],
        hybrid_reranked=[item.chunk_id for item in reranked],
    )
