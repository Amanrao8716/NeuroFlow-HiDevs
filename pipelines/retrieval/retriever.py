"""Hybrid parallel retriever combining dense, sparse, and metadata search."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import logging
from typing import Any, Callable
from uuid import UUID

from db.pool import get_pool, pipeline_connection
from pipelines.retrieval.query_processor import ProcessedQuery, QueryProcessor

logger = logging.getLogger("neuroflow.retrieval.retriever")


@dataclass
class RetrievalResult:
    """A single retrieved chunk with its scores, source, and metadata."""

    chunk_id: str
    document_id: str
    content: str
    score: float
    source_type: str = "text"
    filename: str | None = None
    page_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieval_method: str = "dense"
    rank: int | None = None

    def clone(self, **kwargs) -> RetrievalResult:
        """Create a copy with optional overridden fields."""
        data = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "score": self.score,
            "source_type": self.source_type,
            "filename": self.filename,
            "page_number": self.page_number,
            "metadata": dict(self.metadata),
            "retrieval_method": self.retrieval_method,
            "rank": self.rank,
        }
        data.update(kwargs)
        return RetrievalResult(**data)


def _extract_page_number(metadata: dict[str, Any]) -> int | None:
    """Extract page number from chunk or page metadata if available."""
    if not isinstance(metadata, dict):
        return None
    if "page_number" in metadata:
        try:
            return int(metadata["page_number"])
        except (ValueError, TypeError):
            pass
    pages = metadata.get("pages")
    if isinstance(pages, list) and pages:
        first = pages[0]
        if isinstance(first, dict) and "page_number" in first:
            try:
                return int(first["page_number"])
            except (ValueError, TypeError):
                pass
        elif isinstance(first, int):
            return first
    return None


class HybridRetriever:
    """Parallel hybrid retriever executing dense, sparse, and metadata searches."""

    def __init__(
        self,
        embed_fn: Callable[[list[str]], Any] | None = None,
        query_processor: QueryProcessor | None = None,
        pool: Any | None = None,
    ):
        """Initialize hybrid retriever with embedder, query processor, and connection pool."""
        self._embed_fn = embed_fn
        self.query_processor = query_processor or QueryProcessor()
        self.pool = pool

    async def _get_embed_fn(self) -> Callable[[list[str]], Any]:
        """Resolve or lazily create the embedding function using NeuroFlowClient."""
        if self._embed_fn is not None:
            return self._embed_fn
        from providers.client import NeuroFlowClient

        client = NeuroFlowClient(None)
        self._embed_fn = client.embed
        return self._embed_fn

    async def retrieve_parallel(
        self,
        query: str | ProcessedQuery,
        k: int = 20,
        pipeline_id: UUID | None = None,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[RetrievalResult], list[RetrievalResult], list[RetrievalResult]]:
        """Run dense, sparse, and metadata-filtered retrieval concurrently.

        Returns:
            Tuple of (dense_results, sparse_results, metadata_results)
        """
        if isinstance(query, str):
            processed = await self.query_processor.process(
                query, explicit_filters=filters
            )
        else:
            processed = query
            if filters:
                processed.filters.update(filters)

        queries_to_embed = processed.all_queries
        active_filters = processed.filters

        # 1. Prepare dense query vectors
        embed_fn = await self._get_embed_fn()
        try:
            query_vectors = await embed_fn(queries_to_embed)
        except Exception as exc:
            logger.warning("Query embedding generation failed: %s", exc)
            query_vectors = []

        primary_vector = query_vectors[0] if query_vectors else None

        # 2. Concurrently run retrieval branches
        dense_task = self._dense_search(
            query_vectors, k=k, pipeline_id=pipeline_id
        )
        sparse_task = self._sparse_search(
            processed.query, k=k, pipeline_id=pipeline_id
        )

        if active_filters:
            meta_task = self._metadata_search(
                active_filters,
                query_vector=primary_vector,
                k=k,
                pipeline_id=pipeline_id,
            )
            dense_res, sparse_res, meta_res = await asyncio.gather(
                dense_task, sparse_task, meta_task, return_exceptions=False
            )
        else:
            dense_res, sparse_res = await asyncio.gather(
                dense_task, sparse_task, return_exceptions=False
            )
            meta_res = []

        return dense_res, sparse_res, meta_res

    async def retrieve(
        self,
        query: str | ProcessedQuery,
        k: int = 20,
        pipeline_id: UUID | None = None,
        filters: dict[str, Any] | None = None,
        rrf_k: int = 60,
    ) -> list[RetrievalResult]:
        """Execute parallel hybrid retrieval and fuse candidates using RRF."""
        from pipelines.retrieval.fusion import reciprocal_rank_fusion

        dense_res, sparse_res, meta_res = await self.retrieve_parallel(
            query=query, k=k, pipeline_id=pipeline_id, filters=filters
        )

        result_lists = [dense_res, sparse_res]
        if meta_res:
            result_lists.append(meta_res)

        fused = reciprocal_rank_fusion(result_lists, k=rrf_k)
        return fused[:k]

    async def _acquire_connection(self, pipeline_id: UUID | None):
        """Context manager yielding a database connection with tenant context."""
        if pipeline_id is not None:
            return pipeline_connection(pipeline_id)
        pool = self.pool or get_pool()
        return pool.acquire()

    async def _dense_search(
        self,
        query_vectors: list[list[float]],
        k: int,
        pipeline_id: UUID | None,
    ) -> list[RetrievalResult]:
        """Search chunks using pgvector cosine distance `<=>` for all query vectors."""
        if not query_vectors:
            return []

        best_by_chunk: dict[str, RetrievalResult] = {}

        async with await self._acquire_connection(pipeline_id) as conn:
            for vec in query_vectors:
                vec_str = "[" + ",".join(str(v) for v in vec) + "]"
                rows = await conn.fetch(
                    """
                    SELECT c.id AS chunk_id, c.document_id, c.content, c.chunk_index,
                           c.token_count, c.metadata, d.filename, d.source_type,
                           (1.0 - (c.embedding <=> $1::vector)) AS score
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE c.embedding IS NOT NULL
                    ORDER BY c.embedding <=> $1::vector ASC
                    LIMIT $2
                    """,
                    vec_str,
                    k,
                )

                for row in rows:
                    cid = str(row["chunk_id"])
                    score = float(row["score"])
                    raw_meta = row["metadata"]
                    meta = (
                        json.loads(raw_meta)
                        if isinstance(raw_meta, str)
                        else raw_meta or {}
                    )

                    if cid not in best_by_chunk or score > best_by_chunk[cid].score:
                        best_by_chunk[cid] = RetrievalResult(
                            chunk_id=cid,
                            document_id=str(row["document_id"]),
                            content=row["content"],
                            score=score,
                            source_type=row["source_type"] or "text",
                            filename=row["filename"],
                            page_number=_extract_page_number(meta),
                            metadata=meta,
                            retrieval_method="dense",
                        )

        # Sort descending by score and assign ranks
        sorted_results = sorted(
            best_by_chunk.values(), key=lambda r: r.score, reverse=True
        )[:k]
        for idx, res in enumerate(sorted_results):
            res.rank = idx + 1

        return sorted_results

    async def _sparse_search(
        self,
        query: str,
        k: int,
        pipeline_id: UUID | None,
    ) -> list[RetrievalResult]:
        """Search chunks using PostgreSQL full-text search with cover-density ranking."""
        trimmed = query.strip()
        if not trimmed:
            return []

        async with await self._acquire_connection(pipeline_id) as conn:
            rows = await conn.fetch(
                """
                SELECT c.id AS chunk_id, c.document_id, c.content, c.chunk_index,
                       c.token_count, c.metadata, d.filename, d.source_type,
                       ts_rank_cd(to_tsvector('english', c.content), plainto_tsquery('english', $1)) AS score
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE to_tsvector('english', c.content) @@ plainto_tsquery('english', $1)
                ORDER BY score DESC
                LIMIT $2
                """,
                trimmed,
                k,
            )

            results: list[RetrievalResult] = []
            for idx, row in enumerate(rows):
                raw_meta = row["metadata"]
                meta = (
                    json.loads(raw_meta)
                    if isinstance(raw_meta, str)
                    else raw_meta or {}
                )
                results.append(
                    RetrievalResult(
                        chunk_id=str(row["chunk_id"]),
                        document_id=str(row["document_id"]),
                        content=row["content"],
                        score=float(row["score"]),
                        source_type=row["source_type"] or "text",
                        filename=row["filename"],
                        page_number=_extract_page_number(meta),
                        metadata=meta,
                        retrieval_method="sparse",
                        rank=idx + 1,
                    )
                )

        return results

    async def _metadata_search(
        self,
        filters: dict[str, Any],
        query_vector: list[float] | None,
        k: int,
        pipeline_id: UUID | None,
    ) -> list[RetrievalResult]:
        """Search chunks matching JSONB metadata filters, optionally ordered by vector similarity."""
        if not filters:
            return []

        filters_json = json.dumps(filters)

        async with await self._acquire_connection(pipeline_id) as conn:
            if query_vector:
                vec_str = "[" + ",".join(str(v) for v in query_vector) + "]"
                rows = await conn.fetch(
                    """
                    SELECT c.id AS chunk_id, c.document_id, c.content, c.chunk_index,
                           c.token_count, c.metadata, d.filename, d.source_type,
                           (CASE WHEN c.embedding IS NOT NULL
                                 THEN (1.0 - (c.embedding <=> $2::vector))
                                 ELSE 1.0 END) AS score
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE (c.metadata @> $1::jsonb OR d.metadata @> $1::jsonb)
                    ORDER BY score DESC
                    LIMIT $3
                    """,
                    filters_json,
                    vec_str,
                    k,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT c.id AS chunk_id, c.document_id, c.content, c.chunk_index,
                           c.token_count, c.metadata, d.filename, d.source_type,
                           1.0 AS score
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE (c.metadata @> $1::jsonb OR d.metadata @> $1::jsonb)
                    LIMIT $2
                    """,
                    filters_json,
                    k,
                )

            results: list[RetrievalResult] = []
            for idx, row in enumerate(rows):
                raw_meta = row["metadata"]
                meta = (
                    json.loads(raw_meta)
                    if isinstance(raw_meta, str)
                    else raw_meta or {}
                )
                results.append(
                    RetrievalResult(
                        chunk_id=str(row["chunk_id"]),
                        document_id=str(row["document_id"]),
                        content=row["content"],
                        score=float(row["score"]),
                        source_type=row["source_type"] or "text",
                        filename=row["filename"],
                        page_number=_extract_page_number(meta),
                        metadata=meta,
                        retrieval_method="metadata",
                        rank=idx + 1,
                    )
                )

        return results
