"""RAG Generation Orchestrator, Stream Generator, and Pipeline Run Logging."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import logging
import time
from typing import Any, AsyncGenerator
from uuid import UUID, uuid4

try:
    from db.pool import pipeline_connection
except ImportError:
    pipeline_connection = None
from pipelines.generation.citations import CitationTracker, ParsedCitations
from pipelines.generation.prompt_builder import AssembledPrompt, PromptBuilder
from pipelines.retrieval import (
    ContextAssembler,
    CrossEncoderReranker,
    HybridRetriever,
    QueryProcessor,
    reciprocal_rank_fusion,
)
from pipelines.retrieval.context_assembler import TokenCounter
from providers.base import ChatMessage
from providers.client import NeuroFlowClient

logger = logging.getLogger("neuroflow.generation")


@dataclass
class GenerationResult:
    """Final output from the RAG generation pipeline."""

    run_id: str
    query: str
    query_type: str
    generation: str
    citations: ParsedCitations
    sources: list[dict[str, Any]]
    chunks_used: int
    input_tokens: int
    output_tokens: int
    latency_ms: int
    model_used: str
    status: str
    error_message: str | None = None
    pipeline_version: int | None = None
    retrieval_latency_ms: int | None = None
    cost_usd: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "query": self.query,
            "query_type": self.query_type,
            "generation": self.generation,
            "citations": self.citations.to_dict_list(),
            "sources": self.sources,
            "chunks_used": self.chunks_used,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
            "model_used": self.model_used,
            "status": self.status,
            "error_message": self.error_message,
            "pipeline_version": self.pipeline_version,
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "cost_usd": self.cost_usd,
        }


class GenerationPipeline:
    """Orchestrates query processing, hybrid retrieval, prompt assembly, streaming LLM generation,

    citation tracking, and persistence to the pipeline_runs table.
    """

    def __init__(
        self,
        redis_client: Any = None,
        query_processor: QueryProcessor | None = None,
        retriever: HybridRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        assembler: ContextAssembler | None = None,
        prompt_builder: PromptBuilder | None = None,
        citation_tracker: CitationTracker | None = None,
        token_counter: TokenCounter | None = None,
        llm_client: NeuroFlowClient | None = None,
    ):
        self.redis = redis_client
        self.processor = query_processor or QueryProcessor()
        self.retriever = retriever or HybridRetriever(query_processor=self.processor)
        self.reranker = reranker or CrossEncoderReranker()
        self.assembler = assembler or ContextAssembler()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.citation_tracker = citation_tracker or CitationTracker()
        self.token_counter = token_counter or TokenCounter()
        self.client = llm_client or (NeuroFlowClient(redis_client) if redis_client else None)

    async def _load_pipeline_config(
        self,
        pipeline_id: UUID,
        version: int | None = None,
    ) -> tuple[int, dict[str, Any] | None]:
        """Load pipeline configuration for a specific version or latest active version."""
        if pipeline_connection is None:
            return (1, None)
        try:
            async with pipeline_connection(pipeline_id) as conn:
                if version is not None:
                    row = await conn.fetchrow(
                        """
                        SELECT version, config
                        FROM pipeline_versions
                        WHERE pipeline_id = $1 AND version = $2
                        """,
                        pipeline_id,
                        version,
                    )
                else:
                    row = await conn.fetchrow(
                        """
                        SELECT version, config
                        FROM pipelines
                        WHERE id = $1
                        """,
                        pipeline_id,
                    )
                if row:
                    raw_cfg = row["config"]
                    cfg_dict = json.loads(raw_cfg) if isinstance(raw_cfg, str) else dict(raw_cfg)
                    return (row["version"], cfg_dict)
        except Exception as exc:
            logger.debug("Could not load config for pipeline %s: %s", pipeline_id, exc)
        return (1, None)

    async def _create_pipeline_run(
        self,
        run_id: UUID,
        pipeline_id: UUID,
        query: str,
        retrieved_chunk_ids: list[UUID],
        prompt: str,
        pipeline_version: int | None = None,
        retrieval_latency_ms: int | None = None,
        execution_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist a new pipeline_runs record BEFORE the LLM call with status='running'."""
        try:
            async with pipeline_connection(pipeline_id) as conn:
                await conn.execute(
                    """
                    INSERT INTO pipeline_runs (
                        id, pipeline_id, pipeline_version, query, retrieved_chunk_ids,
                        prompt, status, retrieval_latency_ms, execution_metadata
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, 'running', $7, $8::jsonb)
                    ON CONFLICT (id) DO UPDATE SET
                        pipeline_version = EXCLUDED.pipeline_version,
                        query = EXCLUDED.query,
                        retrieved_chunk_ids = EXCLUDED.retrieved_chunk_ids,
                        prompt = EXCLUDED.prompt,
                        retrieval_latency_ms = EXCLUDED.retrieval_latency_ms,
                        execution_metadata = EXCLUDED.execution_metadata,
                        status = 'running'
                    """,
                    run_id,
                    pipeline_id,
                    pipeline_version,
                    query,
                    retrieved_chunk_ids,
                    prompt,
                    retrieval_latency_ms,
                    json.dumps(execution_metadata or {}),
                )
        except Exception as exc:
            logger.exception("Failed to create pipeline_run %s before LLM call: %s", run_id, exc)
            # Re-raise so pipeline run is strictly logged
            raise

    async def _update_pipeline_run_success(
        self,
        run_id: UUID,
        pipeline_id: UUID,
        generation: str,
        citations: list[dict[str, Any]],
        input_tokens: int,
        output_tokens: int,
        latency_ms: int,
        model_used: str,
        cost_usd: float | None = None,
    ) -> None:
        """Update the pipeline_runs record with completed generation and metrics."""
        try:
            async with pipeline_connection(pipeline_id) as conn:
                await conn.execute(
                    """
                    UPDATE pipeline_runs
                    SET generation = $1,
                        citations = $2::jsonb,
                        input_tokens = $3,
                        output_tokens = $4,
                        latency_ms = $5,
                        model_used = $6,
                        cost_usd = $7,
                        status = 'completed'
                    WHERE id = $8 AND pipeline_id = $9
                    """,
                    generation,
                    json.dumps(citations),
                    input_tokens,
                    output_tokens,
                    latency_ms,
                    model_used,
                    cost_usd,
                    run_id,
                    pipeline_id,
                )
        except Exception as exc:
            logger.exception("Failed to update pipeline_run %s on success: %s", run_id, exc)

    async def _update_pipeline_run_failure(
        self,
        run_id: UUID,
        pipeline_id: UUID,
        error_message: str,
        latency_ms: int = 0,
    ) -> None:
        """Update the pipeline_runs record with failed status and safe error message."""
        try:
            safe_error = error_message[:1000]
            async with pipeline_connection(pipeline_id) as conn:
                await conn.execute(
                    """
                    UPDATE pipeline_runs
                    SET status = 'failed',
                        error_message = $1,
                        latency_ms = $2
                    WHERE id = $3 AND pipeline_id = $4
                    """,
                    safe_error,
                    latency_ms,
                    run_id,
                    pipeline_id,
                )
        except Exception as exc:
            logger.exception("Failed to update pipeline_run %s on failure: %s", run_id, exc)

    async def _enqueue_eval_job(
        self,
        run_id: str,
        pipeline_id: str,
    ) -> None:
        """Enqueue Task 37 evaluation job asynchronously into Redis/arq queue."""
        if not self.redis:
            logger.debug("Redis client not available, skipping eval job enqueue.")
            return

        try:
            # arq job enqueue convention or Redis queue
            payload = json.dumps({
                "job": "evaluate_pipeline_run",
                "run_id": run_id,
                "pipeline_id": pipeline_id,
                "enqueued_at": time.time(),
            })
            # Push into evaluation queue with non-blocking call
            await asyncio.wait_for(
                self.redis.rpush("neuroflow:queue:evaluations", payload),
                timeout=2.0,
            )
            logger.info("Enqueued evaluation job for run %s", run_id)
        except Exception as exc:
            # Queue failure must not fail generation
            logger.warning("Failed to enqueue evaluation job for run %s: %s", run_id, exc)

    async def generate(
        self,
        query: str,
        pipeline_id: UUID,
        run_id: UUID | None = None,
        top_k: int = 10,
        max_context_tokens: int = 4000,
        rerank: bool = True,
        model_name: str | None = None,
        pipeline_version: int | None = None,
        config: Any = None,
        temperature: float | None = None,
    ) -> GenerationResult:
        """Execute complete non-streaming generation pipeline with configuration and versioning."""
        target_run_id = run_id or uuid4()
        start_time = time.perf_counter()

        # Load pipeline configuration if not passed directly
        cfg_dict: dict[str, Any] | None = None
        effective_version = pipeline_version

        if config is not None:
            cfg_dict = config.model_dump() if hasattr(config, "model_dump") else dict(config)
        else:
            ver, loaded_cfg = await self._load_pipeline_config(pipeline_id, pipeline_version)
            cfg_dict = loaded_cfg
            if effective_version is None:
                effective_version = ver

        # Map configuration settings
        if cfg_dict:
            retrieval_cfg = cfg_dict.get("retrieval", {})
            gen_cfg = cfg_dict.get("generation", {})
            eval_cfg = cfg_dict.get("evaluation", {})

            effective_dense_k = retrieval_cfg.get("dense_k", top_k * 2)
            effective_sparse_k = retrieval_cfg.get("sparse_k", top_k * 2)
            effective_top_k = retrieval_cfg.get("top_k_after_rerank", top_k)
            reranker_type = retrieval_cfg.get("reranker", "cross-encoder" if rerank else "none")
            apply_expansion = retrieval_cfg.get("query_expansion", True)
            apply_filters = retrieval_cfg.get("metadata_filters_enabled", True)

            effective_max_tokens = gen_cfg.get("max_context_tokens", max_context_tokens)
            effective_model = model_name or gen_cfg.get("model_routing", "gpt-4o-mini")
            effective_temp = temperature if temperature is not None else gen_cfg.get("temperature", 0.7)
            auto_evaluate = eval_cfg.get("auto_evaluate", True)
        else:
            effective_dense_k = top_k * 2
            effective_sparse_k = top_k * 2
            effective_top_k = top_k
            reranker_type = "cross-encoder" if rerank else "none"
            apply_expansion = True
            apply_filters = True
            effective_max_tokens = max_context_tokens
            effective_model = model_name or "gpt-4o-mini"
            effective_temp = temperature if temperature is not None else 0.7
            auto_evaluate = True

        retrieval_start = time.perf_counter()

        # 1. Query Processing
        if apply_expansion:
            processed = await self.processor.process(query)
        else:
            from pipelines.retrieval.query_processor import ProcessedQuery
            processed = ProcessedQuery(
                query=query,
                query_type="factual",
                expanded_queries=[query],
                filters={},
            )

        # 2. Parallel Multi-Method Retrieval
        filters_to_apply = processed.filters if apply_filters else None
        retrieval_k = max(effective_dense_k, effective_sparse_k)
        dense_res, sparse_res, meta_res = await self.retriever.retrieve_parallel(
            query=processed,
            k=retrieval_k,
            pipeline_id=pipeline_id,
            filters=filters_to_apply,
        )

        # 3. Reciprocal Rank Fusion
        result_lists = [dense_res, sparse_res]
        if meta_res:
            result_lists.append(meta_res)
        fused = reciprocal_rank_fusion(result_lists)

        # 4. Optional Reranking
        if reranker_type != "none" and fused:
            try:
                candidates = await self.reranker.rerank(processed.query, fused, top_n=40)
            except Exception as rerank_err:
                logger.warning("Reranking failed, using RRF candidates: %s", rerank_err)
                candidates = fused
        else:
            candidates = fused

        top_candidates = candidates[:effective_top_k]

        # 5. Context Assembly
        assembled = self.assembler.assemble(top_candidates, max_tokens=effective_max_tokens)
        retrieval_latency_ms = int((time.perf_counter() - retrieval_start) * 1000)

        # 6. Prompt Assembly
        prompt = self.prompt_builder.build(
            query=processed.query,
            query_type=processed.query_type,
            context=assembled,
            pipeline_config=cfg_dict,
        )

        # Convert chunk IDs to UUIDs for DB schema compatibility
        retrieved_uuids: list[UUID] = []
        for s in assembled.sources:
            try:
                retrieved_uuids.append(UUID(str(s.get("chunk_id"))))
            except (ValueError, TypeError):
                pass

        # 7. Persist pipeline_runs record BEFORE LLM call
        await self._create_pipeline_run(
            run_id=target_run_id,
            pipeline_id=pipeline_id,
            query=query,
            retrieved_chunk_ids=retrieved_uuids,
            prompt=prompt.full_prompt,
            pipeline_version=effective_version,
            retrieval_latency_ms=retrieval_latency_ms,
            execution_metadata={"config_used": cfg_dict} if cfg_dict else {},
        )

        # 8. Call LLM
        messages = [
            ChatMessage(role="system", content=prompt.system_prompt),
            ChatMessage(role="user", content=prompt.user_prompt),
        ]

        try:
            cost_usd: float | None = None
            if self.client and hasattr(self.client, "chat"):
                llm_res = await self.client.chat(messages, temperature=effective_temp)
                content = llm_res.content
                in_tokens = llm_res.input_tokens
                out_tokens = llm_res.output_tokens
                used_model = getattr(llm_res, "model", effective_model)
                cost_usd = getattr(llm_res, "cost_usd", None)
            else:
                # Deterministic fallback if mock / testing
                content = f"Based on the provided context [Source 1], the answer is clearly documented."
                in_tokens = self.token_counter.count(prompt.full_prompt)
                out_tokens = self.token_counter.count(content)
                used_model = effective_model
                cost_usd = 0.0001

            # 9. Citation parsing and validation
            parsed_citations = self.citation_tracker.extract_and_validate(
                generation_text=content,
                source_map=prompt.source_map,
            )

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            # 10. Update DB record to completed
            await self._update_pipeline_run_success(
                run_id=target_run_id,
                pipeline_id=pipeline_id,
                generation=content,
                citations=parsed_citations.to_dict_list(),
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                latency_ms=latency_ms,
                model_used=used_model,
                cost_usd=cost_usd,
            )

            # 11. Enqueue async eval job if auto_evaluate is enabled
            if auto_evaluate:
                asyncio.create_task(
                    self._enqueue_eval_job(str(target_run_id), str(pipeline_id))
                )

            return GenerationResult(
                run_id=str(target_run_id),
                query=query,
                query_type=processed.query_type,
                generation=content,
                citations=parsed_citations,
                sources=assembled.sources,
                chunks_used=assembled.chunks_used,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                latency_ms=latency_ms,
                model_used=used_model,
                status="completed",
                pipeline_version=effective_version,
                retrieval_latency_ms=retrieval_latency_ms,
                cost_usd=cost_usd,
            )

        except Exception as exc:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            await self._update_pipeline_run_failure(
                run_id=target_run_id,
                pipeline_id=pipeline_id,
                error_message=str(exc),
                latency_ms=latency_ms,
            )
            raise

    async def stream_events(
        self,
        query: str,
        pipeline_id: UUID,
        run_id: UUID,
        top_k: int = 10,
        max_context_tokens: int = 4000,
        rerank: bool = True,
        model_name: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Progressive streaming generator yielding SSE event dictionaries."""
        start_time = time.perf_counter()

        # 1. Event: retrieval_start
        yield {"type": "retrieval_start"}

        try:
            # 2. Query processing & Retrieval
            processed = await self.processor.process(query)
            dense_res, sparse_res, meta_res = await self.retriever.retrieve_parallel(
                query=processed,
                k=top_k * 2,
                pipeline_id=pipeline_id,
                filters=processed.filters,
            )

            result_lists = [dense_res, sparse_res]
            if meta_res:
                result_lists.append(meta_res)
            fused = reciprocal_rank_fusion(result_lists)

            if rerank and fused:
                try:
                    candidates = await self.reranker.rerank(processed.query, fused, top_n=40)
                except Exception as rerank_err:
                    logger.warning("Reranking failed: %s", rerank_err)
                    candidates = fused
            else:
                candidates = fused

            top_candidates = candidates[:top_k]
            assembled = self.assembler.assemble(top_candidates, max_tokens=max_context_tokens)

            # 3. Event: retrieval_complete
            yield {
                "type": "retrieval_complete",
                "chunk_count": assembled.chunks_used,
                "sources": assembled.sources,
                "retrieval_metadata": {
                    "query": processed.query,
                    "query_type": processed.query_type,
                    "dense_count": len(dense_res),
                    "sparse_count": len(sparse_res),
                    "meta_count": len(meta_res),
                    "fused_count": len(fused),
                    "reranked_count": len(candidates),
                    "top_candidates_count": len(top_candidates),
                    "context_chunks_used": assembled.chunks_used,
                    "total_tokens": assembled.total_tokens,
                },
            }

            # 4. Prompt Assembly
            prompt = self.prompt_builder.build(
                query=processed.query,
                query_type=processed.query_type,
                context=assembled,
            )

            retrieved_uuids: list[UUID] = []
            for s in assembled.sources:
                try:
                    retrieved_uuids.append(UUID(str(s.get("chunk_id"))))
                except (ValueError, TypeError):
                    pass

            # 5. Persist run BEFORE calling LLM
            await self._create_pipeline_run(
                run_id=run_id,
                pipeline_id=pipeline_id,
                query=query,
                retrieved_chunk_ids=retrieved_uuids,
                prompt=prompt.full_prompt,
            )

            # 6. Stream tokens from LLM
            effective_model = model_name or "gpt-4o-mini"
            messages = [
                ChatMessage(role="system", content=prompt.system_prompt),
                ChatMessage(role="user", content=prompt.user_prompt),
            ]

            full_text = ""
            if self.client and hasattr(self.client, "stream"):
                async for delta in self.client.stream(messages):
                    full_text += delta
                    yield {"type": "token", "delta": delta}
            else:
                # Mock token streaming for local testing
                mock_tokens = [
                    "Based ", "on ", "the ", "provided ", "context ", "[Source 1], ",
                    "the ", "answer ", "is ", "clear."
                ]
                for tok in mock_tokens:
                    await asyncio.sleep(0.02)
                    full_text += tok
                    yield {"type": "token", "delta": tok}

            # 7. Parse citations and metrics
            parsed_citations = self.citation_tracker.extract_and_validate(
                generation_text=full_text,
                source_map=prompt.source_map,
            )

            in_tokens = self.token_counter.count(prompt.full_prompt)
            out_tokens = self.token_counter.count(full_text)
            latency_ms = int((time.perf_counter() - start_time) * 1000)

            # 8. Persist completion
            await self._update_pipeline_run_success(
                run_id=run_id,
                pipeline_id=pipeline_id,
                generation=full_text,
                citations=parsed_citations.to_dict_list(),
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                latency_ms=latency_ms,
                model_used=effective_model,
            )

            # 9. Enqueue eval job asynchronously
            asyncio.create_task(
                self._enqueue_eval_job(str(run_id), str(pipeline_id))
            )

            # 10. Event: done
            yield {
                "type": "done",
                "run_id": str(run_id),
                "citations": parsed_citations.to_dict_list(),
            }

        except Exception as exc:
            logger.exception("Error during streaming generation for run %s: %s", run_id, exc)
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            await self._update_pipeline_run_failure(
                run_id=run_id,
                pipeline_id=pipeline_id,
                error_message=str(exc),
                latency_ms=latency_ms,
            )
            yield {"type": "error", "error": str(exc), "run_id": str(run_id)}
