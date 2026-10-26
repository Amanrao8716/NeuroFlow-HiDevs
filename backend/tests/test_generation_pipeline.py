"""Unit tests for GenerationPipeline orchestration, run persistence, and streaming."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4
import pytest

from pipelines.generation.generator import GenerationPipeline, GenerationResult
from pipelines.retrieval.context_assembler import AssembledContext
from pipelines.retrieval.query_processor import ProcessedQuery
from pipelines.retrieval.retriever import RetrievalResult
from providers.base import GenerationResult as LLMResult


@pytest.fixture
def mock_pipeline_components():
    proc = MagicMock()
    proc.process = AsyncMock(return_value=ProcessedQuery(
        query="What is PostgreSQL?",
        query_type="factual",
        expanded_queries=["What is PostgreSQL?"],
        filters={},
    ))

    ret = MagicMock()
    chunk1 = RetrievalResult(
        chunk_id="11111111-1111-1111-1111-111111111111",
        document_id="22222222-2222-2222-2222-222222222222",
        content="PostgreSQL is an open-source object-relational database.",
        score=0.9,
        rank=1,
        source_type="pdf",
        filename="postgres.pdf",
        page_number=1,
        metadata={},
    )
    ret.retrieve_parallel = AsyncMock(return_value=([chunk1], [], []))

    reranker = MagicMock()
    reranker.rerank = AsyncMock(return_value=[chunk1])

    assembler = MagicMock()
    assembler.assemble = MagicMock(return_value=AssembledContext(
        context="[Source 1 — postgres.pdf, page 1]\nPostgreSQL is an open-source object-relational database.",
        chunks_used=1,
        total_tokens=15,
        sources=[{
            "source_index": 1,
            "chunk_id": "11111111-1111-1111-1111-111111111111",
            "document_id": "22222222-2222-2222-2222-222222222222",
            "filename": "postgres.pdf",
            "page_number": 1,
            "score": 0.9,
        }],
    ))

    llm = MagicMock()
    llm.chat = AsyncMock(return_value=LLMResult(
        content="PostgreSQL is a powerful relational database [Source 1].",
        model="gpt-4o-mini",
        input_tokens=50,
        output_tokens=15,
        latency_ms=120.0,
        cost_usd=0.0001,
        finish_reason="stop",
    ))

    async def mock_stream(*args, **kwargs):
        yield "PostgreSQL "
        yield "is "
        yield "a "
        yield "database "
        yield "[Source 1]."

    llm.stream = mock_stream

    redis = MagicMock()
    redis.rpush = AsyncMock(return_value=1)

    return {
        "processor": proc,
        "retriever": ret,
        "reranker": reranker,
        "assembler": assembler,
        "llm": llm,
        "redis": redis,
    }


@pytest.mark.asyncio
async def test_generation_pipeline_runs_and_persists_before_llm(mock_pipeline_components):
    pipeline = GenerationPipeline(
        redis_client=mock_pipeline_components["redis"],
        query_processor=mock_pipeline_components["processor"],
        retriever=mock_pipeline_components["retriever"],
        reranker=mock_pipeline_components["reranker"],
        assembler=mock_pipeline_components["assembler"],
        llm_client=mock_pipeline_components["llm"],
    )

    call_order = []
    fake_llm_result = LLMResult(
        content="PostgreSQL is a powerful relational database [Source 1].",
        model="gpt-4o-mini",
        input_tokens=50,
        output_tokens=15,
        latency_ms=120.0,
        cost_usd=0.0001,
        finish_reason="stop",
    )

    async def mock_create(run_id, pipeline_id, query, retrieved_chunk_ids, prompt, *args, **kwargs):
        call_order.append("create_run")

    async def mock_chat(messages, **kwargs):
        call_order.append("llm_call")
        return fake_llm_result

    pipeline._create_pipeline_run = AsyncMock(side_effect=mock_create)
    pipeline._update_pipeline_run_success = AsyncMock(side_effect=lambda *args, **kwargs: call_order.append("update_run"))
    mock_pipeline_components["llm"].chat = AsyncMock(side_effect=mock_chat)

    p_id = uuid4()
    result = await pipeline.generate(
        query="What is PostgreSQL?",
        pipeline_id=p_id,
    )

    # Verify run was created BEFORE LLM call
    assert call_order == ["create_run", "llm_call", "update_run"]
    assert result.status == "completed"
    assert len(result.citations.citations) == 1
    assert result.citations.citations[0].reference == "Source 1"
    assert result.citations.citations[0].chunk_id == "11111111-1111-1111-1111-111111111111"


@pytest.mark.asyncio
async def test_generation_stream_events_yielded(mock_pipeline_components):
    pipeline = GenerationPipeline(
        redis_client=mock_pipeline_components["redis"],
        query_processor=mock_pipeline_components["processor"],
        retriever=mock_pipeline_components["retriever"],
        reranker=mock_pipeline_components["reranker"],
        assembler=mock_pipeline_components["assembler"],
        llm_client=mock_pipeline_components["llm"],
    )

    pipeline._create_pipeline_run = AsyncMock()
    pipeline._update_pipeline_run_success = AsyncMock()
    pipeline._enqueue_eval_job = AsyncMock()

    run_id = uuid4()
    pipeline_id = uuid4()

    events = []
    async for evt in pipeline.stream_events(
        query="What is PostgreSQL?",
        pipeline_id=pipeline_id,
        run_id=run_id,
    ):
        events.append(evt)

    event_types = [e["type"] for e in events]
    assert event_types[0] == "retrieval_start"
    assert event_types[1] == "retrieval_complete"
    assert "token" in event_types
    assert event_types[-1] == "done"
    assert events[-1]["run_id"] == str(run_id)
    assert len(events[-1]["citations"]) == 1
