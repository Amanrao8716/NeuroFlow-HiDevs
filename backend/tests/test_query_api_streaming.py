"""Integration tests for POST /query and GET /query/{run_id}/stream."""

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
import pytest
from httpx import ASGITransport, AsyncClient

from main import app
from pipelines.generation.generator import GenerationPipeline, GenerationResult
from pipelines.generation.citations import Citation, ParsedCitations


@pytest.fixture
def mock_pipeline():
    pipeline = MagicMock(spec=GenerationPipeline)

    # Mock non-streaming generate
    fake_citations = ParsedCitations(
        citations=[
            Citation(
                reference="Source 1",
                source_number=1,
                chunk_id=str(uuid4()),
                document_name="doc.pdf",
                page_number=2,
                content_preview="PostgreSQL has pgvector.",
                is_valid=True,
            )
        ],
        invalid_citations=[],
        cited_chunk_ids=[str(uuid4())],
    )

    pipeline.generate = AsyncMock(return_value=GenerationResult(
        run_id=str(uuid4()),
        query="What is pgvector?",
        query_type="factual",
        generation="pgvector is an extension [Source 1].",
        citations=fake_citations,
        sources=[{"filename": "doc.pdf"}],
        chunks_used=1,
        input_tokens=100,
        output_tokens=20,
        latency_ms=150,
        model_used="gpt-4o-mini",
        status="completed",
    ))

    async def fake_stream_events(*args, **kwargs):
        yield {"type": "retrieval_start"}
        yield {"type": "retrieval_complete", "chunk_count": 1, "sources": ["doc.pdf"]}
        yield {"type": "token", "delta": "pgvector "}
        yield {"type": "token", "delta": "is "}
        yield {"type": "token", "delta": "fast [Source 1]."}
        yield {"type": "done", "run_id": str(uuid4()), "citations": fake_citations.to_dict_list()}

    pipeline.stream_events = fake_stream_events
    return pipeline


@pytest.mark.asyncio
async def test_post_query_sync_success(mock_pipeline):
    with patch("api.query.get_generation_pipeline", return_value=mock_pipeline):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/query",
                json={
                    "query": "What is pgvector?",
                    "pipeline_id": str(uuid4()),
                    "stream": False,
                },
            )

            assert resp.status_code == 200
            data = resp.json()
            assert data["query"] == "What is pgvector?"
            assert data["status"] == "completed"
            assert "pgvector is an extension" in data["answer"]
            assert len(data["citations"]) == 1
            assert data["citations"][0]["reference"] == "Source 1"


@pytest.mark.asyncio
async def test_post_query_stream_flow(mock_pipeline):
    with patch("api.query.get_generation_pipeline", return_value=mock_pipeline):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. Initiate stream
            resp = await client.post(
                "/query",
                json={
                    "query": "What is pgvector?",
                    "pipeline_id": str(uuid4()),
                    "stream": True,
                },
            )

            assert resp.status_code == 200
            data = resp.json()
            assert "run_id" in data
            assert "stream_url" in data
            run_id = data["run_id"]

            # 2. Connect to SSE stream
            stream_resp = await client.get(f"/query/{run_id}/stream")
            assert stream_resp.status_code == 200
            assert "text/event-stream" in stream_resp.headers["content-type"]

            lines = stream_resp.text.split("\n")
            event_data_lines = [
                line.replace("data: ", "").strip()
                for line in lines
                if line.startswith("data: ")
            ]

            parsed_events = [json.loads(d) for d in event_data_lines if d]
            event_types = [e["type"] for e in parsed_events]

            assert "retrieval_start" in event_types
            assert "retrieval_complete" in event_types
            assert "token" in event_types
            assert "done" in event_types
