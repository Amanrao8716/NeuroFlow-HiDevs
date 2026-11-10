"""Unit tests for NeuroFlowClient core methods and authentication."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from neuroflow import Document, EvaluationResult, NeuroFlowClient, QueryResult
from neuroflow.exceptions import AuthenticationError, IngestionError
from neuroflow.models import Pipeline


@pytest.mark.asyncio
async def test_auth_token_exchange():
    """Verify that client_id and client_secret are exchanged for a token."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/token":
            body = json.loads(request.content)
            assert body["client_id"] == "test-client"
            assert body["client_secret"] == "secret"
            return httpx.Response(200, json={"access_token": "mock-jwt-token"})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    async with NeuroFlowClient(
        "http://testserver",
        client_id="test-client",
        client_secret="secret",
        transport=transport,
    ) as client:
        assert client._token == "mock-jwt-token"
        assert client._auth_headers() == {"Authorization": "Bearer mock-jwt-token"}


@pytest.mark.asyncio
async def test_auth_failure_raises():
    """Verify 401 on /auth/token raises AuthenticationError."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "Invalid credentials"})

    transport = httpx.MockTransport(handler)
    with pytest.raises(AuthenticationError):
        async with NeuroFlowClient(
            "http://testserver",
            client_id="bad-client",
            client_secret="bad-secret",
            transport=transport,
        ):
            pass


@pytest.mark.asyncio
async def test_ingest_file_flow(tmp_path: Path):
    """Verify ingest_file sends multipart upload and polls document status."""
    dummy_file = tmp_path / "test_doc.pdf"
    dummy_file.write_bytes(b"%PDF-1.4 dummy content")

    polls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal polls
        if request.url.path == "/ingest":
            return httpx.Response(202, json={"document_id": "doc-123", "duplicate": False})
        if request.url.path == "/documents/doc-123":
            polls += 1
            if polls == 1:
                return httpx.Response(200, json={"document_id": "doc-123", "status": "processing"})
            return httpx.Response(200, json={
                "document_id": "doc-123",
                "status": "completed",
                "filename": "test_doc.pdf",
                "source_type": "pdf",
                "chunk_count": 5,
            })
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    async with NeuroFlowClient(
        "http://testserver",
        token="mock-token",
        poll_interval=0.01,
        transport=transport,
    ) as client:
        doc = await client.ingest_file(dummy_file, pipeline_id="pipe-1")
        assert isinstance(doc, Document)
        assert doc.document_id == "doc-123"
        assert doc.status == "completed"
        assert doc.chunk_count == 5


@pytest.mark.asyncio
async def test_ingest_url_duplicate():
    """Verify ingest_url returns immediate Document when duplicate is True."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/ingest/url":
            return httpx.Response(202, json={
                "document_id": "doc-existing",
                "status": "completed",
                "duplicate": True,
            })
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    async with NeuroFlowClient(
        "http://testserver",
        token="mock-token",
        transport=transport,
    ) as client:
        doc = await client.ingest_url("https://example.com", pipeline_id="pipe-1")
        assert doc.document_id == "doc-existing"
        assert doc.duplicate is True


@pytest.mark.asyncio
async def test_query_synchronous():
    """Verify query with stream=False returns typed QueryResult with citations."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/query":
            return httpx.Response(200, json={
                "run_id": "run-456",
                "query": "What is NeuroFlow?",
                "query_type": "factual",
                "answer": "NeuroFlow is a RAG pipeline [1].",
                "citations": [{
                    "reference": "[1]",
                    "source_number": 1,
                    "chunk_id": "chunk-1",
                    "document_name": "guide.pdf",
                    "page_number": 2,
                    "content_preview": "NeuroFlow architecture...",
                    "is_valid": True,
                }],
                "sources": [],
                "chunks_used": 1,
                "input_tokens": 150,
                "output_tokens": 40,
                "latency_ms": 320,
                "model_used": "gpt-4o-mini",
                "status": "completed",
            })
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    async with NeuroFlowClient(
        "http://testserver",
        token="mock-token",
        transport=transport,
    ) as client:
        result = await client.query("What is NeuroFlow?", pipeline_id="pipe-1")
        assert isinstance(result, QueryResult)
        assert result.run_id == "run-456"
        assert result.answer == "NeuroFlow is a RAG pipeline [1]."
        assert len(result.citations) == 1
        assert result.citations[0].reference == "[1]"
        assert result.citations[0].chunk_id == "chunk-1"


@pytest.mark.asyncio
async def test_get_evaluation():
    """Verify get_evaluation returns typed EvaluationResult."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/runs/run-456/evaluation":
            return httpx.Response(200, json={
                "id": "eval-1",
                "run_id": "run-456",
                "faithfulness": 0.95,
                "answer_relevance": 0.92,
                "context_precision": 0.88,
                "context_recall": 0.90,
                "overall_score": 0.91,
                "judge_model": "gpt-4o",
                "user_rating": 5,
                "calibration_needed": False,
            })
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    async with NeuroFlowClient(
        "http://testserver",
        token="mock-token",
        transport=transport,
    ) as client:
        ev = await client.get_evaluation("run-456")
        assert isinstance(ev, EvaluationResult)
        assert ev.overall_score == 0.91
        assert ev.faithfulness == 0.95
        assert ev.calibration_needed is False


@pytest.mark.asyncio
async def test_list_and_create_pipelines():
    """Verify list_pipelines and create_pipeline."""
    pipelines_db = [
        {"id": "p-1", "name": "Default", "version": 1, "status": "active", "config": {}},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/pipelines" and request.method == "GET":
            return httpx.Response(200, json=pipelines_db)
        if request.url.path == "/pipelines" and request.method == "POST":
            body = json.loads(request.content)
            new_p = {
                "id": "p-2",
                "name": body["name"],
                "description": body.get("description"),
                "version": 1,
                "status": "active",
                "config": {},
            }
            pipelines_db.append(new_p)
            return httpx.Response(201, json=new_p)
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    async with NeuroFlowClient(
        "http://testserver",
        token="mock-token",
        transport=transport,
    ) as client:
        pipes = await client.list_pipelines()
        assert len(pipes) == 1
        assert pipes[0].name == "Default"

        created = await client.create_pipeline("Staging", description="Staging pipeline")
        assert isinstance(created, Pipeline)
        assert created.name == "Staging"
        assert created.id == "p-2"

        # Also test create_pipeline with config dictionary
        created_dict = await client.create_pipeline({
            "name": "Production",
            "description": "Prod pipeline",
            "retrieval": {"dense_k": 20},
        })
        assert isinstance(created_dict, Pipeline)
        assert created_dict.name == "Production"


@pytest.mark.asyncio
async def test_ingest_missing_pipeline_id_raises(tmp_path: Path):
    """Verify ingest_file and ingest_url raise IngestionError when pipeline_id is None."""
    dummy_file = tmp_path / "dummy.pdf"
    dummy_file.write_bytes(b"%PDF-1.4 test")

    async with NeuroFlowClient("http://testserver", token="mock-token") as client:
        with pytest.raises(IngestionError):
            await client.ingest_file(dummy_file, pipeline_id=None)
        with pytest.raises(IngestionError):
            await client.ingest_url("https://example.com", pipeline_id=None)
