"""Tests for Task 11 backend endpoints and behaviors."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_cors_headers_present():
    """Verify CORS headers are returned for preflight options request."""
    response = client.options(
        "/pipelines",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers


def test_evaluations_stream_redis_unavailable():
    """Verify GET /evaluations/stream returns 503 when Redis is unavailable."""
    app.state.redis = None
    response = client.get("/evaluations/stream")
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()


def test_evaluations_list_no_db():
    """Verify GET /evaluations returns 503 when database pool is unavailable."""
    with patch("api.evaluations.connection", None):
        response = client.get("/evaluations")
        assert response.status_code == 503


def test_documents_list_endpoint():
    """Verify GET /documents returns list when pipeline_connection yields rows."""
    fake_doc_id = uuid4()
    mock_rows = [
        {
            "id": fake_doc_id,
            "filename": "neuroflow_spec.pdf",
            "source_type": "pdf",
            "status": "completed",
            "chunk_count": 12,
            "metadata": {"pages": 5},
            "created_at": MagicMock(isoformat=lambda: "2026-10-26T00:00:00Z"),
        }
    ]

    mock_conn = MagicMock()
    mock_conn.fetch = AsyncMock(return_value=mock_rows)

    @asynccontextmanager
    async def mock_pipeline_connection(pipeline_id):
        yield mock_conn

    with patch("api.ingest.pipeline_connection", mock_pipeline_connection):
        response = client.get(f"/documents?pipeline_id={uuid4()}")
        assert response.status_code == 200
        docs = response.json()
        assert len(docs) == 1
        assert docs[0]["filename"] == "neuroflow_spec.pdf"
        assert docs[0]["chunk_count"] == 12


def test_document_chunks_endpoint():
    """Verify GET /documents/{id}/chunks returns chunk details."""
    doc_id = uuid4()
    chunk_id = uuid4()
    mock_rows = [
        {
            "id": chunk_id,
            "document_id": doc_id,
            "content": "NeuroFlow chunk text content.",
            "chunk_index": 0,
            "token_count": 42,
            "metadata": {"section": "intro"},
            "filename": "spec.pdf",
            "source_type": "pdf",
            "created_at": MagicMock(isoformat=lambda: "2026-10-26T00:00:00Z"),
        }
    ]

    mock_conn = MagicMock()
    mock_conn.fetch = AsyncMock(return_value=mock_rows)

    @asynccontextmanager
    async def mock_conn_mgr():
        yield mock_conn

    with patch("db.pool.connection", mock_conn_mgr):
        response = client.get(f"/documents/{doc_id}/chunks")
        assert response.status_code == 200
        chunks = response.json()
        assert len(chunks) == 1
        assert chunks[0]["chunk_id"] == str(chunk_id)
        assert chunks[0]["content"] == "NeuroFlow chunk text content."


def test_patch_rating_values():
    """Verify PATCH /runs/{id}/rating accepts thumbs up (5) and thumbs down (1)."""
    run_id = uuid4()
    pipeline_id = uuid4()

    mock_conn = MagicMock()
    mock_conn.fetchrow = AsyncMock(return_value={
        "id": uuid4(),
        "faithfulness": 0.85,
        "answer_relevance": 0.88,
        "context_precision": 0.75,
        "context_recall": 0.80,
        "overall_score": 0.83,
        "judge_model": "gpt-4o",
        "user_rating": None,
    })
    mock_conn.execute = AsyncMock()

    @asynccontextmanager
    async def mock_pipeline_connection(p_id):
        yield mock_conn

    with patch("api.evaluations.pipeline_connection", mock_pipeline_connection), \
         patch("api.evaluations._resolve_pipeline_id", AsyncMock(return_value=pipeline_id)):

        # Thumbs up: rating = 5
        res_up = client.patch(
            f"/runs/{run_id}/rating",
            json={"rating": 5, "pipeline_id": str(pipeline_id)},
        )
        assert res_up.status_code == 200
        data_up = res_up.json()
        assert data_up["user_rating"] == 5
        assert data_up["normalized_rating"] == 1.0

        # Thumbs down: rating = 1
        res_down = client.patch(
            f"/runs/{run_id}/rating",
            json={"rating": 1, "pipeline_id": str(pipeline_id)},
        )
        assert res_down.status_code == 200
        data_down = res_down.json()
        assert data_down["user_rating"] == 1
        assert data_down["normalized_rating"] == 0.2
