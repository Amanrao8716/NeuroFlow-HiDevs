"""Unit and Integration tests for Pipeline CRUD, Versioning, History, Analytics, and Suggestions API."""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4
from fastapi.testclient import TestClient

from main import app
from api.pipelines import calculate_percentile, calculate_token_cost
from backend.models.pipeline import PipelineConfig


@pytest.fixture
def client():
    return TestClient(app)


# -------------------------------------------------------------------------
# Deterministic Math / Utility Tests
# -------------------------------------------------------------------------

def test_calculate_percentile_deterministic():
    """Verify linear interpolation percentile calculation across edge cases."""
    # Empty dataset
    assert calculate_percentile([], 50) is None
    assert calculate_percentile([], 95) is None

    # Single element
    assert calculate_percentile([42.0], 50) == 42.0
    assert calculate_percentile([42.0], 99) == 42.0

    # Ordered known list: [10, 20, 30, 40, 50]
    # n=5, p50 rank = 0.50 * 4 = 2.0 -> index 2 is 30.0
    data = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert calculate_percentile(data, 50) == 30.0
    assert calculate_percentile(data, 0) == 10.0
    assert calculate_percentile(data, 100) == 50.0

    # p95 rank = 0.95 * 4 = 3.8 -> 40 + 0.8 * (50 - 40) = 48.0
    assert calculate_percentile(data, 95) == 48.0


def test_calculate_token_cost_pricing():
    """Verify token cost calculation for standard models."""
    # gpt-4o: $2.50 / 1M input, $10.00 / 1M output
    # 1000 input tokens = $0.0025, 1000 output tokens = $0.0100 -> Total = $0.0125
    cost = calculate_token_cost("gpt-4o", input_tokens=1000, output_tokens=1000)
    assert round(cost, 4) == 0.0125

    # gpt-4o-mini: 0.15 / 1M input, 0.60 / 1M output
    # 1000 input = 0.00015, 1000 output = 0.00060 -> Total = 0.00075
    cost_mini = calculate_token_cost("gpt-4o-mini", input_tokens=1000, output_tokens=1000)
    assert round(cost_mini, 6) == 0.000750

    # Unknown model default fallback
    cost_fallback = calculate_token_cost("custom-finetuned", input_tokens=1000, output_tokens=1000)
    assert cost_fallback > 0.0


# -------------------------------------------------------------------------
# Helper Mock Fixture
# -------------------------------------------------------------------------

def _create_mock_db_context(mock_conn):
    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_ctx.__aexit__ = AsyncMock(return_value=None)

    tx_ctx = MagicMock()
    tx_ctx.__aenter__ = AsyncMock(return_value=None)
    tx_ctx.__aexit__ = AsyncMock(return_value=None)
    mock_conn.transaction = MagicMock(return_value=tx_ctx)

    class MockPoolManager:
        def __call__(self, *args, **kwargs):
            return mock_ctx
        async def __aenter__(self):
            return mock_conn
        async def __aexit__(self, *args):
            return None

    return MockPoolManager()


# -------------------------------------------------------------------------
# API Tests with Mocked DB Connection
# -------------------------------------------------------------------------

def test_create_pipeline_api(client):
    """POST /pipelines creates a pipeline and initial version record."""
    mock_conn = AsyncMock()
    mock_conn.fetchval.return_value = None

    mock_mgr = _create_mock_db_context(mock_conn)

    with patch("api.pipelines.pipeline_connection", mock_mgr), \
         patch("api.pipelines.connection", mock_mgr):
        response = client.post(
            "/pipelines",
            json={
                "name": "Customer Support Pipeline",
                "description": "Handles FAQ queries",
                "config": {
                    "name": "Customer Support Pipeline",
                    "description": "Handles FAQ queries",
                    "ingestion": {
                        "chunking_strategy": "auto",
                        "chunk_size_tokens": 512,
                        "chunk_overlap_tokens": 64,
                    },
                    "retrieval": {
                        "dense_k": 20,
                        "sparse_k": 20,
                    },
                    "generation": {
                        "temperature": 0.3,
                    },
                    "evaluation": {
                        "auto_evaluate": True,
                        "training_threshold": 0.8,
                    },
                },
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert UUID(data["id"])
        assert data["name"] == "Customer Support Pipeline"
        assert data["version"] == 1
        assert data["status"] == "active"



def test_list_pipelines_api(client):
    """GET /pipelines lists non-archived pipelines with metrics."""
    pid1 = uuid4()
    mock_conn = AsyncMock()
    mock_conn.fetch.return_value = [
        {
            "id": pid1,
            "name": "Pipeline 1",
            "description": "First",
            "version": 2,
            "status": "active",
            "config": {
                "name": "Pipeline 1",
                "description": "First",
                "ingestion": {"chunking_strategy": "auto"},
                "retrieval": {},
                "generation": {},
                "evaluation": {},
            },
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "run_count": 15,
            "last_run_at": datetime.now(timezone.utc),
        }
    ]

    mock_mgr = _create_mock_db_context(mock_conn)

    with patch("api.pipelines.connection", mock_mgr), \
         patch("api.pipelines.pipeline_connection", mock_mgr):
        response = client.get("/pipelines")
        assert response.status_code == 200
        items = response.json()
        assert len(items) == 1
        assert items[0]["id"] == str(pid1)
        assert items[0]["version"] == 2
        assert items[0]["latest_metrics"]["run_count"] == 15


def test_get_pipeline_detail_api(client):
    """GET /pipelines/{id} returns full configuration and version."""
    pid = uuid4()
    mock_conn = AsyncMock()
    mock_conn.fetchrow.side_effect = [
        # First call: pipeline info
        {
            "id": pid,
            "name": "Pipeline Detail",
            "description": "Detailed view",
            "version": 2,
            "config": {
                "name": "Pipeline Detail",
                "description": "Detailed view",
                "ingestion": {"chunking_strategy": "auto", "chunk_size_tokens": 512, "chunk_overlap_tokens": 50},
                "retrieval": {"dense_k": 20},
                "generation": {"temperature": 0.2},
                "evaluation": {"training_threshold": 0.8},
            },
            "status": "active",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        },
        # Second call: aggregate evaluation scores
        {
            "avg_faith": 0.90,
            "avg_rel": 0.85,
            "avg_prec": 0.80,
            "avg_rec": 0.75,
            "avg_overall": 0.84,
            "total_evaluations": 25,
        },
    ]

    mock_mgr = _create_mock_db_context(mock_conn)

    with patch("api.pipelines.pipeline_connection", mock_mgr), \
         patch("api.pipelines.connection", mock_mgr):
        response = client.get(f"/pipelines/{pid}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(pid)
        assert data["version"] == 2
        assert data["aggregate_scores"]["overall_score"] == 0.84


def test_get_missing_pipeline_returns_404(client):
    """GET /pipelines/{id} with nonexistent ID returns 404."""
    missing_id = uuid4()
    mock_conn = AsyncMock()
    mock_conn.fetchrow.return_value = None

    mock_mgr = _create_mock_db_context(mock_conn)

    with patch("api.pipelines.pipeline_connection", mock_mgr), \
         patch("api.pipelines.connection", mock_mgr):
        response = client.get(f"/pipelines/{missing_id}")
        assert response.status_code == 404


def test_patch_pipeline_creates_new_version(client):
    """PATCH /pipelines/{id} deep-merges changes and increments version."""
    pid = uuid4()
    current_config = {
        "name": "Original Pipeline",
        "description": "Original",
        "ingestion": {
            "chunking_strategy": "auto",
            "chunk_size_tokens": 512,
            "chunk_overlap_tokens": 64,
            "extractors_enabled": [],
        },
        "retrieval": {
            "dense_k": 10,
            "sparse_k": 10,
            "reranker": "cross-encoder",
            "top_k_after_rerank": 5,
            "query_expansion": False,
            "metadata_filters_enabled": False,
        },
        "generation": {
            "model_routing": "gpt-4o-mini",
            "max_context_tokens": 4000,
            "temperature": 0.3,
            "system_prompt_variant": "default",
        },
        "evaluation": {
            "auto_evaluate": True,
            "training_threshold": 0.8,
        },
    }

    mock_conn = AsyncMock()
    mock_conn.fetchrow.side_effect = [
        # Initial check
        {
            "id": pid,
            "name": "Original Pipeline",
            "description": "Original",
            "version": 1,
            "status": "active",
            "config": current_config,
        },
        # Updated record returned
        {
            "id": pid,
            "name": "Original Pipeline",
            "description": "Original",
            "version": 2,
            "status": "active",
            "config": {**current_config, "retrieval": {**current_config["retrieval"], "dense_k": 30}},
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        },
    ]

    mock_mgr = _create_mock_db_context(mock_conn)

    with patch("api.pipelines.pipeline_connection", mock_mgr), \
         patch("api.pipelines.connection", mock_mgr):
        response = client.patch(
            f"/pipelines/{pid}",
            json={
                "config_updates": {
                    "retrieval": {
                        "dense_k": 30,
                    }
                }
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["version"] == 2
        # Verify transaction and insert into pipeline_versions occurred
        calls = [c[0][0] for c in mock_conn.execute.call_args_list]
        assert any("INSERT INTO pipeline_versions" in str(sql) for sql in calls)


def test_soft_delete_pipeline_archives(client):
    """DELETE /pipelines/{id} sets status to 'archived' without deleting runs."""
    pid = uuid4()
    mock_conn = AsyncMock()
    mock_conn.fetchrow.return_value = {
        "id": pid,
        "name": "Old Pipeline",
        "description": "Old",
        "version": 1,
        "config": {
            "name": "Old Pipeline",
            "description": "Old",
            "ingestion": {},
            "retrieval": {},
            "generation": {},
            "evaluation": {},
        },
        "status": "active",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }

    mock_mgr = _create_mock_db_context(mock_conn)

    with patch("api.pipelines.pipeline_connection", mock_mgr), \
         patch("api.pipelines.connection", mock_mgr):
        response = client.delete(f"/pipelines/{pid}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "archived"

        # Verify UPDATE status = 'archived' was called
        calls = [c[0][0] for c in mock_conn.execute.call_args_list]
        assert any("UPDATE pipelines" in str(sql) and "archived" in str(sql) for sql in calls)


def test_paginated_runs_api(client):
    """GET /pipelines/{id}/runs returns paginated run history."""
    pid = uuid4()
    mock_conn = AsyncMock()
    # fetchval is called twice: once for id check, once for count
    mock_conn.fetchval.side_effect = [pid, 100]
    mock_conn.fetch.return_value = [
        {
            "id": uuid4(),
            "pipeline_id": pid,
            "pipeline_version": 2,
            "query": "What is the liability clause?",
            "latency_ms": 250,
            "retrieval_latency_ms": 45,
            "input_tokens": 1200,
            "output_tokens": 150,
            "cost_usd": 0.008,
            "model_used": "gpt-4o",
            "status": "completed",
            "created_at": datetime.now(timezone.utc),
            "faithfulness": 0.95,
            "answer_relevance": 0.90,
            "context_precision": 0.85,
            "context_recall": 0.80,
            "overall_score": 0.88,
            "generation": "The liability clause limits...",
        }
    ]

    mock_mgr = _create_mock_db_context(mock_conn)

    with patch("api.pipelines.pipeline_connection", mock_mgr), \
         patch("api.pipelines.connection", mock_mgr):
        response = client.get(f"/pipelines/{pid}/runs?page=1&page_size=10")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 100
        assert data["page"] == 1
        assert data["page_size"] == 10
        assert len(data["items"]) == 1
        assert data["items"][0]["pipeline_version"] == 2
        assert data["items"][0]["cost_usd"] == 0.008


def test_pipeline_analytics_with_data(client):
    """GET /pipelines/{id}/analytics computes percentiles and 30-day query counts."""
    pid = uuid4()
    mock_conn = AsyncMock()
    # 1. Pipeline existence
    mock_conn.fetchrow.side_effect = [
        {"id": pid, "version": 1},
        # 2. Evaluation averages
        {
            "avg_faith": 0.91,
            "avg_rel": 0.88,
            "avg_prec": 0.82,
            "avg_rec": 0.79,
            "avg_overall": 0.86,
        },
    ]

    # 3. Latencies and tokens fetch
    mock_conn.fetch.side_effect = [
        # Latencies & tokens
        [
            {"retrieval_latency_ms": 20, "latency_ms": 100, "input_tokens": 500, "output_tokens": 100, "cost_usd": 0.004},
            {"retrieval_latency_ms": 40, "latency_ms": 150, "input_tokens": 600, "output_tokens": 120, "cost_usd": 0.005},
            {"retrieval_latency_ms": 60, "latency_ms": 200, "input_tokens": 700, "output_tokens": 140, "cost_usd": 0.006},
        ],
        # Daily counts include a day without recorded costs and a day with real cost data.
        [
            {
                "run_date": datetime.now(timezone.utc).date() - timedelta(days=1),
                "query_count": 2,
                "avg_cost": None,
                "total_cost": None,
            },
            {
                "run_date": datetime.now(timezone.utc).date(),
                "query_count": 3,
                "avg_cost": 0.005,
                "total_cost": 0.015,
            },
        ],
        # Recent failed runs
        [],
        # Daily evaluation scores
        [],
    ]

    mock_mgr = _create_mock_db_context(mock_conn)

    with patch("api.pipelines.pipeline_connection", mock_mgr), \
         patch("api.pipelines.connection", mock_mgr):
        response = client.get(f"/pipelines/{pid}/analytics")
        assert response.status_code == 200
        data = response.json()
        # AnalyticsResponse fields: total_queries_30d, retrieval_latency_p50, avg_generation_latency_ms, daily_query_counts
        assert data["total_queries_30d"] == 5
        assert data["retrieval_latency_p50"] == 40.0
        assert data["retrieval_latency_p95"] is not None
        assert data["avg_generation_latency_ms"] is not None
        assert data["daily_query_counts"] is not None
        # 30 days should be populated (filling 0-count days)
        assert len(data["daily_query_counts"]) == 30
        assert data["cost_trends_30d"] == [
            {
                "date": datetime.now(timezone.utc).date().isoformat(),
                "cost_per_query": 0.005,
                "total_cost": 0.015,
            }
        ]
