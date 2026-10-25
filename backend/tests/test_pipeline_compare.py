"""Unit and Integration tests for A/B Pipeline Comparison endpoint (POST /pipelines/compare)."""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from fastapi.testclient import TestClient

from main import app
from pipelines.generation.generator import GenerationPipeline, GenerationResult
from pipelines.generation.citations import ParsedCitations


@pytest.fixture
def client():
    return TestClient(app)


def test_compare_identical_pipelines_rejected(client):
    """POST /pipelines/compare rejects identical pipeline IDs without distinct versions."""
    same_id = uuid4()
    response = client.post(
        "/pipelines/compare",
        json={
            "query": "Test query",
            "pipeline_a_id": str(same_id),
            "pipeline_b_id": str(same_id),
        },
    )
    assert response.status_code == 400
    assert "Cannot compare identical pipeline IDs" in response.json()["detail"]


def test_compare_missing_pipeline_returns_404(client):
    """POST /pipelines/compare returns 404 when either pipeline does not exist."""
    pid_a = uuid4()
    pid_b = uuid4()

    mock_conn = AsyncMock()
    # pipeline A exists, pipeline B does not
    mock_conn.fetchrow.side_effect = [
        {"id": pid_a, "name": "Pipe A", "status": "active", "version": 1},
        None,
    ]

    mock_pool = MagicMock()
    mock_pool.__aenter__.return_value = mock_conn
    mock_pool.__aexit__.return_value = None

    with patch("api.compare.pipeline_connection", return_value=mock_pool):
        response = client.post(
            "/pipelines/compare",
            json={
                "query": "Test query",
                "pipeline_a_id": str(pid_a),
                "pipeline_b_id": str(pid_b),
            },
        )
        assert response.status_code == 404
        assert f"Pipeline {pid_b} not found" in response.json()["detail"]


def test_compare_archived_pipeline_returns_400(client):
    """POST /pipelines/compare returns 400 when either pipeline is archived."""
    pid_a = uuid4()
    pid_b = uuid4()

    mock_conn = AsyncMock()
    mock_conn.fetchrow.side_effect = [
        {"id": pid_a, "name": "Pipe A", "status": "archived", "version": 1},
        {"id": pid_b, "name": "Pipe B", "status": "active", "version": 2},
    ]

    mock_pool = MagicMock()
    mock_pool.__aenter__.return_value = mock_conn
    mock_pool.__aexit__.return_value = None

    with patch("api.compare.pipeline_connection", return_value=mock_pool):
        response = client.post(
            "/pipelines/compare",
            json={
                "query": "Test query",
                "pipeline_a_id": str(pid_a),
                "pipeline_b_id": str(pid_b),
            },
        )
        assert response.status_code == 400
        assert f"Pipeline {pid_a} is archived" in response.json()["detail"]


def test_compare_one_side_failure(client):
    """POST /pipelines/compare handles one-side execution failure gracefully without failing request."""
    pid_a = uuid4()
    pid_b = uuid4()
    run_id_a = str(uuid4())

    mock_conn = AsyncMock()
    # DB checks for existence
    mock_conn.fetchrow.side_effect = [
        {"id": pid_a, "name": "Pipe A", "status": "active", "version": 1},
        {"id": pid_b, "name": "Pipe B", "status": "active", "version": 2},
        # eval check for pipe A
        {"overall_score": 0.92},
    ]

    mock_pool = MagicMock()
    mock_pool.__aenter__.return_value = mock_conn
    mock_pool.__aexit__.return_value = None

    # Mock generator where pipeline A succeeds and pipeline B raises an exception
    mock_generator = MagicMock(spec=GenerationPipeline)

    async def mock_generate(query, pipeline_id, **kwargs):
        if pipeline_id == pid_a:
            return GenerationResult(
                run_id=run_id_a,
                query=query,
                query_type="factual",
                generation="Answer from Pipeline A",
                citations=ParsedCitations(),
                sources=[],
                chunks_used=3,
                input_tokens=500,
                output_tokens=50,
                latency_ms=120,
                model_used="gpt-4o",
                status="completed",
                pipeline_version=1,
                retrieval_latency_ms=25,
            )
        else:
            raise RuntimeError("Downstream LLM provider timed out")

    mock_generator.generate = mock_generate

    with patch("api.compare.pipeline_connection", return_value=mock_pool), \
         patch("api.compare.get_generation_pipeline", return_value=mock_generator):
        response = client.post(
            "/pipelines/compare",
            json={
                "query": "What is the liability clause?",
                "pipeline_a_id": str(pid_a),
                "pipeline_b_id": str(pid_b),
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "What is the liability clause?"

        # Pipeline A succeeded
        assert data["pipeline_a"]["pipeline_id"] == str(pid_a)
        assert data["pipeline_a"]["answer"] == "Answer from Pipeline A"
        assert data["pipeline_a"]["run_id"] == run_id_a
        assert data["pipeline_a"]["evaluation_score"] == 0.92
        assert data["pipeline_a"]["error"] is None

        # Pipeline B failed
        assert data["pipeline_b"]["pipeline_id"] == str(pid_b)
        assert data["pipeline_b"]["answer"] is None
        assert data["pipeline_b"]["run_id"] is None
        assert "Downstream LLM provider timed out" in data["pipeline_b"]["error"]


@pytest.mark.asyncio
async def test_compare_concurrent_execution_overlapping():
    """Verify that both pipelines execute concurrently using a controlled async double.

    Using synchronization primitives (asyncio.Event), each pipeline signals its start
    and waits for the other to start before finishing. If execution were sequential,
    the first pipeline would block indefinitely waiting for the second to start,
    causing a timeout.
    """
    from api.compare import _execute_single_pipeline

    pid_a = uuid4()
    pid_b = uuid4()

    event_a_started = asyncio.Event()
    event_b_started = asyncio.Event()

    execution_timeline = []

    mock_generator = MagicMock(spec=GenerationPipeline)

    async def mock_generate(query, pipeline_id, **kwargs):
        if pipeline_id == pid_a:
            execution_timeline.append("pipe_a_started")
            event_a_started.set()
            # Wait until pipeline B has also started — proves overlapping execution!
            await asyncio.wait_for(event_b_started.wait(), timeout=1.0)
            execution_timeline.append("pipe_a_finished")
            return GenerationResult(
                run_id=str(uuid4()),
                query=query,
                query_type="analytical",
                generation="Result A",
                citations=ParsedCitations(),
                sources=[],
                chunks_used=4,
                input_tokens=600,
                output_tokens=60,
                latency_ms=50,
                model_used="gpt-4o",
                status="completed",
                pipeline_version=1,
                retrieval_latency_ms=10,
            )
        else:
            execution_timeline.append("pipe_b_started")
            event_b_started.set()
            # Wait until pipeline A has also started
            await asyncio.wait_for(event_a_started.wait(), timeout=1.0)
            execution_timeline.append("pipe_b_finished")
            return GenerationResult(
                run_id=str(uuid4()),
                query=query,
                query_type="analytical",
                generation="Result B",
                citations=ParsedCitations(),
                sources=[],
                chunks_used=2,
                input_tokens=400,
                output_tokens=40,
                latency_ms=50,
                model_used="gpt-4o-mini",
                status="completed",
                pipeline_version=2,
                retrieval_latency_ms=15,
            )

    mock_generator.generate = mock_generate

    # Concurrently execute both via asyncio.gather
    res_a, res_b = await asyncio.gather(
        _execute_single_pipeline(mock_generator, "Test query", pid_a, version=1),
        _execute_single_pipeline(mock_generator, "Test query", pid_b, version=2),
    )

    # Both succeeded
    assert res_a.error is None
    assert res_b.error is None
    assert res_a.answer == "Result A"
    assert res_b.answer == "Result B"

    # Verify overlapping timeline: both pipelines started before either finished
    assert "pipe_a_started" in execution_timeline
    assert "pipe_b_started" in execution_timeline
    idx_a_start = execution_timeline.index("pipe_a_started")
    idx_b_start = execution_timeline.index("pipe_b_started")
    idx_a_end = execution_timeline.index("pipe_a_finished")
    idx_b_end = execution_timeline.index("pipe_b_finished")

    # Both starts must precede both ends
    assert idx_a_start < idx_a_end
    assert idx_b_start < idx_b_end
    assert idx_a_start < idx_b_end
    assert idx_b_start < idx_a_end
