"""Unit and API tests for evaluation human rating calibration and runner."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from fastapi.testclient import TestClient

from main import app
from evaluation.calibration_runner import (
    CalibrationRunner,
    calculate_mae,
    calculate_pearson_correlation,
)


@pytest.fixture
def client():
    return TestClient(app)


def test_pearson_correlation_math():
    # Perfectly correlated
    x = [0.1, 0.4, 0.7, 0.9, 1.0]
    y = [0.1, 0.4, 0.7, 0.9, 1.0]
    corr = calculate_pearson_correlation(x, y)
    assert corr == 1.0

    # Negative correlation
    y_neg = [-v for v in x]
    corr_neg = calculate_pearson_correlation(x, y_neg)
    assert corr_neg == -1.0

    # MAE test
    mae = calculate_mae([1.0, 0.5], [0.8, 0.5])
    assert mae == 0.1


def test_pearson_correlation_constant_vector_returns_none():
    # Zero variance vectors should return None (undefined correlation)
    x = [1.0, 1.0, 1.0, 1.0]
    y = [0.2, 0.4, 0.6, 0.8]
    assert calculate_pearson_correlation(x, y) is None


@pytest.mark.asyncio
async def test_calibration_runner_with_mock_evaluator():
    from evaluation.judge import EvaluationJudge, EvaluationReport

    # Provide a mock judge that produces realistic varying evaluation reports
    mock_judge = MagicMock(spec=EvaluationJudge)
    mock_judge.llm_client = MagicMock()

    call_count = 0
    async def mock_evaluate(query, context, answer, run_id, pipeline_id, **kwargs):
        nonlocal call_count
        call_count += 1
        # Return varying score based on call count to demonstrate evaluator execution
        f_score = 1.0 if "PostgreSQL" in query or "Redis" in query else 0.2
        return EvaluationReport(
            eval_id=str(uuid4()),
            run_id=str(run_id),
            pipeline_id=str(pipeline_id),
            faithfulness=f_score,
            answer_relevance=f_score,
            context_precision=f_score,
            context_recall=f_score,
            overall_score=f_score,
            judge_model="gpt-4o",
            is_training_candidate=f_score > 0.8,
        )

    mock_judge.evaluate = AsyncMock(side_effect=mock_evaluate)

    runner = CalibrationRunner(judge=mock_judge)
    results = await runner.run()

    assert results["sample_count"] == 30
    assert call_count == 30  # Evaluator was actually executed for all 30 examples
    assert results["status"] == "completed"
    assert "faithfulness_pearson_correlation" in results
    assert results["provenance"]["executed_live_evaluation"] is True


@pytest.mark.asyncio
async def test_calibration_runner_no_client_reports_failure():
    # If no LLM client is available, reports failure honestly
    runner = CalibrationRunner(judge=None)
    with patch.object(runner, "_get_default_judge", AsyncMock(return_value=None)):
        results = await runner.run()
        assert results["status"] == "failed"
        assert results["faithfulness_pearson_correlation"] is None
        assert results["is_calibrated"] is False


def test_patch_rating_calibration_needed_true(client):
    run_id = uuid4()
    pipeline_id = uuid4()

    mock_eval_row = {
        "id": uuid4(),
        "faithfulness": 0.95,
        "answer_relevance": 0.95,
        "context_precision": 0.95,
        "context_recall": 0.95,
        "overall_score": 0.95,
        "judge_model": "gpt-4o",
        "user_rating": None,
    }

    mock_conn = MagicMock()
    mock_conn.fetchrow = AsyncMock(return_value=mock_eval_row)
    mock_conn.execute = AsyncMock()

    class MockPipelineConnCtx:
        async def __aenter__(self):
            return mock_conn
        async def __aexit__(self, *args):
            pass

    with patch("api.evaluations.pipeline_connection", return_value=MockPipelineConnCtx()):
        # Human rates 1 star (normalized: 0.20)
        # Discrepancy: |0.95 - 0.20| = 0.75 > 0.3 -> calibration_needed = True
        response = client.patch(
            f"/runs/{run_id}/rating",
            json={"rating": 1, "pipeline_id": str(pipeline_id)},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["user_rating"] == 1
        assert data["normalized_rating"] == 0.20
        assert data["overall_score"] == 0.95
        assert data["calibration_needed"] is True
        assert data["discrepancy"] == 0.75


def test_patch_rating_calibration_needed_false(client):
    run_id = uuid4()
    pipeline_id = uuid4()

    mock_eval_row = {
        "id": uuid4(),
        "faithfulness": 0.90,
        "answer_relevance": 0.85,
        "context_precision": 0.90,
        "context_recall": 0.80,
        "overall_score": 0.875,
        "judge_model": "gpt-4o",
        "user_rating": None,
    }

    mock_conn = MagicMock()
    mock_conn.fetchrow = AsyncMock(return_value=mock_eval_row)
    mock_conn.execute = AsyncMock()

    class MockPipelineConnCtx:
        async def __aenter__(self):
            return mock_conn
        async def __aexit__(self, *args):
            pass

    with patch("api.evaluations.pipeline_connection", return_value=MockPipelineConnCtx()):
        # Human rates 4 stars (normalized: 0.80)
        # Discrepancy: |0.875 - 0.80| = 0.075 <= 0.3 -> calibration_needed = False
        response = client.patch(
            f"/runs/{run_id}/rating",
            json={"rating": 4, "pipeline_id": str(pipeline_id)},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["user_rating"] == 4
        assert data["normalized_rating"] == 0.80
        assert data["calibration_needed"] is False
        assert data["discrepancy"] == 0.075
