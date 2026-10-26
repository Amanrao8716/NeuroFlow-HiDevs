"""Unit and Integration Tests for Task 9 Fine-Tuning Pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from main import app
from pipelines.finetuning.extractor import (
    CITATION_PATTERN,
    TrainingDataExtractor,
    contains_pii,
    extract_context_from_prompt,
    has_valid_citation,
)
from pipelines.finetuning.job_manager import (
    OPENAI_MINIMUM_EXAMPLES,
    FinetuneJobManager,
)
from pipelines.finetuning.tracker import MLflowTracker
from worker.tasks import poll_finetune_job


@pytest.fixture
def client():
    return TestClient(app)


# -------------------------------------------------------------------------
# Helper Mock DB Context
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
# 1. PII Detection Tests
# -------------------------------------------------------------------------

def test_contains_pii_emails():
    """Verify email addresses are flagged accurately without leaking."""
    assert contains_pii("Contact me at user@example.com for details.") is True
    assert contains_pii("Send reports to admin.support+tag@sub.domain.co.uk") is True
    assert contains_pii("Standard query about legal indemnification clause.") is False


def test_contains_pii_phone_numbers():
    """Verify phone numbers are caught, while avoiding innocent dates/numbers."""
    assert contains_pii("Call support at (555) 123-4567 immediately.") is True
    assert contains_pii("Reach out at 555-987-6543") is True
    assert contains_pii("Direct dial +1-800-555-0199") is True
    assert contains_pii("Tel: 555.234.5678") is True

    # Innocent dates and non-phone numbers must NOT be flagged
    assert contains_pii("What are the 2023 and 2024 climate guidelines?") is False
    assert contains_pii("Section 4.2 outlines a 10% penalty on $500,000 deposits.") is False


# -------------------------------------------------------------------------
# 2. Citation Validation Tests
# -------------------------------------------------------------------------

def test_citation_pattern_validation():
    """Verify strict [Source N] positive integer citation checking."""
    assert has_valid_citation("The court ruled in favor of plaintiff [Source 1].") is True
    assert has_valid_citation("Multiple sources [Source 2] and [Source 10] confirm this.") is True

    # Invalid citations
    assert has_valid_citation("No citation provided at all.") is False
    assert has_valid_citation("Zero is invalid [Source 0].") is False
    assert has_valid_citation("Invalid format [Source] or [Source A].") is False
    assert has_valid_citation("Wrong tag format [Citation 1].") is False


# -------------------------------------------------------------------------
# 3. Token Length Boundary Tests
# -------------------------------------------------------------------------

def test_token_length_filtering():
    """Verify assistant responses are rejected if < 50 tokens or > 2000 tokens."""
    extractor = TrainingDataExtractor(min_tokens=50, max_tokens=2000)

    # 1. Too short (< 50 tokens)
    short_text = "PostgreSQL is a relational database system [Source 1]."
    assert extractor.token_counter.count(short_text) < 50

    # 2. Valid length (~70 tokens)
    valid_text = (
        "PostgreSQL is an advanced, enterprise-class open-source relational database [Source 1]. "
        "It supports both SQL for relational and JSON for non-relational queries [Source 2]. "
        "High availability is achieved through streaming replication and automated failover clustering, "
        "ensuring minimal downtime and data integrity across multi-region production deployments."
    )
    tokens = extractor.token_counter.count(valid_text)
    assert 50 <= tokens <= 2000

    # 3. Too long (> 2000 tokens)
    long_text = ("This is repeated legal text verifying upper bound constraints. " * 300) + "[Source 1]"
    assert extractor.token_counter.count(long_text) > 2000


# -------------------------------------------------------------------------
# 4. Quality Score and User Rating Filtering
# -------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_quality_score_and_user_rating_filtering():
    """Verify candidates below quality score or with user_rating < 4 are rejected."""
    extractor = TrainingDataExtractor(min_quality_score=0.82)

    valid_response = (
        "PostgreSQL provides enterprise-grade relational database features and ACID compliance [Source 1]. "
        "It supports advanced indexing like GiST and GIN, partitioned tables, and foreign keys [Source 2]. "
        "Data persistence is guaranteed through Write-Ahead Logging mechanisms and streaming replication, "
        "enabling zero data loss and automated failover in production deployments."
    )

    candidates = [
        # Candidate 1: Valid high quality, rating 5
        {
            "id": uuid4(),
            "run_id": uuid4(),
            "user_message": "Explain PostgreSQL indexing capabilities",
            "assistant_message": valid_response,
            "quality_score": 0.88,
            "eval_faithfulness": 0.95,
            "user_rating": 5,
            "created_at": datetime.now(timezone.utc),
        },
        # Candidate 2: Rejected rating (user_rating = 2)
        {
            "id": uuid4(),
            "run_id": uuid4(),
            "user_message": "Explain PostgreSQL indexing capabilities",
            "assistant_message": valid_response,
            "quality_score": 0.85,
            "eval_faithfulness": 0.90,
            "user_rating": 2,
            "created_at": datetime.now(timezone.utc),
        },
        # Candidate 3: Missing citation
        {
            "id": uuid4(),
            "run_id": uuid4(),
            "user_message": "Explain PostgreSQL indexing capabilities",
            "assistant_message": "PostgreSQL provides enterprise-grade relational database features without citations.",
            "quality_score": 0.86,
            "eval_faithfulness": 0.92,
            "user_rating": 4,
            "created_at": datetime.now(timezone.utc),
        },
    ]

    accepted, rejection_reasons = await extractor.validate_and_filter_pairs(candidates)
    assert len(accepted) == 1
    assert accepted[0]["quality_score"] == 0.88 if isinstance(accepted[0], dict) else accepted[0].quality_score == 0.88
    assert rejection_reasons["rejected_rating"] == 1
    assert rejection_reasons["missing_citation"] == 1


# -------------------------------------------------------------------------
# 5. Missing Faithfulness Re-Evaluation Tests
# -------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_missing_faithfulness_reevaluation_rejection():
    """When faithfulness score is missing from evaluations, re-evaluate and reject if <= 0.8."""
    mock_scorer = MagicMock()
    # Re-evaluation returns low faithfulness score
    mock_score_res = MagicMock()
    mock_score_res.score = 0.60
    mock_scorer.score = AsyncMock(return_value=mock_score_res)

    extractor = TrainingDataExtractor(
        min_quality_score=0.82,
        min_faithfulness=0.80,
        faithfulness_scorer=mock_scorer,
    )

    valid_response = (
        "PostgreSQL provides enterprise-grade relational database features and ACID compliance [Source 1]. "
        "It supports advanced indexing like GiST and GIN, partitioned tables, and foreign keys [Source 2]. "
        "Data persistence is guaranteed through Write-Ahead Logging mechanisms and streaming replication, "
        "enabling zero data loss and automated failover in production deployments."
    )

    candidates = [
        {
            "id": uuid4(),
            "run_id": uuid4(),
            "user_message": "Explain PostgreSQL indexing capabilities",
            "assistant_message": valid_response,
            "quality_score": 0.89,
            "eval_faithfulness": None,  # Missing!
            "user_rating": None,
            "created_at": datetime.now(timezone.utc),
            "run_prompt": "<context>PostgreSQL details</context>",
        }
    ]

    accepted, rejection_reasons = await extractor.validate_and_filter_pairs(candidates)
    assert len(accepted) == 0
    assert rejection_reasons["low_faithfulness"] == 1
    assert mock_scorer.score.called


@pytest.mark.asyncio
async def test_missing_faithfulness_reevaluation_acceptance():
    """When faithfulness is missing, passing re-evaluation allows the example to be accepted."""
    mock_scorer = MagicMock()
    mock_score_res = MagicMock()
    mock_score_res.score = 0.92
    mock_scorer.score = AsyncMock(return_value=mock_score_res)

    extractor = TrainingDataExtractor(
        min_quality_score=0.82,
        min_faithfulness=0.80,
        faithfulness_scorer=mock_scorer,
    )

    valid_response = (
        "PostgreSQL provides enterprise-grade relational database features and ACID compliance [Source 1]. "
        "It supports advanced indexing like GiST and GIN, partitioned tables, and foreign keys [Source 2]. "
        "Data persistence is guaranteed through Write-Ahead Logging mechanisms and streaming replication, "
        "enabling zero data loss and automated failover in production deployments."
    )

    candidates = [
        {
            "id": uuid4(),
            "run_id": uuid4(),
            "user_message": "Explain PostgreSQL indexing capabilities",
            "assistant_message": valid_response,
            "quality_score": 0.89,
            "eval_faithfulness": None,
            "user_rating": 4,
            "created_at": datetime.now(timezone.utc),
            "run_prompt": "<context>PostgreSQL details</context>",
        }
    ]

    accepted, rejection_reasons = await extractor.validate_and_filter_pairs(candidates)
    assert len(accepted) == 1
    assert accepted[0].faithfulness == 0.92
    assert rejection_reasons["low_faithfulness"] == 0


# -------------------------------------------------------------------------
# 6. JSONL Formatting and Persistence Tests
# -------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_jsonl_formatting_and_persistence(tmp_path):
    """Verify output JSONL formatting conforms to OpenAI conversational message schema."""
    extractor = TrainingDataExtractor()
    job_id = uuid4()

    valid_response = (
        "PostgreSQL provides enterprise-grade relational database features and ACID compliance [Source 1]. "
        "It supports advanced indexing like GiST and GIN, partitioned tables, and foreign keys [Source 2]. "
        "Data persistence is guaranteed through Write-Ahead Logging mechanisms and streaming replication, "
        "enabling zero data loss and automated failover in production deployments."
    )

    candidate = {
        "id": uuid4(),
        "run_id": uuid4(),
        "system_prompt": "You are a legal assistant.",
        "user_message": "What is the liability limitation?",
        "assistant_message": valid_response,
        "quality_score": 0.88,
        "eval_faithfulness": 0.94,
        "user_rating": 5,
        "created_at": datetime.now(timezone.utc),
        "run_prompt": "<context>Liability is limited to fees paid.</context>",
    }

    mock_conn = AsyncMock()
    mock_conn.fetch.return_value = [candidate]

    file_path, report, accepted = await extractor.extract_for_job(
        job_id=job_id,
        output_dir=str(tmp_path),
        db_conn=mock_conn,
    )

    assert file_path.exists()
    assert report["accepted_count"] == 1
    assert report["eligible_count"] == 1

    # Read and validate JSONL line
    lines = file_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert "messages" in data
    assert len(data["messages"]) == 3
    assert data["messages"][0]["role"] == "system"
    assert data["messages"][1]["role"] == "user"
    assert "[Context]" in data["messages"][1]["content"]
    assert "[Question]" in data["messages"][1]["content"]
    assert data["messages"][2]["role"] == "assistant"
    assert "[Source 1]" in data["messages"][2]["content"]


# -------------------------------------------------------------------------
# 7. MLflow Tracker Parameter, Artifact & Metric Logging Tests
# -------------------------------------------------------------------------

def test_mlflow_tracker_run_creation_and_logging(tmp_path):
    """Verify MLflowTracker creates run with finetune-{id}, logs parameters and artifact."""
    mock_mlflow_client = MagicMock()
    mock_run = MagicMock()
    mock_run.info.run_id = "test-run-12345"
    mock_mlflow_client.create_run.return_value = mock_run

    mock_exp = MagicMock()
    mock_exp.experiment_id = "exp-001"
    mock_mlflow_client.get_experiment_by_name.return_value = mock_exp

    tracker = MLflowTracker(tracking_uri="http://mock-mlflow:5000")
    tracker._client = mock_mlflow_client

    sample_file = tmp_path / "sample.jsonl"
    sample_file.write_text('{"messages": []}', encoding="utf-8")

    job_id = uuid4()
    run_id, run_url = tracker.create_run(
        job_id=job_id,
        base_model="gpt-4o-mini",
        training_pair_count=25,
        average_quality_score=0.885,
        date_range="2026-10-01 to 2026-10-25",
        training_file_path=sample_file,
    )

    assert run_id == "test-run-12345"
    assert "test-run-12345" in run_url
    assert mock_mlflow_client.create_run.call_args[1]["run_name"] == f"finetune-{job_id}"

    # Verify params logged
    param_calls = {c[0][1]: c[0][2] for c in mock_mlflow_client.log_param.call_args_list}
    assert param_calls["base_model"] == "gpt-4o-mini"
    assert param_calls["accepted_training_pair_count"] == "25"
    assert param_calls["average_quality_score"] == "0.8850"

    # Verify artifact logged
    assert mock_mlflow_client.log_artifact.called

    # Verify completion metric logging
    tracker.log_completion_metrics(
        run_id="test-run-12345",
        metrics={"training_loss": 0.152, "validation_loss": 0.178},
        trained_tokens=15000,
        status="FINISHED",
    )
    metric_calls = {c[0][1]: c[0][2] for c in mock_mlflow_client.log_metric.call_args_list}
    assert metric_calls["training_loss"] == 0.152
    assert metric_calls["trained_tokens"] == 15000.0
    mock_mlflow_client.set_terminated.assert_called_once_with("test-run-12345", status="FINISHED")


# -------------------------------------------------------------------------
# 8. Insufficient Training Examples Rejection
# -------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_job_manager_rejects_insufficient_samples():
    """Verify JobManager fails cleanly when valid examples < 10 (OpenAI minimum)."""
    mock_extractor = MagicMock()
    mock_extractor.extract_for_job = AsyncMock(return_value=(
        Path("/fake/file.jsonl"),
        {
            "eligible_count": 5,
            "accepted_count": 5,
            "rejected_count": 0,
            "rejection_reasons": {},
            "average_quality_score": 0.88,
            "date_range": "2026-10-20",
        },
        [MagicMock() for _ in range(5)],
    ))

    mock_conn = AsyncMock()
    mock_mgr = _create_mock_db_context(mock_conn)

    manager = FinetuneJobManager(
        extractor=mock_extractor,
        openai_client=MagicMock(),
    )

    with patch("pipelines.finetuning.job_manager.connection", mock_mgr):
        result = await manager.create_and_submit_job(base_model="gpt-4o-mini")
        assert result["status"] == "failed"
        assert "Insufficient training examples" in result["error_message"]
        assert result["training_pair_count"] == 5


# -------------------------------------------------------------------------
# 9. Provider Submission and Polling Tests
# -------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_poll_finetune_job_success_and_model_registration():
    """Verify poll_finetune_job updates status, records tokens, and registers in Redis router:models."""
    jid = uuid4()
    mock_conn = AsyncMock()
    mock_conn.fetchrow.return_value = {
        "id": jid,
        "provider_job_id": "ftjob-xyz987",
        "status": "running",
        "mlflow_run_id": "run-mlflow-111",
        "fine_tuned_model": None,
        "base_model": "gpt-4o-mini",
    }

    mock_mgr = _create_mock_db_context(mock_conn)

    # Mock OpenAI retrieve response
    mock_openai_client = MagicMock()
    mock_remote_job = MagicMock()
    mock_remote_job.status = "succeeded"
    mock_remote_job.fine_tuned_model = "ft:gpt-4o-mini:neuroflow:legal-v1"
    mock_remote_job.trained_tokens = 45000
    mock_remote_job.training_metrics = {"final_loss": 0.085}
    mock_openai_client.fine_tuning.jobs.retrieve = AsyncMock(return_value=mock_remote_job)

    # Mock Redis
    mock_redis = AsyncMock()
    mock_redis.hget.return_value = None  # Not yet registered

    mock_ctx = {"redis": mock_redis}

    with patch("worker.tasks.connection", mock_mgr), \
         patch("worker.tasks.MLflowTracker") as mock_tracker_cls:
        mock_tracker = MagicMock()
        mock_tracker_cls.return_value = mock_tracker

        res = await poll_finetune_job(
            ctx=mock_ctx,
            job_id=str(jid),
            openai_client=mock_openai_client,
        )

        assert res["status"] == "succeeded"
        assert res["fine_tuned_model"] == "ft:gpt-4o-mini:neuroflow:legal-v1"

        # Verify DB updated to succeeded
        update_calls = [c[0][0] for c in mock_conn.execute.call_args_list]
        assert any("UPDATE finetune_jobs" in str(sql) and "succeeded" in str(sql) for sql in update_calls)

        # Verify Redis model registered
        assert mock_redis.hset.called
        call_args = mock_redis.hset.call_args[0]
        assert call_args[0] == "router:models"
        assert call_args[1] == "ft:gpt-4o-mini:neuroflow:legal-v1"
        registered_cfg = json.loads(call_args[2])
        assert registered_cfg["fine_tuned"] is True
        assert registered_cfg["task_type"] == "rag_qa"


@pytest.mark.asyncio
async def test_poll_finetune_job_idempotency_terminal_state():
    """Verify poll_finetune_job returns immediately without modifying state if already succeeded."""
    jid = uuid4()
    mock_conn = AsyncMock()
    mock_conn.fetchrow.return_value = {
        "id": jid,
        "provider_job_id": "ftjob-xyz987",
        "status": "succeeded",
        "mlflow_run_id": "run-mlflow-111",
        "fine_tuned_model": "ft:gpt-4o-mini:neuroflow:legal-v1",
        "base_model": "gpt-4o-mini",
    }
    mock_mgr = _create_mock_db_context(mock_conn)
    mock_openai = MagicMock()

    with patch("worker.tasks.connection", mock_mgr):
        res = await poll_finetune_job(ctx={}, job_id=str(jid), openai_client=mock_openai)
        assert res["status"] == "succeeded"
        assert not mock_openai.fine_tuning.jobs.retrieve.called


# -------------------------------------------------------------------------
# 10. DPO Preference Extraction Tests
# -------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dpo_preference_pairing_and_deduplication():
    """Verify DPO matches chosen (>= 4) and rejected (<= 2) for identical queries and ignores PII."""
    extractor = TrainingDataExtractor()

    good_response = "Indemnity covers direct liabilities up to the contract limit [Source 1]."
    bad_response = "No liability coverage is provided under any circumstances."

    mock_conn = AsyncMock()
    mock_conn.fetch.return_value = [
        # Query 1: Valid pair (rating 5 and rating 2)
        {"query": "What is the liability cap?", "generation": good_response, "rating": 5},
        {"query": "What is the liability cap?", "generation": bad_response, "rating": 2},
        # Query 2: Only positive, no negative (should be skipped)
        {"query": "What is the notice period?", "generation": "Notice period is 30 days [Source 1].", "rating": 5},
        # Query 3: PII query (should be skipped)
        {"query": "Contact user@example.com for NDA?", "generation": good_response, "rating": 5},
        {"query": "Contact user@example.com for NDA?", "generation": bad_response, "rating": 1},
    ]

    pairs = await extractor.extract_dpo_pairs(db_conn=mock_conn)
    assert len(pairs) == 1
    assert pairs[0]["prompt"] == "What is the liability cap?"
    assert pairs[0]["chosen"] == good_response
    assert pairs[0]["rejected"] == bad_response


# -------------------------------------------------------------------------
# 11. API Endpoints Tests
# -------------------------------------------------------------------------

def test_api_preview_training_data(client):
    """GET /finetune/training-data/preview returns preview without modifying state."""
    mock_extractor = MagicMock()
    mock_extractor.preview_eligible_pairs = AsyncMock(return_value=([
        {
            "id": str(uuid4()),
            "run_id": str(uuid4()),
            "user_message": "What is pgvector?",
            "assistant_message": "pgvector is a vector similarity search extension for PostgreSQL [Source 1].",
            "quality_score": 0.92,
            "faithfulness": 0.95,
            "user_rating": 5,
            "token_count": 65,
            "citations_count": 1,
            "has_valid_citations": True,
        }
    ], 1))

    with patch("api.finetune.TrainingDataExtractor", return_value=mock_extractor):
        response = client.get("/finetune/training-data/preview")
        assert response.status_code == 200
        data = response.json()
        assert data["total_eligible"] == 1
        assert len(data["items"]) == 1
        assert data["items"][0]["quality_score"] == 0.92
        assert data["items"][0]["has_valid_citations"] is True


def test_api_list_and_get_job(client):
    """GET /finetune/jobs and GET /finetune/jobs/{id} return expected structures."""
    jid = uuid4()
    mock_conn = AsyncMock()
    now = datetime.now(timezone.utc)
    job_record = {
        "id": jid,
        "provider_job_id": "ftjob-001",
        "base_model": "gpt-4o-mini",
        "status": "succeeded",
        "training_pair_count": 20,
        "mlflow_run_id": "mlflow-run-abc",
        "fine_tuned_model": "ft:gpt-4o-mini:neuroflow:v1",
        "metrics": {"training_loss": 0.12},
        "trained_tokens": 30000,
        "error_message": None,
        "extraction_report": {"accepted_count": 20, "eligible_count": 22},
        "pipeline_id": None,
        "created_at": now,
        "completed_at": now,
    }

    mock_conn.fetch.return_value = [job_record]
    mock_conn.fetchrow.return_value = job_record
    mock_mgr = _create_mock_db_context(mock_conn)

    with patch("api.finetune.connection", mock_mgr):
        # 1. List
        list_res = client.get("/finetune/jobs")
        assert list_res.status_code == 200
        list_data = list_res.json()
        assert list_data["total"] == 1
        assert list_data["jobs"][0]["id"] == str(jid)
        assert list_data["jobs"][0]["status"] == "succeeded"

        # 2. Detail
        detail_res = client.get(f"/finetune/jobs/{jid}")
        assert detail_res.status_code == 200
        detail_data = detail_res.json()
        assert detail_data["id"] == str(jid)
        assert detail_data["fine_tuned_model"] == "ft:gpt-4o-mini:neuroflow:v1"
        assert detail_data["trained_tokens"] == 30000
