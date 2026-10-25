"""Unit and integration tests for EvaluationJudge orchestrator and training pair generation."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from evaluation.judge import EvaluationJudge, EvaluationReport
from evaluation.metrics.faithfulness import FaithfulnessScore
from evaluation.metrics.answer_relevance import AnswerRelevanceScore
from evaluation.metrics.context_precision import ContextPrecisionScore
from evaluation.metrics.context_recall import ContextRecallScore
from providers.base import GenerationResult


@pytest.fixture
def mock_llm_judge():
    client = MagicMock()
    client.chat = AsyncMock(return_value=GenerationResult(
        content='{"score": 0.9, "reasoning": "High quality response"}',
        model="gpt-4o",
        input_tokens=100,
        output_tokens=30,
        latency_ms=100.0,
        cost_usd=0.0005,
        finish_reason="stop",
    ))
    router = MagicMock()
    router.route = AsyncMock(return_value="gpt-4o")
    client.router = router
    return client


@pytest.mark.asyncio
async def test_evaluation_judge_weighted_score_and_concurrency(mock_llm_judge):
    judge = EvaluationJudge(llm_client=mock_llm_judge)

    # Mock each metric with specific scores
    judge.faithfulness_metric.score = AsyncMock(return_value=FaithfulnessScore(
        score=1.0, claims=[], reasoning="Faithful", raw_response=""
    ))
    judge.answer_relevance_metric.score = AsyncMock(return_value=AnswerRelevanceScore(
        score=0.9, relevance_verdict="relevant", reasoning="Relevant", raw_response=""
    ))
    judge.context_precision_metric.score = AsyncMock(return_value=ContextPrecisionScore(
        score=0.8, chunk_evaluations=[], reasoning="Precise", raw_response=""
    ))
    judge.context_recall_metric.score = AsyncMock(return_value=ContextRecallScore(
        score=0.7, key_facts=[], reasoning="Good recall", raw_response=""
    ))

    # Mock persistence so DB is not called in unit test
    judge._persist_evaluation = AsyncMock()

    run_id = uuid4()
    pipeline_id = uuid4()

    report = await judge.evaluate(
        query="What is PostgreSQL?",
        context="PostgreSQL is an open-source database.",
        answer="PostgreSQL is an open-source relational database.",
        run_id=run_id,
        pipeline_id=pipeline_id,
    )

    assert judge.weights == {
        "faithfulness": 0.35,
        "answer_relevance": 0.30,
        "context_precision": 0.20,
        "context_recall": 0.15,
    }

    # 0.35 * 1.0 + 0.30 * 0.9 + 0.20 * 0.8 + 0.15 * 0.7 = 0.35 + 0.27 + 0.16 + 0.105 = 0.885
    expected_score = 0.885
    assert report.faithfulness == 1.0
    assert report.answer_relevance == 0.9
    assert report.context_precision == 0.8
    assert report.context_recall == 0.7
    assert report.overall_score == expected_score
    assert report.is_training_candidate is True
    assert report.judge_model == "gpt-4o"

    judge._persist_evaluation.assert_called_once()


@pytest.mark.asyncio
async def test_evaluation_judge_training_candidate_threshold(mock_llm_judge):
    judge = EvaluationJudge(llm_client=mock_llm_judge)
    judge._persist_evaluation = AsyncMock()

    # Case 1: Low score (e.g. 0.50 overall) -> Not a training candidate
    judge.faithfulness_metric.score = AsyncMock(return_value=FaithfulnessScore(score=0.5, claims=[], reasoning="", raw_response=""))
    judge.answer_relevance_metric.score = AsyncMock(return_value=AnswerRelevanceScore(score=0.5, relevance_verdict="", reasoning="", raw_response=""))
    judge.context_precision_metric.score = AsyncMock(return_value=ContextPrecisionScore(score=0.5, chunk_evaluations=[], reasoning="", raw_response=""))
    judge.context_recall_metric.score = AsyncMock(return_value=ContextRecallScore(score=0.5, key_facts=[], reasoning="", raw_response=""))

    low_report = await judge.evaluate(
        query="Q", context="C", answer="A",
        run_id=uuid4(), pipeline_id=uuid4(),
    )
    assert low_report.overall_score == 0.50
    assert low_report.is_training_candidate is False

    # Case 2: High score > 0.8 -> Training candidate
    judge.faithfulness_metric.score = AsyncMock(return_value=FaithfulnessScore(score=0.9, claims=[], reasoning="", raw_response=""))
    judge.answer_relevance_metric.score = AsyncMock(return_value=AnswerRelevanceScore(score=0.9, relevance_verdict="", reasoning="", raw_response=""))
    judge.context_precision_metric.score = AsyncMock(return_value=ContextPrecisionScore(score=0.9, chunk_evaluations=[], reasoning="", raw_response=""))
    judge.context_recall_metric.score = AsyncMock(return_value=ContextRecallScore(score=0.9, key_facts=[], reasoning="", raw_response=""))

    high_report = await judge.evaluate(
        query="Q", context="C", answer="A",
        run_id=uuid4(), pipeline_id=uuid4(),
    )
    assert high_report.overall_score == 0.90
    assert high_report.is_training_candidate is True


@pytest.mark.asyncio
async def test_evaluation_judge_persistence_logic(mock_llm_judge):
    judge = EvaluationJudge(llm_client=mock_llm_judge)

    executed_queries = []
    mock_conn = MagicMock()
    async def mock_execute(sql, *args):
        executed_queries.append((sql, args))

    mock_conn.execute = mock_execute

    class MockPipelineConnCtx:
        async def __aenter__(self):
            return mock_conn
        async def __aexit__(self, *args):
            pass

    with patch("evaluation.judge.pipeline_connection", return_value=MockPipelineConnCtx()):
        # Case A: candidate = True -> inserts into both evaluations and training_pairs
        await judge._persist_evaluation(
            eval_id=uuid4(),
            run_id=uuid4(),
            pipeline_id=uuid4(),
            faithfulness=1.0,
            answer_relevance=1.0,
            context_precision=1.0,
            context_recall=1.0,
            overall_score=0.95,
            judge_model="gpt-4o",
            system_prompt="You are an assistant",
            query="What is RAG?",
            answer="RAG combines retrieval with generation.",
            is_training_candidate=True,
        )

        assert any("INSERT INTO evaluations" in sql for sql, _ in executed_queries)
        assert any("INSERT INTO training_pairs" in sql for sql, _ in executed_queries)

        # Case B: candidate = False -> only inserts into evaluations
        executed_queries.clear()
        await judge._persist_evaluation(
            eval_id=uuid4(),
            run_id=uuid4(),
            pipeline_id=uuid4(),
            faithfulness=0.4,
            answer_relevance=0.4,
            context_precision=0.4,
            context_recall=0.4,
            overall_score=0.40,
            judge_model="gpt-4o",
            system_prompt="You are an assistant",
            query="What is RAG?",
            answer="RAG is bad.",
            is_training_candidate=False,
        )

        assert any("INSERT INTO evaluations" in sql for sql, _ in executed_queries)
        assert not any("INSERT INTO training_pairs" in sql for sql, _ in executed_queries)
