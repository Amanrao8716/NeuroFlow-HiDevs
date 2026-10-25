"""Unit tests for Task 7 RAGAS evaluation metrics."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from evaluation.metrics.faithfulness import FaithfulnessMetric, FaithfulnessScore
from evaluation.metrics.answer_relevance import AnswerRelevanceMetric, AnswerRelevanceScore
from evaluation.metrics.context_precision import ContextPrecisionMetric, ContextPrecisionScore
from evaluation.metrics.context_recall import ContextRecallMetric, ContextRecallScore
from providers.base import GenerationResult


@pytest.fixture
def mock_llm():
    client = MagicMock()
    client.chat = AsyncMock()
    return client


# --- Faithfulness Tests ---

@pytest.mark.asyncio
async def test_faithfulness_metric_success(mock_llm):
    mock_llm.chat.return_value = GenerationResult(
        content='```json\n{"claims": [{"claim": "PostgreSQL is open source", "verdict": "supported"}], "score": 1.0, "reasoning": "Directly supported"}\n```',
        model="gpt-4o",
        input_tokens=100,
        output_tokens=30,
        latency_ms=150.0,
        cost_usd=0.0005,
        finish_reason="stop",
    )

    metric = FaithfulnessMetric(llm_client=mock_llm)
    result = await metric.score(
        answer="PostgreSQL is open source [Source 1].",
        context="PostgreSQL is an open source database.",
        query="What is PostgreSQL?",
    )

    assert isinstance(result, FaithfulnessScore)
    assert result.score == 1.0
    assert len(result.claims) == 1
    assert result.claims[0]["verdict"] == "supported"
    assert "Directly supported" in result.reasoning


@pytest.mark.asyncio
async def test_faithfulness_metric_hallucinated(mock_llm):
    mock_llm.chat.return_value = GenerationResult(
        content='{"claims": [{"claim": "It has 1M stars", "verdict": "hallucinated"}], "score": 0.0, "reasoning": "Not mentioned"}',
        model="gpt-4o",
        input_tokens=100,
        output_tokens=30,
        latency_ms=150.0,
        cost_usd=0.0005,
        finish_reason="stop",
    )

    metric = FaithfulnessMetric(llm_client=mock_llm)
    result = await metric.score(
        answer="It has 1M stars.",
        context="PostgreSQL is an open source database.",
    )

    assert result.score == 0.0
    assert result.claims[0]["verdict"] == "hallucinated"


@pytest.mark.asyncio
async def test_faithfulness_fallback_parsing(mock_llm):
    mock_llm.chat.return_value = GenerationResult(
        content='The faithfulness analysis yielded "score": 0.85 after claim extraction.',
        model="gpt-4o",
        input_tokens=100,
        output_tokens=30,
        latency_ms=150.0,
        cost_usd=0.0005,
        finish_reason="stop",
    )

    metric = FaithfulnessMetric(llm_client=mock_llm)
    result = await metric.score(answer="Test answer", context="Test context")

    assert result.score == 0.85


@pytest.mark.asyncio
async def test_faithfulness_no_client():
    metric = FaithfulnessMetric(llm_client=None)
    result = await metric.score(answer="Test", context="Context")
    assert result.score == 1.0


# --- Answer Relevance Tests ---

@pytest.mark.asyncio
async def test_answer_relevance_metric_success(mock_llm):
    mock_llm.chat.return_value = GenerationResult(
        content='```json\n{"relevance_verdict": "relevant", "score": 0.95, "reasoning": "Direct answer"}\n```',
        model="gpt-4o",
        input_tokens=80,
        output_tokens=25,
        latency_ms=120.0,
        cost_usd=0.0004,
        finish_reason="stop",
    )

    metric = AnswerRelevanceMetric(llm_client=mock_llm)
    result = await metric.score(
        query="What port does Redis use?",
        answer="Redis uses port 6379 by default.",
    )

    assert isinstance(result, AnswerRelevanceScore)
    assert result.score == 0.95
    assert result.relevance_verdict == "relevant"


@pytest.mark.asyncio
async def test_answer_relevance_no_client():
    metric = AnswerRelevanceMetric(llm_client=None)
    result = await metric.score(query="Q", answer="A")
    assert result.score == 1.0
    assert result.relevance_verdict == "relevant"


# --- Context Precision Tests ---

@pytest.mark.asyncio
async def test_context_precision_empty_context():
    metric = ContextPrecisionMetric(llm_client=None)
    result = await metric.score(query="Q", context="   ")
    assert result.score == 0.0
    assert result.reasoning == "Empty context provided."


@pytest.mark.asyncio
async def test_context_precision_success(mock_llm):
    mock_llm.chat.return_value = GenerationResult(
        content='{"chunk_evaluations": [{"chunk_index": 1, "relevant": true}], "score": 1.0, "reasoning": "Top chunk highly relevant"}',
        model="gpt-4o",
        input_tokens=150,
        output_tokens=30,
        latency_ms=140.0,
        cost_usd=0.0005,
        finish_reason="stop",
    )

    metric = ContextPrecisionMetric(llm_client=mock_llm)
    result = await metric.score(
        query="Explain pgvector",
        context="[Source 1] pgvector allows vector storage in Postgres.",
        answer="pgvector adds vector capability.",
    )

    assert isinstance(result, ContextPrecisionScore)
    assert result.score == 1.0
    assert len(result.chunk_evaluations) == 1


# --- Context Recall Tests ---

@pytest.mark.asyncio
async def test_context_recall_empty_context():
    metric = ContextRecallMetric(llm_client=None)
    result = await metric.score(query="Q", context="")
    assert result.score == 0.0


@pytest.mark.asyncio
async def test_context_recall_success(mock_llm):
    mock_llm.chat.return_value = GenerationResult(
        content='{"key_facts": [{"fact": "Uses port 6379", "in_context": true}], "score": 1.0, "reasoning": "All facts covered"}',
        model="gpt-4o",
        input_tokens=150,
        output_tokens=30,
        latency_ms=140.0,
        cost_usd=0.0005,
        finish_reason="stop",
    )

    metric = ContextRecallMetric(llm_client=mock_llm)
    result = await metric.score(
        query="What port does Redis use?",
        context="Redis default port is 6379.",
        answer="Redis uses 6379.",
    )

    assert isinstance(result, ContextRecallScore)
    assert result.score == 1.0
    assert len(result.key_facts) == 1
    assert result.key_facts[0]["in_context"] is True
