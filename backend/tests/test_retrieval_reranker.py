"""Unit tests for CrossEncoderReranker and LLMReranker."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import pytest

from pipelines.retrieval.reranker import CrossEncoderReranker, LLMReranker
from pipelines.retrieval.retriever import RetrievalResult


def _make_candidate(chunk_id: str, content: str, score: float, rank: int) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        content=content,
        score=score,
        rank=rank,
        metadata={"rrf_score": score},
    )


@pytest.mark.asyncio
async def test_cross_encoder_rerank_orders_by_relevance():
    # Candidate A initially ranked 1 with RRF score 0.05
    # Candidate B initially ranked 2 with RRF score 0.03
    cand_a = _make_candidate("A", "The capital of France is Paris.", 0.05, 1)
    cand_b = _make_candidate("B", "Python is an interpreted programming language.", 0.03, 2)

    # Query is about Python, so Mock cross-encoder gives B higher score than A
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.12, 0.95]  # score for A, score for B

    reranker = CrossEncoderReranker(model=mock_model, top_n=10)
    reranked = await reranker.rerank("Tell me about Python language", [cand_a, cand_b])

    assert len(reranked) == 2
    # B should be promoted to rank 1
    assert reranked[0].chunk_id == "B"
    assert reranked[0].score == 0.95
    assert reranked[0].rank == 1
    assert reranked[0].metadata["rrf_score"] == 0.03
    assert reranked[0].metadata["cross_encoder_score"] == 0.95

    # A should be demoted to rank 2
    assert reranked[1].chunk_id == "A"
    assert reranked[1].score == 0.12
    assert reranked[1].rank == 2
    assert reranked[1].metadata["rrf_score"] == 0.05


@pytest.mark.asyncio
async def test_cross_encoder_empty_inputs():
    mock_model = MagicMock()
    reranker = CrossEncoderReranker(model=mock_model)

    assert await reranker.rerank("", [_make_candidate("A", "text", 1.0, 1)]) == []
    assert await reranker.rerank("query", []) == []


@pytest.mark.asyncio
async def test_cross_encoder_error_handling():
    mock_model = MagicMock()
    mock_model.predict.side_effect = RuntimeError("CUDA out of memory")

    reranker = CrossEncoderReranker(model=mock_model)
    with pytest.raises(RuntimeError, match="Reranking failed"):
        await reranker.rerank("query", [_make_candidate("A", "text", 1.0, 1)])


@pytest.mark.asyncio
async def test_llm_reranker_scores_and_ranks():
    mock_llm = SimpleNamespace(
        chat=AsyncMock(
            side_effect=[
                SimpleNamespace(content="8.5"),
                SimpleNamespace(content="3.0"),
            ]
        )
    )

    cand_a = _make_candidate("A", "Text about query", 0.02, 1)
    cand_b = _make_candidate("B", "Unrelated text", 0.04, 2)

    reranker = LLMReranker(llm_client=mock_llm, concurrency=2)
    reranked = await reranker.rerank("sample question", [cand_a, cand_b])

    assert len(reranked) == 2
    assert reranked[0].chunk_id == "A"
    assert reranked[0].score == 8.5
    assert reranked[0].rank == 1

    assert reranked[1].chunk_id == "B"
    assert reranked[1].score == 3.0
    assert reranked[1].rank == 2
