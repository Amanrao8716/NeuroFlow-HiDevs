"""Unit tests for Reciprocal Rank Fusion (RRF)."""

import pytest

from pipelines.retrieval.fusion import reciprocal_rank_fusion
from pipelines.retrieval.retriever import RetrievalResult


def _make_res(chunk_id: str, score: float, rank: int, method: str = "dense") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        content=f"Content of {chunk_id}",
        score=score,
        retrieval_method=method,
        rank=rank,
        metadata={"orig": True},
    )


def test_rrf_hand_calculated_example():
    """Verify hand-calculated RRF scores and expected order."""
    # List 1: A (rank 1), B (rank 2)
    list1 = [
        _make_res("A", 0.9, 1, "dense"),
        _make_res("B", 0.8, 2, "dense"),
    ]
    # List 2: B (rank 1), C (rank 2)
    list2 = [
        _make_res("B", 10.0, 1, "sparse"),
        _make_res("C", 5.0, 2, "sparse"),
    ]

    fused = reciprocal_rank_fusion([list1, list2], k=60)

    # Expected:
    # B: 1/62 + 1/61 = 0.01612903 + 0.01639344 = 0.03252247
    # A: 1/61 = 0.01639344
    # C: 1/62 = 0.01612903
    assert len(fused) == 3
    assert fused[0].chunk_id == "B"
    assert fused[1].chunk_id == "A"
    assert fused[2].chunk_id == "C"

    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61, rel=1e-5)
    assert fused[1].score == pytest.approx(1 / 61, rel=1e-5)
    assert fused[2].score == pytest.approx(1 / 62, rel=1e-5)

    assert fused[0].rank == 1
    assert fused[1].rank == 2
    assert fused[2].rank == 3


def test_rrf_does_not_mutate_inputs():
    list1 = [_make_res("A", 0.9, 1, "dense")]
    list2 = [_make_res("B", 0.8, 1, "sparse")]

    list1_copy = [c.clone() for c in list1]

    fused = reciprocal_rank_fusion([list1, list2], k=60)

    assert list1[0].score == list1_copy[0].score
    assert list1[0].retrieval_method == list1_copy[0].retrieval_method
    assert fused[0].retrieval_method == "rrf_fusion"


def test_rrf_preserves_metadata_and_original_scores():
    list1 = [_make_res("A", 0.85, 1, "dense")]
    list2 = [_make_res("A", 4.2, 2, "sparse")]

    fused = reciprocal_rank_fusion([list1, list2], k=60)
    assert len(fused) == 1
    item = fused[0]

    assert item.metadata.get("orig") is True
    assert item.metadata["original_scores"]["dense"] == 0.85
    assert item.metadata["original_scores"]["sparse"] == 4.2
    assert item.metadata["method_ranks"]["dense"] == 1
    assert item.metadata["method_ranks"]["sparse"] == 2


def test_rrf_deterministic_ties():
    """Ties should be broken deterministically by chunk_id."""
    # A and B appear at rank 1 in different lists of length 1
    list1 = [_make_res("chunk-B", 0.9, 1, "dense")]
    list2 = [_make_res("chunk-A", 0.9, 1, "sparse")]

    fused = reciprocal_rank_fusion([list1, list2], k=60)
    # Both have score 1/61. Secondary sort by chunk_id ascending ('chunk-A' before 'chunk-B')
    assert fused[0].chunk_id == "chunk-A"
    assert fused[1].chunk_id == "chunk-B"


def test_rrf_handles_empty_inputs():
    assert reciprocal_rank_fusion([], k=60) == []
    assert reciprocal_rank_fusion([[], []], k=60) == []
    res = reciprocal_rank_fusion([[_make_res("A", 1.0, 1)], []], k=60)
    assert len(res) == 1
    assert res[0].chunk_id == "A"


def test_rrf_invalid_k_raises():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[]], k=0)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[]], k=-10)
