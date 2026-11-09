from __future__ import annotations

import pytest

from pipelines.retrieval.fusion import reciprocal_rank_fusion
from pipelines.retrieval.retriever import RetrievalResult


def candidate(chunk_id: str, rank: int, method: str = "dense") -> RetrievalResult:
    return RetrievalResult(chunk_id, "doc", chunk_id, 1.0, rank=rank, retrieval_method=method)


def test_rrf_empty_inputs() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []


def test_rrf_known_scores() -> None:
    fused = reciprocal_rank_fusion([[candidate("a", 1), candidate("b", 2)]], k=1)
    assert [item.chunk_id for item in fused] == ["a", "b"]
    assert fused[0].score == pytest.approx(0.5)
    assert fused[1].score == pytest.approx(1 / 3)


def test_rrf_rewards_consensus_and_sets_rank() -> None:
    fused = reciprocal_rank_fusion(
        [[candidate("a", 1), candidate("b", 2)], [candidate("b", 1, "sparse")]], k=1
    )
    assert [item.chunk_id for item in fused] == ["b", "a"]
    assert [item.rank for item in fused] == [1, 2]


def test_rrf_deduplicates_candidate_and_tracks_methods() -> None:
    fused = reciprocal_rank_fusion(
        [[candidate("a", 1)], [candidate("a", 3, "sparse")]], k=60
    )
    assert len(fused) == 1
    assert set(fused[0].metadata["original_scores"]) == {"dense", "sparse"}


def test_rrf_rejects_nonpositive_constant() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[candidate("a", 1)]], k=0)


def test_rrf_ties_are_broken_by_chunk_id() -> None:
    fused = reciprocal_rank_fusion([[candidate("z", 1), candidate("a", 1)]])
    assert [item.chunk_id for item in fused] == ["a", "z"]
