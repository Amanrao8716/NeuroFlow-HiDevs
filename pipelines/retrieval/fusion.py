"""Reciprocal Rank Fusion (RRF) for combining multi-method retrieval candidate lists."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipelines.retrieval.retriever import RetrievalResult

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievalResult]],
    k: int = DEFAULT_RRF_K,
) -> list[RetrievalResult]:
    """Fuse multiple ranked candidate lists using Reciprocal Rank Fusion (RRF).

    Formula:
        score(d) = sum(1 / (k + rank_i(d))) for each list i where document d appears.

    Args:
        result_lists: List of ranked candidate lists.
        k: Smoothing constant (default: 60).

    Returns:
        New list of deduplicated RetrievalResult objects sorted by descending fused score.
    """
    if k <= 0:
        raise ValueError(f"RRF constant k must be positive, got {k}")

    if not result_lists:
        return []

    fused_scores: dict[str, float] = {}
    best_candidates: dict[str, RetrievalResult] = {}
    original_scores: dict[str, dict[str, float]] = {}
    method_ranks: dict[str, dict[str, int]] = {}

    for result_list in result_lists:
        if not result_list:
            continue

        for zero_idx, item in enumerate(result_list):
            rank = item.rank if item.rank is not None else (zero_idx + 1)
            cid = item.chunk_id
            method = item.retrieval_method or "unknown"
            increment = 1.0 / (k + rank)

            fused_scores[cid] = fused_scores.get(cid, 0.0) + increment

            if cid not in original_scores:
                original_scores[cid] = {}
                method_ranks[cid] = {}
            original_scores[cid][method] = item.score
            method_ranks[cid][method] = rank

            if cid not in best_candidates:
                best_candidates[cid] = item

    if not fused_scores:
        return []

    # Sort descending by fused score; break ties deterministically by chunk_id
    sorted_items = sorted(
        fused_scores.items(),
        key=lambda pair: (-pair[1], pair[0]),
    )

    fused_results: list[RetrievalResult] = []
    for rank_idx, (cid, score) in enumerate(sorted_items, start=1):
        source_item = best_candidates[cid]
        new_meta = dict(source_item.metadata)
        new_meta["rrf_score"] = score
        new_meta["original_scores"] = original_scores[cid]
        new_meta["method_ranks"] = method_ranks[cid]

        fused_item = source_item.clone(
            score=score,
            rank=rank_idx,
            retrieval_method="rrf_fusion",
            metadata=new_meta,
        )
        fused_results.append(fused_item)

    return fused_results
