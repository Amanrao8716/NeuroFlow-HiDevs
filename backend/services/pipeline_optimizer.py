"""Rule-based pipeline optimizer offering transparent advisory recommendations based on evaluation history."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from backend.models.pipeline import PipelineConfig, SuggestionItem

logger = logging.getLogger("neuroflow.services.optimizer")

DEFAULT_THRESHOLDS = {
    "context_precision": 0.60,
    "context_recall": 0.60,
    "answer_relevance": 0.70,
    "faithfulness": 0.70,
}
DEFAULT_MIN_SAMPLE_COUNT = 5


class PipelineOptimizer:
    """Analyzes recent pipeline evaluation records and generates deterministic tuning suggestions."""

    def __init__(
        self,
        thresholds: dict[str, float] | None = None,
        min_samples: int = DEFAULT_MIN_SAMPLE_COUNT,
        precision_threshold: float | None = None,
        recall_threshold: float | None = None,
        relevance_threshold: float | None = None,
        faithfulness_threshold: float | None = None,
    ):
        self.thresholds = thresholds.copy() if thresholds else DEFAULT_THRESHOLDS.copy()
        if precision_threshold is not None:
            self.thresholds["context_precision"] = precision_threshold
        if recall_threshold is not None:
            self.thresholds["context_recall"] = recall_threshold
        if relevance_threshold is not None:
            self.thresholds["answer_relevance"] = relevance_threshold
        if faithfulness_threshold is not None:
            self.thresholds["faithfulness"] = faithfulness_threshold
        self.min_samples = min_samples

    def analyze(
        self,
        evaluations_or_config: Any,
        config_or_evaluations: Any,
    ) -> list[SuggestionItem]:
        """Analyze evaluation records against advisory rules.

        Returns empty list if sample count is below min_samples or no metrics breach thresholds.
        """
        if isinstance(evaluations_or_config, list):
            evaluations = evaluations_or_config
            current_config = config_or_evaluations
        else:
            current_config = evaluations_or_config
            evaluations = config_or_evaluations

        if len(evaluations) < self.min_samples:
            logger.info(
                "Insufficient evaluation records (%d < %d) for optimizer analysis",
                len(evaluations),
                self.min_samples,
            )
            return []

        # Compute averages for available metrics
        metrics = ["context_precision", "context_recall", "answer_relevance", "faithfulness"]
        averages: dict[str, float | None] = {}

        for m in metrics:
            vals = [float(e[m]) for e in evaluations if e.get(m) is not None]
            averages[m] = sum(vals) / len(vals) if vals else None

        suggestions: list[SuggestionItem] = []

        # Rule 1: Low Context Precision -> reduce top_k_after_rerank
        prec_val = averages.get("context_precision")
        prec_th = self.thresholds.get("context_precision", 0.60)
        if prec_val is not None and prec_val < prec_th:
            new_top_k = max(1, current_config.retrieval.top_k_after_rerank - 2)
            suggestions.append(
                SuggestionItem(
                    rule_id="reduce_top_k_after_rerank",
                    title="Optimize Context Precision by Reducing Top-K",
                    reason=(
                        f"Average context precision ({prec_val:.2f}) is below threshold ({prec_th:.2f}). "
                        f"Lower-ranked retrieved passages may be diluting answer focus."
                    ),
                    supporting_metric="context_precision",
                    observed_value=round(prec_val, 4),
                    threshold=prec_th,
                    proposed_change={"retrieval": {"top_k_after_rerank": new_top_k}},
                )
            )

        # Rule 2: Low Context Recall -> increase dense_k
        rec_val = averages.get("context_recall")
        rec_th = self.thresholds.get("context_recall", 0.60)
        if rec_val is not None and rec_val < rec_th:
            new_dense = min(100, current_config.retrieval.dense_k + 5)
            suggestions.append(
                SuggestionItem(
                    rule_id="increase_dense_k",
                    title="Improve Context Recall by Expanding Retrieval Depth",
                    reason=(
                        f"Average context recall ({rec_val:.2f}) is below threshold ({rec_th:.2f}). "
                        f"Relevant facts are missing from retrieved passages."
                    ),
                    supporting_metric="context_recall",
                    observed_value=round(rec_val, 4),
                    threshold=rec_th,
                    proposed_change={"retrieval": {"dense_k": new_dense}},
                )
            )

        # Rule 3: Low Answer Relevance -> enable query expansion and review prompt
        rel_val = averages.get("answer_relevance")
        rel_th = self.thresholds.get("answer_relevance", 0.70)
        if rel_val is not None and rel_val < rel_th:
            suggestions.append(
                SuggestionItem(
                    rule_id="enable_query_expansion_or_adjust_prompt",
                    title="Enhance Answer Relevance via Query Expansion and Concise Prompting",
                    reason=(
                        f"Average answer relevance ({rel_val:.2f}) is below threshold ({rel_th:.2f}). "
                        f"Generations are diverging from the user's specific query intent."
                    ),
                    supporting_metric="answer_relevance",
                    observed_value=round(rel_val, 4),
                    threshold=rel_th,
                    proposed_change={
                        "retrieval": {"query_expansion": True},
                        "generation": {"system_prompt_variant": "concise"},
                    },
                )
            )

        # Rule 4: Low Faithfulness -> lower temperature and tighten grounding
        faith_val = averages.get("faithfulness")
        faith_th = self.thresholds.get("faithfulness", 0.70)
        if faith_val is not None and faith_val < faith_th:
            new_temp = max(0.0, round(current_config.generation.temperature - 0.3, 2))
            suggestions.append(
                SuggestionItem(
                    rule_id="reduce_temperature_and_tighten_grounding",
                    title="Suppress Hallucinations by Lowering Sampling Temperature",
                    reason=(
                        f"Average faithfulness ({faith_val:.2f}) is below threshold ({faith_th:.2f}). "
                        f"Model is generating unsubstantiated claims not verified in context."
                    ),
                    supporting_metric="faithfulness",
                    observed_value=round(faith_val, 4),
                    threshold=faith_th,
                    proposed_change={"generation": {"temperature": new_temp}},
                )
            )

        return suggestions
