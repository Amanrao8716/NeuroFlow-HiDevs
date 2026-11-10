"""NeuroFlow evaluation package for retrieval precision, Hit Rate, MRR, and LLM-as-Judge metrics."""

from evaluation.judge import EvaluationJudge, EvaluationReport
from evaluation.metrics import (
    AnswerRelevanceMetric,
    AnswerRelevanceScore,
    ContextPrecisionMetric,
    ContextPrecisionScore,
    ContextRecallMetric,
    ContextRecallScore,
    FaithfulnessMetric,
    FaithfulnessScore,
)
from evaluation.retrieval_eval import run_evaluation

__all__ = [
    "run_evaluation",
    "EvaluationJudge",
    "EvaluationReport",
    "FaithfulnessMetric",
    "FaithfulnessScore",
    "AnswerRelevanceMetric",
    "AnswerRelevanceScore",
    "ContextPrecisionMetric",
    "ContextPrecisionScore",
    "ContextRecallMetric",
    "ContextRecallScore",
]
