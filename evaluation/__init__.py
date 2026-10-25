"""NeuroFlow evaluation package for retrieval precision, Hit Rate, MRR, and LLM-as-Judge metrics."""

from evaluation.retrieval_eval import run_evaluation
from evaluation.judge import EvaluationJudge, EvaluationReport
from evaluation.metrics import (
    FaithfulnessMetric,
    FaithfulnessScore,
    AnswerRelevanceMetric,
    AnswerRelevanceScore,
    ContextPrecisionMetric,
    ContextPrecisionScore,
    ContextRecallMetric,
    ContextRecallScore,
)

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
