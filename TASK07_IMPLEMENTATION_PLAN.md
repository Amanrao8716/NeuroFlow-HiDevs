# Task 07 — Automated Evaluation Framework with LLM-as-Judge and RAGAS Metrics

## Overview
Implement a production-grade automated evaluation framework for the NeuroFlow RAG pipeline.

## Implementation Phases

### Phase 1 — Evaluation Metrics
- [x] `evaluation/metrics/faithfulness.py`
- [x] `evaluation/metrics/answer_relevance.py`
- [x] `evaluation/metrics/context_precision.py`
- [x] `evaluation/metrics/context_recall.py`

### Phase 2 — EvaluationJudge
- [x] `evaluation/judge.py` — orchestrates metrics, computes weighted score, persists to DB

### Phase 3 — Worker Integration
- [x] `backend/worker/evaluate_task.py` — arq background task

### Phase 4 — Human Rating Endpoint
- [x] `PATCH /runs/{run_id}/rating` in `backend/api/evaluations.py`

### Phase 5 — Calibration
- [x] `evaluation/calibration/annotated_set.json` — 30 examples
- [x] `evaluation/calibration_runner.py`

### Phase 6 — Tests
- [x] `backend/tests/test_evaluation_metrics.py`
- [x] `backend/tests/test_evaluation_judge.py`
- [x] `backend/tests/test_evaluation_api.py`

## Metric Weights
| Metric              | Weight |
|---------------------|--------|
| faithfulness        | 0.35   |
| answer_relevance    | 0.30   |
| context_precision   | 0.20   |
| context_recall      | 0.15   |
