"""Repeatable runner for calibration evaluation computing Pearson correlation against human labels."""

from __future__ import annotations

import asyncio
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from evaluation.judge import DEFAULT_WEIGHTS, EvaluationJudge

logger = logging.getLogger("neuroflow.evaluation.calibration")


def calculate_pearson_correlation(x: list[float], y: list[float]) -> float | None:
    """Compute Pearson correlation coefficient between two numeric vectors.

    Returns None if either vector is constant (zero variance) or length < 2.
    """
    n = len(x)
    if n < 2 or len(y) != n:
        return None

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)

    if var_x <= 0.0 or var_y <= 0.0:
        # Zero variance: undefined correlation
        return None

    denom = math.sqrt(var_x * var_y)
    if denom == 0.0:
        return None

    return round(numerator / denom, 4)


def calculate_mae(x: list[float], y: list[float]) -> float | None:
    """Compute Mean Absolute Error."""
    if not x or len(x) != len(y):
        return None
    return round(sum(abs(xi - yi) for xi, yi in zip(x, y)) / len(x), 4)


class CalibrationRunner:
    """Executes evaluation over the 30-sample benchmark and produces calibration metrics."""

    def __init__(
        self,
        dataset_path: Path | str | None = None,
        output_path: Path | str | None = None,
        judge: EvaluationJudge | None = None,
    ):
        base_dir = Path(__file__).resolve().parent
        self.dataset_path = Path(dataset_path or (base_dir / "calibration" / "annotated_set.json"))
        self.output_path = Path(output_path or (base_dir / "calibration_results.json"))
        self.judge = judge

    async def _get_default_judge(self) -> EvaluationJudge | None:
        """Attempt to initialize EvaluationJudge with redis and neuroflow client."""
        try:
            import redis.asyncio as aioredis
            from config import settings
            from providers.client import NeuroFlowClient

            redis_client = aioredis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=3,
            )
            client = NeuroFlowClient(redis_client)
            return EvaluationJudge(llm_client=client)
        except Exception as exc:
            logger.warning("Could not instantiate default live EvaluationJudge: %s", exc)
            return None

    async def run(self) -> dict[str, Any]:
        """Run evaluation on the annotated dataset and compute Pearson correlation."""
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            dataset: list[dict[str, Any]] = json.load(f)

        judge = self.judge
        if judge is None:
            judge = await self._get_default_judge()

        if judge is None or judge.llm_client is None:
            results = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "failed",
                "sample_count": len(dataset),
                "metric_weights": DEFAULT_WEIGHTS.copy(),
                "error": "No LLM judge client or provider available to perform real calibration.",
                "provenance": {
                    "dataset_path": str(self.dataset_path),
                    "evaluator": "None",
                    "executed_live_evaluation": False,
                },
                "faithfulness_pearson_correlation": None,
                "overall_score_pearson_correlation": None,
                "is_calibrated": False,
            }
            with open(self.output_path, "w", encoding="utf-8") as out:
                json.dump(results, out, indent=2)
            return results

        human_faithfulness = [
            float(item.get("human_score", item.get("human_faithfulness", item.get("expected_faithfulness", 0.0))))
            for item in dataset
        ]
        human_overall = [
            float(item.get("ground_truth_overall", 0.0))
            for item in dataset
        ]

        pred_faith: list[float] = []
        pred_rel: list[float] = []
        pred_prec: list[float] = []
        pred_rec: list[float] = []
        pred_overall: list[float] = []
        predictions: list[dict[str, Any]] = []

        judge_model_used = "unknown"
        execution_error = None

        for item in dataset:
            run_id = str(uuid4())
            pipeline_id = str(uuid4())

            try:
                report = await judge.evaluate(
                    query=item["query"],
                    context=item["context"],
                    answer=item["answer"],
                    run_id=run_id,
                    pipeline_id=pipeline_id,
                )
                judge_model_used = report.judge_model
                f_val = report.faithfulness
                rel_val = report.answer_relevance
                p_val = report.context_precision
                r_val = report.context_recall
                o_val = report.overall_score

                # Check if the evaluation metric failed due to LLM provider error
                faith_reason = report.details.get("faithfulness", {}).get("reasoning", "")
                if "LLM call failed" in faith_reason:
                    execution_error = faith_reason
                    break

                pred_faith.append(f_val)
                pred_rel.append(rel_val)
                pred_prec.append(p_val)
                pred_rec.append(r_val)
                pred_overall.append(o_val)

                predictions.append({
                    "id": item["id"],
                    "query": item["query"],
                    "human_faithfulness": item.get("human_score", item.get("human_faithfulness")),
                    "predicted_faithfulness": f_val,
                    "predicted_overall": o_val,
                })
            except Exception as exc:
                logger.exception("Live evaluation failed on item %s: %s", item.get("id"), exc)
                execution_error = str(exc)
                break

        if execution_error or len(pred_faith) != len(dataset):
            results = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "failed",
                "sample_count": len(dataset),
                "evaluated_count": len(pred_faith),
                "metric_weights": DEFAULT_WEIGHTS.copy(),
                "error": f"LLM evaluator call failed during calibration: {execution_error}",
                "provenance": {
                    "dataset_path": str(self.dataset_path),
                    "evaluator_model": judge_model_used,
                    "executed_live_evaluation": True,
                },
                "faithfulness_pearson_correlation": None,
                "overall_score_pearson_correlation": None,
                "is_calibrated": False,
            }
            with open(self.output_path, "w", encoding="utf-8") as out:
                json.dump(results, out, indent=2)
            return results

        faith_corr = calculate_pearson_correlation(human_faithfulness, pred_faith)
        overall_corr = calculate_pearson_correlation(human_overall, pred_overall)
        faith_mae = calculate_mae(human_faithfulness, pred_faith)
        overall_mae = calculate_mae(human_overall, pred_overall)

        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "completed",
            "sample_count": len(dataset),
            "metric_weights": DEFAULT_WEIGHTS.copy(),
            "provenance": {
                "dataset_path": str(self.dataset_path),
                "evaluator_model": judge_model_used,
                "executed_live_evaluation": True,
            },
            "faithfulness_pearson_correlation": faith_corr,
            "overall_score_pearson_correlation": overall_corr,
            "mean_absolute_errors": {
                "faithfulness": faith_mae,
                "overall_score": overall_mae,
            },
            "calibration_threshold": 0.30,
            "is_calibrated": bool(faith_corr is not None and faith_corr >= 0.85),
            "predictions_sample": predictions[:5],
        }

        with open(self.output_path, "w", encoding="utf-8") as out:
            json.dump(results, out, indent=2)

        return results


async def main():
    runner = CalibrationRunner()
    results = await runner.run()
    print("Calibration runner finished:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
