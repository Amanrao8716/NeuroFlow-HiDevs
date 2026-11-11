
"""Run a retrieval hyperparameter grid against a labeled, tenant-scoped dataset.

This script only recommends a config in its JSON report. It never mutates a
stored pipeline. Dataset and database access must point to an approved eval set.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from itertools import product
from pathlib import Path
from typing import Any
from uuid import UUID

from db.pool import close_pool, open_pool, pipeline_connection

from evaluation.retrieval_eval import compute_metrics
from pipelines.retrieval import CrossEncoderReranker, HybridRetriever, QueryProcessor
from pipelines.retrieval.fusion import reciprocal_rank_fusion

logger = logging.getLogger("neuroflow.evaluation.tune_retrieval")
DATASET_PATH = Path(__file__).resolve().parent / "eval_dataset.json"
RESULTS_PATH = Path(__file__).resolve().parent / "retrieval_tuning_results.json"
GRID = {
    "dense_k": (10, 20, 30),
    "sparse_k": (10, 15),
    "top_k_after_rerank": (5, 8, 10),
    "rrf_k": (30, 60, 120),
}


def parameter_grid() -> list[dict[str, int]]:
    """Return every requested retrieval setting combination."""
    keys = tuple(GRID)
    return [dict(zip(keys, values)) for values in product(*(GRID[key] for key in keys))]


def select_best(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the best completed run by reranked MRR, then hit rate and latency."""
    completed = [
        row
        for row in results
        if row.get("status") == "completed"
        and isinstance(row.get("reranked_mrr_at_10"), (int, float))
        and isinstance(row.get("reranked_hit_rate_at_10"), (int, float))
        and isinstance(row.get("mean_query_latency_ms"), (int, float))
    ]
    if not completed:
        raise ValueError("No completed evaluation results are available to select from.")
    return max(
        completed,
        key=lambda row: (
            row["reranked_mrr_at_10"],
            row["reranked_hit_rate_at_10"],
            -row["mean_query_latency_ms"],
        ),
    )


async def run_tuning(
    pipeline_id: UUID,
    dataset_path: Path = DATASET_PATH,
    output_path: Path = RESULTS_PATH,
) -> dict[str, Any]:
    dataset_exists = await asyncio.to_thread(dataset_path.is_file)
    if not dataset_exists:
        raise FileNotFoundError(
            f"Labeled retrieval dataset is missing: {dataset_path}. "
            "Create a reviewed dataset with query IDs and relevant chunk IDs before tuning."
        )

    dataset_text = await asyncio.to_thread(
        dataset_path.read_text,
        encoding="utf-8",
    )
    eval_queries = json.loads(dataset_text)

    if not isinstance(eval_queries, list) or len(eval_queries) < 20:
        raise ValueError("The labeled retrieval dataset must contain at least 20 queries.")

    if any(
        not isinstance(item, dict)
        or not item.get("id")
        or not item.get("query")
        or not item.get("relevant_chunk_ids")
        for item in eval_queries
    ):
        raise ValueError("Every evaluation query needs id, query, and relevant_chunk_ids.")

    import mlflow

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("Set MLFLOW_TRACKING_URI to the evaluation MLflow server.")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("neuroflow-retrieval-tuning")

    await open_pool()
    try:
        async with pipeline_connection(pipeline_id) as conn:
            chunk_count = await conn.fetchval("SELECT count(*) FROM chunks")

        if chunk_count < 50:
            raise ValueError(f"At least 50 tenant chunks are required; found {chunk_count}.")

        retriever = HybridRetriever(query_processor=QueryProcessor())
        reranker = CrossEncoderReranker()
        results: list[dict[str, Any]] = []

        for params in parameter_grid():
            started = time.perf_counter()
            dense_by_query: dict[str, list[str]] = {}
            sparse_by_query: dict[str, list[str]] = {}
            reranked_by_query: dict[str, list[str]] = {}

            with mlflow.start_run(
                run_name=(
                    f"d{params['dense_k']}-s{params['sparse_k']}"
                    f"-t{params['top_k_after_rerank']}-rrf{params['rrf_k']}"
                )
            ) as run:
                mlflow.log_params(params)
                mlflow.log_param("dense_rrf_weight", 0.6)
                mlflow.log_param("sparse_rrf_weight", 0.4)
                mlflow.log_param("pipeline_id", str(pipeline_id))
                mlflow.log_param("evaluation_query_count", len(eval_queries))

                for item in eval_queries:
                    dense, sparse, metadata = await retriever.retrieve_parallel(
                        item["query"],
                        k=max(params["dense_k"], params["sparse_k"]),
                        dense_k=params["dense_k"],
                        sparse_k=params["sparse_k"],
                        pipeline_id=pipeline_id,
                    )
                    query_id = item["id"]
                    dense_by_query[query_id] = [hit.chunk_id for hit in dense]
                    sparse_by_query[query_id] = [hit.chunk_id for hit in sparse]

                    fused = reciprocal_rank_fusion(
                        [dense, sparse, metadata],
                        k=params["rrf_k"],
                        weights={"dense": 0.6, "sparse": 0.4, "metadata": 0.4},
                    )

                    reranked = (
                        await reranker.rerank(item["query"], fused, top_n=40)
                        if fused
                        else []
                    )
                    reranked_by_query[query_id] = [
                        hit.chunk_id for hit in reranked[: params["top_k_after_rerank"]]
                    ]

                dense_hit, dense_mrr, _ = compute_metrics(
                    eval_queries, dense_by_query, k=10
                )
                sparse_hit, sparse_mrr, _ = compute_metrics(
                    eval_queries, sparse_by_query, k=10
                )
                reranked_hit, reranked_mrr, _ = compute_metrics(
                    eval_queries, reranked_by_query, k=10
                )

                mean_latency_ms = (time.perf_counter() - started) * 1000 / len(eval_queries)

                run_result = {
                    "status": "completed",
                    **params,
                    "dense_hit_rate_at_10": dense_hit,
                    "dense_mrr_at_10": dense_mrr,
                    "sparse_hit_rate_at_10": sparse_hit,
                    "sparse_mrr_at_10": sparse_mrr,
                    "reranked_hit_rate_at_10": reranked_hit,
                    "reranked_mrr_at_10": reranked_mrr,
                    "mean_query_latency_ms": round(mean_latency_ms, 3),
                    "mlflow_run_id": run.info.run_id,
                }

                mlflow.log_metrics(
                    {
                        key: run_result[key]
                        for key in (
                            "dense_hit_rate_at_10",
                            "dense_mrr_at_10",
                            "sparse_hit_rate_at_10",
                            "sparse_mrr_at_10",
                            "reranked_hit_rate_at_10",
                            "reranked_mrr_at_10",
                            "mean_query_latency_ms",
                        )
                    }
                )

                results.append(run_result)
                logger.info("Completed MLflow run %s", run.info.run_id)

        best = select_best(results)
        report = {
            "status": "completed",
            "pipeline_id": str(pipeline_id),
            "dataset_path": str(dataset_path),
            "dataset_size": len(eval_queries),
            "chunk_count": chunk_count,
            "selection_metric": "reranked_mrr_at_10, then hit_rate, then latency",
            "best_configuration": best,
            "runs": results,
            "note": "Recommendation only; the persisted pipeline was not modified.",
        }

        report_text = json.dumps(report, indent=2)
        await asyncio.to_thread(
            output_path.write_text,
            report_text,
            encoding="utf-8",
        )
        return report

    finally:
        await close_pool()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-id", required=True, type=UUID)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    arguments = parser.parse_args()
    asyncio.run(run_tuning(arguments.pipeline_id, arguments.dataset, arguments.output))
