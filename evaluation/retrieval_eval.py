"""Retrieval evaluation runner measuring Hit Rate@10 and MRR@10 for RRF and Cross-Encoder."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any
from uuid import UUID

from db.pool import close_pool, open_pool, pipeline_connection
from pipelines.retrieval import (
    CrossEncoderReranker,
    HybridRetriever,
    QueryProcessor,
    reciprocal_rank_fusion,
)

logger = logging.getLogger("neuroflow.evaluation.retrieval_eval")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

RESULTS_PATH = Path(__file__).resolve().parent / "retrieval_results.json"
DATASET_PATH = Path(__file__).resolve().parent / "eval_dataset.json"

MIN_REQUIRED_CHUNKS = 50
HIT_RATE_THRESHOLD = 0.75
MRR_THRESHOLD = 0.55


def compute_metrics(
    eval_queries: list[dict[str, Any]],
    results_by_query: dict[str, list[str]],
    k: int = 10,
) -> tuple[float, float, list[dict[str, Any]]]:
    """Calculate Hit Rate@k and MRR@k against known relevant chunk IDs."""
    hits = 0
    reciprocal_ranks = []
    query_details = []

    for item in eval_queries:
        qid = item["id"]
        query_text = item["query"]
        relevant_chunk_ids = set(item.get("relevant_chunk_ids", []))

        returned_chunk_ids = results_by_query.get(qid, [])[:k]

        first_hit_rank: int | None = None
        for rank_idx, cid in enumerate(returned_chunk_ids, start=1):
            if cid in relevant_chunk_ids:
                first_hit_rank = rank_idx
                break

        if first_hit_rank is not None:
            hits += 1
            rr = 1.0 / first_hit_rank
        else:
            rr = 0.0

        reciprocal_ranks.append(rr)
        query_details.append(
            {
                "id": qid,
                "query": query_text,
                "first_hit_rank": first_hit_rank,
                "hit": first_hit_rank is not None,
                "reciprocal_rank": round(rr, 4),
                "returned_top_3": returned_chunk_ids[:3],
                "relevant_chunks": list(relevant_chunk_ids),
            }
        )

    num_queries = len(eval_queries) if eval_queries else 1
    hit_rate = round(hits / num_queries, 4)
    mrr = round(sum(reciprocal_ranks) / num_queries, 4)
    return hit_rate, mrr, query_details


async def run_evaluation(
    pipeline_id: UUID | None = None,
    eval_dataset: list[dict[str, Any]] | None = None,
    output_path: Path = RESULTS_PATH,
    enable_hyde: bool = False,
) -> dict[str, Any]:
    """Execute end-to-end retrieval benchmarking on hybrid retrieval and cross-encoder reranking."""
    await open_pool()
    try:
        # 1. Check database chunk count
        async with pipeline_connection(pipeline_id) if pipeline_id else await _get_direct_conn() as conn:
            chunk_count = await conn.fetchval("SELECT count(*) FROM chunks;")

        logger.info("Found %d total stored chunks in database.", chunk_count)

        if chunk_count < MIN_REQUIRED_CHUNKS:
            message = (
                f"Insufficient data in database (found {chunk_count} chunks, "
                f"required >= {MIN_REQUIRED_CHUNKS} chunks before full evaluation)."
            )
            logger.warning(message)
            instructions = (
                f"To load sufficient data, use the Task 4 ingestion pipeline:\n"
                f"  POST /ingest (PDF, DOCX, CSV, image) or POST /ingest/url\n"
                f"Ingest documents containing at least {MIN_REQUIRED_CHUNKS} total chunks."
            )

            result_payload = {
                "status": "insufficient_data",
                "chunk_count": chunk_count,
                "required_chunks": MIN_REQUIRED_CHUNKS,
                "thresholds_met": False,
                "evaluation_dataset_size": 0,
                "hit_rate_threshold": HIT_RATE_THRESHOLD,
                "mrr_threshold": MRR_THRESHOLD,
                "instructions": instructions,
                "limitations": message,
            }
            output_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
            return result_payload

        # 2. Load dataset
        if eval_dataset is None:
            if DATASET_PATH.is_file():
                eval_dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
            else:
                raise FileNotFoundError(
                    f"Evaluation dataset not found at {DATASET_PATH}. Please provide 20 questions."
                )

        if len(eval_dataset) < 20:
            logger.warning("Dataset contains %d questions (target is 20).", len(eval_dataset))

        processor = QueryProcessor()
        retriever = HybridRetriever(query_processor=processor)
        reranker = CrossEncoderReranker()

        rrf_results_by_query: dict[str, list[str]] = {}
        rerank_results_by_query: dict[str, list[str]] = {}
        hyde_results_by_query: dict[str, list[str]] = {}

        # 3. Run evaluation across all queries
        for item in eval_dataset:
            qid = item["id"]
            query_text = item["query"]
            filters = item.get("filters")

            # A. Parallel Multi-Method Search & RRF
            dense_res, sparse_res, meta_res = await retriever.retrieve_parallel(
                query=query_text,
                k=20,
                pipeline_id=pipeline_id,
                filters=filters,
            )

            result_lists = [dense_res, sparse_res]
            if meta_res:
                result_lists.append(meta_res)

            fused_candidates = reciprocal_rank_fusion(result_lists, k=60)
            rrf_results_by_query[qid] = [c.chunk_id for c in fused_candidates]

            # B. Cross-Encoder Reranking
            if fused_candidates:
                reranked_candidates = await reranker.rerank(
                    query_text, fused_candidates, top_n=40
                )
                rerank_results_by_query[qid] = [c.chunk_id for c in reranked_candidates]
            else:
                rerank_results_by_query[qid] = []

        # 4. Compute Metrics
        rrf_hit_rate, rrf_mrr, rrf_details = compute_metrics(
            eval_dataset, rrf_results_by_query, k=10
        )
        rerank_hit_rate, rerank_mrr, rerank_details = compute_metrics(
            eval_dataset, rerank_results_by_query, k=10
        )

        thresholds_met = (rerank_hit_rate > HIT_RATE_THRESHOLD) and (
            rerank_mrr > MRR_THRESHOLD
        )

        result_payload = {
            "status": "completed",
            "chunk_count": chunk_count,
            "evaluation_dataset_size": len(eval_dataset),
            "hit_rate_threshold": HIT_RATE_THRESHOLD,
            "mrr_threshold": MRR_THRESHOLD,
            "thresholds_met": thresholds_met,
            "baseline_rrf": {
                "hit_rate_at_10": rrf_hit_rate,
                "mrr_at_10": rrf_mrr,
            },
            "cross_encoder_reranked": {
                "hit_rate_at_10": rerank_hit_rate,
                "mrr_at_10": rerank_mrr,
            },
            "models": {
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "embedding_dimensions": 384,
                "cross_encoder_model": reranker.model_name,
                "rrf_k": 60,
            },
            "queries": [
                {
                    "id": item["id"],
                    "query": item["query"],
                    "rrf_rank": rrf_details[i]["first_hit_rank"],
                    "reranked_rank": rerank_details[i]["first_hit_rank"],
                    "relevant_chunk_ids": item.get("relevant_chunk_ids", []),
                }
                for i, item in enumerate(eval_dataset)
            ],
        }

        output_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
        logger.info(
            "Evaluation complete: RRF (HitRate: %.4f, MRR: %.4f) -> Reranked (HitRate: %.4f, MRR: %.4f) | Pass: %s",
            rrf_hit_rate,
            rrf_mrr,
            rerank_hit_rate,
            rerank_mrr,
            thresholds_met,
        )
        return result_payload

    finally:
        await close_pool()


async def _get_direct_conn():
    from db.pool import get_pool

    pool = get_pool()
    return pool.acquire()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NeuroFlow Retrieval Evaluation")
    parser.add_argument(
        "--pipeline-id",
        type=str,
        default=None,
        help="Optional tenant pipeline UUID",
    )
    args = parser.parse_args()

    pid = UUID(args.pipeline_id) if args.pipeline_id else None
    asyncio.run(run_evaluation(pipeline_id=pid))
