"""Evaluation Judge orchestrating RAGAS-inspired metrics concurrently.

Evaluates RAG pipeline runs using an LLM-as-a-judge, computes a weighted overall score,
persists evaluation records to PostgreSQL, and automatically promotes high-quality runs
(overall_score > 0.8) to the training_pairs table for model fine-tuning.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

try:
    from db.pool import pipeline_connection, connection
except ImportError:
    pipeline_connection = None
    connection = None

from evaluation.metrics.answer_relevance import AnswerRelevanceMetric, AnswerRelevanceScore
from evaluation.metrics.context_precision import ContextPrecisionMetric, ContextPrecisionScore
from evaluation.metrics.context_recall import ContextRecallMetric, ContextRecallScore
from evaluation.metrics.faithfulness import FaithfulnessMetric, FaithfulnessScore
from providers.router import RoutingCriteria

logger = logging.getLogger("neuroflow.evaluation.judge")

DEFAULT_WEIGHTS = {
    "faithfulness": 0.35,
    "answer_relevance": 0.30,
    "context_precision": 0.20,
    "context_recall": 0.15,
}


@dataclass
class EvaluationReport:
    """Comprehensive evaluation results for a single pipeline run."""

    eval_id: str
    run_id: str
    pipeline_id: str
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float
    overall_score: float
    judge_model: str
    is_training_candidate: bool
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "eval_id": self.eval_id,
            "run_id": self.run_id,
            "pipeline_id": self.pipeline_id,
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "overall_score": self.overall_score,
            "judge_model": self.judge_model,
            "is_training_candidate": self.is_training_candidate,
            "details": self.details,
        }


class EvaluationJudge:
    """Orchestrates async evaluation across 4 metrics concurrently and persists outputs."""

    def __init__(
        self,
        llm_client: Any = None,
        weights: dict[str, float] | None = None,
    ):
        self.llm_client = llm_client
        self.weights = weights or DEFAULT_WEIGHTS.copy()

        # Initialize the 4 evaluation metric instances
        self.faithfulness_metric = FaithfulnessMetric(llm_client=llm_client)
        self.answer_relevance_metric = AnswerRelevanceMetric(llm_client=llm_client)
        self.context_precision_metric = ContextPrecisionMetric(llm_client=llm_client)
        self.context_recall_metric = ContextRecallMetric(llm_client=llm_client)

    async def _resolve_judge_model_name(self) -> str:
        """Resolve the judge model via ModelRouter if available, else default to gpt-4o."""
        if self.llm_client and hasattr(self.llm_client, "router"):
            try:
                criteria = RoutingCriteria(task_type="evaluation")
                return await self.llm_client.router.route(criteria)
            except Exception as exc:
                logger.debug("Failed to route judge model via router: %s", exc)
        return "gpt-4o"

    async def evaluate(
        self,
        *,
        query: str,
        context: str,
        answer: str,
        run_id: UUID | str,
        pipeline_id: UUID | str,
        prompt: str = "",
        system_prompt: str | None = None,
        training_threshold: float = 0.8,
    ) -> EvaluationReport:
        """Run all 4 metrics concurrently with asyncio.gather, compute score, and persist."""
        r_id = UUID(str(run_id))
        p_id = UUID(str(pipeline_id))
        eval_id = uuid4()

        # 1. Resolve judge model name
        judge_model = await self._resolve_judge_model_name()

        # 2. Run metrics concurrently using asyncio.gather
        faith_task = self.faithfulness_metric.score(answer=answer, context=context, query=query)
        rel_task = self.answer_relevance_metric.score(query=query, answer=answer, context=context)
        prec_task = self.context_precision_metric.score(query=query, context=context, answer=answer)
        rec_task = self.context_recall_metric.score(query=query, context=context, answer=answer)

        faith_res, rel_res, prec_res, rec_res = await asyncio.gather(
            faith_task, rel_task, prec_task, rec_task
        )

        # 3. Compute weighted overall score
        w = self.weights
        overall_raw = (
            w["faithfulness"] * faith_res.score
            + w["answer_relevance"] * rel_res.score
            + w["context_precision"] * prec_res.score
            + w["context_recall"] * rec_res.score
        )
        overall_score = round(max(0.0, min(1.0, overall_raw)), 4)
        is_training_candidate = overall_score > training_threshold

        report = EvaluationReport(
            eval_id=str(eval_id),
            run_id=str(r_id),
            pipeline_id=str(p_id),
            faithfulness=faith_res.score,
            answer_relevance=rel_res.score,
            context_precision=prec_res.score,
            context_recall=rec_res.score,
            overall_score=overall_score,
            judge_model=judge_model,
            is_training_candidate=is_training_candidate,
            details={
                "faithfulness": {
                    "claims": faith_res.claims,
                    "reasoning": faith_res.reasoning,
                },
                "answer_relevance": {
                    "verdict": rel_res.relevance_verdict,
                    "reasoning": rel_res.reasoning,
                },
                "context_precision": {
                    "chunk_evaluations": prec_res.chunk_evaluations,
                    "reasoning": prec_res.reasoning,
                },
                "context_recall": {
                    "key_facts": rec_res.key_facts,
                    "reasoning": rec_res.reasoning,
                },
            },
        )

        # 4. Persist to database if pool is configured
        await self._persist_evaluation(
            eval_id=eval_id,
            run_id=r_id,
            pipeline_id=p_id,
            faithfulness=faith_res.score,
            answer_relevance=rel_res.score,
            context_precision=prec_res.score,
            context_recall=rec_res.score,
            overall_score=overall_score,
            judge_model=judge_model,
            system_prompt=system_prompt or prompt,
            query=query,
            answer=answer,
            is_training_candidate=is_training_candidate,
        )

        return report

    async def evaluate_run_from_db(
        self,
        run_id: UUID | str,
        pipeline_id: UUID | str,
    ) -> EvaluationReport:
        """Fetch run details directly from pipeline_runs and evaluate."""
        r_id = UUID(str(run_id))
        p_id = UUID(str(pipeline_id))

        if pipeline_connection is None:
            raise RuntimeError("Database pool not available.")

        async with pipeline_connection(p_id) as conn:
            row = await conn.fetchrow(
                """
                SELECT pr.id, pr.query, pr.prompt, pr.generation, pr.retrieved_chunk_ids,
                       COALESCE(pv.config, p.config) AS config
                FROM pipeline_runs pr
                LEFT JOIN pipelines p ON p.id = pr.pipeline_id
                LEFT JOIN pipeline_versions pv ON pv.pipeline_id = pr.pipeline_id AND pv.version = pr.pipeline_version
                WHERE pr.id = $1 AND pr.pipeline_id = $2
                """,
                r_id,
                p_id,
            )
            if not row:
                raise ValueError(f"Pipeline run {run_id} not found for pipeline {pipeline_id}.")

            query = row["query"] or ""
            prompt = row["prompt"] or ""
            answer = row["generation"] or ""
            chunk_ids = row["retrieved_chunk_ids"] or []

            config_data = row["config"]
            if isinstance(config_data, str):
                import json
                try:
                    config_data = json.loads(config_data)
                except Exception:
                    config_data = {}
            elif not isinstance(config_data, dict):
                config_data = {}

            training_threshold = 0.8
            if "evaluation" in config_data and isinstance(config_data["evaluation"], dict):
                threshold = config_data["evaluation"].get("training_threshold")
                if threshold is not None:
                    try:
                        training_threshold = float(threshold)
                    except (ValueError, TypeError):
                        pass

            context = ""
            if chunk_ids:
                chunks = await conn.fetch(
                    """
                    SELECT content FROM chunks WHERE id = ANY($1::uuid[])
                    """,
                    chunk_ids,
                )
                context = "\n\n".join(c["content"] for c in chunks)
            elif prompt:
                # Context can be inferred from the assembled prompt
                context = prompt

        return await self.evaluate(
            query=query,
            context=context,
            answer=answer,
            run_id=r_id,
            pipeline_id=p_id,
            prompt=prompt,
            training_threshold=training_threshold,
        )

    async def _persist_evaluation(
        self,
        eval_id: UUID,
        run_id: UUID,
        pipeline_id: UUID,
        faithfulness: float,
        answer_relevance: float,
        context_precision: float,
        context_recall: float,
        overall_score: float,
        judge_model: str,
        system_prompt: str,
        query: str,
        answer: str,
        is_training_candidate: bool,
    ) -> None:
        """Persist evaluation record and conditionally insert high-quality training pairs."""
        if pipeline_connection is None:
            logger.debug("Database connection pool not available — skipping persistence.")
            return

        try:
            async with pipeline_connection(pipeline_id) as conn:
                # 1. Insert into evaluations table
                await conn.execute(
                    """
                    INSERT INTO evaluations (
                        id, run_id, faithfulness, answer_relevance,
                        context_precision, context_recall, overall_score,
                        judge_model, evaluated_at
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
                    ON CONFLICT (id) DO UPDATE SET
                        faithfulness = EXCLUDED.faithfulness,
                        answer_relevance = EXCLUDED.answer_relevance,
                        context_precision = EXCLUDED.context_precision,
                        context_recall = EXCLUDED.context_recall,
                        overall_score = EXCLUDED.overall_score,
                        judge_model = EXCLUDED.judge_model
                    """,
                    eval_id,
                    run_id,
                    faithfulness,
                    answer_relevance,
                    context_precision,
                    context_recall,
                    overall_score,
                    judge_model,
                )

                # 2. If overall_score > 0.8, promote to training_pairs
                if is_training_candidate:
                    await conn.execute(
                        """
                        INSERT INTO training_pairs (
                            id, run_id, system_prompt, user_message,
                            assistant_message, quality_score, created_at
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, NOW())
                        ON CONFLICT DO NOTHING
                        """,
                        uuid4(),
                        run_id,
                        system_prompt,
                        query,
                        answer,
                        overall_score,
                    )
                    logger.info(
                        "Promoted run %s to training_pairs (overall_score=%.4f)",
                        run_id,
                        overall_score,
                    )

        except Exception as exc:
            logger.exception("Failed to persist evaluation for run %s: %s", run_id, exc)
