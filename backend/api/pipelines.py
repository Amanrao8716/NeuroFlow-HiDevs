"""FastAPI endpoints for Pipeline CRUD, Versioning, History, Analytics, and Optimization."""

from __future__ import annotations

import hmac
import json
import logging
import math
from datetime import date, datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, Field

from config import settings
from backend.models.pipeline import (
    AnalyticsResponse,
    DailyQueryCount,
    PaginatedRunsResponse,
    PipelineConfig,
    PipelineCreateRequest,
    PipelineResponse,
    PipelineRunItem,
    PipelineUpdateRequest,
    SuggestionItem,
    SuggestionsResponse,
)
from backend.services.pipeline_optimizer import PipelineOptimizer
from observability.metrics import QUALITY_ANOMALIES

try:
    from db.pool import connection, pipeline_connection
except ImportError:
    connection = None
    pipeline_connection = None

from backend.security.auth import get_current_user, require_scope
from backend.security.validators import sanitize_text

router = APIRouter(prefix="/pipelines", tags=["pipelines"], dependencies=[Depends(get_current_user)])
logger = logging.getLogger("neuroflow.api.pipelines")


class QualityAnomalyRequest(BaseModel):
    """Worker-reported quality anomaly evidence."""

    run_id: UUID
    latest_score: float = Field(ge=0, le=1)
    baseline_mean: float = Field(ge=0, le=1)
    baseline_stddev: float = Field(ge=0, le=1)


class QualityAnomalyResponse(BaseModel):
    """Persisted quality anomaly and generated pipeline suggestions."""

    id: UUID
    pipeline_id: UUID
    run_id: UUID
    rule_id: str
    latest_score: float
    baseline_mean: float
    baseline_stddev: float
    suggestions: list[SuggestionItem]
    created_at: datetime
    recorded: bool
    cooldown_suppressed: bool


def _decode_suggestions(value: Any) -> list[SuggestionItem]:
    if isinstance(value, str):
        value = json.loads(value)
    return [SuggestionItem(**item) for item in value]


def calculate_percentile(data: list[float], p: float) -> float | None:
    """Calculate percentile using linear interpolation between closest ranks."""
    if not data:
        return None
    s = sorted(data)
    n = len(s)
    if n == 1:
        return round(float(s[0]), 2)
    idx = (p / 100.0) * (n - 1)
    low = int(math.floor(idx))
    high = int(math.ceil(idx))
    if low == high:
        return round(float(s[low]), 2)
    weight = idx - low
    return round(float(s[low] * (1.0 - weight) + s[high] * weight), 2)


MODEL_PRICING: dict[str, tuple[float, float]] = {
    # model_name: (input_price_per_1m, output_price_per_1m)
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-3-5-haiku-latest": (0.80, 4.00),
}


def calculate_token_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate query USD cost based on configured provider pricing per 1M tokens."""
    input_rate, output_rate = MODEL_PRICING.get(model, (0.15, 0.60))
    cost = (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000.0
    return round(cost, 6)


def deep_merge(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge source dictionary into target dictionary."""
    result = target.copy()
    for key, value in source.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@router.post("", response_model=PipelineResponse, status_code=status.HTTP_201_CREATED)
async def create_pipeline(payload: PipelineCreateRequest, _admin=Depends(require_scope("admin"))):
    """Validate and create a new RAG pipeline, establishing version 1."""
    payload.name = sanitize_text(payload.name).strip()
    if payload.description is not None:
        payload.description = sanitize_text(payload.description)
    if not payload.name or len(payload.name) > 100:
        raise HTTPException(status_code=422, detail="Pipeline name must be 1 to 100 characters.")
    if connection is None:
        raise HTTPException(status_code=503, detail="Database connection pool unavailable.")

    config_obj = payload.to_config()
    config_dict = config_obj.model_dump()
    pipeline_id = uuid4()
    now = datetime.now(timezone.utc)

    async with connection() as conn:
        # Check name uniqueness
        existing = await conn.fetchval(
            "SELECT id FROM pipelines WHERE name = $1",
            payload.name,
        )
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Pipeline with name '{payload.name}' already exists.",
            )

        async with conn.transaction():
            # 1. Insert into pipelines table
            await conn.execute(
                """
                INSERT INTO pipelines (id, name, description, config, version, status, created_at, updated_at)
                VALUES ($1, $2, $3, $4::jsonb, 1, 'active', $5, $5)
                """,
                pipeline_id,
                payload.name,
                payload.description,
                json.dumps(config_dict),
                now,
            )

            # 2. Insert immutable version 1 record
            await conn.execute(
                """
                INSERT INTO pipeline_versions (id, pipeline_id, version, config, description, created_at)
                VALUES ($1, $2, 1, $3::jsonb, $4, $5)
                """,
                uuid4(),
                pipeline_id,
                json.dumps(config_dict),
                payload.description,
                now,
            )

    return PipelineResponse(
        id=str(pipeline_id),
        name=payload.name,
        description=payload.description,
        version=1,
        status="active",
        config=config_obj,
        created_at=now,
        updated_at=now,
    )


@router.get("", response_model=list[PipelineResponse])
async def list_pipelines(include_archived: bool = Query(default=False), _scope=Depends(require_scope("query"))):
    """List non-archived pipelines with their current version and configuration."""
    if connection is None:
        raise HTTPException(status_code=503, detail="Database connection pool unavailable.")

    async with connection() as conn:
        status_filter = "" if include_archived else "WHERE p.status = 'active'"
        rows = await conn.fetch(
            f"""
            SELECT p.id, p.name, p.description, p.config, p.version, p.status, p.created_at, p.updated_at,
                   COUNT(r.id) as run_count,
                   MAX(r.created_at) as last_run_at
            FROM pipelines p
            LEFT JOIN pipeline_runs r ON r.pipeline_id = p.id
            {status_filter}
            GROUP BY p.id
            ORDER BY p.created_at DESC
            """
        )

        results: list[PipelineResponse] = []
        for r in rows:
            raw_cfg = r["config"]
            cfg_data = json.loads(raw_cfg) if isinstance(raw_cfg, str) else raw_cfg
            results.append(
                PipelineResponse(
                    id=str(r["id"]),
                    name=r["name"],
                    description=r["description"],
                    version=r["version"],
                    status=r["status"],
                    config=PipelineConfig(**cfg_data),
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                    latest_metrics={
                        "run_count": r["run_count"],
                        "last_run_at": r["last_run_at"].isoformat() if r["last_run_at"] else None,
                    },
                )
            )

    return results


@router.get("/{pipeline_id}", response_model=PipelineResponse)
async def get_pipeline(pipeline_id: UUID, _scope=Depends(require_scope("query"))):
    """Retrieve full pipeline configuration, current version, and aggregate evaluation scores."""
    if connection is None:
        raise HTTPException(status_code=503, detail="Database connection pool unavailable.")

    async with connection() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, name, description, config, version, status, created_at, updated_at
            FROM pipelines
            WHERE id = $1
            """,
            pipeline_id,
        )
        if not row:
            raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found.")

        if row["status"] == "archived":
            raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} has been archived.")

        # Compute aggregate evaluation metrics across all runs of this pipeline
        agg = await conn.fetchrow(
            """
            SELECT AVG(e.faithfulness) as avg_faith,
                   AVG(e.answer_relevance) as avg_rel,
                   AVG(e.context_precision) as avg_prec,
                   AVG(e.context_recall) as avg_rec,
                   AVG(e.overall_score) as avg_overall
            FROM evaluations e
            JOIN pipeline_runs r ON r.id = e.run_id
            WHERE r.pipeline_id = $1
            """,
            pipeline_id,
        )

        raw_cfg = row["config"]
        cfg_data = json.loads(raw_cfg) if isinstance(raw_cfg, str) else raw_cfg

        agg_scores = {
            "faithfulness": round(agg["avg_faith"], 4) if agg and agg["avg_faith"] is not None else None,
            "answer_relevance": round(agg["avg_rel"], 4) if agg and agg["avg_rel"] is not None else None,
            "context_precision": round(agg["avg_prec"], 4) if agg and agg["avg_prec"] is not None else None,
            "context_recall": round(agg["avg_rec"], 4) if agg and agg["avg_rec"] is not None else None,
            "overall_score": round(agg["avg_overall"], 4) if agg and agg["avg_overall"] is not None else None,
        }

        return PipelineResponse(
            id=str(row["id"]),
            name=row["name"],
            description=row["description"],
            version=row["version"],
            status=row["status"],
            config=PipelineConfig(**cfg_data),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            aggregate_scores=agg_scores,
        )


@router.patch("/{pipeline_id}", response_model=PipelineResponse)
async def update_pipeline(pipeline_id: UUID, payload: PipelineUpdateRequest, _admin=Depends(require_scope("admin"))):
    """Update pipeline configuration and create a new immutable version.

    Nested configuration objects (ingestion, retrieval, generation, evaluation)
    are merged into the existing configuration and strictly re-validated.
    """
    if connection is None:
        raise HTTPException(status_code=503, detail="Database connection pool unavailable.")

    async with connection() as conn:
        row = await conn.fetchrow(
            "SELECT id, name, description, config, version, status FROM pipelines WHERE id = $1 FOR UPDATE",
            pipeline_id,
        )
        if not row:
            raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found.")

        if row["status"] == "archived":
            raise HTTPException(status_code=400, detail=f"Cannot update archived pipeline {pipeline_id}.")

        current_cfg = json.loads(row["config"]) if isinstance(row["config"], str) else dict(row["config"])
        current_name = row["name"]
        current_desc = row["description"]
        current_version = row["version"]

        # Merge top-level fields
        new_name = payload.name if payload.name is not None else current_name
        new_desc = payload.description if payload.description is not None else current_desc
        new_name = sanitize_text(new_name).strip()
        new_desc = sanitize_text(new_desc) if new_desc is not None else None
        if not new_name or len(new_name) > 100:
            raise HTTPException(status_code=422, detail="Pipeline name must be 1 to 100 characters.")

        # Check name conflict if renaming
        if new_name != current_name:
            conflict = await conn.fetchval(
                "SELECT id FROM pipelines WHERE name = $1 AND id != $2",
                new_name,
                pipeline_id,
            )
            if conflict:
                raise HTTPException(
                    status_code=409,
                    detail=f"Pipeline with name '{new_name}' already exists.",
                )

        # Merge nested sections
        merged_cfg = current_cfg.copy()
        merged_cfg["name"] = new_name
        merged_cfg["description"] = new_desc

        for section in ("ingestion", "retrieval", "generation", "evaluation"):
            patch_data = getattr(payload, section)
            if patch_data is not None:
                existing_section = merged_cfg.get(section, {})
                merged_cfg[section] = deep_merge(existing_section, patch_data)

        # Strictly validate merged configuration
        try:
            validated_config = PipelineConfig(**merged_cfg)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid configuration update: {str(exc)}",
            ) from exc

        new_version = current_version + 1
        now = datetime.now(timezone.utc)
        validated_dict = validated_config.model_dump()

        async with conn.transaction():
            # 1. Update pipeline record
            await conn.execute(
                """
                UPDATE pipelines
                SET name = $1, description = $2, config = $3::jsonb, version = $4, updated_at = $5
                WHERE id = $6
                """,
                new_name,
                new_desc,
                json.dumps(validated_dict),
                new_version,
                now,
                pipeline_id,
            )

            # 2. Insert new immutable version row
            await conn.execute(
                """
                INSERT INTO pipeline_versions (id, pipeline_id, version, config, description, created_at)
                VALUES ($1, $2, $3, $4::jsonb, $5, $6)
                """,
                uuid4(),
                pipeline_id,
                new_version,
                json.dumps(validated_dict),
                new_desc,
                now,
            )

        return PipelineResponse(
            id=str(pipeline_id),
            name=new_name,
            description=new_desc,
            version=new_version,
            status=row["status"],
            config=validated_config,
            created_at=row["created_at"] if "created_at" in row else now,
            updated_at=now,
        )


@router.delete("/{pipeline_id}")
async def delete_pipeline(pipeline_id: UUID, _admin=Depends(require_scope("admin"))):
    """Soft-delete pipeline by archiving status. Preserves all runs and historical versions."""
    if connection is None:
        raise HTTPException(status_code=503, detail="Database connection pool unavailable.")

    async with connection() as conn:
        row = await conn.fetchrow(
            "SELECT id, status FROM pipelines WHERE id = $1",
            pipeline_id,
        )
        if not row:
            raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found.")

        await conn.execute(
            """
            UPDATE pipelines
            SET status = 'archived', updated_at = NOW()
            WHERE id = $1
            """,
            pipeline_id,
        )

    return {"id": str(pipeline_id), "status": "archived"}


@router.get("/{pipeline_id}/runs", response_model=PaginatedRunsResponse)
async def get_pipeline_runs(
    pipeline_id: UUID,
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    _scope=Depends(require_scope("query")),
):
    """Paginated list of execution runs for this pipeline with metrics and evaluation scores."""
    if connection is None:
        raise HTTPException(status_code=503, detail="Database connection pool unavailable.")

    offset = (page - 1) * page_size

    async with connection() as conn:
        # Check pipeline existence
        p_exists = await conn.fetchval(
            "SELECT id FROM pipelines WHERE id = $1",
            pipeline_id,
        )
        if not p_exists:
            raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found.")

        total = await conn.fetchval(
            "SELECT COUNT(*) FROM pipeline_runs WHERE pipeline_id = $1",
            pipeline_id,
        )

        rows = await conn.fetch(
            """
            SELECT r.id, r.pipeline_id, r.pipeline_version, r.query, r.generation, r.status,
                   r.retrieval_latency_ms, r.latency_ms, r.input_tokens, r.output_tokens,
                   r.cost_usd, r.model_used, r.created_at,
                   e.overall_score, e.faithfulness, e.answer_relevance,
                   e.context_precision, e.context_recall
            FROM pipeline_runs r
            LEFT JOIN LATERAL (
                SELECT overall_score, faithfulness, answer_relevance, context_precision, context_recall
                FROM evaluations
                WHERE run_id = r.id
                ORDER BY evaluated_at DESC
                LIMIT 1
            ) e ON true
            WHERE r.pipeline_id = $1
            ORDER BY r.created_at DESC
            LIMIT $2 OFFSET $3
            """,
            pipeline_id,
            page_size,
            offset,
        )

        items = [
            PipelineRunItem(
                id=str(r["id"]),
                pipeline_id=str(r["pipeline_id"]),
                pipeline_version=r["pipeline_version"],
                query=r["query"],
                generation=r["generation"],
                status=r["status"],
                retrieval_latency_ms=r["retrieval_latency_ms"],
                latency_ms=r["latency_ms"],
                input_tokens=r["input_tokens"],
                output_tokens=r["output_tokens"],
                cost_usd=r["cost_usd"],
                model_used=r["model_used"],
                overall_score=r["overall_score"],
                faithfulness=r["faithfulness"],
                answer_relevance=r["answer_relevance"],
                context_precision=r["context_precision"],
                context_recall=r["context_recall"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

        return PaginatedRunsResponse(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            has_next=(offset + len(items)) < total,
        )


@router.get("/{pipeline_id}/analytics", response_model=AnalyticsResponse)
async def get_pipeline_analytics(pipeline_id: UUID, _scope=Depends(require_scope("query"))):
    """Compute latency percentiles, evaluation aggregates, cost, and 30-day query trends."""
    if connection is None:
        raise HTTPException(status_code=503, detail="Database connection pool unavailable.")

    async with connection() as conn:
        pipe = await conn.fetchrow(
            "SELECT id, version FROM pipelines WHERE id = $1",
            pipeline_id,
        )
        if not pipe:
            raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found.")

        # 1. Fetch retrieval and total latency data
        runs = await conn.fetch(
            """
            SELECT retrieval_latency_ms, latency_ms, input_tokens, output_tokens, cost_usd
            FROM pipeline_runs
            WHERE pipeline_id = $1 AND status = 'completed'
            """,
            pipeline_id,
        )

        retrieval_latencies = [
            float(r["retrieval_latency_ms"])
            for r in runs
            if r["retrieval_latency_ms"] is not None
        ]

        p50 = calculate_percentile(retrieval_latencies, 50)
        p95 = calculate_percentile(retrieval_latencies, 95)
        p99 = calculate_percentile(retrieval_latencies, 99)

        # Average generation latency: (latency_ms - retrieval_latency_ms)
        gen_latencies = [
            float(r["latency_ms"] - (r["retrieval_latency_ms"] or 0))
            for r in runs
            if r["latency_ms"] is not None and r["latency_ms"] >= (r["retrieval_latency_ms"] or 0)
        ]
        avg_gen_latency = round(sum(gen_latencies) / len(gen_latencies), 2) if gen_latencies else None

        # Average cost per query
        cost_vals = [float(r["cost_usd"]) for r in runs if r["cost_usd"] is not None]
        avg_cost = round(sum(cost_vals) / len(cost_vals), 6) if cost_vals else None

        # 2. Evaluation score averages
        eval_row = await conn.fetchrow(
            """
            SELECT AVG(e.faithfulness) as avg_faith,
                   AVG(e.answer_relevance) as avg_rel,
                   AVG(e.context_precision) as avg_prec,
                   AVG(e.context_recall) as avg_rec,
                   AVG(e.overall_score) as avg_overall
            FROM evaluations e
            JOIN pipeline_runs r ON r.id = e.run_id
            WHERE r.pipeline_id = $1
            """,
            pipeline_id,
        )

        avg_evals = {
            "faithfulness": round(eval_row["avg_faith"], 4) if eval_row and eval_row["avg_faith"] is not None else None,
            "answer_relevance": round(eval_row["avg_rel"], 4) if eval_row and eval_row["avg_rel"] is not None else None,
            "context_precision": round(eval_row["avg_prec"], 4) if eval_row and eval_row["avg_prec"] is not None else None,
            "context_recall": round(eval_row["avg_rec"], 4) if eval_row and eval_row["avg_rec"] is not None else None,
            "overall_score": round(eval_row["avg_overall"], 4) if eval_row and eval_row["avg_overall"] is not None else None,
        }

        # 3. 30-day query count and cost buckets
        thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=29)
        daily_rows = await conn.fetch(
            """
            SELECT DATE(created_at) as run_date,
                   COUNT(*) as query_count,
                   AVG(cost_usd) as avg_cost,
                   SUM(cost_usd) as total_cost
            FROM pipeline_runs
            WHERE pipeline_id = $1 AND created_at >= $2
            GROUP BY DATE(created_at)
            """,
            pipeline_id,
            thirty_days_ago,
        )

        daily_map = {str(r["run_date"]): int(r["query_count"]) for r in daily_rows}
        cost_map = {
            str(r["run_date"]): {
                "avg_cost": round(float(r["avg_cost"]), 6) if r["avg_cost"] is not None else None,
                "total_cost": round(float(r["total_cost"]), 6) if r["total_cost"] is not None else None,
            }
            for r in daily_rows
        }

        today = date.today()
        daily_counts: list[DailyQueryCount] = []
        cost_trends: list[dict[str, Any]] = []
        total_30d = 0

        for day_offset in range(29, -1, -1):
            curr_date = today - timedelta(days=day_offset)
            date_str = curr_date.isoformat()
            cnt = daily_map.get(date_str, 0)
            daily_counts.append(DailyQueryCount(date=date_str, count=cnt))
            total_30d += cnt

            c_info = cost_map.get(date_str)
            if c_info and c_info["avg_cost"] is not None:
                cost_trends.append({
                    "date": date_str,
                    "cost_per_query": c_info["avg_cost"],
                    "total_cost": c_info["total_cost"],
                })

        # 4. Recent failed runs
        failed_rows = await conn.fetch(
            """
            SELECT id, query, error_message, latency_ms, created_at
            FROM pipeline_runs
            WHERE pipeline_id = $1 AND status = 'failed'
            ORDER BY created_at DESC
            LIMIT 10
            """,
            pipeline_id,
        )
        recent_failed = [
            {
                "run_id": str(r["id"]),
                "query": r["query"],
                "error_message": r["error_message"] or "Unknown execution error",
                "latency_ms": r["latency_ms"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            }
            for r in failed_rows
        ]

        # 5. 7-day daily score trend for sparkline
        seven_days_ago = datetime.now(timezone.utc) - timedelta(days=6)
        score_rows = await conn.fetch(
            """
            SELECT DATE(e.evaluated_at) as eval_date, AVG(e.overall_score) as avg_score
            FROM evaluations e
            JOIN pipeline_runs r ON r.id = e.run_id
            WHERE r.pipeline_id = $1 AND e.evaluated_at >= $2
            GROUP BY DATE(e.evaluated_at)
            ORDER BY DATE(e.evaluated_at) ASC
            """,
            pipeline_id,
            seven_days_ago,
        )
        score_map = {str(r["eval_date"]): round(float(r["avg_score"]), 4) for r in score_rows}
        daily_scores_7d: list[dict[str, Any]] = []
        for d in range(6, -1, -1):
            cur = (today - timedelta(days=d)).isoformat()
            daily_scores_7d.append({"date": cur, "score": score_map.get(cur)})

        return AnalyticsResponse(
            pipeline_id=str(pipeline_id),
            version=pipe["version"],
            retrieval_latency_p50=p50,
            retrieval_latency_p95=p95,
            retrieval_latency_p99=p99,
            avg_generation_latency_ms=avg_gen_latency,
            avg_evaluation_scores=avg_evals,
            estimated_cost_per_query=avg_cost,
            daily_query_counts=daily_counts,
            total_queries_30d=total_30d,
            percentile_method="linear_interpolation",
            cost_trends_30d=cost_trends,
            recent_failed_runs=recent_failed,
            daily_scores_7d=daily_scores_7d,
        )


@router.get("/{pipeline_id}/suggestions", response_model=SuggestionsResponse)
async def get_pipeline_suggestions(pipeline_id: UUID, _scope=Depends(require_scope("query"))):
    """Analyze recent evaluation metrics and return advisory rule-based optimization suggestions."""
    if connection is None:
        raise HTTPException(status_code=503, detail="Database connection pool unavailable.")

    async with connection() as conn:
        pipe = await conn.fetchrow(
            "SELECT id, config FROM pipelines WHERE id = $1",
            pipeline_id,
        )
        if not pipe:
            raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found.")

        raw_cfg = pipe["config"]
        cfg_data = json.loads(raw_cfg) if isinstance(raw_cfg, str) else raw_cfg
        current_config = PipelineConfig(**cfg_data)

        # Retrieve recent evaluation records
        eval_rows = await conn.fetch(
            """
            SELECT e.faithfulness, e.answer_relevance, e.context_precision, e.context_recall, e.overall_score
            FROM evaluations e
            JOIN pipeline_runs r ON r.id = e.run_id
            WHERE r.pipeline_id = $1
            ORDER BY e.evaluated_at DESC
            LIMIT 50
            """,
            pipeline_id,
        )

        eval_records = [
            {
                "faithfulness": r["faithfulness"],
                "answer_relevance": r["answer_relevance"],
                "context_precision": r["context_precision"],
                "context_recall": r["context_recall"],
                "overall_score": r["overall_score"],
            }
            for r in eval_rows
        ]

        optimizer = PipelineOptimizer()
        suggestions = optimizer.analyze(eval_records, current_config)

        return SuggestionsResponse(
            pipeline_id=str(pipeline_id),
            evaluation_sample_count=len(eval_records),
            has_sufficient_data=len(eval_records) >= optimizer.min_samples,
            suggestions=suggestions,
        )


@router.post(
    "/{pipeline_id}/suggestions",
    response_model=QualityAnomalyResponse,
    status_code=status.HTTP_201_CREATED,
)
async def record_quality_anomaly(
    pipeline_id: UUID,
    payload: QualityAnomalyRequest,
    authorization: str | None = Header(default=None),
    service_token: str | None = Header(default=None, alias="X-Observability-Service-Token"),
    _scope=Depends(require_scope("observability")),
) -> QualityAnomalyResponse:
    """Persist a worker-detected anomaly using the internal bearer credential."""
    expected_token = settings.observability_service_token
    if not expected_token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Internal observability endpoint is not configured.",
        )
    if not service_token or not hmac.compare_digest(service_token, expected_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid service credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if pipeline_connection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection pool unavailable.",
        )

    async with pipeline_connection(pipeline_id) as conn:
        pipeline = await conn.fetchrow(
            "SELECT config FROM pipelines WHERE id = $1",
            pipeline_id,
        )
        run_exists = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1 FROM pipeline_runs
                WHERE id = $1 AND pipeline_id = $2
            )
            """,
            payload.run_id,
            pipeline_id,
        )
        if pipeline is None or not run_exists:
            raise HTTPException(status_code=404, detail="Pipeline run not found.")

        config_data = pipeline["config"]
        if isinstance(config_data, str):
            config_data = json.loads(config_data)
        current_config = PipelineConfig(**config_data)
        rows = await conn.fetch(
            """
            SELECT e.faithfulness, e.answer_relevance, e.context_precision,
                   e.context_recall, e.overall_score
            FROM evaluations e
            JOIN pipeline_runs r ON r.id = e.run_id
            WHERE r.pipeline_id = $1
            ORDER BY e.evaluated_at DESC
            LIMIT 50
            """,
            pipeline_id,
        )
        eval_records = [
            {
                "faithfulness": row["faithfulness"],
                "answer_relevance": row["answer_relevance"],
                "context_precision": row["context_precision"],
                "context_recall": row["context_recall"],
                "overall_score": row["overall_score"],
            }
            for row in rows
        ]
        optimizer = PipelineOptimizer()
        suggestions = optimizer.analyze(eval_records, current_config)
        rule_id = "overall_score_drop_2_stddev"

        async with conn.transaction():
            await conn.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1::text, 0))",
                str(pipeline_id),
            )
            existing = await conn.fetchrow(
                """
                SELECT id, run_id, rule_id, latest_score, baseline_mean,
                       baseline_stddev, suggestions, created_at
                FROM pipeline_quality_anomalies
                WHERE pipeline_id = $1 AND run_id = $2 AND rule_id = $3
                """,
                pipeline_id,
                payload.run_id,
                rule_id,
            )
            if existing:
                return QualityAnomalyResponse(
                    id=existing["id"],
                    pipeline_id=pipeline_id,
                    run_id=existing["run_id"],
                    rule_id=existing["rule_id"],
                    latest_score=existing["latest_score"],
                    baseline_mean=existing["baseline_mean"],
                    baseline_stddev=existing["baseline_stddev"],
                    suggestions=_decode_suggestions(existing["suggestions"]),
                    created_at=existing["created_at"],
                    recorded=False,
                    cooldown_suppressed=False,
                )

            recent = await conn.fetchrow(
                """
                SELECT id, run_id, rule_id, latest_score, baseline_mean,
                       baseline_stddev, suggestions, created_at
                FROM pipeline_quality_anomalies
                WHERE pipeline_id = $1
                  AND rule_id = $2
                  AND created_at >= NOW() - INTERVAL '24 hours'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                pipeline_id,
                rule_id,
            )
            if recent:
                return QualityAnomalyResponse(
                    id=recent["id"],
                    pipeline_id=pipeline_id,
                    run_id=recent["run_id"],
                    rule_id=recent["rule_id"],
                    latest_score=recent["latest_score"],
                    baseline_mean=recent["baseline_mean"],
                    baseline_stddev=recent["baseline_stddev"],
                    suggestions=_decode_suggestions(recent["suggestions"]),
                    created_at=recent["created_at"],
                    recorded=False,
                    cooldown_suppressed=True,
                )

            row = await conn.fetchrow(
                """
                INSERT INTO pipeline_quality_anomalies (
                    pipeline_id, run_id, rule_id, latest_score, baseline_mean,
                    baseline_stddev, suggestions
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                RETURNING id, run_id, rule_id, latest_score, baseline_mean,
                          baseline_stddev, suggestions, created_at
                """,
                pipeline_id,
                payload.run_id,
                rule_id,
                payload.latest_score,
                payload.baseline_mean,
                payload.baseline_stddev,
                json.dumps([item.model_dump(mode="json") for item in suggestions]),
            )

    QUALITY_ANOMALIES.labels(pipeline_id=str(pipeline_id)).inc()
    return QualityAnomalyResponse(
        id=row["id"],
        pipeline_id=pipeline_id,
        run_id=row["run_id"],
        rule_id=row["rule_id"],
        latest_score=row["latest_score"],
        baseline_mean=row["baseline_mean"],
        baseline_stddev=row["baseline_stddev"],
        suggestions=_decode_suggestions(row["suggestions"]),
        created_at=row["created_at"],
        recorded=True,
        cooldown_suppressed=False,
    )
