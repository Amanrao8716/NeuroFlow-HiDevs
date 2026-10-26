"""Quality score anomaly detection and persisted-score metric refresh."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, pstdev
from threading import Lock

from observability.metrics import EVALUATION_SCORE_24H

QUALITY_METRICS = (
    "faithfulness",
    "answer_relevance",
    "context_precision",
    "context_recall",
    "overall_score",
)
MIN_ANOMALY_HISTORY = 5
_known_pipeline_ids: set[str] = set()
_known_pipeline_lock = Lock()


@dataclass(frozen=True)
class QualityAnomaly:
    latest_score: float
    baseline_mean: float
    baseline_stddev: float


def detect_quality_anomaly(
    baseline_scores: list[float],
    latest_score: float,
    *,
    minimum_history: int = MIN_ANOMALY_HISTORY,
) -> QualityAnomaly | None:
    """Detect a latest score more than two population standard deviations low."""
    if len(baseline_scores) < minimum_history:
        return None
    mean = fmean(baseline_scores)
    stddev = pstdev(baseline_scores)
    if latest_score < mean - 2 * stddev:
        return QualityAnomaly(
            latest_score=latest_score,
            baseline_mean=mean,
            baseline_stddev=stddev,
        )
    return None


def remember_pipeline_for_refresh(pipeline_id: str) -> None:
    """Track tenant IDs observed by this worker for rolling-gauge refresh."""
    with _known_pipeline_lock:
        _known_pipeline_ids.add(pipeline_id)


def known_pipeline_ids() -> tuple[str, ...]:
    with _known_pipeline_lock:
        return tuple(_known_pipeline_ids)


async def refresh_rolling_evaluation_metrics(conn, pipeline_id: str) -> None:
    """Refresh rolling 24-hour score gauges from persisted evaluation rows."""
    rows = await conn.fetch(
        """
        SELECT AVG(e.faithfulness) AS faithfulness,
               AVG(e.answer_relevance) AS answer_relevance,
               AVG(e.context_precision) AS context_precision,
               AVG(e.context_recall) AS context_recall,
               AVG(e.overall_score) AS overall_score
        FROM evaluations e
        JOIN pipeline_runs r ON r.id = e.run_id
        WHERE r.pipeline_id = $1
          AND e.evaluated_at >= NOW() - INTERVAL '24 hours'
        """,
        pipeline_id,
    )
    if not rows:
        return
    for metric in QUALITY_METRICS:
        value = rows[0][metric]
        if value is not None:
            EVALUATION_SCORE_24H.labels(
                pipeline_id=pipeline_id,
                metric=metric,
            ).set(float(value))
