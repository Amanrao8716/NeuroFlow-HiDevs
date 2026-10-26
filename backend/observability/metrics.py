"""Prometheus metrics for NeuroFlow API and worker operations."""

from __future__ import annotations

from threading import Lock

from prometheus_client import Counter, Gauge, Histogram

QUERY_TOTAL = Counter(
    "neuroflow_queries_total",
    "Number of query pipeline executions by terminal status.",
    ["status"],
)
QUERY_LATENCY = Histogram(
    "neuroflow_query_duration_seconds",
    "End-to-end query pipeline duration in seconds.",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)
PROVIDER_CALLS = Counter(
    "neuroflow_provider_calls_total",
    "Number of provider operations.",
    ["provider", "operation", "status"],
)
PROVIDER_TOKENS = Counter(
    "neuroflow_provider_tokens_total",
    "Tokens reported by provider responses.",
    ["provider", "direction"],
)
PROVIDER_COST = Counter(
    "neuroflow_provider_cost_usd_total",
    "Provider cost reported by completed non-streaming responses.",
    ["provider"],
)
INGESTION_DOCUMENTS = Counter(
    "neuroflow_ingestion_documents_total",
    "Documents processed by ingestion workers.",
    ["status"],
)
EVALUATION_JOBS = Counter(
    "neuroflow_evaluation_jobs_total",
    "Evaluation jobs completed or failed.",
    ["status"],
)
QUEUE_DEPTH = Gauge(
    "neuroflow_queue_depth",
    "Current number of queued jobs.",
    ["queue"],
)
CIRCUIT_BREAKER_TRIPS = Counter(
    "neuroflow_circuit_breaker_trips_total",
    "Circuit breaker transitions into the open state.",
    ["provider"],
)
CIRCUIT_BREAKER_OPEN = Gauge(
    "neuroflow_circuit_breaker_open",
    "Number of circuit breakers currently in the open state.",
)
CIRCUIT_BREAKER_STATE = Gauge(
    "neuroflow_circuit_breaker_state",
    "Circuit breaker state: 1 when open, 0 otherwise.",
    ["provider"],
)
EVALUATION_SCORE_24H = Gauge(
    "neuroflow_evaluation_score_24h",
    "Rolling 24-hour mean for a persisted evaluation metric.",
    ["pipeline_id", "metric"],
)
QUALITY_ANOMALIES = Counter(
    "neuroflow_quality_anomalies_total",
    "Quality anomalies accepted by the authenticated API.",
    ["pipeline_id"],
)

_open_breakers: set[str] = set()
_breaker_lock = Lock()


def set_circuit_breaker_state(provider: str, is_open: bool) -> None:
    """Update per-provider and process-wide open circuit gauges."""
    with _breaker_lock:
        if is_open:
            _open_breakers.add(provider)
        else:
            _open_breakers.discard(provider)
        CIRCUIT_BREAKER_STATE.labels(provider=provider).set(int(is_open))
        CIRCUIT_BREAKER_OPEN.set(len(_open_breakers))
