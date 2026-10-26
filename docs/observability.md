# NeuroFlow observability

## Stack

The Compose stack provisions Prometheus, Grafana, and Jaeger alongside the API
and worker. Prometheus scrapes `api:8000/metrics` and `worker:8001`; Grafana
provisions the **NeuroFlow Operations** and **NeuroFlow Quality** dashboards.
Jaeger receives OTLP traces on port 4317.

Run `backend/db/migrations/011_quality_anomalies.sql` with the normal database
migration procedure before using the quality dashboard or anomaly endpoint.
Set `OBSERVABILITY_SERVICE_TOKEN` to the same high-entropy value for the API and
worker. The worker sends it as a bearer credential; the token is not logged or
stored in PostgreSQL. Keep Prometheus, Grafana, Jaeger, and `/metrics` behind
private-network or operations access controls in production.

## Metrics

The API and worker expose these custom metric families:

| Metric | Type | Meaning |
| --- | --- | --- |
| `neuroflow_queries_total{status}` | Counter | Completed, failed, or cancelled query executions |
| `neuroflow_query_duration_seconds` | Histogram | End-to-end query duration; seconds; buckets 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30 |
| `neuroflow_provider_calls_total{provider,operation,status}` | Counter | Provider chat, stream, and local embedding operations |
| `neuroflow_provider_tokens_total{provider,direction}` | Counter | Input/output tokens reported by non-streaming provider responses |
| `neuroflow_provider_cost_usd_total{provider}` | Counter | Cost reported by non-streaming provider responses |
| `neuroflow_ingestion_documents_total{status}` | Counter | Successful and failed ingestion executions |
| `neuroflow_evaluation_jobs_total{status}` | Counter | Successful and failed evaluation jobs |
| `neuroflow_queue_depth{queue}` | Gauge | Current Redis ingestion and evaluation queue lengths |
| `neuroflow_circuit_breaker_trips_total{provider}` | Counter | Transitions from closed/half-open to open, not each rejected call |
| `neuroflow_circuit_breaker_open` | Gauge | Number of locally observed open provider circuits |
| `neuroflow_circuit_breaker_state{provider}` | Gauge | `1` while the provider circuit is open, otherwise `0` |
| `neuroflow_evaluation_score_24h{pipeline_id,metric}` | Gauge | Persisted rolling 24-hour mean for five evaluation scores |
| `neuroflow_quality_anomalies_total{pipeline_id}` | Counter | Anomaly events accepted and persisted by the internal endpoint |

Streaming provider responses do not include authoritative token usage or cost
in the current provider contract. Those counters therefore include only
reported non-streaming usage; no stream usage is estimated.

Prometheus instrumentation and FastAPI HTTP metrics are also included. Labels
exclude raw queries, document names, and user identifiers. Per-pipeline score
gauges and anomaly counts use opaque pipeline IDs to support tenant-level
quality trends.

## Tracing

The API and worker initialize OpenTelemetry with OTLP export to Jaeger. Query
traces include query processing, dense/sparse/metadata retrieval, reranking,
context and prompt assembly, and provider calls. Ingestion traces include
extraction, chunking, embedding, and database writes. Evaluation has a parent
worker span, and each queued message carries W3C trace context so the worker
span is a child of the originating generation trace.

Provider spans record the selected model and, when the provider returns them,
token counts, reported cost, and latency. Do not add user prompts or document
contents as span attributes.

## Quality anomaly detection

The worker compares the newly persisted overall score with up to 100 prior
persisted scores from the same pipeline in the preceding seven days. At least
five prior valid scores are required. An anomaly is a score strictly below
`mean - 2 * population_standard_deviation`; zero variance is handled directly
by this comparison and does not divide by zero.

The worker submits anomaly evidence to the authenticated
`POST /pipelines/{pipeline_id}/suggestions` endpoint. The API verifies that the
run belongs to the pipeline, computes suggestions using the existing
`PipelineOptimizer`, and persists the event and suggestions in
`pipeline_quality_anomalies`. A transaction-scoped advisory lock enforces a
24-hour per-pipeline cooldown, and a unique `(pipeline_id, run_id, rule_id)`
constraint deduplicates retries. Persisted anomaly events appear as Grafana
dashboard annotations.

## Alerts

Rules are provisioned from `infra/prometheus/alerts.yml`. Ingestion failure
alerting uses an absolute rate of failed ingestion executions per second, not
a percentage of all jobs. The configured threshold is `0.1` failures/second
over five minutes. Other rules cover query p95 latency, queue depth, open
provider circuits, low rolling evaluation quality, and persisted anomaly
events. Review thresholds for the deployment's workload before enabling
notification routing.
