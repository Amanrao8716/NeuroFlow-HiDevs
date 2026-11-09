# NeuroFlow Architecture & Production Operations Runbook

This runbook documents the standard operating procedures for the five primary production incident scenarios in NeuroFlow. All commands, endpoints, database schemas, and metrics reflect the actual implementation.

---

## 1. High Query Latency (P95 > 10s)

**Symptoms**: Query endpoint (`POST /query`) P95 latency exceeds 10 seconds; Prometheus metric `neuroflow_query_latency_seconds` indicates degradation; API clients observe timeouts or slow response streaming.

### Diagnostic & Triage Procedures

1. **Jaeger Distributed Tracing**:
   - Access the Jaeger UI at `http://localhost:16686` (or OpenTelemetry Collector endpoint at `settings.otel_exporter_otlp_endpoint`, default `http://localhost:4317`).
   - Filter by service `NeuroFlow` and operation `query` or `post_query`.
   - Inspect the span breakdown:
     - `QueryProcessor.process`: query classification and expansion.
     - `HybridRetriever.retrieve`: parallel dense (pgvector HNSW) and sparse (Postgres GIN full-text) retrieval.
     - `reciprocal_rank_fusion`: candidate merging.
     - `CrossEncoderReranker.rerank`: reranking scored pairs.
     - `GenerationPipeline.generate`: LLM generation call (`client.generate`).
   - Identify whether the bottleneck is database retrieval latency, reranker compute, or upstream LLM provider latency.

2. **Redis Memory & Cache Hit Rate**:
   - Check Redis memory usage and hit/miss ratios:
     ```bash
     redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" info memory
     redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" info stats
     ```
   - Calculate cache hit rate:
     $$\text{Hit Rate} = \frac{\text{keyspace\_hits}}{\text{keyspace\_hits} + \text{keyspace\_misses}}$$
   - Verify if full-query caching is operating (`neuroflow:query_cache:*`) and embedding caching is populated (`neuroflow:embeddings:*`).

3. **PostgreSQL pg_stat_statements**:
   - Query the top 10 queries consuming execution time:
     ```sql
     SELECT query,
            calls,
            round(total_exec_time::numeric, 2) AS total_ms,
            round(mean_exec_time::numeric, 2) AS mean_ms,
            rows
     FROM pg_stat_statements
     WHERE query LIKE '%chunks%' OR query LIKE '%pipeline_runs%'
     ORDER BY mean_exec_time DESC
     LIMIT 10;
     ```

### Resolution & Mitigation

1. **Safe Cache Flushing**:
   - *Do not* issue a blind `FLUSHALL` or `FLUSHDB` in production, as this evicts rate limiters and circuit breaker states.
   - Safely clear only query caches using cursor scanning:
     ```bash
     redis-cli -a "$REDIS_PASSWORD" --scan --pattern "neuroflow:query_cache:*" | xargs -L 100 -r redis-cli -a "$REDIS_PASSWORD" del
     ```

2. **Database Indexes & Vacuuming**:
   - Check for bloated indexes on the `chunks` table:
     ```sql
     SELECT indexrelname, pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
     FROM pg_stat_user_indexes
     WHERE relname = 'chunks';
     ```
   - Reindex HNSW vector index and full-text index:
     ```sql
     REINDEX INDEX CONCURRENTLY chunks_embedding_hnsw_idx;
     REINDEX INDEX CONCURRENTLY chunks_content_fts_idx;
     VACUUM ANALYZE chunks;
     ```

3. **API Replica Scaling**:
   - If CPU utilization on API nodes is elevated due to concurrent token parsing or reranking:
     ```bash
     docker compose up -d --scale api=4
     ```

### Verification

```bash
# Verify health latency check
curl -s http://localhost:8000/health | jq '.checks.postgres.latency_ms, .checks.redis.latency_ms'

# Send a test query and verify latency
curl -s -w "\nTime: %{time_total}s\n" -X POST http://localhost:8000/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "Test benchmark latency query", "pipeline_id": "<pipeline-uuid>"}'
```

---

## 2. Evaluation Scores Degrading

**Symptoms**: Automated LLM-as-a-judge scores drop significantly below baseline; alert fires for `pipeline_quality_anomalies` table inserts or `neuroflow_quality_anomalies_total` metric increments.

### Diagnostic & Triage Procedures

1. **Identify Affected Pipeline and Metric**:
   - Query recent evaluation metrics to pinpoint which dimension degraded (`faithfulness`, `answer_relevance`, `context_precision`, `context_recall`):
     ```sql
     SELECT r.pipeline_id,
            p.name AS pipeline_name,
            COUNT(e.id) AS eval_count,
            round(AVG(e.faithfulness)::numeric, 3) AS avg_faithfulness,
            round(AVG(e.answer_relevance)::numeric, 3) AS avg_relevance,
            round(AVG(e.context_precision)::numeric, 3) AS avg_precision,
            round(AVG(e.context_recall)::numeric, 3) AS avg_recall,
            round(AVG(e.overall_score)::numeric, 3) AS avg_overall
     FROM evaluations e
     JOIN pipeline_runs r ON r.id = e.run_id
     JOIN pipelines p ON p.id = r.pipeline_id
     WHERE e.evaluated_at >= NOW() - INTERVAL '24 hours'
     GROUP BY r.pipeline_id, p.name;
     ```
   - Check recent recorded anomalies:
     ```sql
     SELECT id, pipeline_id, run_id, rule_id, latest_score, baseline_mean, baseline_stddev, created_at
     FROM pipeline_quality_anomalies
     ORDER BY created_at DESC
     LIMIT 5;
     ```

2. **Recent Documents Inspection**:
   - Check if low-quality or corrupt documents were recently ingested into the affected pipeline:
     ```sql
     SELECT id, filename, source_type, status, chunk_count, created_at
     FROM documents
     WHERE pipeline_id = '<pipeline-uuid>'
       AND created_at >= NOW() - INTERVAL '48 hours'
     ORDER BY created_at DESC;
     ```

3. **MLflow Model Changes & Tracking**:
   - Access the MLflow Tracking UI at `http://localhost:5000` (or `settings.mlflow_tracking_uri`).
   - Check for recent fine-tuning jobs (`GET /finetune/jobs`) or changes to base models.
   - Inspect run parameters, artifact evaluation metrics, and model tags.

4. **Training-Data Inspection**:
   - Inspect eligible training pairs extracted for the pipeline:
     ```bash
     curl -s http://localhost:8000/finetune/training-data/preview?pipeline_id=<pipeline-uuid>&min_quality_score=0.82 \
       -H "Authorization: Bearer $TOKEN" | jq '.items[] | {quality_score, token_count, user_message}'
     ```
   - Check the `training_pairs` table for low-quality entries:
     ```sql
     SELECT id, quality_score, user_message, assistant_message, created_at
     FROM training_pairs
     WHERE created_at >= NOW() - INTERVAL '7 days' AND quality_score < 0.70;
     ```

### Resolution & Mitigation

1. **Review Optimization Suggestions**:
   - Fetch rule-based advisor suggestions from the optimizer:
     ```bash
     curl -s http://localhost:8000/pipelines/<pipeline-uuid>/suggestions \
       -H "Authorization: Bearer $TOKEN" | jq '.suggestions'
     ```

2. **Pipeline Configuration Rollback**:
   - Query previous immutable configuration versions:
     ```sql
     SELECT version, description, created_at, config
     FROM pipeline_versions
     WHERE pipeline_id = '<pipeline-uuid>'
     ORDER BY version DESC
     LIMIT 5;
     ```
   - Restore the previous known good configuration using `PATCH /pipelines/<pipeline-uuid>`:
     ```bash
     curl -X PATCH http://localhost:8000/pipelines/<pipeline-uuid> \
       -H "Authorization: Bearer $TOKEN" \
       -H "Content-Type: application/json" \
       -d '{"config_updates": <previous-config-json>}'
     ```

### Verification

```bash
# Run a verification query
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "Verify quality recovery", "pipeline_id": "<pipeline-uuid>"}' | jq '.run_id'

# Check evaluation score for the run
curl -s http://localhost:8000/runs/<run-id>/evaluation \
  -H "Authorization: Bearer $TOKEN" | jq '{overall_score, faithfulness, answer_relevance}'
```

---

## 3. LLM Provider Circuit Breaker Open

**Symptoms**: `/health` reports `overall: "degraded"` or `"critical"`, with `checks.circuit_breakers` showing `"state": "open"` or `"half_open"`. Queries return HTTP 500 or fallback responses.

### Diagnostic & Triage Procedures

1. **Check Application Health Endpoint**:
   ```bash
   curl -s http://localhost:8000/health | jq '.checks.circuit_breakers'
   ```
   Inspect which breaker is open (`name`, `state`, `failure_count`, `last_failure`).

2. **Upstream Provider Status Check**:
   - Check third-party provider status pages:
     - OpenAI: `https://status.openai.com`
     - Anthropic: `https://status.anthropic.com`
   - Test connectivity directly from the server environment:
     ```bash
     curl -s -I https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"
     ```

3. **Recovery Timeout Mechanics**:
   - The circuit breaker implementation (`resilience.circuit_breaker.CircuitBreaker`) transitions automatically:
     - `CLOSED` $\xrightarrow{\text{failures} \ge \text{threshold}}$ `OPEN`
     - `OPEN` $\xrightarrow{\text{elapsed} \ge \text{recovery\_timeout}}$ `HALF_OPEN` (applied on next `state()` check)
     - `HALF_OPEN` $\xrightarrow{\text{probe success}}$ `CLOSED`
     - `HALF_OPEN` $\xrightarrow{\text{probe failure}}$ `OPEN`
   - Default recovery timeout is 60 seconds (`OPEN_TIMEOUT = 60`).

### Resolution & Reset Procedure

> **Important**: The API does **not** expose a public HTTP endpoint such as `/admin/circuit-breaker/reset`. Do not attempt to call nonexistent endpoints.

Choose one of the following valid recovery options:

1. **Automatic Half-Open Probe (Preferred)**:
   - Allow the 60-second recovery timeout to elapse.
   - Send a single benign test request. If the upstream provider succeeds, the breaker resets to `CLOSED`.

2. **Direct Redis State Reset**:
   - Delete the circuit breaker state key in Redis:
     ```bash
     redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" keys "neuroflow:cb:*"
     redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" del "neuroflow:cb:llm_provider"
     ```

3. **Process Restart**:
   - If Redis is unavailable or in-memory circuit breaker singletons must be re-initialized:
     ```bash
     docker compose restart api worker
     ```

### Verification

```bash
curl -s http://localhost:8000/health | jq '.checks.circuit_breakers[] | select(.name=="llm_provider")'
# Expected output: {"name": "llm_provider", "state": "closed", ...}
```

---

## 4. Ingestion Queue Depth > 100

**Symptoms**: `/health` reports `checks.ingest_queue_depth > 100`; POST `/ingest` returns HTTP 503 `{"error": "ingestion_queue_full"}` due to backpressure guard; Prometheus metric `neuroflow_queue_depth{queue="ingestion"}` triggers alert.

### Diagnostic & Triage Procedures

1. **Check Queue Depth & Health**:
   ```bash
   curl -s http://localhost:8000/health | jq '{status, ingest_queue_depth: .checks.ingest_queue_depth}'
   ```

2. **Inspect ARQ Worker Logs**:
   - Review worker logs for parsing bottlenecks, out-of-memory errors on large PDFs/images, or worker crashes:
     ```bash
     docker logs neuroflow-worker --tail 100 -f
     ```

3. **Inspect Redis ARQ Keys**:
   - Check pending and in-flight job keys:
     ```bash
     redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" llen "arq:queue:default"
     redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" zcard "arq:retry"
     redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" keys "arq:job:*" | wc -l
     ```

### Resolution & Mitigation

1. **Scale Worker Replicas**:
   - Increase ARQ worker concurrency to drain the queue:
     ```bash
     docker compose up -d --scale worker=4
     ```

2. **Safe Worker Restart**:
   - If an unhandled thread crash occurred or a worker hangs on an external extraction binary:
     ```bash
     docker compose restart worker
     ```

3. **Handling Stuck or Poison-Pill Jobs**:
   - Identify failed documents in PostgreSQL:
     ```sql
     SELECT id, filename, source_type, created_at, metadata
     FROM documents
     WHERE status = 'processing' AND created_at < NOW() - INTERVAL '30 minutes';
     ```
   - Mark stuck processing documents as `failed` so clients are unblocked:
     ```sql
     UPDATE documents
     SET status = 'failed',
         metadata = metadata || '{"error": "Ingestion timed out in worker"}'::jsonb
     WHERE status = 'processing' AND created_at < NOW() - INTERVAL '30 minutes';
     ```
   - If the Redis queue contains unprocessable poison jobs, clear the primary queue key:
     ```bash
     redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" del "arq:queue:default"
     ```

### Verification

```bash
# Verify queue depth is decreasing
watch -n 2 'curl -s http://localhost:8000/health | jq .checks.ingest_queue_depth'

# Test ingestion endpoint accepts new files
curl -X POST http://localhost:8000/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -F "pipeline_id=<pipeline-uuid>" \
  -F "file=@test_doc.pdf"
```

---

## 5. Database Disk Usage > 80%

**Symptoms**: Host or volume disk utilization exceeds 80%; PostgreSQL logs `disk full` warnings; insert queries fail with write errors.

### Diagnostic & Triage Procedures

1. **Table and Index Growth Inspection**:
   - Run storage breakdown query across all NeuroFlow tables:
     ```sql
     SELECT relname AS table_name,
            pg_size_pretty(pg_total_relation_size(relid)) AS total_size,
            pg_size_pretty(pg_relation_size(relid)) AS data_size,
            pg_size_pretty(pg_total_relation_size(relid) - pg_relation_size(relid)) AS index_size
     FROM pg_catalog.pg_statio_user_tables
     ORDER BY pg_total_relation_size(relid) DESC;
     ```
   - Typically, `chunks` (with 384-dim vector embeddings and GIN full-text index) and `pipeline_runs` (with generated text and citations) consume the majority of disk space.

2. **Verify Retention Schedule**:
   - Verify whether the retention cron job (`cron:retention_worker_task`) ran at 02:00 UTC or if it encountered errors:
     ```sql
     SELECT COUNT(*) AS expired_runs
     FROM pipeline_runs r
     LEFT JOIN evaluations e ON e.run_id = r.id
     WHERE r.status = 'completed'
       AND r.created_at < NOW() - INTERVAL '90 days'
       AND e.id IS NULL;

     SELECT COUNT(*) AS expired_evaluations
     FROM evaluations
     WHERE evaluated_at < NOW() - INTERVAL '180 days';

     SELECT COUNT(*) AS archived_chunks
     FROM chunks c
     JOIN documents d ON d.id = c.document_id
     WHERE d.status = 'archived';
     ```

### Resolution & Safe Cleanup

1. **Create Backup Before Deletion**:
   - Always snapshot or take a compressed dump of the database before manual purge:
     ```bash
     pg_dump -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -Fc -d "$POSTGRES_DB" -f "/var/backups/neuroflow_pre_retention_$(date +%Y%m%d_%H%M%S).dump"
     ```

2. **Execute On-Demand Data Retention**:
   - Run the standard retention sweep via the worker task or Python entry point:
     ```bash
     python -c "
     import asyncio
     from backend.tasks.retention import run_retention
     asyncio.run(run_retention())
     "
     ```

3. **Reclaim Disk Space (VACUUM)**:
   - PostgreSQL does not automatically return deleted row space to the operating system filesystem without vacuuming:
     ```sql
     -- Reclaim space and rebuild table statistics
     VACUUM (VERBOSE, ANALYZE) chunks;
     VACUUM (VERBOSE, ANALYZE) pipeline_runs;
     VACUUM (VERBOSE, ANALYZE) evaluations;
     ```

### Verification

```bash
# Check filesystem utilization
df -h /var/lib/postgresql/data

# Verify active document and chunk counts
psql -U neuroflow_app -d neuroflow -c "
SELECT 'documents' AS table, count(*) FROM documents
UNION ALL
SELECT 'chunks', count(*) FROM chunks
UNION ALL
SELECT 'pipeline_runs', count(*) FROM pipeline_runs;
"
```
