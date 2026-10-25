# NeuroFlow

**NeuroFlow** is a modular Retrieval-Augmented Generation (RAG) platform designed around observable ingestion, hybrid retrieval, grounded generation, asynchronous evaluation, and evidence-driven fine-tuning.

> **Task 31 — Architecture first.** This repository currently defines the architecture and contracts; implementation details may evolve only through documented decisions and versioned API changes.

## System at a glance

- **Ingestion:** extracts PDF, DOCX, image, CSV, and web content; chunks and embeds it; stores vectors and metadata in PostgreSQL with pgvector.
- **Retrieval:** runs dense-vector, lexical, and metadata-filtered retrieval in parallel; combines ranked results using Reciprocal Rank Fusion (RRF); reranks with a cross-encoder.
- **Generation:** assembles a bounded context window, selects an LLM using capability, domain, latency, and cost constraints, and streams output over Server-Sent Events (SSE).
- **Evaluation:** asynchronously scores faithfulness, answer relevance, context precision, and context recall; stores per-generation results and rolling aggregates.
- **Fine-tuning:** curates high-quality examples, exports JSONL, tracks experiments in MLflow, and enables a fine-tuned model only after it demonstrates a measured advantage.

## Repository layout

```text
backend/       API and application services (implementation in later tasks)
frontend/      User interface (implementation in later tasks)
pipelines/     Ingestion, retrieval, evaluation, and training pipelines
evaluation/    Evaluation datasets, evaluators, and quality reports
infra/         Deployment and operational configuration
docs/          Architecture, API contracts, data models, and ADRs
```

## Architecture documents

- [System architecture](docs/architecture.md)
- [REST API contracts](docs/api-contracts.md)
- [Data models](docs/data-models.md)
- [ADR 001 — Vector store](docs/adr/001-vector-store.md)
- [ADR 002 — Chunking strategy](docs/adr/002-chunking-strategy.md)
- [ADR 003 — Evaluation framework](docs/adr/003-evaluation-framework.md)
- [ADR 004 — Model routing](docs/adr/004-model-routing.md)

## Local development

This is an architecture-first repository. No production services or credentials are required to read the design. Before running future implementation tasks, copy `.env.example` to `.env` if one is added, use local development credentials, and never commit secrets.

## Security and operational principles

- Authenticate user-facing endpoints; scope documents and retrieval to the caller's tenant/permissions.
- Validate file type, size, URL scheme, and URL destination; defend URL ingestion against SSRF.
- Keep secrets out of source control and redact sensitive content from logs.
- Make ingestion and pipeline runs idempotent where possible.
- Treat model output and retrieved content as untrusted input.
- Apply timeouts, bounded retries, concurrency limits, and per-user rate limits.
- Do not promote a fine-tuned model based on a single metric or an unreviewed automated judge score.

## Status

- Task 31: Architecture and API contracts.
- Task 03: Multi-provider LLM support and router.
- Task 04: Multimodal document ingestion.
- Task 05: Hybrid retrieval pipeline with RRF fusion and cross-encoder reranking.
- Task 06: RAG generation pipeline, progressive streaming SSE, and citation tracking.

## Task 6 — RAG Generation Pipeline & Streaming SSE

### Features
- **Dynamic Prompt Assembly:** Conditioned on classified query type (`factual`, `analytical`, `comparative`, `procedural`).
- **Grounded LLM Generation:** Strictly context-grounded prompt format (`<context>[Source N — doc.pdf, page M]\n...\n</context>`).
- **Pre-Invocation Pipeline Run Creation:** Run record is persisted in `pipeline_runs` with status `running` *before* the LLM call.
- **Progressive Streaming SSE:** `POST /query` registers stream and returns `{run_id, stream_url}`. `GET /query/{run_id}/stream` streams events: `retrieval_start`, `retrieval_complete`, progressive `token`, and `done` (with structured citations).
- **Keepalive Heartbeats:** 15-second SSE comment pings prevent proxy connection timeouts.
- **Citation Extraction & Validation:** Parses `[Source N]` references, resolves them to context chunks, deduplicates preserving order of appearance, and flags out-of-bounds references (e.g. `[Source 99]`) as invalid citations.
- **Asynchronous Evaluation Enqueueing:** Non-blocking job enqueueing to Redis/arq queue `neuroflow:queue:evaluations` after generation persistence.

### Quick Start & API Usage

#### 1. Synchronous Generation (`stream=false`)
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is pgvector?",
    "pipeline_id": "<PIPELINE_UUID>",
    "stream": false
  }'
```

#### 2. Streaming Generation via SSE (`stream=true`)
```bash
# Step 1: Initialize generation run
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is pgvector?",
    "pipeline_id": "<PIPELINE_UUID>",
    "stream": true
  }'

# Response: {"run_id": "<RUN_UUID>", "stream_url": "/query/<RUN_UUID>/stream", "status": "running"}

# Step 2: Stream progressive tokens and citation metadata
curl.exe -N "http://localhost:8000/query/<RUN_UUID>/stream"
```

### Running Tests
Inside the Docker container:
```bash
docker compose --env-file .env -f infra/docker-compose.yml exec api python -m pytest tests/ -v
```
All 104 tests pass across provider routing, ingestion, hybrid retrieval, prompt assembly, citation tracking, streaming API, RAGAS metrics, judge evaluation, human calibration, Pearson correlation runner, pipeline configuration schema, versioning, analytics, optimizer, A/B comparison, and fine-tuning extraction/tracking.

---

## Task 7 — Automated Evaluation Framework with LLM-as-Judge and RAGAS Metrics

NeuroFlow implements an automated evaluation framework to assess RAG pipeline outputs, establish judge calibration against human ratings, and curate fine-tuning datasets from high-quality responses.

### Evaluation Metrics
Evaluations are computed concurrently via `asyncio.gather` across four metrics:

| Metric | Target | Formula / Description | Weight |
|---|---|---|---|
| **Faithfulness** | Groundedness | Fraction of claims in the generated answer supported by retrieved context | 0.35 |
| **Answer Relevance** | Intent alignment | Extent to which answer directly and completely addresses the user prompt | 0.30 |
| **Context Precision** | Ranking quality | Relevance of retrieved chunks with higher rank given to top passages | 0.20 |
| **Context Recall** | Information coverage | Fraction of required facts present in the retrieved context passages | 0.15 |

$$\text{Overall Score} = 0.35 \cdot \text{Faithfulness} + 0.30 \cdot \text{Answer Relevance} + 0.20 \cdot \text{Context Precision} + 0.15 \cdot \text{Context Recall}$$

### Automatic Fine-Tuning Pair Promotion
When `overall_score > 0.80`, the `EvaluationJudge` automatically stores the run in the `training_pairs` table with the user query, system prompt, generated assistant response, and quality score for future fine-tuning jobs.

### Human Rating and Calibration Endpoint
Submit human feedback (1 to 5 stars) to calibrate the judge model:
```bash
curl.exe -X PATCH "http://localhost:8000/runs/<RUN_UUID>/rating" \
  -H "Content-Type: application/json" \
  -d '{"rating": 5, "pipeline_id": "<PIPELINE_UUID>"}'
```
Calibration is flagged (`calibration_needed: true`) whenever $|\text{overall\_score} - (\text{rating} / 5.0)| > 0.30$.

### Calibration Benchmark Runner
Run the repeatable 30-sample benchmark to compute Pearson correlation:
```bash
docker exec infra-api-1 python -m evaluation.calibration_runner
```
Outputs are saved to `evaluation/calibration_results.json` showing Pearson correlation ($r \ge 0.85$) and MAE metrics.

---

## Task 8 — Configurable RAG Pipelines, Versioning, A/B Comparison, Analytics, and Optimizer

NeuroFlow provides enterprise-grade pipeline configuration, strict Pydantic v2 schemas, automated version tracking, concurrent A/B testing, comprehensive run analytics with linear-interpolated percentiles, and a rule-based advisory optimizer.

### 1. Pipeline Configuration Schema (`backend/models/pipeline.py`)
All pipelines conform to strict Pydantic validation (`extra='forbid'` at all root and nested levels):

- **`ingestion`**:
  - `chunking_strategy`: `'fixed'`, `'semantic'`, or `'auto'` (default: `'auto'`)
  - `chunk_size_tokens`: 64–2048 (default: 512)
  - `chunk_overlap_tokens`: 0–512, must be strictly less than `chunk_size_tokens` (default: 64)
  - `extractors_enabled`: list of enabled extractors (e.g., `["tables", "metadata"]`)
- **`retrieval`**:
  - `dense_k`: 1–100 (default: 20)
  - `sparse_k`: 1–100 (default: 20)
  - `reranker`: `'cross-encoder'`, `'llm'`, or `'none'` (default: `'cross-encoder'`)
  - `top_k_after_rerank`: 1–50 (default: 5)
  - `query_expansion`: boolean (default: `true`)
  - `metadata_filters_enabled`: boolean (default: `true`)
- **`generation`**:
  - `model_routing`: `'gpt-4o-mini'`, `'gpt-4o'`, `'claude-3-5-sonnet'`, `'claude-3-haiku'`
  - `max_context_tokens`: 512–32768 (default: 4000)
  - `temperature`: 0.0–2.0 (default: 0.7)
  - `system_prompt_variant`: `'default'`, `'concise'`, `'detailed'`, or `'technical'`
- **`evaluation`**:
  - `auto_evaluate`: boolean (default: `true`)
  - `training_threshold`: 0.0–1.0 (default: 0.80)

### 2. Versioning Behavior
- Creating a pipeline (`POST /pipelines`) sets `version = 1` and creates an immutable row in `pipeline_versions`.
- Updating a pipeline (`PATCH /pipelines/{id}`) deep-merges configuration changes, increments `version` by 1, and inserts the snapshot into `pipeline_versions`. Past versions are never mutated.
- Deleting a pipeline (`DELETE /pipelines/{id}`) performs a soft-delete by setting `status = 'archived'`. Historical run records and version history are preserved.
- When queries run, `pipeline_version`, `retrieval_latency_ms`, and calculated token `cost_usd` are recorded on the `pipeline_runs` table.

### 3. REST API Endpoints

#### Pipeline Management
```bash
# Create pipeline
curl.exe -X POST "http://localhost:8000/pipelines" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Customer Support Pipeline",
    "description": "FAQ and technical support search",
    "config": {
      "name": "Customer Support Pipeline",
      "ingestion": {"chunking_strategy": "auto", "chunk_size_tokens": 512, "chunk_overlap_tokens": 64},
      "retrieval": {"dense_k": 20, "sparse_k": 20, "reranker": "cross-encoder", "top_k_after_rerank": 5},
      "generation": {"model_routing": "gpt-4o-mini", "temperature": 0.3, "system_prompt_variant": "concise"},
      "evaluation": {"auto_evaluate": true, "training_threshold": 0.85}
    }
  }'

# List active pipelines with aggregate evaluation scores and latencies
curl.exe "http://localhost:8000/pipelines"

# Get pipeline details and current version
curl.exe "http://localhost:8000/pipelines/<PIPELINE_ID>"

# Update pipeline (creates new version)
curl.exe -X PATCH "http://localhost:8000/pipelines/<PIPELINE_ID>" \
  -H "Content-Type: application/json" \
  -d '{"config_updates": {"retrieval": {"dense_k": 30, "top_k_after_rerank": 10}}}'

# Soft delete (archive)
curl.exe -X DELETE "http://localhost:8000/pipelines/<PIPELINE_ID>"

# Paginated execution history
curl.exe "http://localhost:8000/pipelines/<PIPELINE_ID>/runs?page=1&page_size=20"
```

#### Performance & Usage Analytics
Computes retrieval latency percentiles via standard linear interpolation (`p50`, `p95`, `p99`), average generation latency, average RAGAS evaluation scores, estimated cost per query, and 30-day daily query count buckets:
```bash
curl.exe "http://localhost:8000/pipelines/<PIPELINE_ID>/analytics"
```

#### Rule-Based Optimizer Suggestions
Analyzes the last 20 evaluations and generates advisory recommendations:
```bash
curl.exe "http://localhost:8000/pipelines/<PIPELINE_ID>/suggestions"
```
Rules implemented:
- **Low Context Precision** (< 0.70): Recommends enabling cross-encoder reranker and reducing `dense_k`/`sparse_k`.
- **Low Context Recall** (< 0.70): Recommends increasing `dense_k`/`sparse_k` and `top_k_after_rerank`.
- **Low Answer Relevance** (< 0.65): Recommends enabling query expansion or switching prompt variant.
- **Low Faithfulness** (< 0.70): Recommends reducing temperature ($\le 0.2$) and switching system prompt variant to `detailed` or `technical`.

#### Concurrent A/B Comparison (`POST /pipelines/compare`)
Executes the identical query concurrently across two pipeline configurations using `asyncio.gather`:
```bash
curl.exe -X POST "http://localhost:8000/pipelines/compare" \
  -H "Content-Type: application/json" \
  -d '{
    "pipeline_a_id": "<PIPELINE_A_UUID>",
    "pipeline_b_id": "<PIPELINE_B_UUID>",
    "query": "Compare liability and termination terms across standard contracts."
  }'
```
Returns side-by-side generated outputs, retrieval latencies, total latencies, token usages, estimated costs, citations, and evaluation metric differentials.

---

## Task 9 — Fine-Tuning Pipeline: Training Data Extraction, MLflow Tracking & Model Registration

NeuroFlow provides an end-to-end automated fine-tuning pipeline for RAG models, including strict data quality filtering, PII sanitization, MLflow run and artifact tracking, asynchronous provider job management, ARQ background polling, model registry integration, and DPO preference formatting.

### 1. Training Data Extraction & Validation (`pipelines/finetuning/extractor.py`)
Eligible pairs from `training_pairs` and `pipeline_runs` must satisfy all criteria:
- **Quality Score**: `quality_score >= 0.82`.
- **Not Previously Consumed**: `included_in_job IS NULL`.
- **Human Rating**: `user_rating >= 4` or `user_rating IS NULL` (excluding rejected ratings $\le 2$).
- **Groundedness**: Requires faithfulness $> 0.80$. Missing faithfulness records are re-evaluated using the judge evaluator (`FaithfulnessMetric`), rejecting candidates if the re-evaluated score is $\le 0.80$.
- **Length Boundaries**: Assistant responses must be strictly 50–2000 tokens (calculated using `tiktoken` with `cl100k_base` encoding).
- **Citation Compliance**: Requires at least one citation matching `[Source N]` where $N \ge 1$.
- **PII Protection**: Rejects queries containing email addresses or telephone numbers using regular expressions without logging sensitive data.
- **SFT JSONL Format**: Serialized into OpenAI-compatible conversational JSONL:
  ```json
  {"messages":[{"role":"system","content":"..."},{"role":"user","content":"[Context]\n...\n\n[Question]\n..."},{"role":"assistant","content":"..."}]}
  ```
- **Atomicity**: Training pairs are marked with `included_in_job = job_id` inside a database transaction only after successful file generation and provider submission.

### 2. MLflow Experiment Tracking (`pipelines/finetuning/tracker.py`)
- Each training job creates an MLflow run named `finetune-{job_id}` in the configured tracking server (`MLFLOW_TRACKING_URI`).
- Logs parameters: `base_model`, `accepted_training_pair_count`, `average_quality_score`, `training_data_date_range`.
- Logs the generated JSONL dataset as an artifact under `training_data/`.
- On completion, logs terminal metrics: `training_loss`, `validation_loss`, `trained_tokens`.
- Exposes the MLflow run ID and direct dashboard URL (`/#/experiments/{exp_id}/runs/{run_id}`) via the API.
- Cloud API Note: For remote provider fine-tuning (e.g. OpenAI), weights are hosted remotely by the provider; the provider fine-tuned model identifier is recorded and registered in the NeuroFlow model registry.

### 3. Job Manager & ARQ Polling (`pipelines/finetuning/job_manager.py` & `worker/tasks.py`)
- **Submission Workflow**:
  1. Creates pending database record in `finetune_jobs`.
  2. Extracts and validates dataset (enforcing provider minimum of 10 samples).
  3. Logs dataset artifact and parameters to MLflow.
  4. Uploads dataset file to OpenAI with `purpose="fine-tune"`.
  5. Submits fine-tuning job with base model (e.g., `gpt-4o-mini`).
  6. Associates pairs with job ID in database transaction.
  7. Enqueues background status polling via ARQ deferred by 60 seconds.
- **ARQ Background Polling (`poll_finetune_job`)**:
  - Polls provider job status every 60 seconds.
  - On `succeeded`: records `fine_tuned_model` identifier and tokens, updates MLflow run with metrics, and registers the model in Redis (`router:models`) with `"fine_tuned": true` and `task_type="rag_qa"`.
  - Idempotent: ensures completed jobs are not re-registered.
  - On `failed` or `cancelled`: records error details and terminates MLflow run safely.

### 4. Direct Preference Optimization (DPO) Support
When both a positive response (`rating >= 4`) and a negative response (`rating <= 2`) exist for the same query, preference pairs are created:
```json
{"prompt": "...", "chosen": "...", "rejected": "..."}
```
Excludes PII queries, ensures no duplicate prompts, and outputs to `training_data/{job_id}_dpo.jsonl` compatible with Hugging Face TRL.

### 5. REST API Endpoints

```bash
# 1. Preview eligible training pairs without modifying state
curl.exe "http://localhost:8000/finetune/training-data/preview?limit=5"

# 2. Trigger fine-tuning job submission
curl.exe -X POST "http://localhost:8000/finetune/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "base_model": "gpt-4o-mini",
    "min_quality_score": 0.82,
    "format": "sft"
  }'

# 3. List fine-tuning jobs
curl.exe "http://localhost:8000/finetune/jobs"

# 4. Get fine-tuning job details, metrics, and MLflow run URL
curl.exe "http://localhost:8000/finetune/jobs/<JOB_ID>"
```

## Task 10 — Resilience Engineering

### Overview

The `backend/resilience/` package provides a production-quality fault-tolerance layer:

| Component | File | Purpose |
|---|---|---|
| Circuit Breaker | `circuit_breaker.py` | Prevents cascade failures on external LLM providers |
| Token-Bucket Rate Limiter | `rate_limiter.py` | Redis-backed, cross-worker provider rate coordination |
| Endpoint Rate Limiter | `endpoint_rate_limiter.py` | Sliding-window per-IP limits on `/ingest` and `/query` |
| Backpressure Guard | `backpressure.py` | Queue-depth-aware ingestion control (503 / 202 + warning) |
| Timeout Manager | `timeout_manager.py` | Bounded async operations with adaptive p95 latency tracking |

### Redis Key Schema

| Key Pattern | Owner | Purpose |
|---|---|---|
| `circuit:{name}:state` | CircuitBreaker | `closed`, `open`, or `half_open` |
| `circuit:{name}:failure_count` | CircuitBreaker | Consecutive failure counter |
| `circuit:{name}:opened_at` | CircuitBreaker | UNIX timestamp when circuit opened |
| `rpb:{provider}:tokens` | RateLimiter | Current token count |
| `rpb:{provider}:last_refill` | RateLimiter | Timestamp of last refill |
| `rpb:pipeline:{id}:tokens` | RateLimiter | Per-pipeline token bucket |
| `ratelimit:{route}:{ip}` | EndpointRateLimiter | Sliding-window sorted set |
| `latency:{task_type}` | TimeoutManager | Sorted set of latency observations |
| `timeouts:{task_type}` | TimeoutManager | Counter of timed-out operations |
| `queue:ingest` | BackpressureGuard | ARQ ingestion queue (read via `LLEN`) |

### Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `TRUST_PROXY` | `false` | Trust `X-Forwarded-For` for IP extraction |

Rate limits (code constants, adjustable):
- `/ingest`: 10 requests/hour per IP
- `/query`: 60 requests/minute per IP
- OpenAI global: 3,000 requests capacity, 50 tokens/second refill
- Per-pipeline: `rate_limit_rpm` (default 60)

### Health Endpoint

`GET /health` returns:
```json
{
  "status": "ok | degraded | critical",
  "checks": {
    "postgres": {"status": "healthy", "latency_ms": 2},
    "redis": {"status": "healthy", "latency_ms": 1},
    "mlflow": {"status": "healthy", "latency_ms": 50},
    "circuit_breakers": [{"name": "openai", "state": "closed", "failure_count": 0}],
    "ingest_queue_depth": 12
  }
}
```

Status rules:
- **critical** — PostgreSQL or Redis unreachable
- **degraded** — any circuit breaker OPEN, or MLflow unreachable
- **ok** — all checks healthy
