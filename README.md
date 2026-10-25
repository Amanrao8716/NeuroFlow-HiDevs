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
All 59 tests pass across provider routing, ingestion, hybrid retrieval, prompt assembly, citation tracking, streaming API, RAGAS metrics, judge evaluation, human calibration, and Pearson correlation runner.

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
