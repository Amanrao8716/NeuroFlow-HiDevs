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

Task 31: architecture and API contracts. Subsequent tasks implement and validate the system incrementally.
