# TASK 6 Implementation Plan: RAG Generation Pipeline, Streaming SSE & Citation Tracking

## Status Overview
- Branch: `task-06`
- Base: Task 5 hybrid retrieval + reranker + context assembly (`2c52cb7`)
- Goal: End-to-end RAG Generation Pipeline with dynamic prompt assembly, token streaming, Server-Sent Events (SSE), citation tracking, pipeline run logging, and async evaluation enqueueing.

---

## Phases & Progress Checklist

- [x] Phase 1: Inspect repository architecture, Git state, providers, retrieval, context assembly, database schema, and Redis integration.
  - *Verification*: Verified Git branch `task-06`, clean status, inspected `providers/client.py`, `providers/base.py`, `worker/tasks.py`, `backend/db/migrations/`, `pipeline_runs` table, `pipelines/retrieval/context_assembler.py`, and `sse-starlette` availability.

- [x] Phase 2: Define generation models, database schema updates, and interfaces.
  - *Verification*: Created migration `008_pipeline_runs_prompt_citations.sql` (adding `prompt TEXT`, `citations JSONB`, `error_message TEXT` to `pipeline_runs`). Applied to running PostgreSQL database and updated `backend/db/migrations/001_schema.sql` and `infra/init/001_schema.sql` consistently. Added `sse-starlette>=2.0.0` to `backend/requirements.txt`.

- [x] Phase 3: Implement dynamic prompt assembly (`pipelines/generation/prompt_builder.py`).
  - *Verification*: Implemented query-type conditioned system instructions (Factual, Analytical, Comparative, Procedural, Default fallback). Formatted context safely as `<context>[Source N — doc, page M]\n...\n</context>`. Implemented deterministic source map creation. Verified via `backend/tests/test_generation_prompt_builder.py` (3 tests passed).

- [x] Phase 4: Implement citation parsing and validation (`pipelines/generation/citations.py`).
  - *Verification*: Implemented typed citation data structures (`Citation`, `ParsedCitations`). Regex extraction of `[Source N]`. Source map resolution. Deduplication preserving order of first appearance. Flagging invalid citations (e.g., `[Source 99]`). Verified via `backend/tests/test_generation_citations.py` (4 tests passed).

- [x] Phase 5: Implement generation orchestration and pipeline run logging (`pipelines/generation/generator.py`).
  - *Verification*: Run record created BEFORE LLM call with status `'running'`. Orchestrates retrieval -> prompt building -> streaming LLM call -> token accumulation -> citation extraction -> DB update (`completed` or `failed`). Verified via `backend/tests/test_generation_pipeline.py` (2 tests passed) and live database inspection.

- [x] Phase 6: Implement progressive SSE streaming and keepalives (`backend/api/query.py`).
  - *Verification*: `GET /query/{run_id}/stream` SSE endpoint emitting `retrieval_start`, `retrieval_complete`, progressive `token`, and `done` (with citations). 15s ping/keepalive configured with `ping=15` in `EventSourceResponse`. Disconnect and error events handled safely.

- [x] Phase 7: Integrate POST /query endpoint with existing retrieval and pipeline configuration.
  - *Verification*: `POST /query` supporting `stream=false` (synchronous complete JSON) and `stream=true` (creates run, registers stream info, returns `{run_id, stream_url}`). Pydantic validation for pipeline_id, query string, top_k, max_context_tokens. Verified via `backend/tests/test_query_api_streaming.py` (2 tests passed).

- [x] Phase 8: Implement asynchronous evaluation job enqueueing.
  - *Verification*: After run persistence succeeds, non-blocking asynchronous enqueueing pushes job payload into `neuroflow:queue:evaluations` via Redis `rpush` with timeout protection. Queue failures safely caught and logged without failing generation. Tested and verified directly against live Redis.

- [x] Phase 9: Add unit, integration, and streaming tests.
  - *Verification*: Created `backend/tests/test_generation_prompt_builder.py`, `backend/tests/test_generation_citations.py`, `backend/tests/test_generation_pipeline.py`, `backend/tests/test_query_api_streaming.py`. All 11 new tests pass.

- [x] Phase 10: Run full test suite, regression verification, and Docker validation.
  - *Verification*: Ran full test suite inside Docker container: 42 passed in 2.70s across Task 3, Task 4, Task 5, and Task 6. Zero regressions. Verified live streaming and database updates on running Docker stack.

- [x] Phase 11: Update documentation and prepare Task 6 submission.
  - *Verification*: Updated `README.md` with Task 6 features, synchronous and streaming SSE API curl examples, test commands, and architecture details. Checked `git diff --check`.
