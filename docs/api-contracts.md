# NeuroFlow REST API Contracts

**Contract version:** `v1` (logical paths shown without a `/v1` prefix to match the Task 31 endpoint list).  
**Format:** JSON request/response unless explicitly noted.  
**Time format:** RFC 3339 UTC timestamps.  
**IDs:** opaque UUIDs unless otherwise stated.

## Common conventions

### Authentication and authorization

- User-facing endpoints require `Authorization: Bearer <access_token>`, except `GET /health` and `GET /metrics`, which are infrastructure endpoints and must be protected by network policy or operations authentication in production.
- Every resource lookup is tenant-scoped. A valid ID does not grant access to another tenant's resource.
- Service-to-service workers use short-lived service credentials. Never put provider API keys in request bodies.

### Common error body

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "The request is invalid.",
    "request_id": "req_01J...",
    "details": [{"field": "top_k", "reason": "must be between 1 and 100"}]
  }
}
```

### Error status meanings

| HTTP | Code examples | Meaning |
|---|---|---|
| 400 | `VALIDATION_ERROR`, `INVALID_JSON` | Malformed body or invalid field |
| 401 | `UNAUTHENTICATED` | Missing/invalid access token |
| 403 | `FORBIDDEN` | Authenticated but not authorized |
| 404 | `NOT_FOUND` | Resource absent or not visible to caller |
| 409 | `CONFLICT`, `IDEMPOTENCY_CONFLICT` | State conflict or reused key with different payload |
| 413 | `PAYLOAD_TOO_LARGE` | Upload/body exceeds configured limit |
| 415 | `UNSUPPORTED_MEDIA_TYPE` | Unsupported file/content type |
| 422 | `UNPROCESSABLE_INPUT` | Input cannot be parsed or processed |
| 429 | `RATE_LIMITED` | Rate limit exceeded; include `Retry-After` |
| 500 | `INTERNAL_ERROR` | Unexpected server failure |
| 502 | `UPSTREAM_ERROR` | Provider returned an invalid/unavailable response |
| 503 | `SERVICE_UNAVAILABLE` | Required dependency unavailable |
| 504 | `UPSTREAM_TIMEOUT` | Upstream operation timed out |

Rate limits below are initial per-tenant defaults; implementation may configure them by plan. Responses that are rate limited include `Retry-After` and standard rate-limit headers. Use `Idempotency-Key` on retryable write requests where indicated.

---

## 1. `POST /ingest`

**Purpose:** Accept a file or URL and enqueue asynchronous ingestion.  
**Authentication:** Required.  
**Rate limit:** 20 requests/minute/tenant; separately enforce file bytes, page/row counts, URL fetch limits, and concurrent jobs.  
**Headers:** `Content-Type: multipart/form-data` for file uploads or `application/json` for URL ingestion; optional `Idempotency-Key`.

### JSON request (URL source)

```json
{
  "source": {
    "type": "url",
    "url": "https://example.org/guide"
  },
  "collection_id": "8d4a2a90-95e0-45bc-8d6b-c9d6dff2f2bd",
  "metadata": {"domain": "product-docs", "language": "en"},
  "chunking_profile": "default",
  "embedding_profile": "default"
}
```

For file upload use multipart fields: `file` (binary, required), `collection_id` (UUID, required), `metadata` (JSON object, optional), `chunking_profile` (string, optional), and `embedding_profile` (string, optional).

### Request schema

- `source.type`: `url` for JSON mode. For multipart mode, source is the uploaded `file`.
- `source.url`: absolute HTTP(S) URL; required for URL mode.
- `collection_id`: UUID; required.
- `metadata`: object with string/number/boolean values; optional; server-managed fields cannot be overridden.
- `chunking_profile`: configured profile ID/name; defaults to `default`.
- `embedding_profile`: configured profile ID/name; defaults to `default`.

### `202 Accepted` response

```json
{
  "job_id": "2a6c5fd7-18e1-41f7-9f07-82e8f9b0a7f8",
  "document_id": "52d64e54-98c1-44df-bb0e-9408ed75f4e7",
  "status": "queued",
  "status_url": "/ingest/2a6c5fd7-18e1-41f7-9f07-82e8f9b0a7f8",
  "created_at": "2026-10-22T12:00:00Z"
}
```

**Errors:** `400`, `401`, `403`, `409`, `413`, `415`, `422`, `429`, `500`, `503`. URL safety violations return `422 UNPROCESSABLE_INPUT`; do not reveal internal network details.

---

## 2. `POST /query`

**Purpose:** Run hybrid retrieval and start an LLM generation.  
**Authentication:** Required.  
**Rate limit:** 30 requests/minute/tenant, plus model-specific token/cost budgets.  
**Headers:** `Idempotency-Key` optional.

### Request

```json
{
  "question": "How does document versioning work?",
  "collection_ids": ["8d4a2a90-95e0-45bc-8d6b-c9d6dff2f2bd"],
  "filters": {"domain": "product-docs"},
  "top_k": 8,
  "max_context_tokens": 6000,
  "response_mode": "stream",
  "routing": {
    "latency": "balanced",
    "cost": "balanced",
    "capability": "general"
  },
  "metadata": {"client": "web"}
}
```

### Request schema

- `question`: string, 1–20,000 characters.
- `collection_ids`: array of UUIDs; at least one authorized collection.
- `filters`: optional JSON object of supported metadata filters; arbitrary SQL is not accepted.
- `top_k`: integer 1–20; default 8.
- `max_context_tokens`: integer 500–20,000; bounded by model and tenant policy.
- `response_mode`: `stream` or `complete`; default `stream`.
- `routing.latency`: `fast`, `balanced`, or `quality`; default `balanced`.
- `routing.cost`: `low`, `balanced`, or `unrestricted`; default `balanced`.
- `routing.capability`: `general`, `reasoning`, `coding`, `long_context`, or `structured_output`; default `general`.
- `metadata`: optional client metadata; size-limited and not trusted for authorization.

### `202 Accepted` response

```json
{
  "query_id": "9f918e88-8e92-4c82-b2c2-7a6cce8db124",
  "status": "queued",
  "stream_url": "/query/9f918e88-8e92-4c82-b2c2-7a6cce8db124/stream",
  "created_at": "2026-10-22T12:00:00Z"
}
```

For `response_mode: "complete"`, the service may wait up to its configured synchronous deadline and return `200`:

```json
{
  "query_id": "9f918e88-8e92-4c82-b2c2-7a6cce8db124",
  "status": "completed",
  "answer": "Document versions are indexed independently...",
  "citations": [
    {
      "chunk_id": "d8b0c7d2-5e51-47d4-b1e5-81c8c1b6c7c1",
      "document_id": "52d64e54-98c1-44df-bb0e-9408ed75f4e7",
      "source": "https://example.org/guide",
      "page": 3,
      "quote": "Each indexed version..."
    }
  ],
  "model": {"provider": "configured-provider", "model_id": "model-id"},
  "usage": {"input_tokens": 2100, "output_tokens": 180, "estimated_cost_usd": 0.004},
  "evaluation_status": "queued"
}
```

**Errors:** `400`, `401`, `403`, `404`, `409`, `422`, `429`, `500`, `502`, `503`, `504`. If generation has already started, provider failures are represented as an SSE error event on the stream and persisted in the trace.

---

## 3. `GET /query/{query_id}/stream`

**Purpose:** Stream generation events for a query using Server-Sent Events.  
**Authentication:** Required; bearer token or short-lived, query-scoped stream token for browser EventSource clients.  
**Rate limit:** 60 stream connections/minute/tenant; concurrent stream cap per tenant.  
**Response content type:** `text/event-stream`; no JSON envelope around SSE frames.

### Event contract

```text
event: meta
data: {"query_id":"9f918e88-8e92-4c82-b2c2-7a6cce8db124","status":"running"}

event: token
data: {"text":"Document "}

event: token
data: {"text":"versions..."}

event: citations
data: {"citations":[{"chunk_id":"d8b0c7d2-5e51-47d4-b1e5-81c8c1b6c7c1","document_id":"52d64e54-98c1-44df-bb0e-9408ed75f4e7","source":"https://example.org/guide","page":3}]}

event: done
data: {"status":"completed","usage":{"input_tokens":2100,"output_tokens":180},"evaluation_status":"queued"}
```

Failure example:

```text
event: error
data: {"code":"UPSTREAM_TIMEOUT","message":"Generation timed out.","retryable":true}
```

### Event schema

- `meta`: query ID and current state.
- `token`: text delta; concatenate in event order.
- `citations`: final citation list; chunk access must be authorized.
- `done`: terminal success event; includes usage and evaluation state.
- `error`: terminal or recoverable error, with safe message and retryability.
- Send periodic SSE comments as heartbeats. Clients reconnecting with `Last-Event-ID` should resume only if event replay is supported; otherwise return a documented conflict rather than duplicating tokens.

**Errors before stream establishment:** `401`, `403`, `404`, `409`, `429`, `500`, `503`. After headers are sent, use an `error` event.

---

## 4. `GET /evaluations`

**Purpose:** List evaluation records with pagination and optional filters.  
**Authentication:** Required; evaluator/admin permission for cross-user records.  
**Rate limit:** 60 requests/minute/tenant.

### Query parameters

`limit` integer 1–100 (default 20); `cursor` opaque optional cursor; `query_id` UUID optional; `model_id` string optional; `status` one of `queued`, `running`, `succeeded`, `failed`; `created_after` and `created_before` RFC 3339 timestamps optional.

### `200 OK` response

```json
{
  "items": [
    {
      "evaluation_id": "a7f2029a-fd82-4c83-a4db-75c3f0edb4f7",
      "query_id": "9f918e88-8e92-4c82-b2c2-7a6cce8db124",
      "status": "succeeded",
      "scores": {
        "faithfulness": 0.93,
        "answer_relevance": 0.91,
        "context_precision": 0.84,
        "context_recall": 0.88
      },
      "evaluator_version": "judge-v1",
      "rubric_version": "1",
      "created_at": "2026-10-22T12:00:05Z"
    }
  ],
  "next_cursor": null
}
```

`status` may be `queued`, `running`, `succeeded`, or `failed`; `scores` is null until a successful evaluation. Each score is in `[0,1]`.

**Errors:** `400`, `401`, `403`, `429`, `500`.

---

## 5. `GET /evaluations/aggregate`

**Purpose:** Return rolling quality metrics.  
**Authentication:** Required; admin/evaluator role for tenant-wide aggregates.  
**Rate limit:** 30 requests/minute/tenant.

### Query parameters

- `window`: `1h`, `24h`, `7d`, or `30d` (default `7d`).
- `group_by`: optional `model_id`, `domain`, `pipeline_id`, or `prompt_version`.
- `created_before`: optional RFC 3339 cutoff for reproducible reports.

### `200 OK` response

```json
{
  "window": "7d",
  "from": "2026-10-15T12:00:00Z",
  "to": "2026-10-22T12:00:00Z",
  "sample_count": 1240,
  "evaluated_count": 1198,
  "metrics": {
    "faithfulness": {"mean": 0.91, "p50": 0.94, "p95": 0.99},
    "answer_relevance": {"mean": 0.89, "p50": 0.91, "p95": 0.98},
    "context_precision": {"mean": 0.82, "p50": 0.85, "p95": 0.96},
    "context_recall": {"mean": 0.86, "p50": 0.88, "p95": 0.97}
  },
  "group_by": null,
  "groups": [],
  "updated_at": "2026-10-22T12:00:05Z"
}
```

Metrics are illustrative response values, not measured NeuroFlow results. Aggregates exclude failed/pending scores and report sample counts to avoid hiding low sample sizes.

**Errors:** `400`, `401`, `403`, `429`, `500`, `503`.

---

## 6. `POST /pipelines`

**Purpose:** Create a named pipeline configuration.  
**Authentication:** Required; pipeline-editor role.  
**Rate limit:** 20 requests/minute/tenant.  
**Headers:** `Idempotency-Key` recommended.

### Request

```json
{
  "name": "support-docs-default",
  "description": "Support documentation RAG pipeline",
  "config": {
    "chunking_profile": "default",
    "embedding_profile": "default",
    "dense_candidates": 40,
    "lexical_candidates": 40,
    "rrf_k": 60,
    "reranker_profile": "default",
    "context_token_budget": 6000,
    "evaluation_profile": "default"
  },
  "enabled": true
}
```

### Schema

- `name`: 1–100 characters; unique within tenant.
- `description`: optional, max 2,000 characters.
- `config`: object validated against versioned pipeline schema; unknown executable fields rejected.
- `enabled`: boolean, default `true`.

### `201 Created` response

```json
{
  "pipeline_id": "7d1cf1d4-4f68-41aa-8d18-40590bc1fd7a",
  "name": "support-docs-default",
  "version": 1,
  "config": {
    "chunking_profile": "default",
    "embedding_profile": "default",
    "dense_candidates": 40,
    "lexical_candidates": 40,
    "rrf_k": 60,
    "reranker_profile": "default",
    "context_token_budget": 6000,
    "evaluation_profile": "default"
  },
  "enabled": true,
  "created_at": "2026-10-22T12:00:00Z"
}
```

**Errors:** `400`, `401`, `403`, `409`, `422`, `429`, `500`.

---

## 7. `GET /pipelines/{id}/runs`

**Purpose:** List pipeline execution history.  
**Authentication:** Required; pipeline viewer/editor role.  
**Rate limit:** 60 requests/minute/tenant.

### Query parameters

`limit` integer 1–100 (default 20); `cursor` optional; `status` optional `queued`, `running`, `succeeded`, `failed`, `cancelled`; `created_after` optional timestamp.

### `200 OK` response

```json
{
  "pipeline_id": "7d1cf1d4-4f68-41aa-8d18-40590bc1fd7a",
  "items": [
    {
      "run_id": "c26d8ed3-8ec8-4e53-baa7-a35bff96f03c",
      "status": "succeeded",
      "started_at": "2026-10-22T11:55:00Z",
      "finished_at": "2026-10-22T11:55:08Z",
      "input_count": 12,
      "success_count": 12,
      "failure_count": 0,
      "metrics": {"duration_ms": 8000}
    }
  ],
  "next_cursor": null
}
```

**Errors:** `400`, `401`, `403`, `404`, `429`, `500`.

---

## 8. `POST /finetune/jobs`

**Purpose:** Validate a dataset and submit a fine-tuning job.  
**Authentication:** Required; fine-tuning-admin role.  
**Rate limit:** 5 requests/hour/tenant; provider-specific quotas and spend limits also apply.  
**Headers:** `Idempotency-Key` required.

### Request

```json
{
  "dataset_id": "0b32a5f6-a7d4-4c80-8bb2-3f6d9b7c0d41",
  "base_model_id": "configured-base-model",
  "experiment_name": "support-answer-v1",
  "hyperparameters": {
    "epochs": 3,
    "learning_rate_multiplier": 0.1
  },
  "promotion_policy": {
    "min_faithfulness": 0.9,
    "min_answer_relevance": 0.85,
    "max_regression_vs_base": 0.02
  }
}
```

### Schema

- `dataset_id`: UUID of a validated, authorized dataset with manifest/hash.
- `base_model_id`: configured provider model identifier; arbitrary provider endpoints are not accepted.
- `experiment_name`: 1–120 characters.
- `hyperparameters`: provider-supported allowlisted fields only.
- `promotion_policy`: predeclared evaluation gates; server policy may enforce stricter limits.

### `202 Accepted` response

```json
{
  "job_id": "de8cb6e1-2b88-4bb2-a48b-90f9d3fd79cc",
  "status": "queued",
  "experiment_name": "support-answer-v1",
  "created_at": "2026-10-22T12:00:00Z"
}
```

**Errors:** `400`, `401`, `403`, `404`, `409`, `422`, `429`, `500`, `502`, `503`.

---

## 9. `GET /finetune/jobs/{id}`

**Purpose:** Read fine-tuning job state and available metrics.  
**Authentication:** Required; fine-tuning viewer/admin role.  
**Rate limit:** 60 requests/minute/tenant.

### `200 OK` response

```json
{
  "job_id": "de8cb6e1-2b88-4bb2-a48b-90f9d3fd79cc",
  "status": "running",
  "base_model_id": "configured-base-model",
  "fine_tuned_model_id": null,
  "provider_job_id": "provider-job-reference",
  "experiment_name": "support-answer-v1",
  "metrics": {},
  "promotion_status": "not_evaluated",
  "failure": null,
  "created_at": "2026-10-22T12:00:00Z",
  "updated_at": "2026-10-22T12:10:00Z"
}
```

- `status`: `queued`, `validating`, `running`, `succeeded`, `failed`, `cancelled`.
- `metrics`: object containing provider metrics and NeuroFlow offline-evaluation metrics when available.
- `promotion_status`: `not_evaluated`, `rejected`, `canary`, `promoted`, or `rolled_back`.
- `failure`: null or safe `{ "code": "...", "message": "..." }`.

**Errors:** `401`, `403`, `404`, `429`, `500`, `503`.

---

## 10. `GET /health`

**Purpose:** Liveness check; process can respond.  
**Authentication:** None for internal liveness probe; restrict network exposure.  
**Rate limit:** 120 requests/minute/IP.

### `200 OK`

```json
{"status": "ok"}
```

Do not include secrets, dependency credentials, or detailed infrastructure data. A separate readiness probe may be added to check critical dependencies.

**Errors:** `429`, `500`.

## 11. `GET /metrics`

**Purpose:** Prometheus-compatible operational metrics.  
**Authentication:** Infrastructure-level authentication or private-network restriction; never expose publicly without controls.  
**Rate limit:** 60 requests/minute/scraper.

**Response:** `200 OK`, `Content-Type: text/plain; version=0.0.4; charset=utf-8`, with Prometheus exposition format, for example:

```text
neuroflow_http_requests_total{method="POST",route="/query",status="202"} 42
neuroflow_ingestion_documents_total{status="succeeded"} 37
neuroflow_queue_depth{queue="evaluation"} 4
```

Labels must avoid user IDs, raw queries, document names, or other high-cardinality/sensitive values.

**Errors:** `401`/`403` if metrics auth is enabled, `429`, `500`, `503`.

## 12. `POST /pipelines/{pipeline_id}/suggestions`

**Purpose:** Persist a quality anomaly reported by the trusted evaluation worker
and return rule-based optimization suggestions generated from that pipeline's
persisted evaluation history.

**Authentication:** Dedicated `Authorization: Bearer <OBSERVABILITY_SERVICE_TOKEN>`
service credential. The endpoint returns `503` when the service credential is
not configured and `401` for a missing or invalid credential.

**Request:**

```json
{
  "run_id": "6c2c56fd-57d0-4b43-8f2b-2eef1e257a33",
  "latest_score": 0.42,
  "baseline_mean": 0.81,
  "baseline_stddev": 0.12
}
```

Scores must be between 0 and 1; standard deviation must be non-negative. The
run must belong to the requested pipeline. A repeated run/rule request is
deduplicated; additional events for a pipeline/rule are suppressed during the
24-hour cooldown. The response includes `recorded` and
`cooldown_suppressed` flags and the persisted advisory suggestions.

The anomaly event is tenant-scoped through PostgreSQL row-level security.
Migration `011_quality_anomalies.sql` must be applied before enabling this
endpoint.

---

## Contract-wide requirements

- Use UTC timestamps and stable opaque IDs.
- Return `request_id` in errors and structured logs.
- Never return stack traces, provider secrets, SQL, or internal network addresses.
- Pagination must be deterministic and tenant-scoped.
- Validate all enums, ranges, lengths, and nested objects.
- Document future breaking changes through API versioning and contract tests.
- Streaming completion is distinct from evaluation completion; `done` means generation finished, not that evaluation has finished.
