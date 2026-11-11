# NeuroFlow Python SDK

Async Python client for the NeuroFlow RAG API.

## Installation

```bash
pip install ./sdk
```

## Quick Start

```python
import asyncio
from neuroflow import NeuroFlowClient

async def main():
    async with NeuroFlowClient(
        "http://localhost:8000",
        client_id="my-app",
        client_secret="",
    ) as client:
        # Ingest a document
        doc = await client.ingest_file("report.pdf", pipeline_id="<uuid>")
        print(f"Ingested: {doc.document_id}, chunks: {doc.chunk_count}")

        # Query the pipeline
        result = await client.query("What is NeuroFlow?", pipeline_id="<uuid>")
        print(result.answer)

asyncio.run(main())
```

## Authentication

The SDK supports two authentication modes:

1. **Client credentials** (recommended): Pass `client_id` and `client_secret` — the SDK
   automatically exchanges them for a JWT token via `POST /auth/token`.

2. **Pre-existing token**: Pass `token="<jwt>"` directly.

## Methods

### `ingest_file(file_path, pipeline_id=None)`

Upload a local file (PDF, DOCX, image, CSV) and wait for processing to complete.

```python
doc = await client.ingest_file("data.pdf", pipeline_id="...")
print(doc.status, doc.chunk_count)
```

### `ingest_url(url, pipeline_id=None)`

Submit a public URL for web scraping and ingestion.

```python
doc = await client.ingest_url("https://example.com/page", pipeline_id="...")
```

### `query(query, pipeline_id, stream=False)`

Execute a RAG query. Returns `QueryResult` when `stream=False`.
When `stream=True`, supports direct token-by-token streaming via `async for`.

```python
# Synchronous response
result = await client.query("Explain X", pipeline_id="...")
print(result.answer, result.citations)

# Direct async streaming
async for token in client.query("Explain X", pipeline_id="...", stream=True):
    print(token, end="", flush=True)
```

### `get_evaluation(run_id, wait=True)`

Retrieve LLM-as-a-judge evaluation scores for a pipeline run.

```python
ev = await client.get_evaluation(run_id="...", wait=True)
print(ev.overall_score, ev.faithfulness)
```

### `list_pipelines()`

List all active pipelines.

```python
pipelines = await client.list_pipelines()
for p in pipelines:
    print(p.name, p.version)
```

### `create_pipeline(config)`

Create a new pipeline configuration. Accepts a configuration dictionary (mapping to `PipelineCreateRequest`) or a pipeline name string.

```python
pipeline = await client.create_pipeline({
    "name": "support-rag",
    "description": "Customer support pipeline",
    "retrieval": {
        "dense_k": 20,
        "sparse_k": 15,
        "rrf_k": 60,
    },
    "generation": {
        "model_routing": "gpt-4o-mini",
    },
})
pipeline = await client.create_pipeline("my-pipeline", description="Test")
```

## Error Handling

```python
from neuroflow.exceptions import (
    NeuroFlowError,      # Base exception
    AuthenticationError,  # 401/403
    RateLimitError,       # 429 (after retries exhausted)
    IngestionError,       # Ingestion failures
    TimeoutError,         # Polling timeout
)
```

The SDK automatically retries HTTP 429 responses up to 3 times, honoring the
`Retry-After` header. File uploads are never retried.

## Configuration

| Parameter       | Default | Description                           |
|----------------|---------|---------------------------------------|
| `timeout`      | 30.0    | HTTP request timeout (seconds)        |
| `poll_interval`| 2.0     | Polling interval for async ops        |
| `poll_timeout` | 300.0   | Max wait for polling operations       |
| `max_retries`  | 3       | Max 429 retry attempts                |
