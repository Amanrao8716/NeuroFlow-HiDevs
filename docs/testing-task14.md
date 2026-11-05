# Task 14 test and benchmark guide

## Isolated API integration tests

The suite's default command is `pytest tests/ -v`. Integration cases skip with a reason unless `NEUROFLOW_INTEGRATION=true`; they never start or mutate a developer stack automatically. Configure an isolated deployment and test-only client with query, ingest, and admin scopes, then set these process environment variables without putting credentials in source control:

```powershell
$env:NEUROFLOW_INTEGRATION = 'true'
$env:NEUROFLOW_TEST_BASE_URL = 'http://127.0.0.1:8000'
$env:NEUROFLOW_TEST_CLIENT_ID = '<test-client-id>'
$env:NEUROFLOW_TEST_CLIENT_SECRET = '<test-client-secret>'
$env:NEUROFLOW_TEST_DATABASE_URL = '<isolated-test-database-url>'
$env:NEUROFLOW_TEST_PIPELINE_ID = '<isolated-pipeline-uuid>'
$env:NEUROFLOW_TEST_FAULT_INJECTION_URL = 'http://127.0.0.1:8765/__test__'
$env:NEUROFLOW_TEST_CONFIRM_ISOLATED = 'I_CONFIRM_THIS_IS_AN_ISOLATED_TEST_DEPLOYMENT'
$env:NEUROFLOW_TEST_CONFIRM_PROVIDER = 'I_CONFIRM_TEST_ONLY_LLM_PROVIDER'
$env:NEUROFLOW_TEST_COMPOSE_PROJECT = 'neuroflow-test-task14'
$env:NEUROFLOW_TEST_CONFIRM_PROJECT = 'neuroflow-test-task14'
pytest tests/integration -v
```

Before authenticating, the integration fixture verifies the explicitly confirmed running Compose project, required service ownership/status, loopback API port publication, and project-owned `postgres_data` volume using read-only Docker inventory. It skips if any check fails. The integration client only contacts loopback or a dedicated `.test` hostname. Confirm the API and worker point to an approved test-only LLM provider before setting the provider confirmation. The fine-tuning test only accepts a PostgreSQL URL using loopback, the Compose `postgres` service, or a dedicated `.test` hostname. The test database must be isolated and disposable.

Run the included OpenAI-compatible mock in a separate test-only terminal and point both the API and worker `OPENAI_BASE_URL` at it (use a host reachable from their containers). Set `OPENAI_API_KEY` to any test-only value. For the fine-tuning check, set `NEUROFLOW_TEST_JSONL_CAPTURE` in the mock process to a temporary path shared with the test runner. The mock's failure-control endpoint accepts the test request to schedule five HTTP 500 responses. Never expose the unauthenticated mock outside the isolated test network.

```powershell
$env:NEUROFLOW_MOCK_PROVIDER_HOST = '0.0.0.0'
$env:NEUROFLOW_TEST_JSONL_CAPTURE = (Join-Path $env:TEMP 'task14-captured.jsonl')
py -m tests.support.mock_openai
```

The A/B test creates and archives two pipelines with distinct `top_k_after_rerank` values. The full RAG test polls bounded ingestion and evaluation endpoints. The breaker test additionally requires `NEUROFLOW_TEST_FAULT_INJECTION_URL` to point to the mock's control endpoint and a short `NEUROFLOW_TEST_BREAKER_RECOVERY_SECONDS` matching the test deployment's breaker configuration. The fine-tuning test inserts and removes 15 valid quality 0.9 training pairs using the isolated database URL, requires `NEUROFLOW_TEST_FINETUNE_PIPELINE_ID`, and checks the mock's captured JSONL. Use a fresh test database and provider adapter; do not point these variables at production or developer data.

To run the existing regression suite, including Task 12 observability and Task 13 security tests:

```powershell
$env:PYTHONPATH = (Resolve-Path -LiteralPath backend).Path
py -m pytest backend/tests/ -v -p no:cacheprovider --basetemp=backend/.pytest-task14-temp
```

## Locust load test

The backend has no client-creation API. A dedicated Locust client must first be provisioned in the isolated API's `NEUROFLOW_AUTH_CLIENTS` registry with `query` and `ingest` scopes (or `admin`, which grants both); store its plaintext secret in an approved test secret store and the corresponding `hash_client_secret()` scrypt hash in the API's test deployment configuration, then restart that API. Do not use a production client's secret. Pipeline creation is available through `POST /pipelines` and requires a test admin client.

Install Locust in the test environment (`py -m pip install locust`). Set the test API URL and test client ID/secret from the test secret store, plus `NEUROFLOW_TEST_CONFIRM_ISOLATED=I_CONFIRM_THIS_IS_AN_ISOLATED_TEST_DEPLOYMENT`. After independently verifying that the isolated API and worker use the approved test-only LLM provider, also set `NEUROFLOW_TEST_CONFIRM_PROVIDER=I_CONFIRM_TEST_ONLY_LLM_PROVIDER`. Set `NEUROFLOW_TEST_COMPOSE_PROJECT` to the exact dedicated project name (matching `neuroflow-test-*`) and `NEUROFLOW_TEST_CONFIRM_PROJECT` to that same name. The URL must use loopback or a dedicated `.test` hostname. Before any HTTP request, the runner performs read-only Docker inventory checks: the named Compose project must be running; `api`, `postgres`, `redis`, `worker`, and `mlflow` containers must belong to and be running in that project; its API container must publish the configured URL port on loopback; and exactly one project-owned `postgres_data` volume must exist. The runner then authenticates the client, checks `/health`, checks pipeline read access and configuration, and only accepts an active pipeline named `load-test-task14`. If `NEUROFLOW_TEST_PIPELINE_ID` is not supplied, the runner needs externally provisioned `NEUROFLOW_TEST_ADMIN_CLIENT_ID` and `NEUROFLOW_TEST_ADMIN_CLIENT_SECRET`; it reuses that reserved pipeline or creates it through the API and uses the returned ID. This pipeline is retained after the run. No IDs or credentials are guessed or printed.

Run the orchestrator rather than invoking Locust directly. It first makes an authenticated representative query and requires a nonempty answer grounded in at least one indexed chunk. It then runs a 1-user, 10-second Locust smoke test and only starts the required 50-user, 5-minute test if both smoke checks succeed:

```powershell
$env:NEUROFLOW_TEST_CONFIRM_ISOLATED = 'I_CONFIRM_THIS_IS_AN_ISOLATED_TEST_DEPLOYMENT'
$env:NEUROFLOW_TEST_COMPOSE_PROJECT = 'neuroflow-test-task14'
$env:NEUROFLOW_TEST_CONFIRM_PROJECT = 'neuroflow-test-task14'
py -m tests.performance.run_locust_test
```

Provide `NEUROFLOW_TEST_BASE_URL`, `NEUROFLOW_TEST_CLIENT_ID`, and `NEUROFLOW_TEST_CLIENT_SECRET` in the process environment; do not save them in this repository. The mock or otherwise approved test-only LLM provider must be configured on the isolated API and confirmed with the provider flag above. The scenario pacing stays under the service's default per-IP limits (60 query/minute and 10 ingest/hour) for one Locust source IP. The runner writes actual smoke and load CSVs and updates `load_test_results.json` with request/failure counts, error rate, throughput, latency percentiles, and min/average/max response time where Locust supplies them. Current status is blocked because the API test URL, dedicated client credentials, suitable test pipeline or test admin provisioning, exact dedicated Compose project, isolation confirmation, and verified provider confirmation are not configured in this workspace. Docker showed a running Compose project named `infra` and persistent volumes; that is not an approved `neuroflow-test-*` project. Detailed container ownership and API port checks could not be completed, so the runner would fail closed before making an API request. No services were started and no API requests or Locust traffic were sent. Targets: query P95 below 5 seconds and total error rate below 2%.

## Retrieval benchmark

The repository's 30 calibration examples do not include indexed chunk IDs or independently annotated relevance labels. In the isolated API deployment, set `NEUROFLOW_TEST_BENCHMARK_ENDPOINT=true` to enable the admin-only `POST /__test__/retrieval-rankings` endpoint. It uses the production dense and sparse retrieval functions, RRF implementation, and cross-encoder reranker, and returns only the actual ordered chunk IDs. Keep the flag off outside an isolated benchmark deployment. Configure that API and worker with the approved test-only LLM provider, and set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` with the required local embedding and cross-encoder models cached before running; this prevents model downloads during collection.

Use [retrieval_questions.template.json](../tests/benchmarks/retrieval_questions.template.json) as a shape-only template. Its single row is deliberately incomplete and contains no benchmark question or chunk label; replace it and expand to exactly 50 uniquely identified questions with independently assigned IDs of real relevant chunks. The template is invalid input and cannot produce scores.

Prepare an input JSON file of exactly 50 rows with unique `question_id`, unique `question`, and independently assigned `relevant_chunk_ids`. The exporter checks this input, checks explicit isolation/provider and exact Compose project confirmations, restricts the target to loopback or `.test`, and verifies the named running Compose project, required running services, loopback-published API port, and project-owned database volume before any API request. It then authenticates a test admin client and verifies that the selected active pipeline is named `retrieval-benchmark-task14` before calling the opt-in endpoint. The test corpus and ground truth must already be seeded in that isolated pipeline. It refuses to overwrite output files, validates all four actual ranking lists, then writes an enriched dataset:

```powershell
$env:NEUROFLOW_BENCHMARK_BASE_URL = 'http://127.0.0.1:8000'
$env:NEUROFLOW_BENCHMARK_ADMIN_CLIENT_ID = '<test-admin-client-id>'
$env:NEUROFLOW_BENCHMARK_ADMIN_CLIENT_SECRET = '<test-admin-client-secret>'
$env:NEUROFLOW_BENCHMARK_PIPELINE_ID = '<retrieval-benchmark-pipeline-uuid>'
$env:NEUROFLOW_TEST_CONFIRM_ISOLATED = 'I_CONFIRM_THIS_IS_AN_ISOLATED_TEST_DEPLOYMENT'
$env:NEUROFLOW_TEST_CONFIRM_PROVIDER = 'I_CONFIRM_TEST_ONLY_LLM_PROVIDER'
$env:NEUROFLOW_TEST_COMPOSE_PROJECT = 'neuroflow-test-task14'
$env:NEUROFLOW_TEST_CONFIRM_PROJECT = 'neuroflow-test-task14'
py -m tests.benchmarks.export_retrieval_rankings tests/benchmarks/retrieval_questions.json --output tests/benchmarks/retrieval_dataset.json
```

Only after this exporter succeeds, calculate metrics from its actual output:

```powershell
py tests/benchmarks/retrieval_benchmark.py tests/benchmarks/retrieval_dataset.json
```

Each enriched row must contain a unique nonempty `question_id`, a unique `question`, independently labeled `relevant_chunk_ids`, and actual ordered chunk IDs under `rankings` for `dense_only`, `sparse_only`, `hybrid_rrf`, and `hybrid_reranked`. Both exporter and scorer reject malformed data before writing rankings or calculating metrics. The scorer computes Hit Rate@5, Hit Rate@10, MRR@10 and binary NDCG@10 for each configuration, writes the comparison to `retrieval_benchmark_results.md`, and reports relative Hybrid+Reranked MRR@10 improvement over Dense-only.

Current status remains blocked: there is no independently labeled 50-question dataset, seeded benchmark corpus/pipeline, test API admin configuration, approved provider confirmation, or verified local model cache in this workspace. No rankings or metrics have been collected.

## Opt-in chaos tests

These tests can stop only named `redis` and `worker` services in a Compose project whose name begins `neuroflow-test-`. They restart the service in `finally`; they do not remove containers or volumes. Use a disposable Compose project and opt in explicitly:

```powershell
$env:NEUROFLOW_CHAOS_ENABLE = '1'
$env:NEUROFLOW_CHAOS_PROJECT = 'neuroflow-test-task14'
$env:NEUROFLOW_CHAOS_CONFIRM_PROJECT = 'neuroflow-test-task14'
$env:NEUROFLOW_INTEGRATION = 'true'
$env:NEUROFLOW_TEST_CONFIRM_ISOLATED = 'I_CONFIRM_THIS_IS_AN_ISOLATED_TEST_DEPLOYMENT'
$env:NEUROFLOW_TEST_CONFIRM_PROVIDER = 'I_CONFIRM_TEST_ONLY_LLM_PROVIDER'
$env:NEUROFLOW_TEST_COMPOSE_PROJECT = 'neuroflow-test-task14'
$env:NEUROFLOW_TEST_CONFIRM_PROJECT = 'neuroflow-test-task14'
pytest tests/chaos -v
```

Docker and a live isolated API are required. The chaos project must equal the test Compose project; the same read-only project/service/port/volume verification runs before any service stop. Tests skip if the opt-in flag, safe project prefix, Docker CLI, confirmations, or test URL are missing. Recovery waits are bounded to 90 seconds.

## Final offline audit status

On 2026-10-28, the Task 14 offline suite completed with 35 passed and 9 skipped; the skipped cases are the opt-in integration and chaos tests. The backend regression suite completed with 185 passed and one Starlette deprecation warning. The focused Locust, retrieval validator/exporter, and ranking endpoint tests completed with 34 passed. No test API smoke request, Locust smoke/load traffic, integration request, chaos action, or retrieval ranking export was run. `tests/performance/load_test_results.json` therefore retains null metric fields. The load test is blocked pending an isolated API URL, dedicated load client and reserved pipeline (or test-only admin provisioning), a running dedicated Compose project and its exact confirmation, and explicit isolation/provider confirmations. Retrieval metrics are blocked pending an isolated benchmark deployment with offline cached models, test-only admin configuration, a reserved seeded pipeline, and an independently labeled dataset of 50 unique questions. Integration/chaos execution remains blocked pending a verified disposable test stack and all test-only credentials/provider configuration.

