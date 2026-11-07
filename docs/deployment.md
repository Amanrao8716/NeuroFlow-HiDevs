# Cloud deployment (Railway)

## Current status

This repository has not been connected to a Railway project, and no production
URL or credentials are present in the workspace. No production checks or load
test have been run. Record the URL, deployment commit, date, and actual result
for every checklist item below after deploying; do not mark a check passed based
only on a successful build. The service layout below is a reproducible manual
Railway setup for the current Docker images and worker architecture.

## Prerequisites

- A Railway account and a project linked to this repository.
- GitHub access to connect the repository and push the deployment branch.
- An OpenAI API key (Anthropic is optional).
- A PostgreSQL 16 database with the `vector` extension available, and Redis 7.
- A persistent volume for MLflow artifacts if artifacts must survive redeploys.
- Railway CLI, Docker, `curl`, and Locust installed locally for operations.

Allow roughly 45–90 minutes for a first manual deployment, plus time to
provision accounts, domains, and provider keys. This is an estimate, not a
measured result.

The app's settings use discrete `POSTGRES_HOST`, `POSTGRES_PORT`,
`POSTGRES_DB`, `POSTGRES_USER`, and `POSTGRES_PASSWORD` fields and discrete
Redis settings. A provider's single `DATABASE_URL` or `REDIS_URL` variable is
not read by the current application configuration. Map the provider's connection
details to these fields. Confirm that the chosen PostgreSQL provider permits
`CREATE EXTENSION vector`; the repository's local database image is
`pgvector/pgvector:pg16`.

## Create services

1. Create a Railway project from this Git repository and select the branch to
   deploy. Use the repository root as the build context.
2. Add a PostgreSQL 16 service. Enable `vector` on the database before starting
   the API. Use its private-network host and port for the API and worker.
3. Add a Redis 7 service and use its private-network host and port. Set an
   authentication password and map it to `REDIS_PASSWORD`.
4. Add an MLflow service built from `infra/Dockerfile.mlflow`. Start it with:

   ```text
   mlflow server --host 0.0.0.0 --port $PORT --backend-store-uri <postgres-uri> --default-artifact-root /mlartifacts
   ```

   Replace `<postgres-uri>` with a URL-encoded PostgreSQL URI for a dedicated
   MLflow database. Attach a persistent Railway volume at `/mlartifacts` and
   expose the service privately to the app. The application health check calls
   `<MLFLOW_TRACKING_URI>/health`.
5. Add the API service with `backend/Dockerfile`, root build context, and start
   command `uvicorn main:app --host 0.0.0.0 --port $PORT --workers 2`. Set the
   Railway health check path to `/health` and a suitable startup timeout.
6. Add a worker service using the same root build context and `backend/Dockerfile`.
   Override its start command to `python -m worker`. Give it the same database,
   Redis, provider, auth, and MLflow settings as the API. The worker must be able
   to reach the API through `NEUROFLOW_API_URL` for worker-authenticated calls.
7. Give the API a public domain. Keep PostgreSQL, Redis, and MLflow private
   unless there is a specific administrative need for public access.
8. Add a frontend service from the same repository using `frontend/Dockerfile`
   and repository root as build context. Set build argument
   `NEXT_PUBLIC_API_URL` to the public HTTPS API URL. The frontend is a static
   export and reads this value at build time. The API currently permits CORS
   from any origin; restrict this before relying on cross-origin deployment
   for sensitive data.

The frontend's Nginx configuration also proxies `/api` to a Docker DNS host
named `api`. Railway services do not automatically share that Docker DNS name.
Use the absolute API URL build argument for browser requests and verify no UI
request falls back to `/api`; adapt the Nginx upstream if proxy routing is
required.

The API container's baked-in Docker health check uses port 8000, while Railway
assigns `$PORT`. If Railway relies on that image health check instead of its
service health check, set the Railway start command to port 8000 and configure
the Railway service to route to container port 8000. Do not mix the two port
models.

## Domain, HTTPS, and CLI

Railway provides an HTTPS domain for a service from its service settings. For a
custom domain, add the hostname to the API or frontend service, publish the DNS
records Railway displays, and wait for certificate issuance before switching
clients. Railway terminates TLS; do not mount local certificates into Nginx.

Install and authenticate Railway CLI using its current official instructions.
Link the selected project/environment in the repository, configure each
service's Dockerfile path, root directory, build arguments, start command,
health check, and variables in Railway, then deploy:

```bash
railway status
railway up --service neuroflow-api --environment production
railway up --service neuroflow-worker --environment production
railway up --service neuroflow-frontend --environment production
railway logs --service neuroflow-api --environment production
```

Service names and CLI flags must match the Railway project configuration.

## Configure environment

Copy `.env.example` as a reference, not as a committed secrets file. Enter each
variable in Railway's service variables, using the same values for API and
worker where applicable. Generate secrets locally and paste them into the
provider's secret manager:

```bash
openssl rand -hex 32
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Use the first value for `NEUROFLOW_JWT_SECRET`; use strong independent values
for PostgreSQL and Redis passwords. Configure `NEUROFLOW_AUTH_CLIENTS` as JSON
with each client ID mapped to an object containing a `secret_hash` (the
`scrypt$...` result of `hash_client_secret()`) and a `scopes` array. The token
endpoint accepts the corresponding plaintext secret; retain it only in an
approved client secret store. Set `APP_ENV=production` and provider keys as
Railway secrets. `ENVIRONMENT` and `LOG_LEVEL` are documentation labels and are
not currently read by application code. `POSTGRES_URL`,
`REDIS_URL`, `PLUGIN_SECRETS_KEY`, `SENTRY_DSN`, and `ENVIRONMENT` are listed
for visibility/compatibility in `.env.example`; the current app does not consume
the URL aliases or optional integration keys. Do not assume they configure the
application until code support is added.

## Deploy and initialize the database

Deploy PostgreSQL, Redis, and MLflow first. Then deploy the API and worker. The
API startup applies migrations from `backend/db/migrations`; confirm this in
the deployment logs and check the health endpoint before sending traffic. The
migration runner applies forward migrations; no reverse migration scripts are
provided. Take a database backup before each release that changes schema.

Example checks (replace the host and provide a valid bearer token for protected
routes):

Set `API_URL` and `MLFLOW_URL` in your shell to the actual Railway HTTPS
domains after deployment. No live domains are configured yet.

```bash
curl -fsS "$API_URL/health"
curl -fsS "$API_URL/metrics"
curl -fsS "$MLFLOW_URL/"
```

Run after deployment. For protected endpoints, first provision a dedicated
test client with `query`, `ingest`, and `admin` scopes in
`NEUROFLOW_AUTH_CLIENTS`, then obtain a token without putting credentials in
the command history:

```bash
curl -fsS "$API_URL/auth/token" \
  -H 'Content-Type: application/json' \
  --data "$(jq -n --arg id "$CLIENT_ID" --arg secret "$CLIENT_SECRET" '{client_id:$id,client_secret:$secret}')"
```

Use the returned token in an `Authorization: Bearer ...` header. First create a
pipeline with `POST /pipelines` and use its returned ID. For ingestion, the
actual endpoint accepts multipart fields `file` and `pipeline_id`:

```bash
curl -fsS "$API_URL/ingest" \
  -H "Authorization: Bearer $TOKEN" \
  -F "pipeline_id=$PIPELINE_ID" \
  -F "file=@tests/fixtures/test_doc.pdf;type=application/pdf"
```

Set `PIPELINE_ID` from the pipeline creation response. The repository does not include that PDF fixture. Add a synthetic,
non-sensitive fixture before executing the command. Poll
`GET "$API_URL/documents/$DOCUMENT_ID"` until status is `complete`. Query using
`POST "$API_URL/query"` with JSON containing a fixture-grounded question,
`PIPELINE_ID`, and `stream:false`. These commands require `jq`:

```bash
QUERY_JSON=$(jq -n --arg id "$PIPELINE_ID" \
  '{query:"Summarize the key points in the test document.",pipeline_id:$id,stream:false}')
curl -fsS "$API_URL/query" -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' --data "$QUERY_JSON"
STREAM_JSON=$(jq -n --arg id "$PIPELINE_ID" \
  '{query:"Cite one fact from the test document.",pipeline_id:$id,stream:true}')
STREAM_URL=$(curl -fsS "$API_URL/query" -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' --data "$STREAM_JSON" | jq -r .stream_url)
curl -N -fsS "$API_URL$STREAM_URL" -H "Authorization: Bearer $TOKEN"
curl -fsS "$API_URL/evaluations?pipeline_id=$PIPELINE_ID" \
  -H "Authorization: Bearer $TOKEN"
```

Confirm a non-empty answer and valid citations. Verify SSE events arrive through
completion and `GET /evaluations` returns an evaluation associated with the run.
Do not treat a queued evaluation as completed.

The stream registry is in-memory in this implementation; route stream init and
SSE requests to the same API instance or use a single API replica for this
check.

`/health` reports PostgreSQL, Redis, MLflow, and related dependency states.
Require HTTP success and every required check to be `healthy`; a degraded
response is not a green deployment. `/metrics` is Prometheus text format and
includes custom counters/histograms registered in the app. Protect its public
availability at the network edge if the service is internet-facing.

## Production acceptance checklist and results

Complete this table against the deployed URL. Store sanitized command output,
run IDs, and timestamps in the release record; never commit credentials,
uploaded private documents, or authorization tokens.

| Check | Procedure | Result |
|---|---|---|
| Live URL and health | `GET "$API_URL/health"`; confirm all required components are healthy. | Pending: no deployed URL configured. |
| Ingestion | Upload `tests/fixtures/test_doc.pdf` to `POST /ingest` with required auth; poll document status until `complete`. | Pending: no deployed URL configured. |
| Cited answer | Create/select a pipeline, submit a question grounded in the fixture to `POST /query`, and verify cited answer and completed generation. | Pending: no deployed URL configured. |
| Evaluation | `GET /evaluations` with required auth; confirm the query has an evaluation and scores. | Pending: no deployed URL configured. |
| Streaming | Use the run ID and `curl -N "$API_URL/query/$RUN_ID/stream"`; confirm progressive token events followed by completion. | Pending: no deployed URL configured. |
| MLflow | Open `$MLFLOW_URL`; confirm the tracking UI loads and experiments are visible after a tracked run. | Pending: no deployed URL configured. |
| Metrics | `GET /metrics`; confirm HTTP 200 and expected custom metric names. | Pending: no deployed URL configured. |
| Load | Run the command below for two minutes; record failures, latency, and service health. | Pending: no deployed URL configured. |

```bash
locust -f tests/performance/locustfile.py -H "$API_URL" --headless -u 10 -r 2 --run-time 2m
```

Before running, set `NEUROFLOW_LOAD_CLIENT_ID`,
`NEUROFLOW_LOAD_CLIENT_SECRET`, and `NEUROFLOW_LOAD_PIPELINE_ID` from a secret
store. The scenario sends real LLM queries and can incur provider charges. Use
a dedicated seeded test pipeline. Its pacing aims to stay below the per-IP
query throttle. Run only against an approved test environment or during an
approved production window.

There is no universal expected throughput: it depends on model/provider
latency, retrieval workload, and Railway resource sizes. Record requests/second,
p50/p95 latency, failure rate, provider errors, and CPU/memory/database usage.
Suggested initial acceptance criteria are under 1% failed requests, no
sustained 5xx responses, p95 below 10 seconds for non-streaming queries, and no
unhealthy dependency during the run. Treat these as rollout targets, not
measured results. Current result: **Run after deployment; no production load
result exists.**

Common failures include a missing `vector` extension (enable before
migrations), invalid credentials/URL encoding, API health 503 from unreachable
MLflow/Redis/Postgres, worker jobs remaining queued (worker cannot connect to
Redis), empty answers (no completed ingestion or provider credentials), and
browser network errors (incorrect API build URL or CORS). Inspect Railway logs
and verify private host/port and secret mappings before redeploying.

## Rollback

1. In Railway, open the API and worker deploy history. Redeploy the last known
   good image/deployment for both services from the same commit. If deploying
   by image tag, select the previous immutable image digest/tag; avoid `latest`.
2. Check whether the release applied database migrations. There are no reverse
   migrations in this repository. For a backward-incompatible schema change,
   restore the pre-release database backup to a recovery database and point the
   rolled-back app at it, or follow a separately reviewed forward-fix plan.
   Do not attempt guessed SQL rollback statements against production.
3. Verify API and worker deployment revisions match, `/health` is all green,
   and a known test pipeline can ingest, query, stream, and evaluate a document.
   Confirm the error rate has returned to baseline before declaring rollback
   complete. Record the restored image digest, database action, and verification
   timestamp here.

## Pull request previews

**Preview automation is not enabled in this repository.** The requested shared
Postgres plus per-preview schema is incompatible with the current code: its
migrations and health checks explicitly target `public`, and API/worker queries
use the default search path. Redis operations use shared fixed key names with
database 0 and no prefix. A workflow that only creates a Railway PR environment
could let preview migrations alter staging tables and preview workers consume
staging jobs. This is unsafe.

Implementing previews requires schema/search-path isolation for every database
connection and migration, and a Redis key prefix (or separate Redis DB/service),
plus cleanup of schemas/keys and fork-safe secret handling. Railway project
token, project ID, and service IDs should only be added as GitHub Actions secrets
after that isolation work is merged. No Railway tokens are currently
configured. Do not expose production credentials to PR workflows from forks.

## Deployment record

- Platform/project: Not configured.
- Public API URL: Not deployed.
- Frontend URL: Not deployed.
- MLflow URL: Not deployed.
- Deployment commit/date: Not deployed.
- Health, end-to-end pipeline, metrics, and load test: Run after deployment.
- Estimated rollback duration: 5–15 minutes for image redeploy; database restore
  time depends on database size and is not included in this estimate.
