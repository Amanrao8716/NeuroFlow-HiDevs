# NeuroFlow security operations

## API clients and JWTs

The API now requires a bearer JWT on every route except `/health`, `/metrics`, and `POST /auth/token`. Interactive OpenAPI documentation endpoints are disabled. Tokens use HS256, expire after 3600 seconds, and contain the configured client's subject and scopes. Query and document read routes require `query`; upload routes require `ingest`; pipeline changes and fine-tuning require `admin`. The Task 12 anomaly callback uses the narrow `observability` scope plus its existing service token. Administrators can perform any scoped operation.

Set `NEUROFLOW_JWT_SECRET` to a randomly generated value with at least 32 bytes. Set `NEUROFLOW_AUTH_CLIENTS` to a JSON object keyed by client ID; each value must contain `secret_hash` and a least-privilege `scopes` list. Client secrets are scrypt hashes (`scrypt$salt$digest`), never plaintext. Generate a hash in a trusted Python environment with:

```python
from backend.security.auth import hash_client_secret
print(hash_client_secret("replace-with-a-generated-client-secret"))
```

Provision clients through the deployment secret manager or environment. Do not put secrets in `.env.example`, source control, logs, or shell history. Issue a token with `POST /auth/token` and JSON fields `client_id` and `client_secret`. The worker anomaly publisher needs a client assigned only the `observability` scope, configured with `NEUROFLOW_WORKER_CLIENT_ID` and `NEUROFLOW_WORKER_CLIENT_SECRET`; it sends the Task 12 service token separately in `X-Observability-Service-Token`.

Required API configuration also includes the existing PostgreSQL and Redis variables. `MAX_UPLOAD_SIZE_BYTES` optionally lowers the default 100 MiB upload limit; values over 100 MiB are rejected by application settings. A failed query injection-classification call returns `503` (fail closed); malicious classifications return `400` with `query_rejected` / `potential_prompt_injection`. The classifier uses one provider call with no tools and no retry. Query text is HTML-stripped, length-limited, and secret-redacted before classification and generation.

## Inputs, URLs, and uploads

Text query input is limited to 5000 characters. Pipeline names are limited to 100 characters. HTML tags are removed using Bleach. URL ingestion accepts HTTP(S) only, blocks non-public DNS/IP destinations and URL credentials, and disables automatic redirects; a single redirect is revalidated. Public DNS answers are checked before requests, though deployments needing protection against a DNS answer changing between validation and socket connection should place outbound fetches behind a network egress proxy that enforces public destinations.

Uploads are limited to 100 MiB and require a supported declared MIME type and matching file signature. DOCX archives are bounded by entry count, expanded size, compression ratio, traversal checks, and archive integrity. PDF JavaScript markers and executable signatures are rejected. Extracted text is scanned for prompt injection and secret patterns; secrets are redacted before chunking so semantic chunking and embedding never receive the originals. Detection events contain only document ID and pattern type.

## Extraction sandbox

Production workers run PDF, DOCX, CSV, and image OCR extraction in a new Docker container with `--network=none`, `--memory=256m`, one CPU, a PID cap, a 60-second host timeout, a read-only root filesystem, dropped capabilities, no-new-privileges, a non-root UID, and only temporary input/output mounts. Docker must be available to production workers, and `NEUROFLOW_EXTRACTION_IMAGE` can select the extraction image (default `neuroflow-api:latest`). `NEUROFLOW_EXTRACTION_SANDBOX=true` enables the same behavior in development. When it is unset in development, the documented local convenience path extracts in-process. Production sandbox failures do not fall back in-process. Isolated image extraction uses OCR only; vision-model image descriptions remain available only in development's in-process path because the extraction container has no network.

The URL fetch itself remains in the worker process to allow outbound HTTP; redirect and destination checks apply before fetching. Use an egress proxy/firewall for DNS-rebinding-resistant enforcement.

## Headers, TLS, and request IDs

Responses include `nosniff`, clickjacking denial, a self-only CSP, HSTS, and a UUID request ID. HSTS advertises HTTPS; terminate TLS correctly in production and ensure the public service is HTTPS before enabling browser-facing HSTS.

## Local checks and secret baseline

Install `backend/requirements.txt`, configure local dependencies, then run the tests from the repository root with the backend package on `PYTHONPATH`. Docker sandbox integration requires a Docker daemon and the configured image; unit tests mock the runner.

The tracked `.secrets.baseline` is produced and checked with the pinned `detect-secrets==1.5.0` dependency:

```powershell
python -m detect_secrets scan --baseline .secrets.baseline
python -m detect_secrets scan --baseline .secrets.baseline
```

When updating the baseline, review every finding and remediate real credentials. Never use baseline entries to legitimize actual secrets. `.env` and credential files stay untracked.

## API compatibility

Bearer authentication is a breaking API change. Existing clients must provision credentials, request tokens, and send `Authorization: Bearer <token>`. Previously open API endpoints now return `401` when the token is absent or invalid, and `403` when the token lacks the required scope.
