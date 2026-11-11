# Task 15: Production containerization

## Production stack

The production stack is separate from `infra/docker-compose.yml` and uses the fixed Compose project name `neuroflow-prod`, its own volumes, and an isolated bridge network. The API and worker use the hardened Python runtime image. Two API replicas and two worker replicas are declared through Compose `deploy.replicas`; resource limits are declared alongside them. The `nginx` service serves the statically exported frontend and reverse proxies `/api/*` to the API replicas. PostgreSQL initialization runs the checked-in migrations and creates the application role for a fresh production data volume. MLflow is included because the API readiness check requires its tracking endpoint.

The frontend is client-rendered and does not use Next.js server-only routes, so the frontend build uses Next.js static export and Nginx serves `out/`. A plain Nginx image cannot execute `.next/standalone/server.js`; a Node runtime would be needed for that mode. API and event-stream URLs use the `/api` prefix and Nginx removes that prefix before proxying.

Provide deployment secrets through the process environment or an approved secret manager; do not put them in Git or edit `.env` for deployment. Required values are `POSTGRES_PASSWORD`, `POSTGRES_APP_PASSWORD`, `REDIS_PASSWORD`, `NEUROFLOW_JWT_SECRET` (at least 32 bytes), and `NEUROFLOW_AUTH_CLIENTS`. `POSTGRES_APP_USER` and `POSTGRES_DB` have defaults. Optional frontend API path is `NEXT_PUBLIC_API_URL` (default `/api`).

```sh
docker compose -f infra/docker-compose.prod.yml up --build -d
```

To stop this production project without removing its persistent data:

```sh
docker compose -f infra/docker-compose.prod.yml down
```

The production compose file has its own named volumes and the project name `neuroflow-prod`; it does not reuse the development Compose project's volumes. Do not run `down -v` unless deleting production data is intended.

## Container security checks

Compose scaling may be provided by `deploy.replicas` in current Docker Compose releases. Verify actual replica counts with:

```sh
docker compose -f infra/docker-compose.prod.yml ps
```

Each API replica runs as the unprivileged `neuroflow` account:

```sh
docker exec neuroflow-prod-api-1 whoami
# expected: neuroflow
```

The API root filesystem is read-only. This should fail with a permission error:

```sh
docker exec neuroflow-prod-api-1 touch /test
```

Inspect dropped capabilities, no-new-privileges, read-only mode, and the image health check:

```sh
docker inspect neuroflow-prod-api-1 --format '{{json .HostConfig.CapDrop}} {{.HostConfig.ReadonlyRootfs}} {{json .HostConfig.SecurityOpt}}'
docker inspect neuroflow-prod-api-1 --format '{{json .State.Health}}'
```

Check frontend/Nginx response and rate limiting from a host with the service running:

```sh
curl -I http://localhost/
curl -i http://localhost/api/health
```

The API container health check runs `curl -f http://localhost:8000/health`. Nginx sends `X-Frame-Options` and `X-Content-Type-Options`, compresses responses, and limits API traffic to 60 requests per minute per client IP (with a short burst allowance).

## TLS setup

Terminate TLS at Nginx or an upstream load balancer. For local-only testing, create a self-signed certificate outside the repository:

```sh
mkdir -p ./certs
openssl req -x509 -newkey rsa:3072 -nodes -days 30 \
  -keyout ./certs/tls.key -out ./certs/tls.crt \
  -subj '/CN=localhost' -addext 'subjectAltName=DNS:localhost,IP:127.0.0.1'
```

Set `TLS_CERT_DIR` to that directory and enable the TLS server block in `infra/nginx/nginx.conf`, then publish port 443 in Compose. The HTTPS server block adds HSTS; do not enable HSTS until HTTPS is consistently available for the host. Keep private keys outside the repository.

## Vulnerability scanning

Install Docker and Trivy, then run:

```sh
make security-scan
```

This builds `neuroflow-api:task15-prod` and returns a failing status if Trivy finds a CRITICAL vulnerability. The GitHub Actions workflow repeats this scan when code is pushed to `main`. Inspect and remediate findings before deployment.

## Image-size note

The API requirements include CPU PyTorch and `sentence-transformers` for local embedding/reranking. Those ML dependencies make a strict 500 MB image target unrealistic while retaining the current feature set. The Dockerfile uses CPU-only PyTorch and a multi-stage build to exclude build artifacts and pip cache. The Task 15 build produced a 2.81 GB API image, so it does not meet the 500 MB target. Check a later build with:

```sh
docker images neuroflow-api:task15-prod
```

If the size limit is mandatory, move embedding/reranking to a separately deployed model service or replace the local ML stack with a remote inference endpoint before setting a hard image-size gate.

The API and worker resource limits alone total 6 GB across the requested replicas; PostgreSQL, Redis, and MLflow add more. Allocate enough Docker host memory for the aggregate limits and normal headroom. In the Task 15 smoke run, the local Docker engine reported 3.9 GB total memory and the API replicas did not become healthy; their Uvicorn child processes exited before the health endpoint accepted connections.
