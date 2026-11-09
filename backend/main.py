"""NeuroFlow FastAPI application."""

import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx
import redis.asyncio as redis
from api.compare import router as compare_router
from api.evaluations import router as evaluations_router
from api.finetune import router as finetune_router
from api.ingest import router as ingest_router
from api.pipelines import router as pipelines_router
from api.query import router as query_router
from api.retrieve import router as retrieve_router
from config import settings
from db.health import check_postgres
from db.pool import close_pool, open_pool
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from observability.metrics import QUEUE_DEPTH
from observability.tracing import configure_tracing
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from redis.exceptions import RedisError
from resilience.backpressure import INGEST_QUEUE_KEY
from resilience.circuit_breaker import CircuitBreakerRegistry
from resilience.endpoint_rate_limiter import RateLimitMiddleware
from resilience.timeout_manager import TimeoutManager

from backend.security.auth import ClientProfile, require_scope
from backend.security.auth import router as auth_router
from backend.security.middleware import SecurityHeadersMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


tracer_provider = configure_tracing(
    settings.app_name,
    settings.otel_exporter_otlp_endpoint,
    settings.app_env,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize and clean up shared application resources."""
    await open_pool()

    try:
        app.state.redis = redis.from_url(  # type: ignore[no-untyped-call]
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=3,
        )

        await app.state.redis.ping()
        logger.info("Redis connection established.")
        QUEUE_DEPTH.labels(queue="ingestion").set(await app.state.redis.llen(INGEST_QUEUE_KEY))
        QUEUE_DEPTH.labels(queue="evaluation").set(
            await app.state.redis.llen("neuroflow:queue:evaluations")
        )

        # Initialize resilience components
        app.state.circuit_breakers = CircuitBreakerRegistry(app.state.redis)
        app.state.timeout_manager = TimeoutManager(app.state.redis)
        logger.info("Resilience layer initialized.")

        # Whether to trust X-Forwarded-For headers (set via environment or config)
        app.state.trust_proxy = getattr(settings, "trust_proxy", False)

        yield

    finally:
        redis_client = getattr(app.state, "redis", None)

        if redis_client is not None:
            await redis_client.aclose()

        await close_pool()
        tracer_provider.force_flush()


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    description=(
        "NeuroFlow is a production-grade Retrieval-Augmented Generation (RAG) API. "
        "Ingest documents, run hybrid search queries, evaluate generation quality, "
        "and manage pipeline configurations."
    ),
    contact={"name": "NeuroFlow Team"},
    license_info={"name": "Proprietary"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add sliding-window rate-limiting middleware
# (Redis is not available at module level; the middleware handles None redis gracefully)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(SecurityHeadersMiddleware)

app.include_router(ingest_router)
app.include_router(auth_router)
app.include_router(retrieve_router)
app.include_router(query_router)
app.include_router(evaluations_router)
app.include_router(pipelines_router)
app.include_router(compare_router)
app.include_router(finetune_router)


@app.get(
    "/",
    tags=["system"],
    summary="Service info",
    description="Returns service name and running status. Requires query scope.",
)
async def root(_user: ClientProfile = Depends(require_scope("query"))) -> dict[str, str]:
    """Return basic service information."""
    return {
        "service": settings.app_name,
        "status": "running",
    }


@app.get(
    "/health",
    tags=["system"],
    summary="Health check",
    description=(
        "Returns health status of Postgres, Redis, MLflow, circuit breakers, "
        "and ingestion queue. No authentication required."
    ),
)
async def health() -> JSONResponse:
    """
    Resilience-aware health check.

    Returns:
        status: "ok" | "degraded" | "critical"
        checks: per-component result dict

    Status rules:
        critical  – PostgreSQL or Redis unreachable
        degraded  – any circuit breaker OPEN, or MLflow/optional dep unhealthy
        ok        – all required components healthy
    """
    checks: dict[str, object] = {}
    critical = False
    degraded = False

    # ── PostgreSQL ─────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        pg_result = await check_postgres()
        pg_latency_ms = round((time.perf_counter() - t0) * 1000)
        pg_result["latency_ms"] = pg_latency_ms
    except Exception as exc:
        pg_result = {"status": "unhealthy", "error": str(exc)}
        pg_latency_ms = round((time.perf_counter() - t0) * 1000)
        pg_result["latency_ms"] = pg_latency_ms

    checks["postgres"] = pg_result
    if pg_result.get("status") != "healthy":
        critical = True

    # ── Redis ──────────────────────────────────────────────────────────
    redis_client = getattr(app.state, "redis", None)
    t0 = time.perf_counter()
    try:
        if redis_client:
            await redis_client.ping()
            redis_latency_ms = round((time.perf_counter() - t0) * 1000)
            checks["redis"] = {"status": "healthy", "latency_ms": redis_latency_ms}
        else:
            checks["redis"] = {"status": "unhealthy", "error": "Not initialized"}
            critical = True
    except Exception:
        redis_latency_ms = round((time.perf_counter() - t0) * 1000)
        logger.exception("Redis health check failed.")
        checks["redis"] = {
            "status": "unhealthy",
            "latency_ms": redis_latency_ms,
        }
        critical = True

    # ── MLflow (optional) ──────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{settings.mlflow_tracking_uri.rstrip('/')}/health")
        mlflow_latency_ms = round((time.perf_counter() - t0) * 1000)
        mlflow_status = "healthy" if response.status_code == 200 else "unhealthy"
        checks["mlflow"] = {
            "status": mlflow_status,
            "latency_ms": mlflow_latency_ms,
        }
        if mlflow_status != "healthy":
            degraded = True
    except httpx.HTTPError:
        mlflow_latency_ms = round((time.perf_counter() - t0) * 1000)
        logger.exception("MLflow health check failed.")
        checks["mlflow"] = {
            "status": "unhealthy",
            "latency_ms": mlflow_latency_ms,
        }
        degraded = True

    # ── Circuit Breakers ───────────────────────────────────────────────
    cb_registry = getattr(app.state, "circuit_breakers", None)
    circuit_info: list[dict[str, object]] = []
    if cb_registry:
        for name, cb in cb_registry.all().items():
            try:
                # state() applies an elapsed recovery timeout (OPEN -> HALF_OPEN).
                # Health should report the live breaker state, not a stale Redis value.
                state = getattr(cb, "state", None)
                if state is not None:
                    await state()
                info = await cb.get_info()
                circuit_info.append(info)
                if info.get("state") in {"open", "half_open"}:
                    degraded = True
            except Exception as exc:
                circuit_info.append({"name": name, "error": str(exc)})
    checks["circuit_breakers"] = circuit_info

    # ── Ingestion queue depth ──────────────────────────────────────────
    try:
        if redis_client:
            depth = await redis_client.llen(INGEST_QUEUE_KEY)
            checks["ingest_queue_depth"] = int(depth or 0)
        else:
            checks["ingest_queue_depth"] = None
    except Exception:
        checks["ingest_queue_depth"] = None

    # ── Overall status ─────────────────────────────────────────────────
    if critical:
        overall = "critical"
        http_status = 503
    elif degraded:
        overall = "degraded"
        http_status = 503
    else:
        overall = "ok"
        http_status = 200

    return JSONResponse(
        status_code=http_status,
        content={
            "status": overall,
            "checks": checks,
        },
    )


@app.get(
    "/metrics",
    response_class=PlainTextResponse,
    tags=["system"],
    summary="Prometheus metrics",
    description="Exposes Prometheus-format metrics for scraping. No authentication required.",
)
async def metrics() -> PlainTextResponse:
    """Expose Prometheus metrics."""
    redis_client = getattr(app.state, "redis", None)
    if redis_client is not None:
        try:
            QUEUE_DEPTH.labels(queue="ingestion").set(await redis_client.llen(INGEST_QUEUE_KEY))
            QUEUE_DEPTH.labels(queue="evaluation").set(
                await redis_client.llen("neuroflow:queue:evaluations")
            )
        except (RedisError, OSError, TimeoutError):
            logger.exception("Could not refresh queue metrics from Redis.")
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


FastAPIInstrumentor.instrument_app(app)
