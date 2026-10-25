
"""NeuroFlow FastAPI application."""

from contextlib import asynccontextmanager
import logging

import httpx
import redis.asyncio as redis
from api.evaluations import router as evaluations_router
from api.ingest import router as ingest_router
from api.query import router as query_router
from api.retrieve import router as retrieve_router
from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from config import settings
from db.health import check_postgres
from db.pool import close_pool, open_pool


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configure OpenTelemetry tracing and export spans to Jaeger.
resource = Resource.create(
    {
        "service.name": settings.app_name,
        "deployment.environment": settings.app_env,
    }
)

tracer_provider = TracerProvider(resource=resource)

otlp_exporter = OTLPSpanExporter(
    endpoint=settings.otel_exporter_otlp_endpoint,
    insecure=True,
)

tracer_provider.add_span_processor(
    BatchSpanProcessor(otlp_exporter)
)

trace.set_tracer_provider(tracer_provider)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and clean up shared application resources."""
    await open_pool()

    try:
        app.state.redis = redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=3,
        )

        await app.state.redis.ping()
        logger.info("Redis connection established.")

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
)
app.include_router(ingest_router)
app.include_router(retrieve_router)
app.include_router(query_router)
app.include_router(evaluations_router)


@app.get("/")
async def root():
    """Return basic service information."""
    return {
        "service": settings.app_name,
        "status": "running",
    }


@app.get("/health")
async def health():
    """Check PostgreSQL, Redis, and MLflow connectivity."""
    checks = {}

    checks["postgres"] = await check_postgres()

    try:
        await app.state.redis.ping()
        checks["redis"] = {"status": "healthy"}
    except Exception:
        logger.exception("Redis health check failed.")
        checks["redis"] = {"status": "unhealthy"}

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                f"{settings.mlflow_tracking_uri.rstrip('/')}/health"
            )

        checks["mlflow"] = {
            "status": (
                "healthy"
                if response.status_code == 200
                else "unhealthy"
            )
        }
    except httpx.HTTPError:
        logger.exception("MLflow health check failed.")
        checks["mlflow"] = {"status": "unhealthy"}

    healthy = all(
        result["status"] == "healthy"
        for result in checks.values()
    )

    return JSONResponse(
        status_code=200 if healthy else 503,
        content={
            "status": "healthy" if healthy else "degraded",
            "checks": checks,
        },
    )


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Expose Prometheus metrics."""
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


FastAPIInstrumentor.instrument_app(app)
