"""Shared OpenTelemetry initialization and W3C trace-context helpers."""

from __future__ import annotations

from opentelemetry import propagate, trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_provider: TracerProvider | None = None


def configure_tracing(service_name: str, endpoint: str, environment: str) -> TracerProvider:
    """Configure one OTLP tracer provider per process and reuse it on startup."""
    global _provider
    if _provider is None:
        _provider = TracerProvider(
            resource=Resource.create(
                {
                    "service.name": service_name,
                    "deployment.environment": environment,
                }
            )
        )
        _provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
        )
        trace.set_tracer_provider(_provider)
    return _provider


def inject_trace_context() -> dict[str, str]:
    """Return the active W3C trace context for propagation through a queue."""
    carrier: dict[str, str] = {}
    propagate.inject(carrier)
    return carrier


def extract_trace_context(carrier: dict[str, str] | None = None):
    """Extract a W3C context received from a queue message."""
    return propagate.extract(carrier or {})
