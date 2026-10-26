from __future__ import annotations

from observability.metrics import QUERY_LATENCY
from observability.quality import detect_quality_anomaly
from observability.tracing import extract_trace_context, inject_trace_context
from opentelemetry.sdk.trace import TracerProvider


def test_quality_anomaly_requires_five_prior_scores():
    assert detect_quality_anomaly([0.9] * 4, 0.1) is None


def test_quality_anomaly_handles_zero_variance_baseline():
    anomaly = detect_quality_anomaly([0.9] * 5, 0.8)

    assert anomaly is not None
    assert anomaly.baseline_mean == 0.9
    assert anomaly.baseline_stddev == 0


def test_quality_anomaly_uses_two_population_standard_deviations():
    baseline = [0.7, 0.8, 0.8, 0.9, 0.8]

    assert detect_quality_anomaly(baseline, 0.1) is not None
    assert detect_quality_anomaly(baseline, 0.7) is None


def test_query_latency_histogram_uses_documented_buckets():
    assert QUERY_LATENCY._upper_bounds == [
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        30.0,
        float("inf"),
    ]


def test_queue_trace_context_propagates_with_w3c_traceparent():
    tracer = TracerProvider().get_tracer("test.observability")

    with tracer.start_as_current_span("parent"):
        carrier = inject_trace_context()
        extracted = extract_trace_context(carrier)

    assert "traceparent" in carrier
    assert extracted is not None
