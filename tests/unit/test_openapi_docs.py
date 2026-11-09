"""Unit tests for OpenAPI documentation coverage and quality."""

from __future__ import annotations

import pytest
from backend.main import app


def test_openapi_schema_generated():
    """Verify that OpenAPI schema can be generated and is accessible."""
    schema = app.openapi()
    assert schema is not None
    assert schema["info"]["title"] == "NeuroFlow"
    assert "description" in schema["info"]
    assert len(schema["info"]["description"]) > 20
    assert "paths" in schema
    assert len(schema["paths"]) > 5


def test_core_endpoints_have_documentation():
    """Verify core endpoints have summary, description, and tags."""
    schema = app.openapi()
    paths = schema["paths"]

    core_endpoints = [
        ("/query", "post"),
        ("/ingest", "post"),
        ("/ingest/url", "post"),
        ("/pipelines", "get"),
        ("/pipelines", "post"),
        ("/evaluations", "get"),
        ("/health", "get"),
        ("/metrics", "get"),
    ]

    for path, method in core_endpoints:
        assert path in paths, f"Path {path} not found in OpenAPI schema"
        operation = paths[path].get(method)
        assert operation is not None, f"Method {method.upper()} {path} not found"
        assert "summary" in operation, f"Missing summary for {method.upper()} {path}"
        assert len(operation["summary"]) > 0, f"Empty summary for {method.upper()} {path}"
        assert "description" in operation, f"Missing description for {method.upper()} {path}"
        assert len(operation["description"]) > 10, f"Description too short for {method.upper()} {path}"
        assert "tags" in operation, f"Missing tags for {method.upper()} {path}"
        assert len(operation["tags"]) > 0, f"Empty tags for {method.upper()} {path}"


def test_openapi_tags_categorization():
    """Verify endpoints have requested logical tags: Ingestion, Query, Evaluation, Fine-Tuning, Admin."""
    schema = app.openapi()
    paths = schema["paths"]

    assert "Ingestion" in paths["/ingest"]["post"]["tags"]
    assert "Query" in paths["/query"]["post"]["tags"]
    assert "Admin" in paths["/pipelines"]["get"]["tags"]
    assert "Evaluation" in paths["/evaluations"]["get"]["tags"]
    assert "Fine-Tuning" in paths["/finetune/jobs"]["get"]["tags"]


def test_model_examples_in_schema():
    """Verify representative request and response models contain examples."""
    schema = app.openapi()
    components = schema.get("components", {}).get("schemas", {})

    assert "QueryRequest" in components
    query_req = components["QueryRequest"]
    assert "example" in query_req or "examples" in query_req.get("properties", {}).get("query", {})

    assert "URLIngestRequest" in components
    url_req = components["URLIngestRequest"]
    assert "example" in url_req or "examples" in url_req.get("properties", {}).get("url", {})


def test_docs_urls_configured():
    """Verify Swagger UI and Redoc URLs are enabled on the FastAPI app."""
    assert app.docs_url == "/docs"
    assert app.redoc_url == "/redoc"
    assert app.openapi_url == "/openapi.json"
