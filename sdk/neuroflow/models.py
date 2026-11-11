"""Typed data models for NeuroFlow SDK responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    """Represents an ingested document returned by the NeuroFlow API."""

    document_id: str
    """Server-assigned UUID for this document."""

    status: str
    """Processing status: 'queued', 'processing', 'completed', or 'failed'."""

    filename: str | None = None
    """Original filename of the uploaded document."""

    source_type: str | None = None
    """Document type: 'pdf', 'docx', 'image', 'csv', or 'url'."""

    chunk_count: int = 0
    """Number of text chunks extracted after processing completes."""

    duplicate: bool = False
    """True if the content hash already existed in the pipeline."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Arbitrary metadata returned by the server."""

    created_at: str | None = None
    """ISO-8601 timestamp of document creation."""


@dataclass(frozen=True)
class Citation:
    """A single citation from a generated answer."""

    reference: str
    source_number: int
    chunk_id: str | None = None
    document_name: str | None = None
    page_number: int | None = None
    content_preview: str | None = None
    is_valid: bool = True
    error: str | None = None


@dataclass(frozen=True)
class QueryResult:
    """Result of a non-streaming RAG query."""

    run_id: str
    """UUID of the pipeline run."""

    query: str
    """Original query text."""

    answer: str
    """Generated answer text."""

    query_type: str = ""
    """Detected query classification (e.g. 'factual', 'comparative')."""

    citations: list[Citation] = field(default_factory=list)
    """Structured citations extracted from the answer."""

    sources: list[dict[str, Any]] = field(default_factory=list)
    """Raw source metadata for retrieved chunks."""

    chunks_used: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    model_used: str = ""
    status: str = "completed"


@dataclass(frozen=True)
class EvaluationResult:
    """LLM-as-a-judge evaluation scores for a pipeline run."""

    id: str
    run_id: str
    faithfulness: float | None = None
    answer_relevance: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    overall_score: float | None = None
    judge_model: str | None = None
    user_rating: int | None = None
    calibration_needed: bool = False


@dataclass(frozen=True)
class Pipeline:
    """A NeuroFlow RAG pipeline configuration."""

    id: str
    name: str
    description: str | None = None
    version: int = 1
    status: str = "active"
    config: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None
