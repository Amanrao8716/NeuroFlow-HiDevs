"""Strict Pydantic configuration schemas and API models for RAG pipelines."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ChunkingStrategy(str, Enum):
    AUTO = "auto"
    FIXED = "fixed"
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"


class RerankerType(str, Enum):
    CROSS_ENCODER = "cross-encoder"
    NONE = "none"
    LLM = "llm"


class SystemPromptVariant(str, Enum):
    DEFAULT = "default"
    CONCISE = "concise"
    DETAILED = "detailed"
    TECHNICAL = "technical"


class ExtractorType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    CSV = "csv"
    IMAGE = "image"
    URL = "url"
    TEXT = "text"


class IngestionConfig(BaseModel):
    """Configuration for document parsing, chunking, and text extraction."""

    model_config = ConfigDict(extra="forbid")

    chunking_strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.AUTO,
        description="Strategy for partitioning document content",
    )
    chunk_size_tokens: int = Field(
        default=512,
        gt=0,
        le=8192,
        description="Target maximum token count per chunk (must be positive)",
    )
    chunk_overlap_tokens: int = Field(
        default=50,
        ge=0,
        description="Token overlap between adjacent chunks (must be >= 0 and < chunk_size_tokens)",
    )
    extractors_enabled: list[ExtractorType] = Field(
        default_factory=lambda: [
            ExtractorType.PDF,
            ExtractorType.DOCX,
            ExtractorType.CSV,
            ExtractorType.IMAGE,
            ExtractorType.URL,
            ExtractorType.TEXT,
        ],
        description="List of document formats enabled for automated extraction",
    )

    @model_validator(mode="after")
    def validate_overlap(self) -> IngestionConfig:
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError(
                f"chunk_overlap_tokens ({self.chunk_overlap_tokens}) must be strictly less than "
                f"chunk_size_tokens ({self.chunk_size_tokens})"
            )
        return self


class RetrievalConfig(BaseModel):
    """Configuration for multi-method hybrid search, expansion, and reranking."""

    model_config = ConfigDict(extra="forbid")

    dense_k: int = Field(
        default=10,
        gt=0,
        le=100,
        description="Number of candidates to retrieve via dense semantic search",
    )
    sparse_k: int = Field(
        default=10,
        gt=0,
        le=100,
        description="Number of candidates to retrieve via sparse lexical/keyword search",
    )
    reranker: RerankerType = Field(
        default=RerankerType.CROSS_ENCODER,
        description="Reranking method to apply across reciprocal rank fusion candidates",
    )
    top_k_after_rerank: int = Field(
        default=5,
        gt=0,
        le=100,
        description="Top K passages to retain following cross-encoder reranking",
    )
    query_expansion: bool = Field(
        default=True,
        description="Whether to generate query expansions for improved lexical/dense recall",
    )
    metadata_filters_enabled: bool = Field(
        default=True,
        description="Whether to extract and apply date/source metadata filters",
    )

    @model_validator(mode="after")
    def validate_rerank_limits(self) -> RetrievalConfig:
        max_possible = self.dense_k + self.sparse_k
        if self.top_k_after_rerank > max_possible:
            raise ValueError(
                f"top_k_after_rerank ({self.top_k_after_rerank}) cannot exceed total candidates "
                f"retrieved from dense_k ({self.dense_k}) + sparse_k ({self.sparse_k}) = {max_possible}"
            )
        return self


class GenerationConfig(BaseModel):
    """Configuration for LLM model routing, context window budgeting, and prompt assembly."""

    model_config = ConfigDict(extra="forbid")

    model_routing: str = Field(
        default="gpt-4o-mini",
        min_length=1,
        max_length=100,
        description="Target model identifier or router criteria ('gpt-4o', 'gpt-4o-mini', etc.)",
    )
    max_context_tokens: int = Field(
        default=4000,
        gt=0,
        le=32000,
        description="Maximum token budget allocated for context passages in prompt",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for LLM generation (0.0 to 2.0)",
    )
    system_prompt_variant: SystemPromptVariant = Field(
        default=SystemPromptVariant.DEFAULT,
        description="System prompt style variant ('default', 'concise', 'detailed', 'technical')",
    )


class EvaluationConfig(BaseModel):
    """Configuration for automated LLM-as-a-judge evaluation and fine-tuning pair curation."""

    model_config = ConfigDict(extra="forbid")

    auto_evaluate: bool = Field(
        default=True,
        description="Whether to automatically enqueue background evaluation for completed runs",
    )
    training_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum overall evaluation score (0.0 - 1.0) required to promote run to training pairs",
    )


class PipelineConfig(BaseModel):
    """Complete, strict, versionable configuration for a NeuroFlow RAG pipeline."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique human-readable pipeline name",
    )
    description: str | None = Field(
        default=None,
        max_length=1000,
        description="Optional detailed description of pipeline purpose and tuning",
    )
    ingestion: IngestionConfig = Field(
        default_factory=IngestionConfig,
        description="Document extraction and chunking settings",
    )
    retrieval: RetrievalConfig = Field(
        default_factory=RetrievalConfig,
        description="Search, fusion, and reranking parameters",
    )
    generation: GenerationConfig = Field(
        default_factory=GenerationConfig,
        description="LLM provider routing and prompt settings",
    )
    evaluation: EvaluationConfig = Field(
        default_factory=EvaluationConfig,
        description="Judge evaluation and training dataset curation rules",
    )


# --- API Request and Response Models ---

class PipelineCreateRequest(BaseModel):
    """Request payload for POST /pipelines."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=1000)
    config: PipelineConfig | None = None
    ingestion: IngestionConfig | None = None
    retrieval: RetrievalConfig | None = None
    generation: GenerationConfig | None = None
    evaluation: EvaluationConfig | None = None

    @model_validator(mode="after")
    def populate_from_config(self) -> PipelineCreateRequest:
        if self.config:
            if not self.name:
                self.name = self.config.name
            if self.description is None:
                self.description = self.config.description
            if self.ingestion is None:
                self.ingestion = self.config.ingestion
            if self.retrieval is None:
                self.retrieval = self.config.retrieval
            if self.generation is None:
                self.generation = self.config.generation
            if self.evaluation is None:
                self.evaluation = self.config.evaluation
        if not self.name:
            raise ValueError("name is required")
        if self.ingestion is None:
            self.ingestion = IngestionConfig()
        if self.retrieval is None:
            self.retrieval = RetrievalConfig()
        if self.generation is None:
            self.generation = GenerationConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()
        return self

    def to_config(self) -> PipelineConfig:
        return PipelineConfig(
            name=self.name,
            description=self.description,
            ingestion=self.ingestion,
            retrieval=self.retrieval,
            generation=self.generation,
            evaluation=self.evaluation,
        )


class PipelineUpdateRequest(BaseModel):
    """Request payload for PATCH /pipelines/{id}.

    Nested objects are merged with existing configuration values.
    """

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=1000)
    config_updates: dict[str, Any] | None = None
    ingestion: dict[str, Any] | None = None
    retrieval: dict[str, Any] | None = None
    generation: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None

    @model_validator(mode="after")
    def populate_config_updates(self) -> PipelineUpdateRequest:
        if self.config_updates:
            if self.ingestion is None and "ingestion" in self.config_updates:
                self.ingestion = self.config_updates["ingestion"]
            if self.retrieval is None and "retrieval" in self.config_updates:
                self.retrieval = self.config_updates["retrieval"]
            if self.generation is None and "generation" in self.config_updates:
                self.generation = self.config_updates["generation"]
            if self.evaluation is None and "evaluation" in self.config_updates:
                self.evaluation = self.config_updates["evaluation"]
        return self


class PipelineResponse(BaseModel):
    """Full pipeline details with current version and aggregate metrics."""

    id: str
    name: str
    description: str | None
    version: int
    status: str
    config: PipelineConfig
    created_at: datetime
    updated_at: datetime
    aggregate_scores: dict[str, float | None] | None = None
    latest_metrics: dict[str, Any] | None = None


class PipelineRunItem(BaseModel):
    """Summary of a single run for run history."""

    id: str
    pipeline_id: str
    pipeline_version: int | None
    query: str
    generation: str | None
    status: str
    retrieval_latency_ms: int | None
    latency_ms: int | None
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    model_used: str | None
    overall_score: float | None
    faithfulness: float | None
    answer_relevance: float | None
    context_precision: float | None
    context_recall: float | None
    created_at: datetime


class PaginatedRunsResponse(BaseModel):
    """Paginated list of pipeline runs."""

    items: list[PipelineRunItem]
    total: int
    page: int
    page_size: int
    has_next: bool


class DailyQueryCount(BaseModel):
    date: str
    count: int


class AnalyticsResponse(BaseModel):
    """Aggregate latency, quality, cost, and usage analytics for a pipeline."""

    pipeline_id: str
    version: int
    retrieval_latency_p50: float | None
    retrieval_latency_p95: float | None
    retrieval_latency_p99: float | None
    avg_generation_latency_ms: float | None
    avg_evaluation_scores: dict[str, float | None]
    estimated_cost_per_query: float | None
    daily_query_counts: list[DailyQueryCount]
    total_queries_30d: int
    percentile_method: str = "linear_interpolation"
    cost_trends_30d: list[dict[str, Any]] | None = None
    recent_failed_runs: list[dict[str, Any]] | None = None
    daily_scores_7d: list[dict[str, Any]] | None = None


class SuggestionItem(BaseModel):
    """Actionable improvement recommendation based on recent evaluation records."""

    rule_id: str
    title: str
    reason: str
    supporting_metric: str
    observed_value: float | None
    threshold: float
    proposed_change: dict[str, Any]


class SuggestionsResponse(BaseModel):
    """Rule-based optimizer recommendations."""

    pipeline_id: str
    evaluation_sample_count: int
    has_sufficient_data: bool
    suggestions: list[SuggestionItem]
