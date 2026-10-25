"""Pydantic models for Task 9 Fine-tuning Pipeline, Preview, and DPO."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field


class CreateFinetuneJobRequest(BaseModel):
    """Request payload to create a fine-tuning job."""

    model_config = ConfigDict(extra="forbid")

    base_model: str = Field(
        default="gpt-4o-mini",
        description="Base model to fine-tune (e.g., gpt-4o-mini, gpt-4o, babbage-002)",
    )
    pipeline_id: UUID | None = Field(
        default=None,
        description="Optional pipeline ID to restrict training data extraction",
    )
    min_quality_score: float = Field(
        default=0.82,
        ge=0.0,
        le=1.0,
        description="Minimum quality score threshold for training pairs",
    )
    format: str = Field(
        default="sft",
        pattern="^(sft|dpo)$",
        description="Training format: 'sft' (OpenAI JSONL) or 'dpo' (Direct Preference Optimization)",
    )


class ExtractionReport(BaseModel):
    """Extraction and validation summary for training pairs."""

    model_config = ConfigDict(extra="forbid")

    eligible_count: int
    accepted_count: int
    rejected_count: int
    rejection_reasons: dict[str, int] = Field(default_factory=dict)
    average_quality_score: float | None = None
    date_range: str | None = None


class FinetuneJobResponse(BaseModel):
    """Detailed response for a fine-tuning job."""

    id: str
    provider_job_id: str | None = None
    base_model: str
    status: str
    training_pair_count: int | None = None
    mlflow_run_id: str | None = None
    mlflow_run_url: str | None = None
    fine_tuned_model: str | None = None
    metrics: dict[str, Any] | None = None
    trained_tokens: int | None = None
    error_message: str | None = None
    extraction_report: dict[str, Any] | None = None
    pipeline_id: str | None = None
    created_at: datetime
    completed_at: datetime | None = None


class FinetuneJobListResponse(BaseModel):
    """List of fine-tuning jobs."""

    jobs: list[FinetuneJobResponse]
    total: int


class TrainingDataPreviewItem(BaseModel):
    """Single item preview of eligible training pairs."""

    id: str
    run_id: str
    user_message: str
    assistant_message: str
    quality_score: float | None
    faithfulness: float | None
    user_rating: int | None
    token_count: int
    citations_count: int
    has_valid_citations: bool


class TrainingDataPreviewResponse(BaseModel):
    """Read-only preview response of eligible training pairs without mutating state."""

    items: list[TrainingDataPreviewItem]
    total_eligible: int


class DPOPreferenceItem(BaseModel):
    """Pair of chosen and rejected responses for a query."""

    prompt: str
    chosen: str
    rejected: str


class DPOPreferenceResponse(BaseModel):
    """List of DPO preference pairs."""

    items: list[DPOPreferenceItem]
    total: int
