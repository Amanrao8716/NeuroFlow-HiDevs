from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.models.pipeline import (
    ChunkingStrategy,
    IngestionConfig,
    PipelineConfig,
    RetrievalConfig,
)


def test_missing_pipeline_name_is_rejected() -> None:
    with pytest.raises(ValidationError):
        PipelineConfig()


def test_invalid_chunk_size_is_rejected() -> None:
    with pytest.raises(ValidationError):
        IngestionConfig(chunk_size_tokens=0)


def test_invalid_provider_enum_is_rejected() -> None:
    with pytest.raises(ValidationError):
        IngestionConfig(chunking_strategy="magical")


def test_malformed_nested_config_is_rejected() -> None:
    with pytest.raises(ValidationError):
        PipelineConfig(name="test", retrieval={"dense_k": 0})


def test_invalid_overlap_is_rejected() -> None:
    with pytest.raises(ValidationError):
        IngestionConfig(chunk_size_tokens=10, chunk_overlap_tokens=10)


def test_valid_pipeline_config_is_accepted() -> None:
    config = PipelineConfig(name="support", ingestion={"chunking_strategy": "fixed"})
    assert config.name == "support"
    assert config.ingestion.chunking_strategy is ChunkingStrategy.FIXED
    assert RetrievalConfig().dense_k > 0
