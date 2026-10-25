"""Tests for strict Pydantic PipelineConfig and Rule-Based Optimizer."""

import pytest
from pydantic import ValidationError

from backend.models.pipeline import (
    ChunkingStrategy,
    EvaluationConfig,
    ExtractorType,
    GenerationConfig,
    IngestionConfig,
    PipelineConfig,
    PipelineCreateRequest,
    PipelineUpdateRequest,
    RerankerType,
    RetrievalConfig,
    SystemPromptVariant,
)
from backend.services.pipeline_optimizer import PipelineOptimizer


def test_valid_pipeline_config():
    """Verify that a valid pipeline configuration validates cleanly."""
    config = PipelineConfig(
        name="Production Enterprise RAG",
        description="Standard QA pipeline for customer documentation",
        ingestion=IngestionConfig(
            chunking_strategy=ChunkingStrategy.AUTO,
            chunk_size_tokens=512,
            chunk_overlap_tokens=64,
            extractors_enabled=[ExtractorType.PDF, ExtractorType.CSV],
        ),
        retrieval=RetrievalConfig(
            dense_k=20,
            sparse_k=20,
            reranker=RerankerType.CROSS_ENCODER,
            top_k_after_rerank=5,
            query_expansion=True,
            metadata_filters_enabled=True,
        ),
        generation=GenerationConfig(
            model_routing="gpt-4o",
            max_context_tokens=4000,
            temperature=0.3,
            system_prompt_variant="detailed",
        ),
        evaluation=EvaluationConfig(
            auto_evaluate=True,
            training_threshold=0.85,
        ),
    )
    assert config.name == "Production Enterprise RAG"
    assert config.ingestion.chunk_size_tokens == 512
    assert config.retrieval.dense_k == 20
    assert config.generation.temperature == 0.3
    assert config.evaluation.training_threshold == 0.85


def test_missing_required_fields():
    """Verify that missing required top-level or request fields raise ValidationError."""
    with pytest.raises(ValidationError) as exc:
        PipelineConfig.model_validate({})
    assert "name" in str(exc.value)

    with pytest.raises(ValidationError) as exc:
        PipelineCreateRequest.model_validate({})
    assert "name" in str(exc.value)


def test_overlap_greater_than_or_equal_to_chunk_size():
    """Chunk overlap must be strictly less than chunk_size_tokens."""
    with pytest.raises(ValidationError) as exc:
        IngestionConfig(
            chunking_strategy=ChunkingStrategy.FIXED,
            chunk_size_tokens=256,
            chunk_overlap_tokens=256,
        )
    assert "must be strictly less than chunk_size_tokens" in str(exc.value)

    with pytest.raises(ValidationError) as exc:
        IngestionConfig(
            chunking_strategy=ChunkingStrategy.FIXED,
            chunk_size_tokens=256,
            chunk_overlap_tokens=300,
        )
    assert "must be strictly less than chunk_size_tokens" in str(exc.value)


def test_invalid_ranges_and_types():
    """Verify numeric range boundaries."""
    # Negative chunk_size
    with pytest.raises(ValidationError):
        IngestionConfig(chunk_size_tokens=-10)

    # Zero or negative dense_k
    with pytest.raises(ValidationError):
        RetrievalConfig(dense_k=0)

    # Temperature out of bounds
    with pytest.raises(ValidationError):
        GenerationConfig(temperature=2.5)
    with pytest.raises(ValidationError):
        GenerationConfig(temperature=-0.1)

    # Training threshold out of bounds
    with pytest.raises(ValidationError):
        EvaluationConfig(training_threshold=1.5)
    with pytest.raises(ValidationError):
        EvaluationConfig(training_threshold=-0.05)


def test_invalid_enums():
    """Verify that unsupported strategies raise ValidationError."""
    with pytest.raises(ValidationError):
        IngestionConfig(chunking_strategy="unsupported_chunking")

    with pytest.raises(ValidationError):
        RetrievalConfig(reranker="unsupported_reranker")

    with pytest.raises(ValidationError):
        GenerationConfig(system_prompt_variant="unsupported_variant")

    with pytest.raises(ValidationError):
        GenerationConfig(model_routing="")


def test_rejects_unknown_fields_top_level():
    """Extra fields at top level must be strictly forbidden."""
    raw = {
        "name": "Pipeline",
        "description": "Desc",
        "unknown_extra_field": 12345,
        "ingestion": {
            "chunking_strategy": "auto",
            "chunk_size_tokens": 512,
            "chunk_overlap_tokens": 50,
        },
        "retrieval": {},
        "generation": {},
        "evaluation": {},
    }
    with pytest.raises(ValidationError) as exc:
        PipelineConfig.model_validate(raw)
    assert "extra_forbidden" in str(exc.value) or "Extra inputs are not permitted" in str(exc.value)


def test_rejects_unknown_fields_nested_levels():
    """Extra fields at nested configuration levels must be strictly forbidden."""
    # Unknown in ingestion
    with pytest.raises(ValidationError) as exc:
        IngestionConfig.model_validate({
            "chunking_strategy": "auto",
            "chunk_size_tokens": 512,
            "chunk_overlap_tokens": 50,
            "unsupported_nested_field": "error",
        })
    assert "Extra inputs are not permitted" in str(exc.value)

    # Unknown in retrieval
    with pytest.raises(ValidationError) as exc:
        RetrievalConfig.model_validate({
            "dense_k": 10,
            "unknown_retrieval_option": True,
        })
    assert "Extra inputs are not permitted" in str(exc.value)

    # Unknown in generation
    with pytest.raises(ValidationError) as exc:
        GenerationConfig.model_validate({
            "temperature": 0.5,
            "bogus_setting": "nope",
        })
    assert "Extra inputs are not permitted" in str(exc.value)

    # Unknown in evaluation
    with pytest.raises(ValidationError) as exc:
        EvaluationConfig.model_validate({
            "auto_evaluate": True,
            "hallucination_meter": 99,
        })
    assert "Extra inputs are not permitted" in str(exc.value)


# -------------------------------------------------------------------------
# Optimizer Tests
# -------------------------------------------------------------------------

def test_optimizer_insufficient_data():
    """When fewer than min_sample_count evaluations exist, no recommendations are made."""
    optimizer = PipelineOptimizer(min_samples=5)
    config = PipelineConfig(
        name="Test",
        description="Desc",
        ingestion=IngestionConfig(),
        retrieval=RetrievalConfig(top_k_after_rerank=10, dense_k=20),
        generation=GenerationConfig(),
        evaluation=EvaluationConfig(),
    )

    # Only 3 evaluations (below minimum of 5)
    recent_evals = [
        {"context_precision": 0.2, "context_recall": 0.2, "answer_relevance": 0.2, "faithfulness": 0.2}
        for _ in range(3)
    ]
    suggestions = optimizer.analyze(config, recent_evals)
    assert len(suggestions) == 0


def test_optimizer_healthy_pipeline_no_suggestions():
    """A well-performing pipeline with high scores yields no suggestions."""
    optimizer = PipelineOptimizer(min_samples=5)
    config = PipelineConfig(
        name="Healthy Pipeline",
        description="High scores",
        ingestion=IngestionConfig(),
        retrieval=RetrievalConfig(top_k_after_rerank=5, dense_k=20),
        generation=GenerationConfig(),
        evaluation=EvaluationConfig(),
    )
    recent_evals = [
        {"context_precision": 0.90, "context_recall": 0.88, "answer_relevance": 0.92, "faithfulness": 0.95}
        for _ in range(6)
    ]
    suggestions = optimizer.analyze(config, recent_evals)
    assert len(suggestions) == 0


def test_optimizer_low_context_precision_rule():
    """Low context precision recommends reducing top_k_after_rerank."""
    optimizer = PipelineOptimizer(min_samples=5, precision_threshold=0.60)
    config = PipelineConfig(
        name="Test",
        description="Desc",
        ingestion=IngestionConfig(),
        retrieval=RetrievalConfig(top_k_after_rerank=8, dense_k=20),
        generation=GenerationConfig(),
        evaluation=EvaluationConfig(),
    )
    # Average precision = 0.40 (< 0.60)
    recent_evals = [
        {"context_precision": 0.40, "context_recall": 0.85, "answer_relevance": 0.85, "faithfulness": 0.85}
        for _ in range(5)
    ]
    suggestions = optimizer.analyze(config, recent_evals)
    rule_ids = [s.rule_id for s in suggestions]
    assert "reduce_top_k_after_rerank" in rule_ids
    sug = next(s for s in suggestions if s.rule_id == "reduce_top_k_after_rerank")
    assert sug.proposed_change["retrieval"]["top_k_after_rerank"] < 8
    assert sug.observed_value == 0.40


def test_optimizer_low_context_recall_rule():
    """Low context recall recommends increasing dense_k."""
    optimizer = PipelineOptimizer(min_samples=5, recall_threshold=0.60)
    config = PipelineConfig(
        name="Test",
        description="Desc",
        ingestion=IngestionConfig(),
        retrieval=RetrievalConfig(dense_k=15, top_k_after_rerank=5),
        generation=GenerationConfig(),
        evaluation=EvaluationConfig(),
    )
    recent_evals = [
        {"context_precision": 0.85, "context_recall": 0.45, "answer_relevance": 0.85, "faithfulness": 0.85}
        for _ in range(5)
    ]
    suggestions = optimizer.analyze(config, recent_evals)
    rule_ids = [s.rule_id for s in suggestions]
    assert "increase_dense_k" in rule_ids
    sug = next(s for s in suggestions if s.rule_id == "increase_dense_k")
    assert sug.proposed_change["retrieval"]["dense_k"] > 15
    assert sug.observed_value == 0.45


def test_optimizer_low_answer_relevance_rule():
    """Low answer relevance recommends enabling query expansion or tuning prompt."""
    optimizer = PipelineOptimizer(min_samples=5, relevance_threshold=0.65)
    config = PipelineConfig(
        name="Test",
        description="Desc",
        ingestion=IngestionConfig(),
        retrieval=RetrievalConfig(query_expansion=False),
        generation=GenerationConfig(system_prompt_variant="default"),
        evaluation=EvaluationConfig(),
    )
    recent_evals = [
        {"context_precision": 0.85, "context_recall": 0.85, "answer_relevance": 0.50, "faithfulness": 0.85}
        for _ in range(5)
    ]
    suggestions = optimizer.analyze(config, recent_evals)
    rule_ids = [s.rule_id for s in suggestions]
    # Rule id from pipeline_optimizer.py is "enable_query_expansion_or_adjust_prompt"
    assert any("query_expansion" in r for r in rule_ids)


def test_optimizer_low_faithfulness_rule():
    """Low faithfulness recommends precision prompt or reranking."""
    optimizer = PipelineOptimizer(min_samples=5, faithfulness_threshold=0.70)
    config = PipelineConfig(
        name="Test",
        description="Desc",
        ingestion=IngestionConfig(),
        retrieval=RetrievalConfig(),
        generation=GenerationConfig(temperature=0.7, system_prompt_variant="default"),
        evaluation=EvaluationConfig(),
    )
    recent_evals = [
        {"context_precision": 0.85, "context_recall": 0.85, "answer_relevance": 0.85, "faithfulness": 0.55}
        for _ in range(5)
    ]
    suggestions = optimizer.analyze(config, recent_evals)
    rule_ids = [s.rule_id for s in suggestions]
    # Rule id from pipeline_optimizer.py is "reduce_temperature_and_tighten_grounding"
    assert any("faithfulness" in r or "temperature" in r or "grounding" in r for r in rule_ids)
    sug = next(s for s in suggestions if "temperature" in s.rule_id or "faithfulness" in s.rule_id or "grounding" in s.rule_id)
    # Temperature should be reduced (lowered) in proposed change
    assert sug.proposed_change["generation"]["temperature"] <= 0.5
