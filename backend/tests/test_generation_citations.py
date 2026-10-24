"""Tests for citation extraction, validation, and invalid reference handling."""

import pytest
from pipelines.generation.citations import CitationTracker
from pipelines.generation.prompt_builder import SourceMapEntry


@pytest.fixture
def sample_source_map() -> dict[int, SourceMapEntry]:
    return {
        1: SourceMapEntry(
            source_number=1,
            source_label="Source 1",
            chunk_id="chunk-uuid-1",
            document_id="doc-uuid-1",
            document_name="deep_learning.pdf",
            page_number=5,
            content_preview="Deep learning is part of a broader family of machine learning methods.",
        ),
        2: SourceMapEntry(
            source_number=2,
            source_label="Source 2",
            chunk_id="chunk-uuid-2",
            document_id="doc-uuid-2",
            document_name="transformers.pdf",
            page_number=10,
            content_preview="Attention is all you need proposes the Transformer architecture.",
        ),
    }


def test_citation_tracker_valid_citations(sample_source_map):
    tracker = CitationTracker()
    text = (
        "According to research, neural networks learn representations [Source 1]. "
        "Furthermore, attention mechanisms improve performance significantly [Source 2]."
    )
    parsed = tracker.extract_and_validate(text, sample_source_map)

    assert len(parsed.citations) == 2
    assert len(parsed.invalid_citations) == 0
    assert parsed.citations[0].reference == "Source 1"
    assert parsed.citations[0].chunk_id == "chunk-uuid-1"
    assert parsed.citations[0].document_name == "deep_learning.pdf"
    assert parsed.citations[0].page_number == 5
    assert parsed.citations[1].reference == "Source 2"
    assert parsed.citations[1].chunk_id == "chunk-uuid-2"
    assert parsed.cited_chunk_ids == ["chunk-uuid-1", "chunk-uuid-2"]


def test_citation_tracker_deduplication(sample_source_map):
    tracker = CitationTracker()
    text = (
        "As stated [Source 1], neural networks learn. Also, repeat reference [Source 1] "
        "and another one [Source 2] followed by [Source 1] again."
    )
    parsed = tracker.extract_and_validate(text, sample_source_map)

    assert len(parsed.citations) == 2
    assert parsed.citations[0].reference == "Source 1"
    assert parsed.citations[1].reference == "Source 2"
    assert parsed.cited_chunk_ids == ["chunk-uuid-1", "chunk-uuid-2"]


def test_citation_tracker_invalid_and_out_of_bounds(sample_source_map):
    tracker = CitationTracker()
    text = (
        "Here is a valid claim [Source 1]. Here is an hallucinated citation [Source 99] "
        "and another invalid reference [Source 42]."
    )
    parsed = tracker.extract_and_validate(text, sample_source_map)

    assert len(parsed.citations) == 1
    assert parsed.citations[0].reference == "Source 1"
    assert parsed.citations[0].is_valid is True

    assert len(parsed.invalid_citations) == 2
    assert parsed.invalid_citations[0].reference == "Source 99"
    assert parsed.invalid_citations[0].is_valid is False
    assert parsed.invalid_citations[0].chunk_id is None
    assert "not provided in the retrieval context" in parsed.invalid_citations[0].error

    assert parsed.invalid_citations[1].reference == "Source 42"
    assert parsed.invalid_citations[1].is_valid is False


def test_citation_tracker_empty_or_no_citations(sample_source_map):
    tracker = CitationTracker()
    assert len(tracker.extract_and_validate("", sample_source_map).all_citations) == 0
    assert len(tracker.extract_and_validate("Just text with no citations", sample_source_map).all_citations) == 0
