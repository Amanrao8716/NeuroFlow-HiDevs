"""Unit tests for query processor expansion, classification, and metadata extraction."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest

from pipelines.retrieval.query_processor import ProcessedQuery, QueryProcessor


@pytest.mark.asyncio
async def test_classify_all_query_types():
    processor = QueryProcessor()

    # Factual
    assert processor._classify_query_type("What is the speed of light?") == "factual"
    assert processor._classify_query_type("Who was Nikola Tesla?") == "factual"

    # Analytical
    assert processor._classify_query_type("Why do leaves turn yellow in autumn?") == "analytical"
    assert processor._classify_query_type("Explain the causes of inflation.") == "analytical"
    assert processor._classify_query_type("How does photosynthesis work?") == "analytical"

    # Comparative
    assert processor._classify_query_type("Compare Postgres vs MySQL performance") == "comparative"
    assert processor._classify_query_type("What is the difference between TCP and UDP?") == "comparative"
    assert processor._classify_query_type("Pros and cons of monolithic vs microservices") == "comparative"

    # Procedural
    assert processor._classify_query_type("How to install Docker on Ubuntu") == "procedural"
    assert processor._classify_query_type("Step by step guide to baking bread") == "procedural"
    assert processor._classify_query_type("Instructions for resetting router password") == "procedural"


@pytest.mark.asyncio
async def test_extract_metadata_filters_for_2023_climate_documents():
    processor = QueryProcessor()

    filters1 = processor._extract_metadata_filters("2023 climate documents")
    assert filters1.get("year") == 2023
    assert filters1.get("topic") == "climate"

    filters2 = processor._extract_metadata_filters("documents from 2023 about climate change")
    assert filters2.get("year") == 2023
    assert filters2.get("topic") == "climate"

    filters3 = processor._extract_metadata_filters("in 2021 financial pdf reports")
    assert filters3.get("year") == 2021
    assert filters3.get("source_type") == "pdf"


@pytest.mark.asyncio
async def test_extract_metadata_filters_ignores_non_year_numbers():
    processor = QueryProcessor()
    # Code or port numbers like 8000 or error 4040 should not be extracted as years
    filters = processor._extract_metadata_filters("Error code 5432 occurred on port 8080")
    assert "year" not in filters


@pytest.mark.asyncio
async def test_query_expansion_with_llm_json():
    mock_llm = SimpleNamespace(
        chat=AsyncMock(
            return_value=SimpleNamespace(
                content='["how does indexing function in postgres", "database index mechanisms"]'
            )
        )
    )
    processor = QueryProcessor(llm_client=mock_llm)
    result = await processor.process("How does indexing work?")

    assert result.query == "How does indexing work?"
    assert len(result.expanded_queries) == 2
    assert "how does indexing function in postgres" in result.expanded_queries
    assert len(result.all_queries) == 3


@pytest.mark.asyncio
async def test_query_expansion_fallback_on_llm_failure():
    mock_llm = SimpleNamespace(
        chat=AsyncMock(side_effect=RuntimeError("LLM API rate limit"))
    )
    processor = QueryProcessor(llm_client=mock_llm)

    # Should not raise exception; falls back gracefully
    result = await processor.process("What is pgvector?")
    assert result.query == "What is pgvector?"
    assert isinstance(result.expanded_queries, list)
    assert len(result.all_queries) >= 1


@pytest.mark.asyncio
async def test_query_expansion_deduplication():
    # LLM returns identical or duplicate phrases
    mock_llm = SimpleNamespace(
        chat=AsyncMock(
            return_value=SimpleNamespace(
                content='["what is pgvector?", "WHAT IS PGVECTOR", "overview of pgvector"]'
            )
        )
    )
    processor = QueryProcessor(llm_client=mock_llm)
    result = await processor.process("what is pgvector?")

    # Duplicates of original should be removed
    assert "overview of pgvector" in result.expanded_queries
    assert len(result.expanded_queries) == 1
    assert len(result.all_queries) == 2
