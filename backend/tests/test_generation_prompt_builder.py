"""Tests for prompt builder and dynamic query instructions."""

import pytest
from pipelines.generation.prompt_builder import (
    BASE_SYSTEM_PROMPT,
    PromptBuilder,
    SourceMapEntry,
)
from pipelines.retrieval.context_assembler import AssembledContext
from pipelines.retrieval.retriever import RetrievalResult


def test_prompt_builder_query_types():
    builder = PromptBuilder()

    # Factual
    prompt_f = builder.build("What is pgvector?", query_type="factual")
    assert "Provide a direct, concise answer" in prompt_f.system_prompt
    assert "What is pgvector?" in prompt_f.user_prompt

    # Analytical
    prompt_a = builder.build("Why compare HNSW and IVFFlat?", query_type="analytical")
    assert "Analyze and synthesize across the provided sources" in prompt_a.system_prompt

    # Comparative
    prompt_c = builder.build("Compare Dense and Sparse search", query_type="comparative")
    assert "Organize your response as a structured comparison" in prompt_c.system_prompt

    # Procedural
    prompt_p = builder.build("How do I install postgres?", query_type="procedural")
    assert "Provide numbered steps" in prompt_p.system_prompt

    # Unknown type defaults safely
    prompt_unknown = builder.build("Hello world", query_type="conversational")
    assert "Provide a clear and well-reasoned answer directly based on the context" in prompt_unknown.system_prompt


def test_prompt_builder_source_map_and_context():
    builder = PromptBuilder()

    chunk1 = RetrievalResult(
        chunk_id="c1-uuid",
        document_id="d1-uuid",
        content="PostgreSQL pgvector extension provides vector search.",
        score=0.95,
        rank=1,
        source_type="pdf",
        filename="pgvector_guide.pdf",
        page_number=3,
        metadata={},
    )
    chunk2 = RetrievalResult(
        chunk_id="c2-uuid",
        document_id="d2-uuid",
        content="HNSW graphs offer fast approximate nearest neighbors.",
        score=0.88,
        rank=2,
        source_type="pdf",
        filename="hnsw_paper.pdf",
        page_number=12,
        metadata={},
    )

    assembled = AssembledContext(
        context="[Source 1 — pgvector_guide.pdf, page 3]\nPostgreSQL pgvector extension provides vector search.\n\n[Source 2 — hnsw_paper.pdf, page 12]\nHNSW graphs offer fast approximate nearest neighbors.",
        chunks_used=2,
        total_tokens=40,
        sources=[
            {
                "source_index": 1,
                "chunk_id": "c1-uuid",
                "document_id": "d1-uuid",
                "filename": "pgvector_guide.pdf",
                "page_number": 3,
                "score": 0.95,
            },
            {
                "source_index": 2,
                "chunk_id": "c2-uuid",
                "document_id": "d2-uuid",
                "filename": "hnsw_paper.pdf",
                "page_number": 12,
                "score": 0.88,
            },
        ],
    )

    prompt = builder.build("Tell me about pgvector", query_type="factual", context=assembled)

    assert 1 in prompt.source_map
    assert 2 in prompt.source_map
    assert prompt.source_map[1].chunk_id == "c1-uuid"
    assert prompt.source_map[1].document_name == "pgvector_guide.pdf"
    assert prompt.source_map[1].page_number == 3
    assert prompt.source_map[2].chunk_id == "c2-uuid"
    assert prompt.source_map[2].document_name == "hnsw_paper.pdf"
    assert prompt.source_map[2].page_number == 12

    assert "<context>" in prompt.user_prompt
    assert "</context>" in prompt.user_prompt
    assert "[Source 1 — pgvector_guide.pdf, page 3]" in prompt.user_prompt


def test_prompt_builder_empty_context():
    builder = PromptBuilder()
    prompt = builder.build("Empty test", query_type="factual", context=None)
    assert len(prompt.source_map) == 0
    assert "No relevant context found" in prompt.user_prompt
