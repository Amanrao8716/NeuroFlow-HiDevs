from __future__ import annotations

import pytest

from pipelines.ingestion.chunker import (
    chunk_pages,
    fixed_size_chunks,
    hierarchical_chunks,
    select_strategy,
    semantic_chunks,
)
from pipelines.ingestion.models import ExtractedPage


@pytest.fixture
def sample_pages() -> list[ExtractedPage]:
    return [
        ExtractedPage(1, "Alpha is about astronomy. Stars shine brightly."),
        ExtractedPage(2, "Beta studies oceans. Waves move constantly."),
    ]


def test_fixed_strategy_splits_and_preserves_metadata(sample_pages: list[ExtractedPage]) -> None:
    chunks = fixed_size_chunks(sample_pages, chunk_size=6, overlap=0)
    assert len(chunks) >= 2
    assert chunks[0]["metadata"]["strategy"] == "fixed_size"
    assert chunks[0]["metadata"]["pages"]


def test_fixed_strategy_rejects_invalid_sizes(sample_pages: list[ExtractedPage]) -> None:
    with pytest.raises(ValueError):
        fixed_size_chunks(sample_pages, chunk_size=2, overlap=2)


@pytest.mark.asyncio
async def test_semantic_strategy_groups_similar_sentences(
    sample_pages: list[ExtractedPage],
) -> None:
    async def embeddings(sentences: list[str]) -> list[list[float]]:
        return [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]

    chunks = await semantic_chunks(sample_pages, embeddings)
    assert len(chunks) == 2
    assert chunks[0]["metadata"]["strategy"] == "semantic"


def test_hierarchical_strategy_creates_parent_and_child() -> None:
    pages = [
        ExtractedPage(
            1,
            "Ocean facts.",
            metadata={"is_heading": True, "heading_hierarchy": ["Science", "Ocean"]},
        ),
        ExtractedPage(1, "Waves carry energy."),
    ]
    chunks = hierarchical_chunks(pages)
    assert [item["metadata"]["level"] for item in chunks] == ["parent", "child"]
    assert chunks[0]["metadata"]["heading"] == "Science > Ocean"


def test_auto_strategy_selection(sample_pages: list[ExtractedPage]) -> None:
    assert select_strategy(sample_pages) == "fixed_size"


@pytest.mark.asyncio
async def test_auto_chunk_dispatch(sample_pages: list[ExtractedPage]) -> None:
    assert await chunk_pages(sample_pages, strategy="auto")
