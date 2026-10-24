
"""Chunk extracted document content using configurable strategies."""

import re
from collections.abc import Callable
from typing import Any

import tiktoken

from pipelines.ingestion.models import ExtractedPage


DEFAULT_CHUNK_SIZE = 512
DEFAULT_OVERLAP = 64
SEMANTIC_THRESHOLD = 0.7

_ENCODER = tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    return len(_ENCODER.encode(text))


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _make_chunk(
    content: str,
    index: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "content": content,
        "chunk_index": index,
        "token_count": _token_count(content),
        "metadata": metadata,
    }


def fixed_size_chunks(
    pages: list[ExtractedPage],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[dict[str, Any]]:
    """Create overlapping chunks, preferring nearby sentence boundaries."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0.")

    chunks = []
    current_sentences: list[str] = []
    current_tokens = 0
    page_metadata: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal current_sentences, current_tokens, page_metadata

        if not current_sentences:
            return

        content = " ".join(current_sentences).strip()
        if content:
            chunks.append(
                _make_chunk(
                    content,
                    len(chunks),
                    {"pages": page_metadata.copy(), "strategy": "fixed_size"},
                )
            )

        overlap_sentences: list[str] = []
        overlap_count = 0

        for sentence in reversed(current_sentences):
            count = _token_count(sentence)
            if overlap_count + count > overlap:
                break
            overlap_sentences.insert(0, sentence)
            overlap_count += count

        current_sentences = overlap_sentences
        current_tokens = overlap_count
        page_metadata = []

    for page in pages:
        sentences = _split_sentences(page.content)
        if not sentences and page.content.strip():
            sentences = [page.content.strip()]

        for sentence in sentences:
            count = _token_count(sentence)

            if current_sentences and current_tokens + count > chunk_size:
                flush()

            current_sentences.append(sentence)
            current_tokens += count
            page_info = {
                "page_number": page.page_number,
                "content_type": page.content_type,
            }
            if page_info not in page_metadata:
                page_metadata.append(page_info)

    flush()
    return chunks


async def semantic_chunks(
    pages: list[ExtractedPage],
    embed: Callable,
    threshold: float = SEMANTIC_THRESHOLD,
) -> list[dict[str, Any]]:
    """Split between adjacent sentences when embedding similarity drops."""
    import math

    sentences: list[tuple[str, int, str]] = []
    for page in pages:
        for sentence in _split_sentences(page.content):
            sentences.append((sentence, page.page_number, page.content_type))

    if not sentences:
        return []

    vectors = await embed([sentence[0] for sentence in sentences])
    if len(vectors) != len(sentences):
        raise ValueError("Embedding count does not match sentence count.")

    groups: list[list[tuple[str, int, str]]] = [[]]

    for index, sentence in enumerate(sentences):
        if index > 0:
            left, right = vectors[index - 1], vectors[index]
            dot = sum(a * b for a, b in zip(left, right))
            left_norm = math.sqrt(sum(a * a for a in left))
            right_norm = math.sqrt(sum(b * b for b in right))
            similarity = (
                dot / (left_norm * right_norm)
                if left_norm and right_norm
                else 0.0
            )

            if similarity < threshold:
                groups.append([])

        groups[-1].append(sentence)

    return [
        _make_chunk(
            " ".join(item[0] for item in group),
            index,
            {
                "strategy": "semantic",
                "pages": sorted({item[1] for item in group}),
                "content_types": sorted({item[2] for item in group}),
            },
        )
        for index, group in enumerate(groups)
    ]


def hierarchical_chunks(
    pages: list[ExtractedPage],
) -> list[dict[str, Any]]:
    """Create section parent chunks and subsection child chunks."""
    chunks = []
    sections: list[dict[str, Any]] = []
    current_heading = "Document"
    current_section: list[str] = []
    current_page = 1

    def save_section() -> None:
        if not current_section:
            return

        content = "\n".join(current_section).strip()
        if not content:
            return

        parent_index = len(chunks)
        chunks.append(
            _make_chunk(
                content,
                parent_index,
                {
                    "strategy": "hierarchical",
                    "level": "parent",
                    "heading": current_heading,
                },
            )
        )
        sections.append(
            {
                "heading": current_heading,
                "content": content,
                "parent_chunk_index": parent_index,
            }
        )

    for page in pages:
        heading_hierarchy = page.metadata.get("heading_hierarchy", [])
        is_heading = page.metadata.get("is_heading", False)

        if is_heading and heading_hierarchy:
            save_section()
            current_heading = " > ".join(heading_hierarchy)
            current_section = []
            current_page = page.page_number
        else:
            if not current_section:
                current_page = page.page_number
            current_section.append(page.content)

    save_section()

    for section in sections:
        sentences = _split_sentences(section["content"])
        child_content = ""
        child_tokens = 0

        for sentence in sentences:
            count = _token_count(sentence)

            if child_content and child_tokens + count > DEFAULT_CHUNK_SIZE:
                chunks.append(
                    _make_chunk(
                        child_content.strip(),
                        len(chunks),
                        {
                            "strategy": "hierarchical",
                            "level": "child",
                            "parent_chunk_index": section["parent_chunk_index"],
                            "heading": section["heading"],
                        },
                    )
                )
                child_content = ""
                child_tokens = 0

            child_content += sentence + " "
            child_tokens += count

        if child_content.strip():
            chunks.append(
                _make_chunk(
                    child_content.strip(),
                    len(chunks),
                    {
                        "strategy": "hierarchical",
                        "level": "child",
                        "parent_chunk_index": section["parent_chunk_index"],
                        "heading": section["heading"],
                    },
                )
            )

    return chunks


def select_strategy(pages: list[ExtractedPage]) -> str:
    """Choose a chunking strategy based on extracted document structure."""
    if any(page.content_type == "table" for page in pages):
        return "fixed_size"

    if any(
        page.metadata.get("source_type") == "docx"
        and page.metadata.get("heading_hierarchy")
        for page in pages
    ):
        return "hierarchical"

    if any(
        page.metadata.get("source_type") == "pdf"
        and page.metadata.get("page_count", 0) > 50
        for page in pages
    ):
        return "semantic"

    return "fixed_size"


async def chunk_pages(
    pages: list[ExtractedPage],
    embed: Callable | None = None,
    strategy: str = "auto",
) -> list[dict[str, Any]]:
    """Chunk pages using an explicit or automatically selected strategy."""
    if strategy == "auto":
        strategy = select_strategy(pages)

    if strategy == "fixed_size":
        return fixed_size_chunks(pages)

    if strategy == "hierarchical":
        return hierarchical_chunks(pages)

    if strategy == "semantic":
        if embed is None:
            raise ValueError("Semantic chunking requires an embedding function.")
        return await semantic_chunks(pages, embed)

    raise ValueError(f"Unknown chunking strategy: {strategy}")
