"""Citation extraction, validation, and chunk mapping for Task 6 RAG generation."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from pipelines.generation.prompt_builder import SourceMapEntry

# Matches [Source N], [Source 1, Source 2], [Source 1, 2], and variations
CITATION_REGEX = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)


@dataclass
class Citation:
    """Structured representation of a single cited source."""

    reference: str  # e.g. "Source 1"
    source_number: int  # 1
    chunk_id: str | None
    document_name: str | None
    page_number: int | None
    content_preview: str | None
    is_valid: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference": self.reference,
            "source_number": self.source_number,
            "chunk_id": self.chunk_id,
            "document_name": self.document_name,
            "page_number": self.page_number,
            "content_preview": self.content_preview,
            "is_valid": self.is_valid,
            "error": self.error,
        }


@dataclass
class ParsedCitations:
    """Aggregated citations extracted from generation output."""

    citations: list[Citation] = field(default_factory=list)
    invalid_citations: list[Citation] = field(default_factory=list)
    cited_chunk_ids: list[str] = field(default_factory=list)

    @property
    def all_citations(self) -> list[Citation]:
        return self.citations + self.invalid_citations

    def to_dict_list(self) -> list[dict[str, Any]]:
        return [c.to_dict() for c in self.all_citations]


class CitationTracker:
    """Extracts, validates, deduplicates, and maps source citations against the prompt source map."""

    def __init__(self, citation_regex: re.Pattern = CITATION_REGEX):
        self.citation_regex = citation_regex

    def extract_and_validate(
        self,
        generation_text: str,
        source_map: dict[int, SourceMapEntry],
    ) -> ParsedCitations:
        """Parse all [Source N] citations in generation_text and validate against source_map.

        Rules:
        - Extracts all cited source numbers.
        - Resolves valid source numbers to source map entries.
        - Deduplicates repeated citations while preserving first-seen order.
        - Flags source numbers not in source_map as invalid (e.g. [Source 99]).
        - Never silently maps invalid citations to valid chunks.

        Args:
            generation_text: The complete generated response string.
            source_map: Mapping of integer source_number to SourceMapEntry.

        Returns:
            ParsedCitations with valid citations, invalid citations, and cited chunk IDs.
        """
        if not generation_text:
            return ParsedCitations()

        matches = self.citation_regex.findall(generation_text)
        if not matches:
            return ParsedCitations()

        seen_sources: set[int] = set()
        valid_citations: list[Citation] = []
        invalid_citations: list[Citation] = []
        cited_chunk_ids: list[str] = []

        for match_str in matches:
            try:
                src_num = int(match_str)
            except ValueError:
                continue

            if src_num in seen_sources:
                continue
            seen_sources.add(src_num)

            reference_label = f"Source {src_num}"

            if src_num in source_map:
                entry = source_map[src_num]
                c = Citation(
                    reference=reference_label,
                    source_number=src_num,
                    chunk_id=entry.chunk_id,
                    document_name=entry.document_name,
                    page_number=entry.page_number,
                    content_preview=entry.content_preview,
                    is_valid=True,
                    error=None,
                )
                valid_citations.append(c)
                if entry.chunk_id and entry.chunk_id not in cited_chunk_ids:
                    cited_chunk_ids.append(entry.chunk_id)
            else:
                c = Citation(
                    reference=reference_label,
                    source_number=src_num,
                    chunk_id=None,
                    document_name=None,
                    page_number=None,
                    content_preview=None,
                    is_valid=False,
                    error=f"Source {src_num} was not provided in the retrieval context.",
                )
                invalid_citations.append(c)

        return ParsedCitations(
            citations=valid_citations,
            invalid_citations=invalid_citations,
            cited_chunk_ids=cited_chunk_ids,
        )
