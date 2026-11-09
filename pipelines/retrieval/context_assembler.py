"""Context assembly with strict token budgeting, source headers, and sentence boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
from typing import Any

from pipelines.retrieval.retriever import RetrievalResult

logger = logging.getLogger("neuroflow.retrieval.context_assembler")

DEFAULT_TOKEN_BUDGET = 4000
DEFAULT_ENCODING = "cl100k_base"


class TokenCounter:
    """Reusable token counter backed by tiktoken with character ratio fallback."""

    def __init__(self, encoding_name: str = DEFAULT_ENCODING):
        try:
            import tiktoken

            self._encoder = tiktoken.get_encoding(encoding_name)
        except Exception:
            self._encoder = None

    def count(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        if self._encoder is not None:
            return len(self._encoder.encode(text))
        # Deterministic fallback (~4 chars per token)
        return max(1, (len(text) + 3) // 4)


@dataclass
class AssembledContext:
    """Structured result of assembled retrieval context."""

    context: str
    chunks_used: int
    total_tokens: int
    sources: list[dict[str, Any]] = field(default_factory=list)


def format_source_header(index: int, item: RetrievalResult) -> str:
    """Format a source header like [Source 1 — doc.pdf, page 3]."""
    details = []
    if item.filename:
        details.append(item.filename)
    elif item.source_type and item.source_type != "text":
        details.append(item.source_type)

    if item.page_number is not None:
        details.append(f"page {item.page_number}")

    if not details:
        return f"[Source {index}]"
    if len(details) == 1:
        return f"[Source {index} — {details[0]}]"
    return f"[Source {index} — {details[0]}, {', '.join(details[1:])}]"


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences cleanly."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


class ContextAssembler:
    """Assembles ranked chunks into a context string within a strict token budget."""

    def __init__(
        self,
        token_counter: TokenCounter | None = None,
        default_budget: int = DEFAULT_TOKEN_BUDGET,
    ):
        self.counter = token_counter or TokenCounter()
        self.default_budget = default_budget

    def assemble(
        self,
        chunks: list[RetrievalResult],
        max_tokens: int | None = None,
    ) -> AssembledContext:
        """Assemble top chunks into context without exceeding max_tokens.

        Args:
            chunks: Ranked candidate chunks.
            max_tokens: Maximum allowed token budget (inclusive).

        Returns:
            AssembledContext with final text, counts, and source references.
        """
        budget = max_tokens if max_tokens is not None else self.default_budget
        if budget <= 0 or not chunks:
            return AssembledContext(
                context="",
                chunks_used=0,
                total_tokens=0,
                sources=[],
            )

        used_chunks = 0
        seen_chunk_ids: set[str] = set()
        sources: list[dict[str, Any]] = []
        assembled_sections: list[str] = []
        current_tokens = 0

        for chunk in chunks:
            if chunk.chunk_id in seen_chunk_ids:
                continue

            content = chunk.content.strip()
            if not content:
                continue

            source_index = len(sources) + 1
            header = format_source_header(source_index, chunk)

            # Check if adding separator is required
            separator = "\n\n" if assembled_sections else ""
            candidate_section = f"{header}\n{content}"
            proposed_text = separator + candidate_section
            cost = self.counter.count(proposed_text)

            if current_tokens + cost <= budget:
                assembled_sections.append(candidate_section)
                current_tokens += cost
                used_chunks += 1
                seen_chunk_ids.add(chunk.chunk_id)
                sources.append(
                    {
                        "source_index": source_index,
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "filename": chunk.filename,
                        "page_number": chunk.page_number,
                        "score": chunk.score,
                    }
                )
            else:
                # If chunk cannot fit as a whole, try sentence-boundary aware fit
                # only if at least one complete sentence can fit with header
                remaining_tokens = budget - current_tokens
                header_cost = self.counter.count(separator + header + "\n")
                if remaining_tokens <= header_cost:
                    # Not even header fits
                    continue

                available_for_content = remaining_tokens - header_cost
                sentences = _split_into_sentences(content)
                partial_sentences: list[str] = []
                partial_cost = 0

                for s in sentences:
                    s_cost = self.counter.count(" " + s if partial_sentences else s)
                    if partial_cost + s_cost <= available_for_content:
                        partial_sentences.append(s)
                        partial_cost += s_cost
                    else:
                        break

                if partial_sentences:
                    partial_content = " ".join(partial_sentences).strip()
                    partial_section = f"{header}\n{partial_content}"
                    full_addition = separator + partial_section
                    full_cost = self.counter.count(full_addition)

                    if current_tokens + full_cost <= budget:
                        assembled_sections.append(partial_section)
                        current_tokens += full_cost
                        used_chunks += 1
                        seen_chunk_ids.add(chunk.chunk_id)
                        sources.append(
                            {
                                "source_index": source_index,
                                "chunk_id": chunk.chunk_id,
                                "document_id": chunk.document_id,
                                "filename": chunk.filename,
                                "page_number": chunk.page_number,
                                "score": chunk.score,
                                "truncated_sentence_boundary": True,
                            }
                        )

        final_context = "\n\n".join(assembled_sections)
        final_tokens = self.counter.count(final_context)

        return AssembledContext(
            context=final_context,
            chunks_used=used_chunks,
            total_tokens=final_tokens,
            sources=sources,
        )
