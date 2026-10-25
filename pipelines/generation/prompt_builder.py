"""Dynamic prompt assembly for Task 6 RAG generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
import html
import re
from typing import Any

from pipelines.retrieval.context_assembler import AssembledContext
from pipelines.retrieval.retriever import RetrievalResult

BASE_SYSTEM_PROMPT = (
    "You are a precise research assistant. Answer the user's question using ONLY "
    "the provided context. If the context does not contain enough information to answer "
    "fully, say so explicitly. For every factual claim, include a citation in the format "
    "[Source N]. Do not introduce information not present in the context."
)

QUERY_TYPE_INSTRUCTIONS: dict[str, str] = {
    "factual": "Provide a direct, concise answer. If multiple sources agree, cite all of them.",
    "analytical": "Analyze and synthesize across the provided sources. Identify agreements and contradictions.",
    "comparative": "Organize your response as a structured comparison. Use a table if appropriate.",
    "procedural": "Provide numbered steps. Each step must be cited.",
}

DEFAULT_INSTRUCTION = "Provide a clear and well-reasoned answer directly based on the context."


@dataclass
class SourceMapEntry:
    """Explicit mapping from [Source N] identifier to chunk details."""

    source_number: int
    source_label: str
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int | None
    content_preview: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_number": self.source_number,
            "source_label": self.source_label,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "page_number": self.page_number,
            "content_preview": self.content_preview,
        }


@dataclass
class AssembledPrompt:
    """Result of prompt construction, including system instructions and context."""

    system_prompt: str
    user_prompt: str
    source_map: dict[int, SourceMapEntry] = field(default_factory=dict)
    query_type: str = "factual"

    @property
    def full_prompt(self) -> str:
        """Full representation for logging and persistence."""
        return f"[SYSTEM]\n{self.system_prompt}\n\n[USER]\n{self.user_prompt}"


class PromptBuilder:
    """Builds query-type conditioned prompts with structured context and deterministic source mapping."""

    def __init__(self, base_system_prompt: str = BASE_SYSTEM_PROMPT):
        self.base_system_prompt = base_system_prompt

    def build(
        self,
        query: str,
        query_type: str | None = None,
        context: AssembledContext | list[RetrievalResult] | None = None,
        pipeline_config: dict[str, Any] | None = None,
    ) -> AssembledPrompt:
        """Construct the prompt and source map.

        Args:
            query: The original user search query.
            query_type: Type from query classification (factual, analytical, comparative, procedural).
            context: Either an AssembledContext or a list of RetrievalResult candidates.
            pipeline_config: Optional pipeline configuration dictionary.

        Returns:
            AssembledPrompt with system_prompt, user_prompt, and source_map.
        """
        clean_query = query.strip()
        norm_type = (query_type or "factual").strip().lower()
        type_instruction = QUERY_TYPE_INSTRUCTIONS.get(norm_type, DEFAULT_INSTRUCTION)
        system_prompt = f"{self.base_system_prompt}\n\n{type_instruction}"

        if pipeline_config:
            gen_cfg = pipeline_config.get("generation", {})
            variant = gen_cfg.get("system_prompt_variant", "default")
            if variant == "concise":
                system_prompt += "\n\nStyle Instruction: Keep answers extremely concise and direct, omitting conversational filler."
            elif variant == "detailed":
                system_prompt += "\n\nStyle Instruction: Provide comprehensive, in-depth explanations with thorough source analysis."
            elif variant == "technical":
                system_prompt += "\n\nStyle Instruction: Use rigorous, precise engineering terminology and architectural detail."

        # Build context blocks and deterministic source map
        context_blocks: list[str] = []
        source_map: dict[int, SourceMapEntry] = {}

        if context is None:
            raw_sources = []
        elif isinstance(context, AssembledContext):
            raw_sources = context.sources
        elif isinstance(context, list):
            raw_sources = [
                {
                    "source_index": idx + 1,
                    "chunk_id": getattr(r, "chunk_id", str(idx + 1)),
                    "document_id": getattr(r, "document_id", ""),
                    "filename": getattr(r, "filename", None) or "document",
                    "page_number": getattr(r, "page_number", None),
                    "content": getattr(r, "content", ""),
                }
                for idx, r in enumerate(context)
            ]
        else:
            raw_sources = []

        # If context is AssembledContext, parse out chunk contents from sources or fallback
        for idx, src in enumerate(raw_sources, start=1):
            source_num = src.get("source_index", idx)
            chunk_id = str(src.get("chunk_id", ""))
            doc_id = str(src.get("document_id", ""))
            doc_name = str(src.get("filename") or "document")
            page_no = src.get("page_number")
            raw_content = src.get("content", "")

            # If content wasn't directly in sources dict, extract from assembled text or preview
            preview = (raw_content[:100] if raw_content else "").strip()

            header_parts = [f"Source {source_num} — {doc_name}"]
            if page_no is not None:
                header_parts.append(f"page {page_no}")
            header_str = ", ".join(header_parts)

            entry = SourceMapEntry(
                source_number=source_num,
                source_label=f"Source {source_num}",
                chunk_id=chunk_id,
                document_id=doc_id,
                document_name=doc_name,
                page_number=page_no,
                content_preview=preview,
            )
            source_map[source_num] = entry

            if raw_content:
                # Sanitize and safely delimit chunk content
                safe_content = raw_content.replace("</context>", "")
                context_blocks.append(f"[{header_str}]\n{safe_content}")

        # If context was AssembledContext with pre-formatted context string
        if isinstance(context, AssembledContext) and context.context and not context_blocks:
            # Context text is already formatted with headers from ContextAssembler
            safe_context_str = context.context.replace("</context>", "")
            context_section = f"<context>\n{safe_context_str}\n</context>"
            # Also populate preview for source map entries if possible
            for s_num, entry in source_map.items():
                pattern = rf"\[Source {s_num}[^\]]*\]\s*([\s\S]*?)(?=\[Source \d+|$)"
                m = re.search(pattern, context.context)
                if m:
                    extracted = m.group(1).strip()
                    entry.content_preview = extracted[:100]
        elif context_blocks:
            context_section = "<context>\n" + "\n\n".join(context_blocks) + "\n</context>"
        else:
            context_section = "<context>\nNo relevant context found.\n</context>"

        user_prompt = f"{context_section}\n\nUser question: {clean_query}"

        return AssembledPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            source_map=source_map,
            query_type=norm_type,
        )
