"""Query processor for expansion, metadata extraction, and classification."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import re
from typing import Any, Literal

logger = logging.getLogger("neuroflow.retrieval.query_processor")

QueryType = Literal["factual", "analytical", "comparative", "procedural"]


@dataclass
class ProcessedQuery:
    """Structured representation of a processed query."""

    query: str
    expanded_queries: list[str] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)
    query_type: QueryType = "factual"

    @property
    def all_queries(self) -> list[str]:
        """Return the original query and all unique non-empty alternatives."""
        seen = {self.query.strip().lower()}
        result = [self.query.strip()]
        for q in self.expanded_queries:
            cleaned = q.strip()
            if cleaned and cleaned.lower() not in seen:
                seen.add(cleaned.lower())
                result.append(cleaned)
        return result


class QueryProcessor:
    """Asynchronous query processor for hybrid retrieval."""

    def __init__(self, llm_client: Any | None = None):
        """Initialize query processor with optional LLM client for expansion."""
        self.llm_client = llm_client

    async def process(
        self,
        query: str,
        explicit_filters: dict[str, Any] | None = None,
    ) -> ProcessedQuery:
        """Process a raw user query into expanded variants, filters, and query type."""
        trimmed = query.strip()
        if not trimmed:
            return ProcessedQuery(query="", query_type="factual")

        query_type = self._classify_query_type(trimmed)
        extracted_filters = self._extract_metadata_filters(trimmed)

        # Merge explicit filters with extracted filters (explicit overrides extracted)
        merged_filters = dict(extracted_filters)
        if explicit_filters:
            merged_filters.update(explicit_filters)

        expanded = await self._expand_query(trimmed)

        return ProcessedQuery(
            query=trimmed,
            expanded_queries=expanded,
            filters=merged_filters,
            query_type=query_type,
        )

    def _classify_query_type(self, query: str) -> QueryType:
        """Classify query type deterministically using structural linguistic cues."""
        text = query.lower()

        # Comparative indicators
        if any(
            pattern in text
            for pattern in (
                " versus ",
                " vs ",
                " vs. ",
                "difference between",
                "differences between",
                "compare ",
                "comparing ",
                "comparison of",
                "similarities between",
                "pros and cons",
                "better than",
            )
        ):
            return "comparative"

        # Procedural indicators
        if any(
            pattern in text
            for pattern in (
                "how to ",
                "step by step",
                "steps to ",
                "steps for ",
                "instructions for",
                "guide to ",
                "procedure for",
                "how do i ",
                "how can i ",
                "how to configure",
                "how to setup",
                "how to install",
            )
        ):
            return "procedural"

        # Analytical indicators
        if any(
            pattern in text
            for pattern in (
                "why ",
                "how does ",
                "how did ",
                "explain ",
                "explanation of",
                "impact of",
                "causes of",
                "analyze ",
                "analysis of",
                "reasons for",
                "implications of",
            )
        ):
            return "analytical"

        # Default to factual
        return "factual"

    def _extract_metadata_filters(self, query: str) -> dict[str, Any]:
        """Extract structured metadata filters such as year, topic, and source_type."""
        filters: dict[str, Any] = {}
        text = query.strip()

        # 1. Year extraction (contextual: must be preceded or followed by indicator words or standalone year)
        # e.g., '2023 climate documents', 'from 2023', 'in 2023', 'year 2023', '2020-2029'
        year_match = re.search(
            r"\b(?:(?:from|in|year|during|since)\s+)?(19\d{2}|20\d{2})(?:\s+(?:documents|data|reports|climate))?\b",
            text,
            re.IGNORECASE,
        )
        if year_match:
            try:
                year_val = int(year_match.group(1))
                if 1970 <= year_val <= 2099:
                    filters["year"] = year_val
            except (ValueError, IndexError):
                pass

        # 2. Topic extraction
        # Specific known topic patterns
        topic_match = re.search(
            r"\b(?:about|regarding|topic|on)\s+([a-zA-Z0-9_\-\s]+?)(?:\s+documents|\s+files|\s+data|$|\.|\?)",
            text,
            re.IGNORECASE,
        )
        if topic_match:
            topic = topic_match.group(1).strip().lower()
            # Clean stop words
            if topic in ("climate", "climate change"):
                filters["topic"] = "climate"
            elif topic:
                filters["topic"] = topic
        elif re.search(r"\bclimate\b", text, re.IGNORECASE):
            filters["topic"] = "climate"

        # 3. Source type extraction
        if re.search(r"\b(?:pdf|pdfs)\b", text, re.IGNORECASE):
            filters["source_type"] = "pdf"
        elif re.search(r"\b(?:csv|tabular|spreadsheet)\b", text, re.IGNORECASE):
            filters["source_type"] = "csv"
        elif re.search(r"\b(?:docx|word document)\b", text, re.IGNORECASE):
            filters["source_type"] = "docx"
        elif re.search(r"\b(?:image|images|png|jpg|jpeg)\b", text, re.IGNORECASE):
            filters["source_type"] = "image"
        elif re.search(r"\b(?:url|webpage|web page|website)\b", text, re.IGNORECASE):
            filters["source_type"] = "url"

        return filters

    async def _expand_query(self, query: str) -> list[str]:
        """Generate 2-3 semantically equivalent alternative query phrasings."""
        if self.llm_client is None:
            return self._heuristic_expansion(query)

        prompt = (
            f"Generate 2 or 3 distinct alternative search queries that have the exact same intent as:\n"
            f"'{query}'\n\n"
            f"Respond ONLY with a JSON array of strings, e.g. [\"variant 1\", \"variant 2\"]. "
            f"Do not include explanation."
        )

        try:
            from providers.base import ChatMessage

            messages = [ChatMessage(role="user", content=prompt)]
            result = await self.llm_client.chat(messages)
            raw = result.content.strip()

            # Attempt JSON parsing
            json_match = re.search(r"\[.*?\]", raw, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                if isinstance(parsed, list):
                    normalized = []
                    # Normalize for dedup: lowercase + strip trailing punctuation
                    def _norm(s: str) -> str:
                        return re.sub(r"[\W_]+$", "", s.lower()).strip()

                    query_norm = _norm(query.strip())
                    seen_norm: set[str] = {query_norm, query.strip().lower()}
                    for item in parsed:
                        if isinstance(item, str):
                            item_clean = item.strip().strip('"').strip("'")
                            item_norm = _norm(item_clean)
                            if item_clean and item_norm not in seen_norm:
                                seen_norm.add(item_norm)
                                normalized.append(item_clean)
                    if normalized:
                        return normalized[:3]
        except Exception as exc:
            logger.warning("LLM query expansion failed, falling back to heuristics: %s", exc)

        return self._heuristic_expansion(query)

    def _heuristic_expansion(self, query: str) -> list[str]:
        """Deterministic heuristic query variants when LLM is unavailable."""
        variants: list[str] = []
        q_lower = query.lower()

        # Procedural / question inversion
        if q_lower.startswith("how does ") or q_lower.startswith("how do "):
            clean = re.sub(r"^how (?:does|do) ", "", query, flags=re.IGNORECASE).rstrip("?")
            variants.append(f"{clean} mechanism and overview")
            variants.append(f"explaining {clean}")
        elif q_lower.startswith("what is ") or q_lower.startswith("what are "):
            clean = re.sub(r"^what (?:is|are) ", "", query, flags=re.IGNORECASE).rstrip("?")
            variants.append(f"definition of {clean}")
            variants.append(f"{clean} specification and details")
        elif " vs " in q_lower or " versus " in q_lower:
            parts = re.split(r" vs\.? | versus ", query, flags=re.IGNORECASE)
            if len(parts) == 2:
                variants.append(f"comparing {parts[0].strip()} and {parts[1].strip()}")
                variants.append(f"difference between {parts[0].strip()} and {parts[1].strip()}")

        # Deduplicate against original
        results = []
        seen = {query.strip().lower()}
        for v in variants:
            v_clean = v.strip()
            if v_clean and v_clean.lower() not in seen:
                seen.add(v_clean.lower())
                results.append(v_clean)

        return results[:3]
