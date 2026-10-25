"""Context Precision metric — measure ranking quality of retrieved context.

Evaluates whether relevant chunks are retrieved at the top of the context ranking.
Calculates a score between 0.0 and 1.0 (mean average precision style or proportion of relevant passages).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from providers.base import ChatMessage, GenerationResult

logger = logging.getLogger("neuroflow.evaluation.context_precision")

_SYSTEM_PROMPT = """\
You are an expert retrieval evaluator. You will receive:
1. QUERY — the user's question.
2. CONTEXT — the retrieved context chunks, typically numbered or separated by sources.
3. ANSWER (optional) — the reference or generated answer.

Assess whether the retrieved context passages are relevant to answering the query, and whether
the most relevant information is presented near the top.

Respond ONLY with valid JSON in the following schema:
{
  "chunk_evaluations": [
    {"chunk_index": <int>, "relevant": <true|false>, "reason": "<brief note>"}
  ],
  "score": <float 0.0-1.0>,
  "reasoning": "<brief explanation of precision>"
}
score should reflect the precision of retrieved chunks (e.g. fraction of relevant chunks weighted by top ranking).
If the context is completely empty, return 0.0. If all retrieved context is relevant, return 1.0.
"""

_USER_TEMPLATE = """\
QUERY:
{query}

CONTEXT:
{context}

ANSWER:
{answer}
"""


@dataclass
class ContextPrecisionScore:
    score: float
    chunk_evaluations: list[dict[str, Any]]
    reasoning: str
    raw_response: str


class ContextPrecisionMetric:
    """Async context precision scorer using an LLM judge."""

    def __init__(self, llm_client: Any = None):
        self.llm_client = llm_client

    async def score(
        self,
        *,
        query: str,
        context: str,
        answer: str = "",
    ) -> ContextPrecisionScore:
        """Compute a context precision score for the given query/context/answer combination."""
        if not context.strip():
            return ContextPrecisionScore(
                score=0.0,
                chunk_evaluations=[],
                reasoning="Empty context provided.",
                raw_response="",
            )

        if self.llm_client is None:
            logger.warning("No LLM client configured — returning default context precision score.")
            return ContextPrecisionScore(
                score=1.0,
                chunk_evaluations=[],
                reasoning="No LLM client available.",
                raw_response="",
            )

        user_msg = _USER_TEMPLATE.format(query=query[:2000], context=context[:8000], answer=answer[:2000])
        messages = [
            ChatMessage(role="system", content=_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_msg),
        ]

        try:
            from providers.router import RoutingCriteria

            criteria = RoutingCriteria(task_type="evaluation")
            result: GenerationResult = await self.llm_client.chat(messages, routing_criteria=criteria)
            raw = result.content
        except Exception as exc:
            logger.exception("LLM call failed for context_precision metric: %s", exc)
            return ContextPrecisionScore(
                score=0.0,
                chunk_evaluations=[],
                reasoning=f"LLM call failed: {exc}",
                raw_response="",
            )

        return self._parse_response(raw)

    def _parse_response(self, raw: str) -> ContextPrecisionScore:
        """Extract structured JSON from the LLM response."""
        try:
            cleaned = re.sub(r"```(?:json)?", "", raw).strip()
            data = json.loads(cleaned)
            score = float(data.get("score", 0.0))
            score = max(0.0, min(1.0, score))
            return ContextPrecisionScore(
                score=score,
                chunk_evaluations=data.get("chunk_evaluations", []),
                reasoning=data.get("reasoning", ""),
                raw_response=raw,
            )
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning("Failed to parse context_precision JSON: %s | raw=%r", exc, raw[:200])
            match = re.search(r'"score"\s*:\s*([0-9.]+)', raw)
            score = float(match.group(1)) if match else 0.0
            return ContextPrecisionScore(
                score=max(0.0, min(1.0, score)),
                chunk_evaluations=[],
                reasoning="Parse error — extracted score only.",
                raw_response=raw,
            )
