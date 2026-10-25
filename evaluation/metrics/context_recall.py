"""Context Recall metric — measure coverage of required information in retrieved context.

Evaluates whether the retrieved context contains all necessary facts and information to completely
answer the query. Scores between 0.0 and 1.0.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from providers.base import ChatMessage, GenerationResult

logger = logging.getLogger("neuroflow.evaluation.context_recall")

_SYSTEM_PROMPT = """\
You are an expert evaluator assessing retrieval recall in a RAG pipeline.
You will receive:
1. QUERY — the user's question.
2. CONTEXT — the retrieved context passages.
3. ANSWER (or ground truth reference) — the answer containing key facts.

Evaluate whether the CONTEXT contains all the facts, data points, or context necessary
to fully answer the QUERY.
Identify the key facts or claims required, and check if each is present in the CONTEXT.

Respond ONLY with valid JSON in the following schema:
{
  "key_facts": [
    {"fact": "<key fact>", "in_context": <true|false>}
  ],
  "score": <float 0.0-1.0>,
  "reasoning": "<brief explanation of recall>"
}
score = (number of key facts present in context) / (total key facts required).
If no facts are required or question is trivial, return 1.0.
If context is empty and facts are required, return 0.0.
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
class ContextRecallScore:
    score: float
    key_facts: list[dict[str, Any]]
    reasoning: str
    raw_response: str


class ContextRecallMetric:
    """Async context recall scorer using an LLM judge."""

    def __init__(self, llm_client: Any = None):
        self.llm_client = llm_client

    async def score(
        self,
        *,
        query: str,
        context: str,
        answer: str = "",
    ) -> ContextRecallScore:
        """Compute a context recall score for the given query/context/answer combination."""
        if not context.strip():
            return ContextRecallScore(
                score=0.0,
                key_facts=[],
                reasoning="Empty context provided.",
                raw_response="",
            )

        if self.llm_client is None:
            logger.warning("No LLM client configured — returning default context recall score.")
            return ContextRecallScore(
                score=1.0,
                key_facts=[],
                reasoning="No LLM client available.",
                raw_response="",
            )

        user_msg = _USER_TEMPLATE.format(query=query[:2000], context=context[:8000], answer=answer[:3000])
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
            logger.exception("LLM call failed for context_recall metric: %s", exc)
            return ContextRecallScore(
                score=0.0,
                key_facts=[],
                reasoning=f"LLM call failed: {exc}",
                raw_response="",
            )

        return self._parse_response(raw)

    def _parse_response(self, raw: str) -> ContextRecallScore:
        """Extract structured JSON from the LLM response."""
        try:
            cleaned = re.sub(r"```(?:json)?", "", raw).strip()
            data = json.loads(cleaned)
            score = float(data.get("score", 0.0))
            score = max(0.0, min(1.0, score))
            return ContextRecallScore(
                score=score,
                key_facts=data.get("key_facts", []),
                reasoning=data.get("reasoning", ""),
                raw_response=raw,
            )
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning("Failed to parse context_recall JSON: %s | raw=%r", exc, raw[:200])
            match = re.search(r'"score"\s*:\s*([0-9.]+)', raw)
            score = float(match.group(1)) if match else 0.0
            return ContextRecallScore(
                score=max(0.0, min(1.0, score)),
                key_facts=[],
                reasoning="Parse error — extracted score only.",
                raw_response=raw,
            )
