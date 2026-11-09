"""Answer Relevance metric — evaluate how directly and completely the answer addresses the query.

Scores the response between 0.0 and 1.0. A score of 1.0 means the answer directly,
completely, and concisely addresses the user's question without extraneous digression or avoidance.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from providers.base import ChatMessage, GenerationResult

logger = logging.getLogger("neuroflow.evaluation.answer_relevance")

_SYSTEM_PROMPT = """\
You are an expert evaluator assessing answer relevance in a question-answering system.
You will receive:
1. QUERY — the user's question or prompt.
2. ANSWER — the generated answer.

Evaluate how directly, completely, and appropriately the ANSWER addresses the QUERY.
Ignore whether the information is factually true; focus strictly on whether it answers what was asked.
Penalize answers that are vague, evasive, repetitive, or diverge from the topic.

Respond ONLY with valid JSON in the following schema:
{
  "relevance_verdict": "relevant"|"partially_relevant"|"irrelevant",
  "score": <float 0.0-1.0>,
  "reasoning": "<brief explanation of relevance>"
}
"""

_USER_TEMPLATE = """\
QUERY:
{query}

ANSWER:
{answer}
"""


@dataclass
class AnswerRelevanceScore:
    score: float
    relevance_verdict: str
    reasoning: str
    raw_response: str


class AnswerRelevanceMetric:
    """Async answer relevance scorer using an LLM judge."""

    def __init__(self, llm_client: Any = None):
        self.llm_client = llm_client

    async def score(
        self,
        *,
        query: str,
        answer: str,
        context: str = "",
    ) -> AnswerRelevanceScore:
        """Compute an answer relevance score for the given query/answer pair."""
        if self.llm_client is None:
            logger.warning("No LLM client configured — returning default answer relevance score.")
            return AnswerRelevanceScore(
                score=1.0,
                relevance_verdict="relevant",
                reasoning="No LLM client available.",
                raw_response="",
            )

        user_msg = _USER_TEMPLATE.format(query=query[:2000], answer=answer[:4000])
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
            logger.exception("LLM call failed for answer_relevance metric: %s", exc)
            return AnswerRelevanceScore(
                score=0.0,
                relevance_verdict="error",
                reasoning=f"LLM call failed: {exc}",
                raw_response="",
            )

        return self._parse_response(raw)

    def _parse_response(self, raw: str) -> AnswerRelevanceScore:
        """Extract structured JSON from the LLM response."""
        try:
            cleaned = re.sub(r"```(?:json)?", "", raw).strip()
            data = json.loads(cleaned)
            score = float(data.get("score", 0.0))
            score = max(0.0, min(1.0, score))
            return AnswerRelevanceScore(
                score=score,
                relevance_verdict=data.get("relevance_verdict", "unknown"),
                reasoning=data.get("reasoning", ""),
                raw_response=raw,
            )
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning("Failed to parse answer_relevance JSON: %s | raw=%r", exc, raw[:200])
            match = re.search(r'"score"\s*:\s*([0-9.]+)', raw)
            score = float(match.group(1)) if match else 0.0
            return AnswerRelevanceScore(
                score=max(0.0, min(1.0, score)),
                relevance_verdict="unknown",
                reasoning="Parse error — extracted score only.",
                raw_response=raw,
            )
