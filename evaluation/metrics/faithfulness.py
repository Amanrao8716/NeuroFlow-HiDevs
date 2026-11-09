"""Faithfulness metric — verify the answer is grounded in the provided context.

Uses an LLM judge to check each claim in the generated answer against the retrieved
context passages.  Returns a normalised score in [0, 1] where 1.0 means every claim
is faithfully supported by the context.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from providers.base import ChatMessage, GenerationResult

logger = logging.getLogger("neuroflow.evaluation.faithfulness")

_SYSTEM_PROMPT = """\
You are a rigorous fact-checking judge.  You will receive:
1. CONTEXT — the source passages retrieved for a query.
2. ANSWER — the generated answer to evaluate.

Your task is to decompose the ANSWER into atomic claims, then verify whether each claim
is directly supported by the CONTEXT (true), not mentioned (hallucinated), or
contradicted (false).

Respond ONLY with valid JSON in the following schema:
{
  "claims": [
    {"claim": "<claim text>", "verdict": "supported"|"hallucinated"|"contradicted"}
  ],
  "score": <float 0.0-1.0>,
  "reasoning": "<brief explanation>"
}
score = (number of supported claims) / (total number of claims).
If there are no claims, return score 1.0.
"""

_USER_TEMPLATE = """\
CONTEXT:
{context}

ANSWER:
{answer}
"""


@dataclass
class FaithfulnessScore:
    score: float
    claims: list[dict[str, str]]
    reasoning: str
    raw_response: str


class FaithfulnessMetric:
    """Async faithfulness scorer using an LLM judge."""

    def __init__(self, llm_client: Any = None):
        self.llm_client = llm_client

    async def score(
        self,
        *,
        answer: str,
        context: str,
        query: str = "",  # kept for API symmetry; not used in faithfulness
    ) -> FaithfulnessScore:
        """Compute a faithfulness score for the given answer/context pair."""
        if self.llm_client is None:
            logger.warning("No LLM client configured — returning default faithfulness score.")
            return FaithfulnessScore(
                score=1.0,
                claims=[],
                reasoning="No LLM client available.",
                raw_response="",
            )

        user_msg = _USER_TEMPLATE.format(context=context[:8000], answer=answer[:4000])
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
            logger.exception("LLM call failed for faithfulness metric: %s", exc)
            return FaithfulnessScore(
                score=0.0,
                claims=[],
                reasoning=f"LLM call failed: {exc}",
                raw_response="",
            )

        return self._parse_response(raw)

    def _parse_response(self, raw: str) -> FaithfulnessScore:
        """Extract structured JSON from the LLM response."""
        try:
            # Strip markdown code fences if present
            cleaned = re.sub(r"```(?:json)?", "", raw).strip()
            data = json.loads(cleaned)
            score = float(data.get("score", 0.0))
            score = max(0.0, min(1.0, score))
            return FaithfulnessScore(
                score=score,
                claims=data.get("claims", []),
                reasoning=data.get("reasoning", ""),
                raw_response=raw,
            )
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning("Failed to parse faithfulness JSON: %s | raw=%r", exc, raw[:200])
            # Fallback: look for a bare float in the text
            match = re.search(r'"score"\s*:\s*([0-9.]+)', raw)
            score = float(match.group(1)) if match else 0.0
            return FaithfulnessScore(
                score=max(0.0, min(1.0, score)),
                claims=[],
                reasoning="Parse error — extracted score only.",
                raw_response=raw,
            )
