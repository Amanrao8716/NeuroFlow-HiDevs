"""Optional Hypothetical Document Embeddings (HyDE) module."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Protocol

from providers.base import ChatMessage

logger = logging.getLogger("neuroflow.retrieval.hyde")


class ChatResponse(Protocol):
    """Expected response structure from an LLM client."""

    content: str


class LLMClient(Protocol):
    """Expected interface for an LLM client."""

    async def chat(self, messages: list[ChatMessage]) -> ChatResponse: ...


class HyDEGenerator:
    """Generate hypothetical passage embeddings for enhanced semantic matching."""

    def __init__(
        self,
        llm_client: LLMClient,
        embed_fn: Callable[[list[str]], Awaitable[list[list[float]]]],
    ) -> None:
        self.llm_client = llm_client
        self.embed_fn = embed_fn

    async def generate_hypothetical_document(self, query: str) -> str:
        """Prompt LLM to write a hypothetical passage answering the user's question."""

        prompt = (
            "Write a short, authoritative, factual paragraph answering the following question. "
            "Include typical domain keywords, concepts, and technical terminology:\n\n"
            f"Question: {query}\n\n"
            "Passage:"
        )

        try:
            response = await self.llm_client.chat([ChatMessage(role="user", content=prompt)])
            return response.content.strip()
        except Exception as exc:
            logger.warning("HyDE document generation failed: %s", exc)
            return query

    async def embed_hyde_query(self, query: str) -> list[float]:
        """Generate a hypothetical document and return its embedding vector."""

        hypothetical_doc = await self.generate_hypothetical_document(query)
        vectors = await self.embed_fn([hypothetical_doc])
        return vectors[0]
