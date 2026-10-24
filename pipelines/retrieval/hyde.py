"""Optional Hypothetical Document Embeddings (HyDE) module."""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger("neuroflow.retrieval.hyde")


class HyDEGenerator:
    """Generate hypothetical passage embeddings for enhanced semantic matching."""

    def __init__(self, llm_client: Any, embed_fn: Callable[[list[str]], Any]):
        self.llm_client = llm_client
        self.embed_fn = embed_fn

    async def generate_hypothetical_document(self, query: str) -> str:
        """Prompt LLM to write a hypothetical passage answering the user's question."""
        from providers.base import ChatMessage

        prompt = (
            f"Write a short, authoritative, factual paragraph answering the following question. "
            f"Include typical domain keywords, concepts, and technical terminology:\n\n"
            f"Question: {query}\n\n"
            f"Passage:"
        )

        try:
            response = await self.llm_client.chat([ChatMessage(role="user", content=prompt)])
            return response.content.strip()
        except Exception as exc:
            logger.warning("HyDE document generation failed: %s", exc)
            return query

    async def embed_hyde_query(self, query: str) -> list[float]:
        """Generate a hypothetical document and return its 384-dimensional vector."""
        hypothetical_doc = await self.generate_hypothetical_document(query)
        vectors = await self.embed_fn([hypothetical_doc])
        return vectors[0]
