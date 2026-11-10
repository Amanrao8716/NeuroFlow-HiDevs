
"""Cross-encoder and LLM rerankers for candidate precision optimization."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import Sequence
from threading import Lock
from typing import Protocol, cast

from providers.base import ChatMessage, GenerationResult

from pipelines.retrieval.retriever import RetrievalResult

logger = logging.getLogger("neuroflow.retrieval.reranker")

DEFAULT_CROSS_ENCODER_MODEL = os.getenv(
    "CROSS_ENCODER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)


class CrossEncoderModel(Protocol):
    """Interface required from a cross-encoder model."""

    def predict(self, pairs: list[tuple[str, str]]) -> object:
        """Predict relevance scores for query-document pairs."""
        ...


class LLMClient(Protocol):
    """Interface required from the LLM client."""

    async def chat(
        self,
        messages: list[ChatMessage],
        **kwargs: object,
    ) -> GenerationResult:
        """Generate a response from the supplied messages."""
        ...


_MODEL_INSTANCE: CrossEncoderModel | None = None
_MODEL_LOCK = Lock()


def _get_cross_encoder_model(
    model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
) -> CrossEncoderModel:
    """Lazily load and cache the cross-encoder model singleton."""
    global _MODEL_INSTANCE

    if _MODEL_INSTANCE is None:
        with _MODEL_LOCK:
            if _MODEL_INSTANCE is None:
                from sentence_transformers import CrossEncoder

                logger.info("Loading cross-encoder model: %s", model_name)
                _MODEL_INSTANCE = cast(
                    CrossEncoderModel,
                    CrossEncoder(model_name),
                )

    return _MODEL_INSTANCE


def _predict_pairs(
    model: CrossEncoderModel,
    pairs: list[tuple[str, str]],
) -> list[float]:
    """Execute cross-encoder inference on (query, text) pairs."""
    if not pairs:
        return []

    scores = model.predict(pairs)

    tolist = getattr(scores, "tolist", None)
    if callable(tolist):
        values = tolist()
        if isinstance(values, list):
            return [float(value) for value in values]

    if isinstance(scores, Sequence):
        return [float(value) for value in scores]

    raise TypeError("Cross-encoder returned an unsupported score format.")


class CrossEncoderReranker:
    """Local cross-encoder reranker using sentence-transformers CrossEncoder."""

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        model: CrossEncoderModel | None = None,
        top_n: int = 40,
    ) -> None:
        """Initialize reranker with optional injected model and candidate count."""
        self.model_name = model_name
        self._model = model
        self.top_n = top_n

    def _resolve_model(self) -> CrossEncoderModel:
        """Get the injected or singleton model."""
        if self._model is not None:
            return self._model
        return _get_cross_encoder_model(self.model_name)

    async def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_n: int | None = None,
    ) -> list[RetrievalResult]:
        """Rerank candidates using cross-encoder scores.

        Args:
            query: The search query string.
            candidates: Ranked candidate list (e.g. from RRF).
            top_n: Maximum candidates to rerank (defaults to self.top_n).

        Returns:
            Candidates sorted by cross-encoder score descending, with ranks updated.
        """
        if not candidates or not query.strip():
            return []

        limit = top_n if top_n is not None else self.top_n
        pool_to_rerank = candidates[:limit]
        remaining = candidates[limit:]

        model = self._resolve_model()
        pairs = [(query, candidate.content) for candidate in pool_to_rerank]

        try:
            scores = await asyncio.to_thread(_predict_pairs, model, pairs)
        except Exception as exc:
            logger.error("Cross-encoder inference failed: %s", exc)
            raise RuntimeError(f"Reranking failed: {exc}") from exc

        if len(scores) != len(pool_to_rerank):
            raise ValueError(
                f"Cross-encoder output length {len(scores)} "
                f"does not match candidates {len(pool_to_rerank)}"
            )

        scored_candidates: list[RetrievalResult] = []

        for candidate, ce_score in zip(pool_to_rerank, scores):
            new_meta = dict(candidate.metadata)
            new_meta["cross_encoder_score"] = ce_score
            new_meta["reranked"] = True

            if "rrf_score" not in new_meta:
                new_meta["rrf_score"] = candidate.score

            new_candidate = candidate.clone(
                score=ce_score,
                metadata=new_meta,
            )
            scored_candidates.append(new_candidate)

        scored_candidates.sort(key=lambda candidate: -candidate.score)

        all_results = scored_candidates + remaining

        for idx, item in enumerate(all_results, start=1):
            item.rank = idx

        return all_results


class LLMReranker:
    """Optional LLM-based reranker with score validation and bounded concurrency."""

    def __init__(
        self,
        llm_client: LLMClient,
        concurrency: int = 5,
        top_n: int = 20,
    ) -> None:
        self.llm_client = llm_client
        self.semaphore = asyncio.Semaphore(concurrency)
        self.top_n = top_n

    async def _score_single(
        self,
        query: str,
        candidate: RetrievalResult,
    ) -> float:
        """Prompt LLM to score relevance from 0 to 10."""
        prompt = (
            "Rate the relevance of the following passage to the question "
            "on a scale of 0 to 10, where 0 is completely irrelevant "
            "and 10 is an exact, comprehensive answer.\n\n"
            f"Question: {query}\n"
            f"Passage: {candidate.content[:1000]}\n\n"
            "Respond ONLY with a single numeric number between "
            "0.0 and 10.0, nothing else."
        )

        async with self.semaphore:
            try:
                response = await self.llm_client.chat(
                    [ChatMessage(role="user", content=prompt)]
                )
                text = response.content.strip()
                match = re.search(r"(\d+(?:\.\d+)?)", text)

                if match:
                    value = float(match.group(1))
                    return max(0.0, min(10.0, value))

            except Exception as exc:
                logger.warning(
                    "LLM reranker failed for chunk %s: %s",
                    candidate.chunk_id,
                    exc,
                )

        return 0.0

    async def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_n: int | None = None,
    ) -> list[RetrievalResult]:
        """Score candidates using LLM in parallel with bounded concurrency."""
        if not candidates or not query.strip():
            return []

        limit = top_n if top_n is not None else self.top_n
        to_score = candidates[:limit]
        remaining = candidates[limit:]

        tasks = [
            self._score_single(query, candidate)
            for candidate in to_score
        ]
        scores = await asyncio.gather(*tasks)

        scored: list[RetrievalResult] = []

        for candidate, llm_score in zip(to_score, scores):
            new_meta = dict(candidate.metadata)
            new_meta["llm_rerank_score"] = llm_score
            new_meta["reranked"] = True
            new_meta["rrf_score"] = candidate.score

            scored.append(
                candidate.clone(
                    score=llm_score,
                    metadata=new_meta,
                )
            )

        scored.sort(key=lambda candidate: -candidate.score)

        all_results = scored + remaining

        for idx, item in enumerate(all_results, start=1):
            item.rank = idx

        return all_results
