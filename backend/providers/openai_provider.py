
import asyncio
import os
import time
from collections.abc import AsyncGenerator

from openai import AsyncOpenAI, RateLimitError

from .base import BaseLLMProvider, ChatMessage, GenerationResult


class OpenAIProvider(BaseLLMProvider):
    """OpenAI-compatible LLM provider."""

    MODEL_PRICING = {
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
    }

    MODEL_CONTEXT = {
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
    }

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        client: AsyncOpenAI | None = None,
        embedding_model: str = "text-embedding-3-small",
    ):
        self.model = model
        self.embedding_model = embedding_model
        self.client = client or AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
        )

    @property
    def cost_per_input_token(self) -> float:
        return self.MODEL_PRICING.get(self.model, (0.15, 0.60))[0] / 1_000_000

    @property
    def cost_per_output_token(self) -> float:
        return self.MODEL_PRICING.get(self.model, (0.15, 0.60))[1] / 1_000_000

    @property
    def context_window(self) -> int:
        return self.MODEL_CONTEXT.get(self.model, 128000)

    @staticmethod
    def _format_messages(messages: list[ChatMessage]) -> list[dict]:
        return [
            {"role": message.role, "content": message.content}
            for message in messages
        ]

    async def _create_completion(self, **kwargs):
        """Retry rate-limit errors up to three times."""
        for attempt in range(4):
            try:
                return await self.client.chat.completions.create(**kwargs)
            except RateLimitError as exc:
                if attempt == 3:
                    raise

                retry_after = exc.response.headers.get("retry-after")
                try:
                    delay = float(retry_after) if retry_after else 2**attempt
                except (TypeError, ValueError):
                    delay = 2**attempt

                await asyncio.sleep(max(0, delay))

    async def complete(
        self,
        messages: list[ChatMessage],
        **kwargs,
    ) -> GenerationResult:
        start = time.perf_counter()

        response = await self._create_completion(
            model=kwargs.pop("model", self.model),
            messages=self._format_messages(messages),
            **kwargs,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        choice = response.choices[0]
        usage = response.usage

        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        model = response.model or self.model

        input_price, output_price = self.MODEL_PRICING.get(
            model, self.MODEL_PRICING.get(self.model, (0.15, 0.60))
        )
        cost = (
            input_tokens * input_price
            + output_tokens * output_price
        ) / 1_000_000

        return GenerationResult(
            content=choice.message.content or "",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed_ms,
            cost_usd=cost,
            finish_reason=choice.finish_reason or "unknown",
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        model = kwargs.pop("model", self.model)

        response = await self.client.chat.completions.create(
            model=model,
            messages=self._format_messages(messages),
            stream=True,
            **kwargs,
        )

        async for chunk in response:
            if not chunk.choices:
                continue

            text = chunk.choices[0].delta.content
            if text:
                yield text

    async def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = []

        for start in range(0, len(texts), 100):
            batch = texts[start:start + 100]

            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=batch,
            )

            embeddings.extend(
                item.embedding
                for item in sorted(response.data, key=lambda item: item.index)
            )

        return embeddings
