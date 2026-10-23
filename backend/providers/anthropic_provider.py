
import os
import time
from collections.abc import AsyncGenerator

from anthropic import AsyncAnthropic

from .base import BaseLLMProvider, ChatMessage, GenerationResult


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude LLM provider."""

    MODEL_PRICING = {
        "claude-sonnet-4-20250514": (3.00, 15.00),
        "claude-3-5-haiku-latest": (0.80, 4.00),
    }

    MODEL_CONTEXT = {
        "claude-sonnet-4-20250514": 200000,
        "claude-3-5-haiku-latest": 200000,
    }

    def __init__(
        self,
        model: str = "claude-3-5-haiku-latest",
        client: AsyncAnthropic | None = None,
    ):
        self.model = model
        self.client = client or AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

    @property
    def cost_per_input_token(self) -> float:
        return self.MODEL_PRICING.get(self.model, (0.80, 4.00))[0] / 1_000_000

    @property
    def cost_per_output_token(self) -> float:
        return self.MODEL_PRICING.get(self.model, (0.80, 4.00))[1] / 1_000_000

    @property
    def context_window(self) -> int:
        return self.MODEL_CONTEXT.get(self.model, 200000)

    @staticmethod
    def _format_messages(
        messages: list[ChatMessage],
    ) -> tuple[str | None, list[dict]]:
        system_parts = []
        formatted = []

        for message in messages:
            if message.role == "system":
                system_parts.append(str(message.content))
            else:
                formatted.append({
                    "role": message.role,
                    "content": message.content,
                })

        system = "\n".join(system_parts) or None
        return system, formatted

    async def complete(
        self,
        messages: list[ChatMessage],
        **kwargs,
    ) -> GenerationResult:
        start = time.perf_counter()
        system, formatted = self._format_messages(messages)

        request = {
            "model": kwargs.pop("model", self.model),
            "messages": formatted,
            **kwargs,
        }
        if system:
            request["system"] = system

        response = await self.client.messages.create(**request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        content = "".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        )

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        model = response.model or self.model

        input_price, output_price = self.MODEL_PRICING.get(
            model, self.MODEL_PRICING.get(self.model, (0.80, 4.00))
        )
        cost = (
            input_tokens * input_price
            + output_tokens * output_price
        ) / 1_000_000

        return GenerationResult(
            content=content,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed_ms,
            cost_usd=cost,
            finish_reason=response.stop_reason or "unknown",
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        system, formatted = self._format_messages(messages)

        request = {
            "model": kwargs.pop("model", self.model),
            "messages": formatted,
            **kwargs,
        }
        if system:
            request["system"] = system

        async with self.client.messages.stream(**request) as stream:
            async for text in stream.text_stream:
                if text:
                    yield text

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError(
            "Anthropic does not provide a native embeddings API. "
            "Use the OpenAI provider or a dedicated embedding model."
        )
