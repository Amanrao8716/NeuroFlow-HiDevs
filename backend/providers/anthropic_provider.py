import os
import time
from collections.abc import AsyncGenerator

from anthropic import AsyncAnthropic
from anthropic.types import AssistantMessageParam, MessageParam, UserMessageParam
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming

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
    ) -> None:
        self.model = model
        self.client = client or AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

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
    ) -> tuple[str | None, list[MessageParam]]:
        system_parts: list[str] = []
        formatted: list[MessageParam] = []

        for message in messages:
            if not isinstance(message.content, str):
                raise ValueError("AnthropicProvider currently supports text message content only.")

            if message.role == "system":
                system_parts.append(message.content)
            elif message.role == "user":
                user_message: UserMessageParam = {
                    "role": "user",
                    "content": message.content,
                }
                formatted.append(user_message)
            elif message.role == "assistant":
                assistant_message: AssistantMessageParam = {
                    "role": "assistant",
                    "content": message.content,
                }
                formatted.append(assistant_message)
            else:
                raise ValueError(f"Unsupported Anthropic message role: {message.role}")

        system = "\n".join(system_parts) or None
        return system, formatted

    @staticmethod
    def _extract_request_options(
        kwargs: dict[str, object],
        default_model: str,
    ) -> tuple[str, int]:
        model = kwargs.pop("model", default_model)
        max_tokens = kwargs.pop("max_tokens", None)

        if not isinstance(model, str):
            raise TypeError("Anthropic model must be a string.")

        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise TypeError("Anthropic max_tokens must be an integer.")

        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported Anthropic request options: {unsupported}")

        return model, max_tokens

    async def complete(
        self,
        messages: list[ChatMessage],
        **kwargs: object,
    ) -> GenerationResult:
        start = time.perf_counter()
        system, formatted = self._format_messages(messages)

        model, max_tokens = self._extract_request_options(dict(kwargs), self.model)

        request: MessageCreateParamsNonStreaming = {
            "model": model,
            "messages": formatted,
            "max_tokens": max_tokens,
        }

        if system:
            request["system"] = system

        response = await self.client.messages.create(**request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        content = "".join(block.text for block in response.content if block.type == "text")

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        response_model = response.model or self.model

        input_price, output_price = self.MODEL_PRICING.get(
            response_model,
            self.MODEL_PRICING.get(self.model, (0.80, 4.00)),
        )
        cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000

        return GenerationResult(
            content=content,
            model=response_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed_ms,
            cost_usd=cost,
            finish_reason=response.stop_reason or "unknown",
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        **kwargs: object,
    ) -> AsyncGenerator[str, None]:
        system, formatted = self._format_messages(messages)

        model, max_tokens = self._extract_request_options(dict(kwargs), self.model)

        if system:
            async with self.client.messages.stream(
                model=model,
                messages=formatted,
                max_tokens=max_tokens,
                system=system,
            ) as stream:
                async for text in stream.text_stream:
                    if text:
                        yield text
        else:
            async with self.client.messages.stream(
                model=model,
                messages=formatted,
                max_tokens=max_tokens,
            ) as stream:
                async for text in stream.text_stream:
                    if text:
                        yield text

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError(
            "Anthropic does not provide a native embeddings API. "
            "Use the OpenAI provider or a dedicated embedding model."
        )
