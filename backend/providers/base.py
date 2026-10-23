
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncGenerator


@dataclass
class ChatMessage:
    """A message sent to or received from an LLM."""
    role: str
    content: str | list


@dataclass
class GenerationResult:
    """Standard result returned by a provider."""
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    finish_reason: str


class BaseLLMProvider(ABC):
    """Common interface for all LLM providers."""

    @abstractmethod
    async def complete(
        self, messages: list[ChatMessage], **kwargs
    ) -> GenerationResult:
        """Generate a complete response."""
        ...

    @abstractmethod
    async def stream(
        self, messages: list[ChatMessage], **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream response text progressively."""
        ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for text inputs."""
        ...

    @property
    @abstractmethod
    def cost_per_input_token(self) -> float:
        """Input price in USD per token."""
        ...

    @property
    @abstractmethod
    def cost_per_output_token(self) -> float:
        """Output price in USD per token."""
        ...

    @property
    @abstractmethod
    def context_window(self) -> int:
        """Maximum context length in tokens."""
        ...
