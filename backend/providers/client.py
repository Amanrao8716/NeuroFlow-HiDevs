
import asyncio
import time
from threading import Lock

from opentelemetry import trace

from .anthropic_provider import AnthropicProvider
from .base import ChatMessage, GenerationResult
from .openai_provider import OpenAIProvider
from .router import ModelRouter, RoutingCriteria

_LOCAL_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model_instance = None
_model_lock = Lock()


def _get_embedding_model():
    global _model_instance
    if _model_instance is None:
        with _model_lock:
            if _model_instance is None:
                from sentence_transformers import SentenceTransformer

                _model_instance = SentenceTransformer(_LOCAL_EMBEDDING_MODEL_NAME)
    return _model_instance


def _encode_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = _get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()



class NeuroFlowClient:
    """Main client for routed LLM calls, embeddings, and metrics."""

    def __init__(self, redis_client):
        self.redis = redis_client
        self.providers = {
            "gpt-4o": OpenAIProvider(model="gpt-4o"),
            "gpt-4o-mini": OpenAIProvider(model="gpt-4o-mini"),
            "claude-sonnet-4-20250514": AnthropicProvider(
                model="claude-sonnet-4-20250514"
            ),
            "claude-3-5-haiku-latest": AnthropicProvider(
                model="claude-3-5-haiku-latest"
            ),
        }
        self.router = ModelRouter(redis_client, self.providers)
        self.tracer = trace.get_tracer("neuroflow.providers")

    async def chat(
        self,
        messages: list[ChatMessage],
        routing_criteria: RoutingCriteria | None = None,
        **kwargs,
    ) -> GenerationResult:
        model_name = await self.router.route(routing_criteria)
        provider = self.providers[model_name]
        start = time.perf_counter()

        with self.tracer.start_as_current_span("llm.chat") as span:
            span.set_attribute("llm.model", model_name)

            result = await provider.complete(messages, **kwargs)
            latency_ms = (time.perf_counter() - start) * 1000

            span.set_attribute("llm.input_tokens", result.input_tokens)
            span.set_attribute("llm.output_tokens", result.output_tokens)
            span.set_attribute("llm.cost_usd", result.cost_usd)
            span.set_attribute("llm.latency_ms", latency_ms)

        await self._record_metrics(model_name, result)
        return result

    async def stream(
        self,
        messages: list[ChatMessage],
        routing_criteria: RoutingCriteria | None = None,
        **kwargs,
    ):
        model_name = await self.router.route(routing_criteria)
        provider = self.providers[model_name]

        with self.tracer.start_as_current_span("llm.stream") as span:
            span.set_attribute("llm.model", model_name)

            async for text in provider.stream(messages, **kwargs):
                yield text

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate 384-dimensional embeddings using local SentenceTransformer."""
        if not texts:
            return []

        start = time.perf_counter()
        with self.tracer.start_as_current_span("embedding.local") as span:
            span.set_attribute("embedding.model", _LOCAL_EMBEDDING_MODEL_NAME)
            span.set_attribute("embedding.count", len(texts))

            embeddings = await asyncio.to_thread(_encode_texts, texts)

            latency_ms = (time.perf_counter() - start) * 1000
            span.set_attribute("embedding.latency_ms", latency_ms)
            return embeddings

    async def _record_metrics(
        self,
        model_name: str,
        result: GenerationResult,
    ) -> None:
        calls_key = f"metrics:model:{model_name}:calls"
        cost_key = f"metrics:model:{model_name}:cost_usd"

        # Increment the call count and accumulated cost.
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.incr(calls_key)
            pipe.incrbyfloat(cost_key, result.cost_usd)
            await pipe.execute()
