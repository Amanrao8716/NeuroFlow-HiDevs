
import asyncio
import time
from threading import Lock

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from .anthropic_provider import AnthropicProvider
from .base import ChatMessage, GenerationResult
from .openai_provider import OpenAIProvider
from .router import ModelRouter, RoutingCriteria
from observability.metrics import PROVIDER_CALLS, PROVIDER_COST, PROVIDER_TOKENS

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

    def __init__(self, redis_client, circuit_breakers=None):
        self.redis = redis_client
        self.circuit_breakers = circuit_breakers
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
        provider_name = self._provider_name(provider)
        start = time.perf_counter()

        with self.tracer.start_as_current_span("llm.chat") as span:
            span.set_attribute("llm.model", model_name)
            span.set_attribute("llm.provider", provider_name)

            try:
                if self.circuit_breakers:
                    async with self.circuit_breakers.get(provider_name):
                        result = await provider.complete(messages, **kwargs)
                else:
                    result = await provider.complete(messages, **kwargs)
            except Exception as exc:
                PROVIDER_CALLS.labels(
                    provider=provider_name,
                    operation="chat",
                    status="error",
                ).inc()
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

            latency_ms = (time.perf_counter() - start) * 1000
            span.set_attribute("llm.input_tokens", result.input_tokens)
            span.set_attribute("llm.output_tokens", result.output_tokens)
            span.set_attribute("llm.cost_usd", result.cost_usd)
            span.set_attribute("llm.latency_ms", latency_ms)

        PROVIDER_CALLS.labels(
            provider=provider_name,
            operation="chat",
            status="success",
        ).inc()
        PROVIDER_TOKENS.labels(provider=provider_name, direction="input").inc(
            result.input_tokens
        )
        PROVIDER_TOKENS.labels(provider=provider_name, direction="output").inc(
            result.output_tokens
        )
        PROVIDER_COST.labels(provider=provider_name).inc(result.cost_usd)
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
        provider_name = self._provider_name(provider)

        with self.tracer.start_as_current_span("llm.stream") as span:
            span.set_attribute("llm.model", model_name)
            span.set_attribute("llm.provider", provider_name)
            completed = False
            try:
                if self.circuit_breakers:
                    async with self.circuit_breakers.get(provider_name):
                        async for text in provider.stream(messages, **kwargs):
                            yield text
                else:
                    async for text in provider.stream(messages, **kwargs):
                        yield text
                completed = True
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise
            finally:
                PROVIDER_CALLS.labels(
                    provider=provider_name,
                    operation="stream",
                    status="success" if completed else "error",
                ).inc()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate 384-dimensional embeddings using local SentenceTransformer."""
        if not texts:
            return []

        start = time.perf_counter()
        with self.tracer.start_as_current_span("embedding.local") as span:
            span.set_attribute("embedding.model", _LOCAL_EMBEDDING_MODEL_NAME)
            span.set_attribute("embedding.count", len(texts))

            try:
                embeddings = await asyncio.to_thread(_encode_texts, texts)
            except Exception as exc:
                PROVIDER_CALLS.labels(
                    provider="local",
                    operation="embedding",
                    status="error",
                ).inc()
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

            latency_ms = (time.perf_counter() - start) * 1000
            span.set_attribute("embedding.latency_ms", latency_ms)
            PROVIDER_CALLS.labels(
                provider="local",
                operation="embedding",
                status="success",
            ).inc()
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

    @staticmethod
    def _provider_name(provider) -> str:
        return "anthropic" if isinstance(provider, AnthropicProvider) else "openai"
