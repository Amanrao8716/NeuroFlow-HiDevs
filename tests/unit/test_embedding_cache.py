from __future__ import annotations

import json

import pytest
from opentelemetry import trace

from providers import client as client_module


class MemoryRedis:
    def __init__(self):
        self.values: dict[str, str] = {}
        self.ttls: dict[str, int] = {}

    async def mget(self, keys):
        return [self.values.get(key) for key in keys]

    async def set(self, key, value, ex):
        self.values[key] = value
        self.ttls[key] = ex


class BrokenRedis:
    async def mget(self, _keys):
        raise ConnectionError("redis unavailable")

    async def set(self, *_args, **_kwargs):
        raise ConnectionError("redis unavailable")


def _client(redis):
    instance = object.__new__(client_module.NeuroFlowClient)
    instance.redis = redis
    instance.embedding_cache_ttl_seconds = 1800
    instance.tracer = trace.get_tracer("test.embedding-cache")
    return instance

@pytest.mark.asyncio
async def test_embedding_cache_hits_and_deduplicates_misses(monkeypatch):
    redis = MemoryRedis()
    calls: list[list[str]] = []

    def encode(texts):
        calls.append(texts)
        return [[float(len(text))] + [1.0] * 383 for text in texts]

    monkeypatch.setattr(client_module, "_encode_texts", encode)
    client = _client(redis)

    first = await client.embed(["same input", "same input"])
    second = await client.embed(["same input"])

    assert first == second * 2
    assert calls == [["same input"]]
    assert list(redis.ttls.values()) == [1800]
    assert json.loads(next(iter(redis.values.values()))) == first[0]


@pytest.mark.asyncio
async def test_embedding_cache_failure_falls_back_to_local_encoder(monkeypatch):
    calls = []
    monkeypatch.setattr(
        client_module,
        "_encode_texts",
        lambda texts: calls.append(texts) or [[1.0] * 384],
    )
    client = _client(BrokenRedis())

    result = await client.embed(["uncached"])

    assert result == [[1.0] * 384]
    assert calls == [["uncached"]]
