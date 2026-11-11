"""Unit tests for SDK SSE streaming parser."""

from __future__ import annotations

import httpx
import pytest
from neuroflow import NeuroFlowClient
from neuroflow.exceptions import NeuroFlowError


@pytest.mark.asyncio
async def test_streaming_token_assembly():
    """Verify stream yields tokens from SSE format."""
    sse_body = (
        ": keepalive comment\n\n"
        'data: {"type": "token", "content": "Hello"}\n\n'
        'data: {"type": "token", "content": " world"}\n\n'
        'data: {"type": "token", "content": "!"}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/query":
            return httpx.Response(200, json={"run_id": "r-1", "stream_url": "/query/r-1/stream"})
        if request.url.path == "/query/r-1/stream":
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                text=sse_body,
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    async with NeuroFlowClient(
        "http://testserver",
        token="mock-token",
        transport=transport,
    ) as client:
        stream = await client.query("test", pipeline_id="p-1", stream=True)
        tokens = [token async for token in stream]
        assert tokens == ["Hello", " world", "!"]
        assert "".join(tokens) == "Hello world!"


@pytest.mark.asyncio
async def test_streaming_direct_async_for():
    """Verify direct async-iteration: async for token in client.query(..., stream=True)."""
    sse_body = (
        'data: {"type": "token", "content": "Async"}\n\n'
        'data: {"type": "token", "content": " generator"}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/query":
            return httpx.Response(200, json={"run_id": "r-direct", "stream_url": "/query/r-direct/stream"})
        if request.url.path == "/query/r-direct/stream":
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                text=sse_body,
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    async with NeuroFlowClient(
        "http://testserver",
        token="mock-token",
        transport=transport,
    ) as client:
        tokens = []
        async for token in client.query("test", pipeline_id="p-1", stream=True):
            tokens.append(token)
        assert tokens == ["Async", " generator"]
        assert "".join(tokens) == "Async generator"


@pytest.mark.asyncio
async def test_streaming_handles_malformed_and_comments():
    """Verify stream ignores malformed events and comments."""
    sse_body = (
        ": comment line\n"
        "\n"
        "data: NOT_VALID_JSON\n\n"
        'data: {"type": "token", "content": "Valid"}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=sse_body,
        )

    transport = httpx.MockTransport(handler)
    async with NeuroFlowClient(
        "http://testserver",
        token="mock-token",
        transport=transport,
    ) as client:
        tokens = []
        async for token in client._stream_tokens("/stream"):
            tokens.append(token)
        assert tokens == ["Valid"]


@pytest.mark.asyncio
async def test_streaming_error_event_raises():
    """Verify error event from SSE raises NeuroFlowError."""
    sse_body = (
        'data: {"type": "token", "content": "Partial"}\n\n'
        'data: {"type": "error", "message": "Model generation timeout"}\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=sse_body,
        )

    transport = httpx.MockTransport(handler)
    async with NeuroFlowClient(
        "http://testserver",
        token="mock-token",
        transport=transport,
    ) as client:
        with pytest.raises(NeuroFlowError) as exc_info:
            async for _ in client._stream_tokens("/stream"):
                pass
        assert "Model generation timeout" in str(exc_info.value)
