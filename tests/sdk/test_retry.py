"""Unit tests for NeuroFlowClient retry and rate-limiting behavior."""

from __future__ import annotations

import httpx
import pytest
from neuroflow import NeuroFlowClient
from neuroflow.exceptions import NeuroFlowError, RateLimitError


@pytest.mark.asyncio
async def test_retry_on_429_success():
    """Verify client retries on 429 and succeeds on subsequent response."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "0.01"})
        return httpx.Response(200, json=[])

    transport = httpx.MockTransport(handler)
    async with NeuroFlowClient(
        "http://testserver",
        token="mock-token",
        transport=transport,
    ) as client:
        result = await client.list_pipelines()
        assert attempts == 2
        assert result == []


@pytest.mark.asyncio
async def test_retry_on_429_exhaustion():
    """Verify RateLimitError is raised when max_retries is exceeded."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "0.01"})

    transport = httpx.MockTransport(handler)
    async with NeuroFlowClient(
        "http://testserver",
        token="mock-token",
        max_retries=2,
        transport=transport,
    ) as client:
        with pytest.raises(RateLimitError) as exc_info:
            await client.list_pipelines()
        assert exc_info.value.status_code == 429
        assert exc_info.value.retry_after == 0.01


@pytest.mark.asyncio
async def test_no_retry_on_400():
    """Verify non-429 4xx errors are not retried."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, json={"detail": "Bad request"})

    transport = httpx.MockTransport(handler)
    async with NeuroFlowClient(
        "http://testserver",
        token="mock-token",
        transport=transport,
    ) as client:
        with pytest.raises(NeuroFlowError) as exc_info:
            await client._request("GET", "/test")
        assert attempts == 1
        assert exc_info.value.status_code == 400
