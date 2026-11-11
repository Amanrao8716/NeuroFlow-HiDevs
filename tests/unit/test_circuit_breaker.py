from __future__ import annotations

import time

import pytest
from resilience.circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState


class InMemoryRedis:
    """Small Redis script simulator for circuit breaker state transition tests."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.scripts: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def set(self, key: str, value: object) -> None:
        self.values[key] = str(value)

    async def delete(self, *keys: str) -> None:
        for key in keys:
            self.values.pop(key, None)

    async def script_load(self, script: str) -> str:
        sha = str(len(self.scripts))
        self.scripts[sha] = script
        return sha

    async def evalsha(self, sha: str, numkeys: int, *args: object) -> object:
        script = self.scripts[sha]
        keys = [str(key) for key in args[:numkeys]]
        argv = args[numkeys:]
        if "threshold" in script and "half_open" in script and "INCR" in script:
            state, count, opened, inflight = keys
            current = self.values.get(state, "closed")
            if current == "half_open":
                self.values.update({state: "open", opened: str(argv[1]), count: "1"})
                self.values.pop(inflight, None)
                return "reopened"
            failures = int(self.values.get(count, "0")) + 1
            self.values[count] = str(failures)
            if failures >= int(argv[0]):
                self.values[state], self.values[opened] = "open", str(argv[1])
                return "opened"
            return "counted"
        if "still_open" in script:
            state, opened, required, inflight = keys
            if self.values.get(state, "closed") != "open":
                return self.values.get(state, "closed")
            if float(argv[0]) - float(self.values.get(opened, "0")) < float(argv[1]):
                return "still_open"
            self.values.update({state: "half_open", required: str(argv[2]), inflight: "0"})
            return "half_open"
        if "inflight >= max_calls" in script:
            state, inflight = keys
            count = int(self.values.get(inflight, "0"))
            if self.values.get(state) != "half_open" or count >= int(argv[0]):
                return 0
            self.values[inflight] = str(count + 1)
            return 1
        if "required_key" in script:
            state, count, inflight, required = keys
            if self.values.get(state, "closed") == "closed":
                self.values[count] = "0"
                return "reset"
            remaining = int(self.values[required]) - 1
            self.values[required] = str(remaining)
            self.values[inflight] = str(max(0, int(self.values.get(inflight, "0")) - 1))
            if remaining <= 0:
                self.values.update({state: "closed", count: "0"})
                self.values.pop(required, None)
                self.values.pop(inflight, None)
                return "closed"
            return "counted"
        return 1


@pytest.fixture
def breaker() -> CircuitBreaker:
    return CircuitBreaker(
        InMemoryRedis(),
        "unit",
        failure_threshold=2,
        recovery_timeout=0.01,
        half_open_max_calls=1,
    )


@pytest.mark.asyncio
async def test_failure_threshold_opens_circuit(breaker: CircuitBreaker) -> None:
    for _ in range(2):
        with pytest.raises(RuntimeError):
            async with breaker:
                raise RuntimeError("provider failure")
    assert await breaker.state() is CircuitState.OPEN


@pytest.mark.asyncio
async def test_open_circuit_rejects_requests(breaker: CircuitBreaker) -> None:
    redis = breaker._redis
    await redis.set("circuit:unit:state", "open")
    await redis.set("circuit:unit:opened_at", time.time())
    with pytest.raises(CircuitOpenError):
        async with breaker:
            pass


@pytest.mark.asyncio
async def test_open_timeout_transitions_to_half_open(breaker: CircuitBreaker) -> None:
    redis = breaker._redis
    await redis.set("circuit:unit:state", "open")
    await redis.set("circuit:unit:opened_at", time.time() - 1)
    assert await breaker.state() is CircuitState.HALF_OPEN


@pytest.mark.asyncio
async def test_half_open_success_recovers_to_closed(breaker: CircuitBreaker) -> None:
    redis = breaker._redis
    await redis.set("circuit:unit:state", "open")
    await redis.set("circuit:unit:opened_at", time.time() - 1)
    async with breaker:
        pass
    assert await breaker.state() is CircuitState.CLOSED


@pytest.mark.asyncio
async def test_success_resets_closed_failure_count(breaker: CircuitBreaker) -> None:
    redis = breaker._redis
    await redis.set("circuit:unit:failure_count", 1)
    async with breaker:
        pass
    assert await redis.get("circuit:unit:failure_count") == "0"


@pytest.mark.asyncio
async def test_half_open_failure_reopens_circuit(breaker: CircuitBreaker) -> None:
    redis = breaker._redis
    await redis.set("circuit:unit:state", "open")
    await redis.set("circuit:unit:opened_at", time.time() - 1)
    with pytest.raises(RuntimeError):
        async with breaker:
            raise RuntimeError("still unavailable")
    assert await breaker.state() is CircuitState.OPEN
