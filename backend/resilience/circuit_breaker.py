"""
Redis-backed circuit breaker.

States:
    CLOSED    – normal operation; failures increment a counter.
    OPEN      – all calls rejected immediately with CircuitOpenError.
    HALF_OPEN – limited test calls permitted; a failure re-opens, successes
                eventually close the circuit.

All state mutations are performed atomically through Redis Lua scripts.
Cancellation is not counted as a provider failure.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from enum import StrEnum

from observability.metrics import (
    CIRCUIT_BREAKER_TRIPS,
    set_circuit_breaker_state,
)

from resilience._redis_protocol import AsyncRedisProtocol

logger = logging.getLogger(__name__)


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when a call is attempted against an open circuit."""

    def __init__(self, name: str, retry_after: float | None = None) -> None:
        self.name = name
        self.retry_after = retry_after

        message = f"Circuit '{name}' is OPEN."
        if retry_after is not None:
            message += f" Retry after {retry_after:.1f}s."

        super().__init__(message)


_LUA_RECORD_FAILURE = """
local state_key = KEYS[1]
local count_key = KEYS[2]
local opened_key = KEYS[3]
local inflight_key = KEYS[4]
local threshold = tonumber(ARGV[1])
local now = tonumber(ARGV[2])

local state = redis.call('GET', state_key) or 'closed'

if state == 'open' then
    return 'already_open'
end

if state == 'half_open' then
    redis.call('SET', state_key, 'open')
    redis.call('SET', opened_key, now)
    redis.call('SET', count_key, 1)
    redis.call('DEL', inflight_key)
    return 'reopened'
end

local count = redis.call('INCR', count_key)
if count >= threshold then
    redis.call('SET', state_key, 'open')
    redis.call('SET', opened_key, now)
    return 'opened'
end

return 'counted'
"""

_LUA_RECORD_SUCCESS = """
local state_key = KEYS[1]
local count_key = KEYS[2]
local inflight_key = KEYS[3]
local required_key = KEYS[4]

local state = redis.call('GET', state_key) or 'closed'

if state == 'closed' then
    redis.call('SET', count_key, 0)
    return 'reset'
end

if state == 'half_open' then
    local remaining = redis.call('DECR', required_key)
    redis.call('DECR', inflight_key)

    if remaining <= 0 then
        redis.call('SET', state_key, 'closed')
        redis.call('SET', count_key, 0)
        redis.call('DEL', required_key)
        redis.call('DEL', inflight_key)
        return 'closed'
    end

    return 'counted'
end

return 'noop'
"""

_LUA_ACQUIRE_HALF_OPEN = """
local state_key = KEYS[1]
local inflight_key = KEYS[2]
local max_calls = tonumber(ARGV[1])

local state = redis.call('GET', state_key) or 'closed'
if state ~= 'half_open' then
    return 0
end

local inflight = tonumber(redis.call('GET', inflight_key) or '0')
if inflight >= max_calls then
    return 0
end

redis.call('INCR', inflight_key)
return 1
"""

_LUA_RELEASE_HALF_OPEN_SLOT = """
local inflight_key = KEYS[1]
local value = tonumber(redis.call('GET', inflight_key) or '0')

if value > 0 then
    redis.call('DECR', inflight_key)
end

return 1
"""

_LUA_TRY_HALF_OPEN = """
local state_key = KEYS[1]
local opened_key = KEYS[2]
local required_key = KEYS[3]
local inflight_key = KEYS[4]
local now = tonumber(ARGV[1])
local timeout = tonumber(ARGV[2])
local max_calls = tonumber(ARGV[3])

local state = redis.call('GET', state_key) or 'closed'
if state ~= 'open' then
    return state
end

local opened_at = tonumber(redis.call('GET', opened_key) or '0')
if (now - opened_at) < timeout then
    return 'still_open'
end

redis.call('SET', state_key, 'half_open')
redis.call('SET', required_key, max_calls)
redis.call('SET', inflight_key, 0)

return 'half_open'
"""


class CircuitBreaker:
    """Redis-backed circuit breaker."""

    def __init__(
        self,
        redis: AsyncRedisProtocol,
        name: str,
        *,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if recovery_timeout < 0:
            raise ValueError("recovery_timeout cannot be negative")
        if half_open_max_calls < 1:
            raise ValueError("half_open_max_calls must be at least 1")

        self._redis = redis
        self._name = name
        self._threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max = half_open_max_calls

        self._k_state = f"circuit:{name}:state"
        self._k_count = f"circuit:{name}:failure_count"
        self._k_opened = f"circuit:{name}:opened_at"
        self._k_inflight = f"circuit:{name}:half_open_in_flight"
        self._k_required = f"circuit:{name}:half_open_required"

        self._sha_failure: str | None = None
        self._sha_success: str | None = None
        self._sha_acquire: str | None = None
        self._sha_release: str | None = None
        self._sha_try_half_open: str | None = None

        self._current_state = CircuitState.CLOSED

    @staticmethod
    def _require_sha(sha: str | None) -> str:
        """Return a loaded script SHA or raise a clear runtime error."""
        if sha is None:
            raise RuntimeError("Redis Lua script has not been loaded.")
        return sha

    @staticmethod
    def _decode(value: str | bytes | None) -> str | None:
        """Normalize Redis string responses."""
        if value is None:
            return None
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return value

    async def _load_scripts(self) -> None:
        """Load Lua scripts into Redis and cache their SHA values."""
        if self._sha_failure is None:
            self._sha_failure = await self._redis.script_load(_LUA_RECORD_FAILURE)

        if self._sha_success is None:
            self._sha_success = await self._redis.script_load(_LUA_RECORD_SUCCESS)

        if self._sha_acquire is None:
            self._sha_acquire = await self._redis.script_load(_LUA_ACQUIRE_HALF_OPEN)

        if self._sha_release is None:
            self._sha_release = await self._redis.script_load(_LUA_RELEASE_HALF_OPEN_SLOT)

        if self._sha_try_half_open is None:
            self._sha_try_half_open = await self._redis.script_load(_LUA_TRY_HALF_OPEN)

    async def _run_script(
        self,
        sha: str,
        keys: list[str],
        args: list[str | int | float],
    ) -> object:
        """Execute a previously loaded Redis Lua script."""
        return await self._redis.evalsha(
            sha,
            len(keys),
            *keys,
            *args,
        )

    async def state(self) -> CircuitState:
        """Return the current state, possibly transitioning OPEN to HALF_OPEN."""
        raw_state = self._decode(await self._redis.get(self._k_state))
        state = raw_state or CircuitState.CLOSED.value

        if state == CircuitState.OPEN:
            await self._load_scripts()

            result = await self._run_script(
                self._require_sha(self._sha_try_half_open),
                [
                    self._k_state,
                    self._k_opened,
                    self._k_required,
                    self._k_inflight,
                ],
                [
                    time.time(),
                    self._recovery_timeout,
                    self._half_open_max,
                ],
            )

            result_state = str(result)

            if result_state == "still_open":
                set_circuit_breaker_state(self._name, True)
                return CircuitState.OPEN

            state = result_state

            if state == CircuitState.HALF_OPEN:
                set_circuit_breaker_state(self._name, False)

        try:
            return CircuitState(state)
        except ValueError:
            logger.warning(
                "Unexpected circuit state name=%s state=%s",
                self._name,
                state,
            )
            return CircuitState.OPEN

    async def get_info(self) -> dict[str, object]:
        """Return observable state for the health endpoint."""
        raw_state = self._decode(await self._redis.get(self._k_state))
        raw_count = self._decode(await self._redis.get(self._k_count))
        raw_opened = self._decode(await self._redis.get(self._k_opened))

        state = raw_state or CircuitState.CLOSED.value
        failure_count = int(raw_count or "0")
        opened_at = float(raw_opened) if raw_opened else None

        set_circuit_breaker_state(
            self._name,
            state == CircuitState.OPEN.value,
        )

        return {
            "name": self._name,
            "state": state,
            "failure_count": failure_count,
            "opened_at": opened_at,
        }

    @asynccontextmanager
    async def __call__(
        self,
        name: str | None = None,
    ) -> AsyncGenerator[None, None]:
        """Allow usage as async with circuit_breaker('openai')."""
        async with self:
            yield

    async def __aenter__(self) -> CircuitBreaker:
        await self._load_scripts()
        now = time.time()

        transition = await self._run_script(
            self._require_sha(self._sha_try_half_open),
            [
                self._k_state,
                self._k_opened,
                self._k_required,
                self._k_inflight,
            ],
            [
                now,
                self._recovery_timeout,
                self._half_open_max,
            ],
        )

        if transition == CircuitState.HALF_OPEN.value:
            set_circuit_breaker_state(self._name, False)

        raw_state = self._decode(await self._redis.get(self._k_state))
        state = CircuitState(raw_state or CircuitState.CLOSED.value)

        if state == CircuitState.OPEN:
            set_circuit_breaker_state(self._name, True)

            raw_opened = self._decode(await self._redis.get(self._k_opened))
            opened_at = float(raw_opened) if raw_opened else now

            retry_after = max(
                0.0,
                self._recovery_timeout - (now - opened_at),
            )

            raise CircuitOpenError(self._name, retry_after=retry_after)

        if state == CircuitState.HALF_OPEN:
            granted = await self._run_script(
                self._require_sha(self._sha_acquire),
                [self._k_state, self._k_inflight],
                [self._half_open_max],
            )

            if not granted:
                raise CircuitOpenError(self._name, retry_after=1.0)

        self._current_state = state
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> bool:
        if exc_type in (asyncio.CancelledError, GeneratorExit):
            if self._current_state == CircuitState.HALF_OPEN:
                await self._run_script(
                    self._require_sha(self._sha_release),
                    [self._k_inflight],
                    [],
                )
            return False

        if exc_type is not None:
            transition = await self._run_script(
                self._require_sha(self._sha_failure),
                [
                    self._k_state,
                    self._k_count,
                    self._k_opened,
                    self._k_inflight,
                ],
                [self._threshold, time.time()],
            )

            if transition in ("opened", "reopened"):
                CIRCUIT_BREAKER_TRIPS.labels(provider=self._name).inc()
                set_circuit_breaker_state(self._name, True)

            logger.warning(
                "circuit_breaker name=%s recorded failure exc=%s",
                self._name,
                exc_type.__name__,
            )
            return False

        transition = await self._run_script(
            self._require_sha(self._sha_success),
            [
                self._k_state,
                self._k_count,
                self._k_inflight,
                self._k_required,
            ],
            [self._half_open_max],
        )

        if transition == "closed":
            set_circuit_breaker_state(self._name, False)

        return False


class CircuitBreakerRegistry:
    """Singleton-per-name registry for circuit breaker instances."""

    def __init__(self, redis: AsyncRedisProtocol) -> None:
        self._redis = redis
        self._breakers: dict[str, CircuitBreaker] = {}

    def get(
        self,
        name: str,
        *,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
    ) -> CircuitBreaker:
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                self._redis,
                name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                half_open_max_calls=half_open_max_calls,
            )

        return self._breakers[name]

    def all(self) -> dict[str, CircuitBreaker]:
        return dict(self._breakers)
