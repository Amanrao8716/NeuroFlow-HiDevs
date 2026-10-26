"""
Redis-backed circuit breaker.

States:
    CLOSED    – normal operation; failures increment a counter.
    OPEN      – all calls rejected immediately with CircuitOpenError.
    HALF_OPEN – limited test calls permitted; a failure re-opens, successes
                eventually close the circuit.

Redis keys (per named circuit):
    circuit:{name}:state         – "closed" | "open" | "half_open"
    circuit:{name}:failure_count – consecutive failure counter (integer)
    circuit:{name}:opened_at     – UNIX timestamp (float) when circuit opened
    circuit:{name}:half_open_in_flight – number of concurrent half-open calls

Design notes
────────────
- All state mutations go through Lua scripts so they are atomic even with
  multiple workers sharing the same Redis instance.
- asyncio.CancelledError is NOT counted as a provider failure; the caller may
  cancel for reasons unrelated to provider health (request timeout, shutdown).
- A timeout *wrapping* the provider call (e.g. asyncio.wait_for) will surface
  as a regular exception if the wrapped coroutine raises; we count that as a
  failure.  The circuit breaker itself never imposes timeouts – that is
  TimeoutManager's responsibility.
- Half-open close policy: every *successful* half-open call decrements
  remaining_required (initialised to half_open_max_calls).  When the counter
  reaches zero the circuit moves to CLOSED.  A single failure re-opens
  immediately.  This is conservative and deterministic.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from enum import Enum
from typing import AsyncGenerator

from observability.metrics import (
    CIRCUIT_BREAKER_TRIPS,
    set_circuit_breaker_state,
)

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when a call is attempted against an open circuit."""

    def __init__(self, name: str, retry_after: float | None = None) -> None:
        self.name = name
        self.retry_after = retry_after
        msg = f"Circuit '{name}' is OPEN."
        if retry_after is not None:
            msg += f"  Retry after {retry_after:.1f}s."
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Lua scripts – executed atomically by Redis
# ---------------------------------------------------------------------------

# Record a failure.  Transitions:
#   CLOSED  -> OPEN if failure_count >= threshold
#   HALF_OPEN -> OPEN immediately (any failure reopens)
_LUA_RECORD_FAILURE = """
local state_key   = KEYS[1]
local count_key   = KEYS[2]
local opened_key  = KEYS[3]
local inflight_key = KEYS[4]
local threshold   = tonumber(ARGV[1])
local now         = tonumber(ARGV[2])

local state = redis.call('GET', state_key) or 'closed'

if state == 'open' then
    return 'already_open'
end

if state == 'half_open' then
    -- Re-open immediately on any half-open failure
    redis.call('SET', state_key, 'open')
    redis.call('SET', opened_key, now)
    redis.call('SET', count_key, 1)
    redis.call('DEL', inflight_key)
    return 'reopened'
end

-- CLOSED state: increment failure count
local count = redis.call('INCR', count_key)
if count >= threshold then
    redis.call('SET', state_key, 'open')
    redis.call('SET', opened_key, now)
    return 'opened'
end

return 'counted'
"""

# Record a success.  Transitions:
#   HALF_OPEN -> CLOSED when remaining required successes drop to zero
# In CLOSED we just reset the failure count.
_LUA_RECORD_SUCCESS = """
local state_key    = KEYS[1]
local count_key    = KEYS[2]
local inflight_key = KEYS[3]
local required_key = KEYS[4]
local max_calls    = tonumber(ARGV[1])

local state = redis.call('GET', state_key) or 'closed'

if state == 'closed' then
    redis.call('SET', count_key, 0)
    return 'reset'
end

if state == 'half_open' then
    -- Decrement remaining-required counter atomically
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

# Try to acquire a half-open slot (atomic gate).
# Returns '1' if a slot was granted, '0' if at capacity or not half-open.
_LUA_ACQUIRE_HALF_OPEN = """
local state_key    = KEYS[1]
local inflight_key = KEYS[2]
local max_calls    = tonumber(ARGV[1])

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

# Release a half-open slot without recording success/failure (used on
# CancelledError so the slot doesn't leak).
_LUA_RELEASE_HALF_OPEN_SLOT = """
local inflight_key = KEYS[1]
local v = tonumber(redis.call('GET', inflight_key) or '0')
if v > 0 then
    redis.call('DECR', inflight_key)
end
return 1
"""

# Transition OPEN -> HALF_OPEN after recovery_timeout has elapsed.
# Only transitions if the state is still 'open' AND the elapsed time is met.
_LUA_TRY_HALF_OPEN = """
local state_key    = KEYS[1]
local opened_key   = KEYS[2]
local required_key = KEYS[3]
local inflight_key = KEYS[4]
local now          = tonumber(ARGV[1])
local timeout      = tonumber(ARGV[2])
local max_calls    = tonumber(ARGV[3])

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
    """
    Redis-backed circuit breaker.

    Usage::

        cb = CircuitBreaker(redis_client, "openai")
        async with cb:
            result = await provider.complete(...)
    """

    def __init__(
        self,
        redis,
        name: str,
        *,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
    ) -> None:
        self._redis = redis
        self._name = name
        self._threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max = half_open_max_calls

        # Redis key names
        self._k_state = f"circuit:{name}:state"
        self._k_count = f"circuit:{name}:failure_count"
        self._k_opened = f"circuit:{name}:opened_at"
        self._k_inflight = f"circuit:{name}:half_open_in_flight"
        self._k_required = f"circuit:{name}:half_open_required"

        # Cache Lua script SHAs after first load
        self._sha_failure: str | None = None
        self._sha_success: str | None = None
        self._sha_acquire: str | None = None
        self._sha_release: str | None = None
        self._sha_try_half_open: str | None = None

    # ------------------------------------------------------------------
    # Lua script helpers
    # ------------------------------------------------------------------

    async def _load_scripts(self) -> None:
        """Load all Lua scripts into Redis and cache SHAs."""
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

    async def _run_script(self, sha: str, keys: list, args: list):
        return await self._redis.evalsha(sha, len(keys), *keys, *args)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def state(self) -> CircuitState:
        """Return the current circuit state (may trigger OPEN->HALF_OPEN)."""
        raw = await self._redis.get(self._k_state)
        s = (raw or "closed")
        if s == CircuitState.OPEN:
            # Check recovery timeout
            await self._load_scripts()
            result = await self._run_script(
                self._sha_try_half_open,  # type: ignore[arg-type]
                [self._k_state, self._k_opened, self._k_required, self._k_inflight],
                [time.time(), self._recovery_timeout, self._half_open_max],
            )
            # 'still_open' means recovery timeout hasn't elapsed yet → stay OPEN
            if result == "still_open":
                set_circuit_breaker_state(self._name, True)
                return CircuitState.OPEN
            s = result or s
            if s == CircuitState.HALF_OPEN:
                set_circuit_breaker_state(self._name, False)
        # Guard against any unexpected values returned by the Lua sim
        try:
            return CircuitState(s)
        except ValueError:
            return CircuitState.OPEN


    async def get_info(self) -> dict:
        """Return observable state for the health endpoint."""
        raw_state = await self._redis.get(self._k_state)
        raw_count = await self._redis.get(self._k_count)
        raw_opened = await self._redis.get(self._k_opened)
        set_circuit_breaker_state(
            self._name,
            (raw_state or CircuitState.CLOSED) == CircuitState.OPEN,
        )
        return {
            "name": self._name,
            "state": raw_state or CircuitState.CLOSED,
            "failure_count": int(raw_count or 0),
            "opened_at": float(raw_opened) if raw_opened else None,
        }

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def __call__(self, name: str | None = None) -> AsyncGenerator[None, None]:  # type: ignore[override]
        """Use as: async with circuit_breaker('openai'):"""
        # 'name' parameter ignored – the instance was already constructed with a name.
        # Kept for compatibility with `async with circuit_breaker("openai"):`
        async with self:
            yield

    async def __aenter__(self):
        await self._load_scripts()
        now = time.time()

        # Possibly transition OPEN -> HALF_OPEN
        transition = await self._run_script(
            self._sha_try_half_open,  # type: ignore[arg-type]
            [self._k_state, self._k_opened, self._k_required, self._k_inflight],
            [now, self._recovery_timeout, self._half_open_max],
        )
        if transition == CircuitState.HALF_OPEN:
            set_circuit_breaker_state(self._name, False)

        raw_state = await self._redis.get(self._k_state)
        state = CircuitState(raw_state or "closed")

        if state == CircuitState.OPEN:
            set_circuit_breaker_state(self._name, True)
            raw_opened = await self._redis.get(self._k_opened)
            opened_at = float(raw_opened) if raw_opened else now
            retry_after = max(0.0, self._recovery_timeout - (now - opened_at))
            raise CircuitOpenError(self._name, retry_after=retry_after)

        if state == CircuitState.HALF_OPEN:
            granted = await self._run_script(
                self._sha_acquire,  # type: ignore[arg-type]
                [self._k_state, self._k_inflight],
                [self._half_open_max],
            )
            if not granted:
                raise CircuitOpenError(
                    self._name,
                    retry_after=1.0,
                )

        self._current_state = state
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type in (asyncio.CancelledError, GeneratorExit):
            # Cancellation is not a provider failure.  Release the half-open
            # slot if we had one, but do not modify the failure counter.
            if self._current_state == CircuitState.HALF_OPEN:
                await self._run_script(
                    self._sha_release,  # type: ignore[arg-type]
                    [self._k_inflight],
                    [],
                )
            return False  # re-raise CancelledError

        if exc_type is not None:
            # Any other exception counts as a failure.
            transition = await self._run_script(
                self._sha_failure,  # type: ignore[arg-type]
                [self._k_state, self._k_count, self._k_opened, self._k_inflight],
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
            return False  # re-raise the exception

        # Success
        transition = await self._run_script(
            self._sha_success,  # type: ignore[arg-type]
            [self._k_state, self._k_count, self._k_inflight, self._k_required],
            [self._half_open_max],
        )
        if transition == "closed":
            set_circuit_breaker_state(self._name, False)
        return False


# ---------------------------------------------------------------------------
# Module-level factory
# ---------------------------------------------------------------------------

class CircuitBreakerRegistry:
    """
    Singleton-per-name registry so the same CircuitBreaker instance is reused
    across the application.  Pass the Redis client once at startup.
    """

    def __init__(self, redis) -> None:
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
