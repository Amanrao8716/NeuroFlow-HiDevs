"""
Redis-backed token-bucket rate limiter.

Algorithm
─────────
Each bucket stores:
    rpb:{provider}:tokens       – current token count (float stored as string)
    rpb:{provider}:last_refill  – UNIX timestamp of last refill (float)

A single Lua script atomically:
  1. Reads current tokens and last-refill timestamp.
  2. Computes tokens to add based on elapsed time × refill_rate, capped at capacity.
  3. Checks if enough tokens are available for the requested cost.
  4. If yes, deducts and returns (1, wait=0).
  5. If no, returns (0, seconds_to_wait) without mutating the bucket.

Per-pipeline buckets:
    rpb:pipeline:{pipeline_id}:tokens
    rpb:pipeline:{pipeline_id}:last_refill

Global provider defaults are configurable; OpenAI defaults are provided as
named constants but are not hardcoded for other providers.

Concurrency
───────────
All mutations go through a single Lua script executed atomically in Redis,
so multiple workers cannot overspend a bucket.

Waiting
───────
If tokens are insufficient, the caller waits for the computed refill duration
(capped at max_wait_seconds) and retries once.  asyncio.CancelledError
propagates cleanly so the caller is not stuck.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider-specific configuration
# ---------------------------------------------------------------------------

@dataclass
class BucketConfig:
    """Token-bucket parameters for a named provider or pipeline."""
    capacity: float         # maximum tokens in the bucket
    refill_rate: float      # tokens added per second
    cost_per_request: float = 1.0

    def key_prefix(self, name: str) -> str:
        return f"rpb:{name}"


# Preset for global OpenAI limits (per task spec)
OPENAI_CONFIG = BucketConfig(
    capacity=3_000.0,
    refill_rate=50.0,       # 3000 tokens / 60s = 50 /s
    cost_per_request=1.0,
)

# Anthropic has no fixed public limit; use a generous default.
ANTHROPIC_CONFIG = BucketConfig(
    capacity=1_000.0,
    refill_rate=16.67,      # 1000 / 60s
    cost_per_request=1.0,
)

PROVIDER_CONFIGS: dict[str, BucketConfig] = {
    "openai": OPENAI_CONFIG,
    "anthropic": ANTHROPIC_CONFIG,
}


class RateLimitExceeded(Exception):
    """Raised when the rate limiter cannot grant tokens within max_wait."""

    def __init__(self, bucket: str, retry_after: float) -> None:
        self.bucket = bucket
        self.retry_after = retry_after
        super().__init__(
            f"Rate limit exceeded for bucket '{bucket}'. "
            f"Retry after {retry_after:.1f}s."
        )


# ---------------------------------------------------------------------------
# Lua script – atomic consume-or-refuse
# ---------------------------------------------------------------------------

_LUA_CONSUME = """
local tokens_key       = KEYS[1]
local last_refill_key  = KEYS[2]
local capacity         = tonumber(ARGV[1])
local refill_rate      = tonumber(ARGV[2])
local cost             = tonumber(ARGV[3])
local now              = tonumber(ARGV[4])

local tokens      = tonumber(redis.call('GET', tokens_key) or capacity)
local last_refill = tonumber(redis.call('GET', last_refill_key) or now)

-- Refill
local elapsed = math.max(0, now - last_refill)
tokens = math.min(capacity, tokens + elapsed * refill_rate)
redis.call('SET', last_refill_key, now)

if tokens >= cost then
    redis.call('SET', tokens_key, tokens - cost)
    return {1, '0'}
else
    -- How many seconds until we have enough tokens?
    local needed = cost - tokens
    local wait_s = needed / refill_rate
    redis.call('SET', tokens_key, tokens)  -- persist refilled but un-spent tokens
    return {0, tostring(wait_s)}
end
"""


class RateLimiter:
    """
    Redis token-bucket rate limiter.

    Usage::

        rl = RateLimiter(redis, "openai", config=OPENAI_CONFIG)
        await rl.acquire()          # blocks until token available
        ...
        # Per-pipeline:
        pl_rl = RateLimiter(redis, f"pipeline:{pipeline_id}",
                            config=BucketConfig(capacity=60, refill_rate=1.0))
        await pl_rl.acquire()
    """

    def __init__(
        self,
        redis,
        name: str,
        *,
        config: BucketConfig | None = None,
        max_wait_seconds: float = 60.0,
        _clock=None,            # injectable for tests
    ) -> None:
        self._redis = redis
        self._name = name
        self._config = config or PROVIDER_CONFIGS.get(name, OPENAI_CONFIG)
        self._max_wait = max_wait_seconds
        self._clock = _clock or time.time

        prefix = f"rpb:{name}"
        self._k_tokens = f"{prefix}:tokens"
        self._k_refill = f"{prefix}:last_refill"
        self._sha: str | None = None

    async def _load_script(self) -> None:
        if self._sha is None:
            self._sha = await self._redis.script_load(_LUA_CONSUME)

    async def _try_consume(
        self,
        cost: float = 1.0,
    ) -> tuple[bool, float]:
        """Attempt to consume tokens. Returns (success, wait_seconds)."""
        await self._load_script()
        now = self._clock()
        result = await self._redis.evalsha(
            self._sha,
            2,
            self._k_tokens,
            self._k_refill,
            self._config.capacity,
            self._config.refill_rate,
            cost,
            now,
        )
        granted = bool(int(result[0]))
        wait_s = float(result[1])
        return granted, wait_s

    async def acquire(
        self,
        cost: float | None = None,
    ) -> None:
        """
        Block until a token is available (or raise RateLimitExceeded).

        Waits at most max_wait_seconds.  CancelledError propagates cleanly.
        """
        if cost is None:
            cost = self._config.cost_per_request

        granted, wait_s = await self._try_consume(cost)
        if granted:
            return

        if wait_s > self._max_wait:
            raise RateLimitExceeded(self._name, retry_after=wait_s)

        logger.debug(
            "rate_limiter bucket=%s waiting=%.2fs",
            self._name,
            wait_s,
        )
        await asyncio.sleep(wait_s)

        granted, wait_s2 = await self._try_consume(cost)
        if not granted:
            raise RateLimitExceeded(self._name, retry_after=wait_s2)

    async def peek(self) -> dict:
        """Return the current bucket state (for observability)."""
        raw_tokens = await self._redis.get(self._k_tokens)
        raw_refill = await self._redis.get(self._k_refill)
        capacity = self._config.capacity
        tokens = float(raw_tokens) if raw_tokens else capacity
        return {
            "name": self._name,
            "tokens": round(tokens, 2),
            "capacity": capacity,
            "refill_rate": self._config.refill_rate,
            "last_refill": float(raw_refill) if raw_refill else None,
        }


# ---------------------------------------------------------------------------
# Pipeline-scoped rate limiter factory
# ---------------------------------------------------------------------------

def pipeline_rate_limiter(
    redis,
    pipeline_id: str,
    rate_limit_rpm: int = 60,
    max_wait_seconds: float = 60.0,
) -> RateLimiter:
    """
    Return a per-pipeline token-bucket rate limiter.

    Capacity = rate_limit_rpm (1 min window → refill_rate = rpm/60 per second).
    """
    config = BucketConfig(
        capacity=float(rate_limit_rpm),
        refill_rate=rate_limit_rpm / 60.0,
        cost_per_request=1.0,
    )
    return RateLimiter(
        redis,
        f"pipeline:{pipeline_id}",
        config=config,
        max_wait_seconds=max_wait_seconds,
    )
