"""
Sliding-window endpoint rate limiting.

Implements per-IP, per-route rate limits using a Redis sliding window algorithm.

Key schema:
    ratelimit:{route}:{ip} → sorted set of request timestamps (score = timestamp)

Algorithm
─────────
1. Compute window_start = now - window_seconds.
2. Atomically:
   a. Remove members with score < window_start  (expired entries)
   b. Count remaining members
   c. If count < limit: add current timestamp as new member, set TTL.
   d. Else: return (0, oldest_member_score) so caller knows retry_after.

IP extraction
─────────────
Client IP is extracted from:
  - X-Forwarded-For header IF app.state.trust_proxy is True
  - Request.client.host otherwise

This prevents IP spoofing in environments that do not sit behind a trusted proxy.

Configured routes
─────────────────
POST /ingest   : 10 requests / hour
POST /query    : 60 requests / minute

Health endpoints are NOT rate-limited.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from fastapi import Request, Response
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# Lua script: sliding-window consume
_LUA_SLIDE = """
local key          = KEYS[1]
local now          = tonumber(ARGV[1])
local window_start = tonumber(ARGV[2])
local limit        = tonumber(ARGV[3])
local ttl          = tonumber(ARGV[4])

-- Remove expired entries
redis.call('ZREMRANGEBYSCORE', key, '-inf', window_start)

-- Count in current window
local count = redis.call('ZCARD', key)

if count < limit then
    -- Admit this request
    redis.call('ZADD', key, now, now)
    redis.call('EXPIRE', key, ttl)
    return {1, '0'}
else
    -- Oldest entry tells us when the window clears
    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    local oldest_score = tonumber(oldest[2] or now)
    return {0, tostring(oldest_score)}
end
"""


@dataclass
class RouteLimit:
    limit: int          # max requests
    window_s: int       # window in seconds
    route_key: str      # used in Redis key


# Route configuration
ROUTE_LIMITS: dict[str, RouteLimit] = {
    "POST:/ingest": RouteLimit(
        limit=10, window_s=3600, route_key="ingest"
    ),
    "POST:/query": RouteLimit(
        limit=60, window_s=60, route_key="query"
    ),
}


def _extract_ip(request: Request) -> str:
    """
    Safely extract the client IP.

    Honours X-Forwarded-For only when app.state.trust_proxy == True.
    Falls back to request.client.host.
    """
    trust_proxy = getattr(getattr(request, "app", None), "state", None)
    trust_proxy = getattr(trust_proxy, "trust_proxy", False) if trust_proxy else False

    if trust_proxy:
        xff = request.headers.get("x-forwarded-for", "")
        if xff:
            # First address is the original client
            return xff.split(",")[0].strip()

    if request.client:
        return request.client.host or "unknown"
    return "unknown"


class SlidingWindowRateLimiter:
    """
    FastAPI middleware/dependency for per-IP endpoint rate limiting.

    Use as a FastAPI middleware::

        app.add_middleware(RateLimitMiddleware, redis=redis_client)

    Or as a dependency on individual routes::

        @router.post("/ingest")
        async def ingest(req: Request, _=Depends(ingest_rate_limit)):
            ...
    """

    def __init__(self, redis, *, limits: dict[str, RouteLimit] | None = None) -> None:
        self._redis = redis
        self._limits = limits or ROUTE_LIMITS
        self._sha: str | None = None

    async def _load_script(self) -> None:
        if self._sha is None:
            self._sha = await self._redis.script_load(_LUA_SLIDE)

    async def check(self, request: Request) -> JSONResponse | None:
        """
        Check rate limit for the request.

        Returns a JSONResponse(429) if limit exceeded, else None.
        """
        route_key = f"{request.method}:{request.url.path}"
        cfg = self._limits.get(route_key)
        if cfg is None:
            return None  # route not rate-limited

        ip = _extract_ip(request)
        redis_key = f"ratelimit:{cfg.route_key}:{ip}"
        now = time.time()
        window_start = now - cfg.window_s

        try:
            await self._load_script()
            result = await self._redis.evalsha(
                self._sha,
                1,
                redis_key,
                now,
                window_start,
                cfg.limit,
                cfg.window_s + 10,  # TTL slightly longer than window
            )
            granted = bool(int(result[0]))
            if not granted:
                oldest_score = float(result[1])
                retry_after = max(1, int(cfg.window_s - (now - oldest_score)) + 1)
                logger.info(
                    "rate_limit exceeded route=%s ip=%s retry_after=%d",
                    route_key,
                    ip,
                    retry_after,
                )
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "rate_limit_exceeded",
                        "route": route_key,
                        "retry_after": retry_after,
                    },
                    headers={"Retry-After": str(retry_after)},
                )
        except Exception:
            logger.warning(
                "rate_limit: Redis error for route=%s ip=%s; admitting request",
                route_key,
                ip,
                exc_info=True,
            )
        return None  # admit on Redis failure (fail-open)


class RateLimitMiddleware:
    """
    Starlette-compatible ASGI middleware wrapping SlidingWindowRateLimiter.

    Add to FastAPI with::

        app.add_middleware(RateLimitMiddleware)

    Redis is retrieved from ``app.state.redis`` at request time so the
    middleware can be added during application construction (before lifespan
    initialises Redis).
    """

    def __init__(self, app, limits: dict[str, RouteLimit] | None = None) -> None:
        self.app = app
        self._limits = limits or ROUTE_LIMITS
        self._limiters: dict[int, SlidingWindowRateLimiter] = {}

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        from starlette.requests import Request as StarletteRequest
        from starlette.applications import Starlette

        request = StarletteRequest(scope, receive)

        # Fetch Redis from app state at request time (available after lifespan)
        app_state = getattr(scope.get("app", None), "state", None)
        redis_client = getattr(app_state, "redis", None)

        if redis_client is None:
            # Redis not yet initialised; skip rate-limiting
            await self.app(scope, receive, send)
            return

        # Obtain (or create) the per-redis limiter
        limiter_key = id(redis_client)
        if limiter_key not in self._limiters:
            self._limiters[limiter_key] = SlidingWindowRateLimiter(
                redis_client, limits=self._limits
            )
        limiter = self._limiters[limiter_key]

        response = await limiter.check(request)

        if response is not None:
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
