"""
Sliding-window endpoint rate limiting.

Implements per-IP, per-route rate limits using a Redis sliding window algorithm.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from resilience._redis_protocol import AsyncRedisProtocol

logger = logging.getLogger(__name__)

_LUA_SLIDE = """
local key          = KEYS[1]
local now          = tonumber(ARGV[1])
local window_start = tonumber(ARGV[2])
local limit        = tonumber(ARGV[3])
local ttl          = tonumber(ARGV[4])

redis.call('ZREMRANGEBYSCORE', key, '-inf', window_start)

local count = redis.call('ZCARD', key)

if count < limit then
    redis.call('ZADD', key, now, now)
    redis.call('EXPIRE', key, ttl)
    return {1, '0'}
else
    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    local oldest_score = tonumber(oldest[2] or now)
    return {0, tostring(oldest_score)}
end
"""


@dataclass
class RouteLimit:
    limit: int
    window_s: int
    route_key: str


ROUTE_LIMITS: dict[str, RouteLimit] = {
    "POST:/ingest": RouteLimit(
        limit=10,
        window_s=3600,
        route_key="ingest",
    ),
    "POST:/query": RouteLimit(
        limit=60,
        window_s=60,
        route_key="query",
    ),
}


def _extract_ip(request: Request) -> str:
    """Safely extract the client IP."""
    app_state = getattr(
        getattr(request, "app", None),
        "state",
        None,
    )
    trust_proxy = getattr(app_state, "trust_proxy", False)

    if trust_proxy:
        xff = request.headers.get("x-forwarded-for", "")
        if xff:
            return xff.split(",")[0].strip()

    if request.client:
        return request.client.host or "unknown"

    return "unknown"


def _require_sha(sha: str | None) -> str:
    """Return a validated Redis script SHA."""
    if sha is None:
        raise RuntimeError("Redis rate-limit script SHA is unavailable.")

    return sha


def _to_float(value: object) -> float:
    """Convert a Redis Lua result value to float."""
    if isinstance(value, bytes):
        return float(value.decode("utf-8"))

    if isinstance(value, (str, int, float)):
        return float(value)

    raise TypeError("Invalid Redis rate-limit response value.")


class SlidingWindowRateLimiter:
    """Per-IP endpoint rate limiter."""

    def __init__(
        self,
        redis: AsyncRedisProtocol,
        *,
        limits: dict[str, RouteLimit] | None = None,
    ) -> None:
        self._redis = redis
        self._limits = limits or ROUTE_LIMITS
        self._sha: str | None = None

    async def _load_script(self) -> None:
        if self._sha is None:
            self._sha = await self._redis.script_load(_LUA_SLIDE)

    async def check(self, request: Request) -> JSONResponse | None:
        """Return HTTP 429 when the configured rate limit is exceeded."""
        route_key = f"{request.method}:{request.url.path}"
        cfg = self._limits.get(route_key)

        if cfg is None:
            return None

        ip = _extract_ip(request)
        redis_key = f"ratelimit:{cfg.route_key}:{ip}"

        now = time.time()
        window_start = now - cfg.window_s

        try:
            await self._load_script()

            result = await self._redis.evalsha(
                _require_sha(self._sha),
                1,
                redis_key,
                now,
                window_start,
                cfg.limit,
                cfg.window_s + 10,
            )

            if not isinstance(result, (list, tuple)) or len(result) < 2:
                raise ValueError("Invalid Redis rate-limit script response.")

            granted = bool(int(_to_float(result[0])))

            if not granted:
                oldest_score = _to_float(result[1])

                retry_after = max(
                    1,
                    int(cfg.window_s - (now - oldest_score)) + 1,
                )

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
                    headers={
                        "Retry-After": str(retry_after),
                    },
                )

        except Exception:
            logger.warning(
                "rate_limit: Redis error for route=%s ip=%s; admitting request",
                route_key,
                ip,
                exc_info=True,
            )

        return None


class RateLimitMiddleware:
    """Starlette-compatible ASGI rate-limit middleware."""

    def __init__(
        self,
        app: ASGIApp,
        limits: dict[str, RouteLimit] | None = None,
    ) -> None:
        self.app = app
        self._limits = limits or ROUTE_LIMITS
        self._limiters: dict[int, SlidingWindowRateLimiter] = {}

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        from starlette.requests import Request as StarletteRequest

        request = StarletteRequest(scope, receive)

        app_state = getattr(
            scope.get("app"),
            "state",
            None,
        )
        redis_client = getattr(app_state, "redis", None)

        if redis_client is None:
            await self.app(scope, receive, send)
            return

        if not isinstance(redis_client, AsyncRedisProtocol):
            logger.warning(
                "Rate limiting skipped because app.state.redis "
                "does not implement AsyncRedisProtocol."
            )
            await self.app(scope, receive, send)
            return

        limiter_key = id(redis_client)

        if limiter_key not in self._limiters:
            self._limiters[limiter_key] = SlidingWindowRateLimiter(
                redis_client,
                limits=self._limits,
            )

        limiter = self._limiters[limiter_key]

        response = await limiter.check(request)

        if response is not None:
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
