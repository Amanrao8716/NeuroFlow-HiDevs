"""
Timeout manager with adaptive p95 latency tracking.

Default timeouts (seconds):
    embedding        : 10
    chat_completion  : 60
    reranking        : 15
    evaluation       : 120
    file_extraction  : 30
    url_fetch        : 15

Adaptive mode
─────────────
Latency observations per task type are stored in a Redis sorted set:
    Key:   latency:{task_type}
    Score: UNIX timestamp of observation
    Value: "{timestamp}:{latency_ms}" (unique member)

At most MAX_OBSERVATIONS (1 000) are retained; older entries are trimmed.
When ≥ MIN_OBSERVATIONS (30) exist, an adaptive timeout is computed as:
    timeout = clamp(p95_latency_seconds × 1.5,
                    min=static_default × 0.5,
                    max=static_default × 4.0)

The adaptive value is used for the NEXT call (not the current one); callers
always see a consistent timeout for the duration of their await.

Upward trend detection
──────────────────────
We compare the median latency in the most recent window (TREND_WINDOW_S = 300s)
with the median latency of the preceding window of equal length.  If the recent
window is ≥ TREND_RATIO (1.25×) larger AND both windows have ≥ TREND_MIN_OBS (5)
observations, a warning is emitted.  Warning are rate-limited: at most one
warning per task type per WARN_COOLDOWN_S (60 s).

Redis failure
─────────────
If Redis is temporarily unavailable:
- Latency recording is silently skipped (metric failure must not block callers).
- Static default timeouts are used instead of adaptive values.
- asyncio.wait_for still enforces the timeout, so the operation remains bounded.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Coroutine
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Default timeouts in seconds
TIMEOUT_DEFAULTS: dict[str, float] = {
    "embedding": 10.0,
    "chat_completion": 60.0,
    "reranking": 15.0,
    "evaluation": 120.0,
    "file_extraction": 30.0,
    "url_fetch": 15.0,
}

# Adaptive timeout parameters
MAX_OBSERVATIONS = 1_000
MIN_OBSERVATIONS = 30
P95_MULTIPLIER = 1.5        # adaptive = p95 × 1.5
MIN_RATIO = 0.5             # lower-bound: static × 0.5
MAX_RATIO = 4.0             # upper-bound: static × 4.0

# Trend detection parameters
TREND_WINDOW_S = 300        # seconds – recent window size
TREND_RATIO = 1.25          # flag if recent_median ≥ 1.25 × prior_median
TREND_MIN_OBS = 5           # minimum observations per window
WARN_COOLDOWN_S = 60        # min seconds between warnings per task type


class TaskTimeout(asyncio.TimeoutError):
    """Raised when a task exceeds its allowed execution time."""

    def __init__(self, task_type: str, timeout_s: float) -> None:
        self.task_type = task_type
        self.timeout_s = timeout_s
        super().__init__(
            f"Task '{task_type}' timed out after {timeout_s:.1f}s."
        )


class TimeoutManager:
    """
    Wraps coroutines with a bounded timeout, tracks latency in Redis,
    and emits warnings when sustained upward latency trends are detected.

    Usage::

        tm = TimeoutManager(redis)
        result = await tm.run("chat_completion", provider.complete(messages))
    """

    def __init__(
        self,
        redis=None,
        *,
        custom_defaults: dict[str, float] | None = None,
    ) -> None:
        self._redis = redis
        self._defaults: dict[str, float] = {**TIMEOUT_DEFAULTS}
        if custom_defaults:
            self._defaults.update(custom_defaults)

        # Per-task-type timestamp of last trend warning (local state, not Redis,
        # to avoid the overhead of another Redis round-trip on every call).
        self._last_warn: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        task_type: str,
        coro: Awaitable[T] | Coroutine[Any, Any, T],
        *,
        extra_timeout: float | None = None,
    ) -> T:
        """
        Execute *coro* within a bounded timeout.

        Parameters
        ----------
        task_type:
            One of the named task types in TIMEOUT_DEFAULTS, or a custom key
            if custom_defaults was passed at construction.
        coro:
            The coroutine to execute.
        extra_timeout:
            Override the computed timeout for this single call.
        """
        timeout_s = extra_timeout if extra_timeout is not None else await self._get_timeout(task_type)

        start = time.perf_counter()
        try:
            result = await asyncio.wait_for(asyncio.ensure_future(coro), timeout=timeout_s)
        except asyncio.TimeoutError as exc:
            elapsed = time.perf_counter() - start
            logger.warning(
                "timeout task_type=%s timeout_s=%.1f elapsed_s=%.1f",
                task_type,
                timeout_s,
                elapsed,
            )
            await self._increment_timeout_counter(task_type)
            raise TaskTimeout(task_type, timeout_s) from exc
        except asyncio.CancelledError:
            # Propagate without recording as a timeout
            raise

        elapsed_s = time.perf_counter() - start
        await self._record_latency(task_type, elapsed_s * 1000)
        return result

    # ------------------------------------------------------------------
    # Adaptive timeout calculation
    # ------------------------------------------------------------------

    async def _get_timeout(self, task_type: str) -> float:
        """Return the effective timeout; uses adaptive value when available."""
        static = self._defaults.get(task_type, 60.0)
        if self._redis is None:
            return static

        try:
            adaptive = await self._compute_adaptive_timeout(task_type, static)
            return adaptive
        except Exception:
            logger.debug(
                "timeout_manager: could not compute adaptive timeout for %s; using static %.1fs",
                task_type,
                static,
                exc_info=True,
            )
            return static

    async def _compute_adaptive_timeout(self, task_type: str, static: float) -> float:
        """Compute p95-based adaptive timeout from Redis sorted set."""
        key = f"latency:{task_type}"
        # Retrieve all latency values (the member is "{ts}:{latency_ms}")
        entries = await self._redis.zrange(key, 0, -1)
        if not entries or len(entries) < MIN_OBSERVATIONS:
            return static

        latencies_ms: list[float] = []
        for entry in entries:
            try:
                _, lat_str = entry.rsplit(":", 1)
                latencies_ms.append(float(lat_str))
            except (ValueError, AttributeError):
                pass

        if len(latencies_ms) < MIN_OBSERVATIONS:
            return static

        latencies_ms.sort()
        p95_ms = latencies_ms[int(len(latencies_ms) * 0.95)]
        p95_s = p95_ms / 1000.0

        adaptive = p95_s * P95_MULTIPLIER
        min_bound = static * MIN_RATIO
        max_bound = static * MAX_RATIO
        result = max(min_bound, min(max_bound, adaptive))

        # Upward trend check
        await self._check_upward_trend(task_type, entries)

        return result

    async def _check_upward_trend(self, task_type: str, entries: list[str]) -> None:
        """Emit a warning if recent latency shows a sustained upward trend."""
        now = time.time()
        # Rate-limit warnings
        last_warn = self._last_warn.get(task_type, 0.0)
        if (now - last_warn) < WARN_COOLDOWN_S:
            return

        cutoff_recent = now - TREND_WINDOW_S
        cutoff_prior = now - 2 * TREND_WINDOW_S

        recent_lats: list[float] = []
        prior_lats: list[float] = []

        for entry in entries:
            try:
                ts_str, lat_str = entry.rsplit(":", 1)
                ts = float(ts_str)
                lat = float(lat_str)
                if ts >= cutoff_recent:
                    recent_lats.append(lat)
                elif ts >= cutoff_prior:
                    prior_lats.append(lat)
            except (ValueError, AttributeError):
                pass

        if len(recent_lats) < TREND_MIN_OBS or len(prior_lats) < TREND_MIN_OBS:
            return

        recent_lats.sort()
        prior_lats.sort()
        recent_median = recent_lats[len(recent_lats) // 2]
        prior_median = prior_lats[len(prior_lats) // 2]

        if prior_median > 0 and recent_median >= prior_median * TREND_RATIO:
            logger.warning(
                "timeout_manager: upward latency trend detected task_type=%s "
                "prior_median_ms=%.1f recent_median_ms=%.1f ratio=%.2f",
                task_type,
                prior_median,
                recent_median,
                recent_median / prior_median,
            )
            self._last_warn[task_type] = now

    # ------------------------------------------------------------------
    # Latency recording
    # ------------------------------------------------------------------

    async def _record_latency(self, task_type: str, latency_ms: float) -> None:
        """Store a latency observation in Redis; trim to MAX_OBSERVATIONS."""
        if self._redis is None:
            return
        try:
            now = time.time()
            key = f"latency:{task_type}"
            member = f"{now}:{latency_ms:.2f}"
            async with self._redis.pipeline(transaction=True) as pipe:
                pipe.zadd(key, {member: now})
                # Keep only the newest MAX_OBSERVATIONS entries
                pipe.zremrangebyrank(key, 0, -(MAX_OBSERVATIONS + 1))
            await pipe.execute()
        except Exception:
            logger.debug(
                "timeout_manager: failed to record latency for %s",
                task_type,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Metrics counter
    # ------------------------------------------------------------------

    async def _increment_timeout_counter(self, task_type: str) -> None:
        """Increment Redis timeout counter; failure is non-fatal."""
        if self._redis is None:
            return
        try:
            await self._redis.incr(f"timeouts:{task_type}")
        except Exception:
            logger.debug(
                "timeout_manager: failed to increment counter for %s",
                task_type,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    async def get_stats(self, task_type: str) -> dict:
        """Return observable latency stats for a task type."""
        static = self._defaults.get(task_type, 60.0)
        result: dict = {
            "task_type": task_type,
            "static_default_s": static,
            "adaptive_timeout_s": None,
            "observation_count": 0,
            "p95_ms": None,
        }

        if self._redis is None:
            return result

        try:
            key = f"latency:{task_type}"
            entries = await self._redis.zrange(key, 0, -1)
            count = len(entries)
            result["observation_count"] = count

            if count >= MIN_OBSERVATIONS:
                lats = []
                for entry in entries:
                    try:
                        _, lat_str = entry.rsplit(":", 1)
                        lats.append(float(lat_str))
                    except (ValueError, AttributeError):
                        pass
                if lats:
                    lats.sort()
                    p95 = lats[int(len(lats) * 0.95)]
                    result["p95_ms"] = round(p95, 2)
                    adaptive = max(
                        static * MIN_RATIO,
                        min(static * MAX_RATIO, (p95 / 1000.0) * P95_MULTIPLIER),
                    )
                    result["adaptive_timeout_s"] = round(adaptive, 2)
        except Exception:
            pass

        return result
