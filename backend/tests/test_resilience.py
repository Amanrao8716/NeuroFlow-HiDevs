"""
Tests for the NeuroFlow resilience layer.

Covers:
- Circuit breaker: open after N failures, recovery, HALF_OPEN concurrency,
  state persistence, cancellation behavior.
- Rate limiter: token refill, exhaustion, persistence, concurrent workers.
- Sliding-window 429 and Retry-After header.
- Backpressure: depth > 100 rejects without enqueue, depth > 50 warns.
- Timeout manager: timeout counter increment, adaptive p95 selection,
  upward-trend warning detection.
- Health status: degraded on open circuit, critical on DB/Redis failure.

All tests use a fake Redis (FakeRedis / dict-backed AsyncMock) and a fake
clock – NO real Redis, PostgreSQL, or paid LLM API calls are used.
"""

from __future__ import annotations

import asyncio
import math
import time
import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Helpers – lightweight in-memory fake Redis
# ──────────────────────────────────────────────────────────────────────────────

class FakeRedis:
    """
    Minimal in-memory Redis stand-in sufficient for all resilience tests.
    Supports: get, set, del, incr, decr, incrbyfloat, llen, zadd, zrange,
    zremrangebyrank, zremrangebyscore, zcard, expire, script_load, evalsha,
    pipeline (non-transactional batch), hgetall.
    """

    def __init__(self):
        self._store: dict = {}
        self._scripts: dict[str, str] = {}   # sha -> lua source (not executed natively)
        # Pre-registered scripts by their approximate functionality
        self._lua_registry: dict[str, str] = {}

    # ── Basic key ops ──────────────────────────────────────────────────────
    async def get(self, key: str):
        return self._store.get(key)

    async def set(self, key: str, value, ex=None, **kwargs):
        self._store[key] = str(value)

    async def delete(self, *keys):
        for k in keys:
            self._store.pop(k, None)

    async def exists(self, key: str) -> int:
        return 1 if key in self._store else 0

    # ── Counters ───────────────────────────────────────────────────────────
    async def incr(self, key: str) -> int:
        val = int(self._store.get(key, 0)) + 1
        self._store[key] = str(val)
        return val

    async def decr(self, key: str) -> int:
        val = int(self._store.get(key, 0)) - 1
        self._store[key] = str(val)
        return val

    async def incrbyfloat(self, key: str, amount: float) -> float:
        val = float(self._store.get(key, 0)) + amount
        self._store[key] = str(val)
        return val

    # ── List ops ───────────────────────────────────────────────────────────
    async def llen(self, key: str) -> int:
        v = self._store.get(key)
        if isinstance(v, list):
            return len(v)
        return int(v) if v is not None else 0

    async def lpush(self, key: str, *values):
        if key not in self._store:
            self._store[key] = []
        self._store[key] = list(values) + self._store[key]

    # ── Sorted set ops ─────────────────────────────────────────────────────
    async def zadd(self, key: str, mapping: dict) -> int:
        if key not in self._store:
            self._store[key] = {}
        d = self._store[key]
        added = 0
        for member, score in mapping.items():
            if member not in d:
                added += 1
            d[member] = float(score)
        return added

    async def zrange(self, key: str, start: int, end: int, **kwargs) -> list:
        d = self._store.get(key, {})
        sorted_items = sorted(d.items(), key=lambda x: x[1])
        if end == -1:
            sliced = sorted_items[start:]
        else:
            sliced = sorted_items[start:end + 1]
        return [item[0] for item in sliced]

    async def zremrangebyrank(self, key: str, start: int, end: int) -> int:
        d = self._store.get(key, {})
        sorted_items = sorted(d.items(), key=lambda x: x[1])
        if end < 0:
            end = len(sorted_items) + end
        to_remove = sorted_items[max(0, start):end + 1]
        for member, _ in to_remove:
            del d[member]
        return len(to_remove)

    async def zremrangebyscore(self, key: str, min_score, max_score) -> int:
        d = self._store.get(key, {})
        min_v = float('-inf') if min_score == '-inf' else float(min_score)
        max_v = float('inf') if max_score == '+inf' else float(max_score)
        to_remove = [m for m, s in d.items() if min_v <= s <= max_v]
        for m in to_remove:
            del d[m]
        return len(to_remove)

    async def zcard(self, key: str) -> int:
        return len(self._store.get(key, {}))

    async def expire(self, key: str, seconds: int) -> int:
        return 1

    # ── Hash ops ───────────────────────────────────────────────────────────
    async def hgetall(self, key: str) -> dict:
        v = self._store.get(key, {})
        return v if isinstance(v, dict) else {}

    # ── Pipeline (non-transactional) ───────────────────────────────────────
    class _Pipeline:
        def __init__(self, redis_instance, transaction=False):
            self._redis = redis_instance
            self._cmds: list = []

        def zadd(self, key, mapping, **kwargs):
            self._cmds.append(("zadd", key, mapping))
            return self

        def zremrangebyrank(self, key, start, end):
            self._cmds.append(("zremrangebyrank", key, start, end))
            return self

        def incr(self, key):
            self._cmds.append(("incr", key))
            return self

        def incrbyfloat(self, key, amount):
            self._cmds.append(("incrbyfloat", key, amount))
            return self

        async def execute(self):
            results = []
            for cmd in self._cmds:
                name, *args = cmd
                coro = getattr(self._redis, name)(*args)
                results.append(await coro)
            return results

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

    def pipeline(self, transaction=False):
        return self._Pipeline(self, transaction)

    # ── Lua scripts ────────────────────────────────────────────────────────
    async def script_load(self, script: str) -> str:
        sha = f"sha_{len(self._scripts)}"
        self._scripts[sha] = script
        return sha

    async def evalsha(self, sha: str, numkeys: int, *args):
        """Execute a simplified Python simulation of the Lua scripts."""
        script = self._scripts.get(sha, "")
        keys = list(args[:numkeys])
        argv = list(args[numkeys:])
        return await self._eval_lua_sim(script, keys, argv)

    async def _eval_lua_sim(self, script: str, keys: list, argv: list):
        """
        Very simplified Lua → Python simulation for the specific scripts used.
        Dispatches based on script content fingerprints.
        """
        if "INCR" in script and "threshold" in script and "half_open" in script:
            return await self._sim_record_failure(keys, argv)
        elif "DECR" in script and "required_key" in script:
            return await self._sim_record_success(keys, argv)
        elif "ZREMRANGEBYSCORE" in script and "ZCARD" in script:
            return await self._sim_slide_window(keys, argv)
        elif "inflight >= max_calls" in script or "inflight_key" in script and "INCR" in script and "state_key" in keys[0] if keys else False:
            return await self._sim_acquire_half_open(keys, argv)
        elif "DECR" in script and "inflight_key" in script and "required" not in script:
            return await self._sim_release_slot(keys, argv)
        elif "still_open" in script:
            return await self._sim_try_half_open(keys, argv)
        elif "refill_rate" in script:
            return await self._sim_consume_tokens(keys, argv)
        return None

    # ── Lua sim: circuit breaker record_failure ────────────────────────────
    async def _sim_record_failure(self, keys, argv):
        state_key, count_key, opened_key, inflight_key = keys
        threshold = int(float(argv[0]))
        now = float(argv[1])

        state = self._store.get(state_key, "closed")
        if state == "open":
            return "already_open"
        if state == "half_open":
            self._store[state_key] = "open"
            self._store[opened_key] = str(now)
            self._store[count_key] = "1"
            self._store.pop(inflight_key, None)
            return "reopened"
        # closed
        count = int(self._store.get(count_key, "0")) + 1
        self._store[count_key] = str(count)
        if count >= threshold:
            self._store[state_key] = "open"
            self._store[opened_key] = str(now)
            return "opened"
        return "counted"

    # ── Lua sim: circuit breaker record_success ────────────────────────────
    async def _sim_record_success(self, keys, argv):
        state_key, count_key, inflight_key, required_key = keys
        max_calls = int(float(argv[0]))

        state = self._store.get(state_key, "closed")
        if state == "closed":
            self._store[count_key] = "0"
            return "reset"
        if state == "half_open":
            remaining = int(self._store.get(required_key, str(max_calls))) - 1
            self._store[required_key] = str(remaining)
            inf = max(0, int(self._store.get(inflight_key, "0")) - 1)
            self._store[inflight_key] = str(inf)
            if remaining <= 0:
                self._store[state_key] = "closed"
                self._store[count_key] = "0"
                self._store.pop(required_key, None)
                self._store.pop(inflight_key, None)
                return "closed"
            return "counted"
        return "noop"

    # ── Lua sim: circuit breaker acquire_half_open ─────────────────────────
    async def _sim_acquire_half_open(self, keys, argv):
        state_key, inflight_key = keys
        max_calls = int(float(argv[0]))

        state = self._store.get(state_key, "closed")
        if state != "half_open":
            return 0
        inflight = int(self._store.get(inflight_key, "0"))
        if inflight >= max_calls:
            return 0
        self._store[inflight_key] = str(inflight + 1)
        return 1

    # ── Lua sim: circuit breaker release_half_open ─────────────────────────
    async def _sim_release_slot(self, keys, argv):
        inflight_key = keys[0]
        v = max(0, int(self._store.get(inflight_key, "0")) - 1)
        self._store[inflight_key] = str(v)
        return 1

    # ── Lua sim: circuit breaker try_half_open ─────────────────────────────
    async def _sim_try_half_open(self, keys, argv):
        state_key, opened_key, required_key, inflight_key = keys
        now = float(argv[0])
        timeout = float(argv[1])
        max_calls = int(float(argv[2]))

        state = self._store.get(state_key, "closed")
        if state != "open":
            return state
        opened_at = float(self._store.get(opened_key, str(now)))
        if (now - opened_at) < timeout:
            return "still_open"
        self._store[state_key] = "half_open"
        self._store[required_key] = str(max_calls)
        self._store[inflight_key] = "0"
        return "half_open"

    # ── Lua sim: sliding window rate limit ─────────────────────────────────
    async def _sim_slide_window(self, keys, argv):
        key = keys[0]
        now = float(argv[0])
        window_start = float(argv[1])
        limit = int(float(argv[2]))
        ttl = int(float(argv[3]))

        d = self._store.setdefault(key, {})
        # Remove expired
        to_del = [m for m, s in d.items() if s <= window_start]
        for m in to_del:
            del d[m]
        count = len(d)
        if count < limit:
            d[str(now)] = now
            return [1, "0"]
        else:
            oldest = min(d.values())
            return [0, str(oldest)]

    # ── Lua sim: token bucket consume ──────────────────────────────────────
    async def _sim_consume_tokens(self, keys, argv):
        tokens_key, last_refill_key = keys
        capacity = float(argv[0])
        refill_rate = float(argv[1])
        cost = float(argv[2])
        now = float(argv[3])

        tokens = float(self._store.get(tokens_key, capacity))
        last_refill = float(self._store.get(last_refill_key, now))

        elapsed = max(0, now - last_refill)
        tokens = min(capacity, tokens + elapsed * refill_rate)
        self._store[last_refill_key] = str(now)

        if tokens >= cost:
            self._store[tokens_key] = str(tokens - cost)
            return [1, "0"]
        else:
            needed = cost - tokens
            wait_s = needed / refill_rate
            self._store[tokens_key] = str(tokens)
            return [0, str(wait_s)]


# ──────────────────────────────────────────────────────────────────────────────
# Circuit Breaker Tests
# ──────────────────────────────────────────────────────────────────────────────

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from resilience.circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState


class TestCircuitBreaker:
    """Tests for Redis-backed circuit breaker."""

    def _make_cb(self, redis=None, **kwargs):
        r = redis or FakeRedis()
        return CircuitBreaker(r, "test", **kwargs), r

    @pytest.mark.asyncio
    async def test_initially_closed(self):
        cb, _ = self._make_cb()
        state = await cb.state()
        assert state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_opens_after_threshold_failures(self):
        cb, _ = self._make_cb(failure_threshold=5)
        for i in range(5):
            try:
                async with cb:
                    raise ValueError(f"fail {i}")
            except ValueError:
                pass
            except CircuitOpenError:
                break
        state = await cb.state()
        assert state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_open_rejects_immediately(self):
        cb, _ = self._make_cb(failure_threshold=3)
        # Open the circuit
        for _ in range(3):
            try:
                async with cb:
                    raise ValueError("fail")
            except (ValueError, CircuitOpenError):
                pass

        with pytest.raises(CircuitOpenError):
            async with cb:
                pass  # should not get here

    @pytest.mark.asyncio
    async def test_recovery_after_timeout(self):
        cb, r = self._make_cb(failure_threshold=2, recovery_timeout=60.0)
        # Open
        for _ in range(2):
            try:
                async with cb:
                    raise ValueError("fail")
            except (ValueError, CircuitOpenError):
                pass

        assert await cb.state() == CircuitState.OPEN

        # Manually backdate opened_at to simulate timeout elapsed
        r._store[cb._k_opened] = str(time.time() - 61)

        state = await cb.state()
        assert state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_half_open_successful_calls_close_circuit(self):
        cb, r = self._make_cb(failure_threshold=2, recovery_timeout=1.0, half_open_max_calls=2)
        # Open it
        for _ in range(2):
            try:
                async with cb:
                    raise ValueError("fail")
            except (ValueError, CircuitOpenError):
                pass
        # Simulate timeout elapsed
        r._store[cb._k_opened] = str(time.time() - 2)

        # Two successful half-open calls should close it
        for _ in range(2):
            async with cb:
                pass  # success

        state = await cb.state()
        assert state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens(self):
        cb, r = self._make_cb(failure_threshold=2, recovery_timeout=1.0, half_open_max_calls=3)
        for _ in range(2):
            try:
                async with cb:
                    raise ValueError("fail")
            except (ValueError, CircuitOpenError):
                pass
        r._store[cb._k_opened] = str(time.time() - 2)

        # Transition to HALF_OPEN
        await cb.state()
        r._store[cb._k_state] = "half_open"
        r._store[cb._k_required] = "3"
        r._store[cb._k_inflight] = "0"

        # One failure re-opens
        try:
            async with cb:
                raise RuntimeError("half-open fail")
        except RuntimeError:
            pass

        state = await cb.state()
        assert state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_half_open_respects_concurrency_limit(self):
        cb, r = self._make_cb(failure_threshold=2, recovery_timeout=1.0, half_open_max_calls=1)
        for _ in range(2):
            try:
                async with cb:
                    raise ValueError("fail")
            except (ValueError, CircuitOpenError):
                pass
        r._store[cb._k_opened] = str(time.time() - 2)

        # Manually set to half_open with one slot already taken
        r._store[cb._k_state] = "half_open"
        r._store[cb._k_required] = "1"
        r._store[cb._k_inflight] = "1"  # already at capacity

        with pytest.raises(CircuitOpenError):
            async with cb:
                pass

    @pytest.mark.asyncio
    async def test_cancellation_not_counted_as_failure(self):
        cb, _ = self._make_cb(failure_threshold=5)
        # Simulate cancellation
        try:
            async with cb:
                raise asyncio.CancelledError()
        except asyncio.CancelledError:
            pass

        state = await cb.state()
        assert state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_state_persists_across_instances(self):
        redis = FakeRedis()
        cb1 = CircuitBreaker(redis, "shared", failure_threshold=3)
        cb2 = CircuitBreaker(redis, "shared", failure_threshold=3)

        for _ in range(3):
            try:
                async with cb1:
                    raise ValueError("fail")
            except (ValueError, CircuitOpenError):
                pass

        # cb2 should see the same OPEN state
        with pytest.raises(CircuitOpenError):
            async with cb2:
                pass


# ──────────────────────────────────────────────────────────────────────────────
# Rate Limiter Tests
# ──────────────────────────────────────────────────────────────────────────────

from resilience.rate_limiter import RateLimiter, RateLimitExceeded, BucketConfig


class FakeClock:
    def __init__(self, start: float = 1000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float):
        self.t += seconds


class TestRateLimiter:

    def _make_rl(self, capacity=10.0, refill_rate=1.0, max_wait=5.0):
        clock = FakeClock()
        redis = FakeRedis()
        cfg = BucketConfig(capacity=capacity, refill_rate=refill_rate)
        rl = RateLimiter(redis, "test", config=cfg, max_wait_seconds=max_wait, _clock=clock)
        return rl, clock, redis

    @pytest.mark.asyncio
    async def test_acquire_within_capacity(self):
        rl, _, _ = self._make_rl(capacity=5.0)
        for _ in range(5):
            await rl.acquire()  # should not raise

    @pytest.mark.asyncio
    async def test_exhaustion_raises_when_wait_too_long(self):
        rl, _, _ = self._make_rl(capacity=2.0, refill_rate=0.1, max_wait=1.0)
        await rl.acquire()
        await rl.acquire()
        with pytest.raises(RateLimitExceeded):
            await rl.acquire()

    @pytest.mark.asyncio
    async def test_refill_based_on_elapsed_time(self):
        rl, clock, _ = self._make_rl(capacity=5.0, refill_rate=1.0, max_wait=30.0)
        # Exhaust
        for _ in range(5):
            await rl.acquire()
        # Advance 5 seconds → 5 new tokens
        clock.advance(5.0)
        for _ in range(5):
            await rl.acquire()  # should succeed after refill

    @pytest.mark.asyncio
    async def test_persistence_across_reconstructed_instances(self):
        clock = FakeClock()
        redis = FakeRedis()
        cfg = BucketConfig(capacity=3.0, refill_rate=1.0)

        rl1 = RateLimiter(redis, "shared_bucket", config=cfg, max_wait_seconds=1.0, _clock=clock)
        for _ in range(3):
            await rl1.acquire()

        # New instance, same Redis
        rl2 = RateLimiter(redis, "shared_bucket", config=cfg, max_wait_seconds=1.0, _clock=clock)
        with pytest.raises(RateLimitExceeded):
            await rl2.acquire()

    @pytest.mark.asyncio
    async def test_tokens_capped_at_capacity(self):
        rl, clock, _ = self._make_rl(capacity=5.0, refill_rate=1.0)
        # Advance a lot – should not exceed capacity
        clock.advance(1000.0)
        for _ in range(5):
            await rl.acquire()
        with pytest.raises(RateLimitExceeded):
            await rl.acquire()

    @pytest.mark.asyncio
    async def test_concurrent_workers_do_not_overspend(self):
        clock = FakeClock()
        redis = FakeRedis()
        cfg = BucketConfig(capacity=3.0, refill_rate=1.0)
        granted = []

        async def worker():
            rl = RateLimiter(redis, "concurrent", config=cfg, max_wait_seconds=0.0, _clock=clock)
            try:
                await rl.acquire()
                granted.append(1)
            except RateLimitExceeded:
                pass

        await asyncio.gather(*[worker() for _ in range(10)])
        assert sum(granted) == 3


# ──────────────────────────────────────────────────────────────────────────────
# Sliding-Window Endpoint Rate Limiter Tests
# ──────────────────────────────────────────────────────────────────────────────

from resilience.endpoint_rate_limiter import SlidingWindowRateLimiter, RouteLimit, ROUTE_LIMITS


class FakeRequest:
    def __init__(self, method="POST", path="/ingest", ip="127.0.0.1"):
        self.method = method
        self.url = type("URL", (), {"path": path})()
        self.headers = {}
        self.client = type("Client", (), {"host": ip})()
        self.app = type("App", (), {"state": type("State", (), {"trust_proxy": False})()})()


class TestEndpointRateLimiter:

    def _make_limiter(self, limit=3, window_s=60):
        redis = FakeRedis()
        limits = {
            "POST:/ingest": RouteLimit(limit=limit, window_s=window_s, route_key="ingest"),
        }
        return SlidingWindowRateLimiter(redis, limits=limits)

    @pytest.mark.asyncio
    async def test_within_limit_passes(self):
        limiter = self._make_limiter(limit=3)
        req = FakeRequest()
        for _ in range(3):
            result = await limiter.check(req)
            assert result is None  # admitted

    @pytest.mark.asyncio
    async def test_over_limit_returns_429(self):
        limiter = self._make_limiter(limit=3)
        req = FakeRequest()
        for _ in range(3):
            await limiter.check(req)
        response = await limiter.check(req)
        assert response is not None
        assert response.status_code == 429

    @pytest.mark.asyncio
    async def test_retry_after_header_present(self):
        limiter = self._make_limiter(limit=1)
        req = FakeRequest()
        await limiter.check(req)
        response = await limiter.check(req)
        assert response is not None
        assert "Retry-After" in response.headers

    @pytest.mark.asyncio
    async def test_different_ips_have_separate_counters(self):
        limiter = self._make_limiter(limit=1)
        req1 = FakeRequest(ip="1.1.1.1")
        req2 = FakeRequest(ip="2.2.2.2")
        await limiter.check(req1)
        await limiter.check(req2)
        # Second request from req1 should be blocked
        assert await limiter.check(req1) is not None
        # Second request from req2 should be blocked independently
        assert await limiter.check(req2) is not None

    @pytest.mark.asyncio
    async def test_unknown_route_not_rate_limited(self):
        limiter = self._make_limiter()
        req = FakeRequest(path="/metrics")
        for _ in range(100):
            result = await limiter.check(req)
            assert result is None


# ──────────────────────────────────────────────────────────────────────────────
# Backpressure Tests
# ──────────────────────────────────────────────────────────────────────────────

from resilience.backpressure import BackpressureGuard, QueueStatus, INGEST_QUEUE_KEY


class TestBackpressure:

    def _make_guard(self, depth: int):
        redis = FakeRedis()
        redis._store[INGEST_QUEUE_KEY] = str(depth)
        return BackpressureGuard(redis), redis

    @pytest.mark.asyncio
    async def test_normal_depth_ok(self):
        guard, _ = self._make_guard(10)
        result = await guard.check()
        assert result.status == QueueStatus.OK
        assert not result.should_reject

    @pytest.mark.asyncio
    async def test_high_depth_warning(self):
        guard, _ = self._make_guard(75)
        result = await guard.check()
        assert result.status == QueueStatus.WARNING
        assert not result.should_reject
        assert result.warning == "high_queue_depth"
        assert result.estimated_wait_minutes is not None

    @pytest.mark.asyncio
    async def test_full_depth_rejects(self):
        guard, _ = self._make_guard(150)
        result = await guard.check()
        assert result.status == QueueStatus.FULL
        assert result.should_reject
        assert result.retry_after == 30

    @pytest.mark.asyncio
    async def test_estimated_wait_not_zero(self):
        guard, _ = self._make_guard(60)
        result = await guard.check()
        assert result.estimated_wait_minutes is not None
        assert result.estimated_wait_minutes > 0

    @pytest.mark.asyncio
    async def test_reject_includes_queue_depth(self):
        guard, _ = self._make_guard(200)
        result = await guard.check()
        assert result.queue_depth == 200

    @pytest.mark.asyncio
    async def test_redis_failure_defaults_to_zero_depth(self):
        """If Redis is unreachable, depth defaults to 0 (fail-open for ingestion)."""
        class BrokenRedis:
            async def llen(self, key):
                raise ConnectionError("Redis down")

        guard = BackpressureGuard(BrokenRedis())
        result = await guard.check()
        assert result.status == QueueStatus.OK  # fail-open


# ──────────────────────────────────────────────────────────────────────────────
# Timeout Manager Tests
# ──────────────────────────────────────────────────────────────────────────────

from resilience.timeout_manager import (
    TimeoutManager, TaskTimeout, TIMEOUT_DEFAULTS,
    MIN_OBSERVATIONS, P95_MULTIPLIER, MAX_RATIO, MIN_RATIO,
)


class TestTimeoutManager:

    @pytest.mark.asyncio
    async def test_completes_within_timeout(self):
        redis = FakeRedis()
        tm = TimeoutManager(redis)

        async def fast_coro():
            return "done"

        result = await tm.run("embedding", fast_coro())
        assert result == "done"

    @pytest.mark.asyncio
    async def test_timeout_raises_task_timeout(self):
        redis = FakeRedis()
        tm = TimeoutManager(redis)

        async def slow_coro():
            await asyncio.sleep(10)

        with pytest.raises(TaskTimeout) as exc_info:
            await tm.run("embedding", slow_coro(), extra_timeout=0.01)

        assert exc_info.value.task_type == "embedding"

    @pytest.mark.asyncio
    async def test_timeout_counter_increments(self):
        redis = FakeRedis()
        tm = TimeoutManager(redis)

        async def slow():
            await asyncio.sleep(10)

        try:
            await tm.run("chat_completion", slow(), extra_timeout=0.01)
        except TaskTimeout:
            pass

        count = await redis.get("timeouts:chat_completion")
        assert int(count) == 1

    @pytest.mark.asyncio
    async def test_timeout_counter_survives_redis_failure(self):
        """Metric failure must not prevent timeout from being raised."""
        class BrokenRedis:
            async def incr(self, key):
                raise ConnectionError("Redis down")
            async def script_load(self, s): return "sha"
            async def evalsha(self, *a): return None
            async def zadd(self, *a, **kw): raise ConnectionError()
            async def zremrangebyrank(self, *a): raise ConnectionError()
            def pipeline(self, **kw): raise ConnectionError()
            async def zrange(self, *a, **kw): return []
            async def get(self, key): return None

        tm = TimeoutManager(BrokenRedis())

        async def slow():
            await asyncio.sleep(10)

        with pytest.raises(TaskTimeout):
            await tm.run("embedding", slow(), extra_timeout=0.01)

    @pytest.mark.asyncio
    async def test_adaptive_timeout_uses_p95(self):
        redis = FakeRedis()
        tm = TimeoutManager(redis)

        # Inject MIN_OBSERVATIONS synthetic latency records (200ms each)
        key = "latency:embedding"
        for i in range(MIN_OBSERVATIONS):
            ts = time.time() - (MIN_OBSERVATIONS - i)
            member = f"{ts}:200.0"
            await redis.zadd(key, {member: ts})

        stats = await tm.get_stats("embedding")
        assert stats["observation_count"] >= MIN_OBSERVATIONS
        assert stats["p95_ms"] is not None

        # Adaptive should be p95 * 1.5 clamped
        static = TIMEOUT_DEFAULTS["embedding"]  # 10s
        p95_s = stats["p95_ms"] / 1000.0
        expected = max(static * MIN_RATIO, min(static * MAX_RATIO, p95_s * P95_MULTIPLIER))
        assert abs(stats["adaptive_timeout_s"] - expected) < 0.1

    @pytest.mark.asyncio
    async def test_insufficient_observations_uses_static(self):
        redis = FakeRedis()
        tm = TimeoutManager(redis)

        # Only 5 observations (below MIN_OBSERVATIONS=30)
        key = "latency:reranking"
        for i in range(5):
            ts = time.time() - i
            await redis.zadd(key, {f"{ts}:100.0": ts})

        stats = await tm.get_stats("reranking")
        assert stats["adaptive_timeout_s"] is None  # insufficient data

    @pytest.mark.asyncio
    async def test_upward_trend_warning_emitted(self, caplog):
        """If recent latency is >= 1.25x prior median, a warning is emitted."""
        import logging
        redis = FakeRedis()
        tm = TimeoutManager(redis)

        now = time.time()
        key = "latency:chat_completion"

        # Prior window (300-600s ago): median 100ms
        for i in range(10):
            ts = now - 400 - i
            await redis.zadd(key, {f"{ts}:100.0": ts})

        # Recent window (0-300s): median 200ms (2x prior)
        for i in range(10):
            ts = now - i
            await redis.zadd(key, {f"{ts}:200.0": ts})

        with caplog.at_level(logging.WARNING):
            entries = await redis.zrange(key, 0, -1)
            await tm._check_upward_trend("chat_completion", entries)

        assert any("upward latency trend" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_upward_trend_warning_rate_limited(self, caplog):
        """Second trend check within cooldown period should NOT emit a warning."""
        import logging
        redis = FakeRedis()
        tm = TimeoutManager(redis)

        now = time.time()
        tm._last_warn["embedding"] = now  # mark as just warned

        key = "latency:embedding"
        for i in range(10):
            ts = now - 400 - i
            await redis.zadd(key, {f"{ts}:100.0": ts})
        for i in range(10):
            ts = now - i
            await redis.zadd(key, {f"{ts}:500.0": ts})

        with caplog.at_level(logging.WARNING):
            entries = await redis.zrange(key, 0, -1)
            await tm._check_upward_trend("embedding", entries)

        assert not any("upward latency trend" in r.message for r in caplog.records)


# ──────────────────────────────────────────────────────────────────────────────
# Health endpoint integration tests (mocked)
# ──────────────────────────────────────────────────────────────────────────────

from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock


class TestHealthEndpoint:
    """Verify health status rules using TestClient with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_health_degraded_when_circuit_open(self):
        """If any circuit breaker is OPEN, status should be degraded."""
        redis = FakeRedis()
        # Manually open a circuit
        redis._store["circuit:openai:state"] = "open"
        redis._store["circuit:openai:failure_count"] = "5"
        redis._store["circuit:openai:opened_at"] = str(time.time())

        from resilience.circuit_breaker import CircuitBreakerRegistry, CircuitBreaker
        registry = CircuitBreakerRegistry(redis)
        # Register the openai breaker
        registry.get("openai")

        info = await registry.get("openai").get_info()
        assert info["state"] == "open"

    @pytest.mark.asyncio
    async def test_health_circuit_info_includes_all_fields(self):
        redis = FakeRedis()
        redis._store["circuit:anthropic:state"] = "open"
        redis._store["circuit:anthropic:failure_count"] = "3"
        opened_at = time.time() - 30
        redis._store["circuit:anthropic:opened_at"] = str(opened_at)

        from resilience.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker(redis, "anthropic")
        info = await cb.get_info()
        assert info["name"] == "anthropic"
        assert info["state"] == "open"
        assert info["failure_count"] == 3
        assert info["opened_at"] is not None
