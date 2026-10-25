"""
NeuroFlow resilience layer.

Provides production-quality fault-tolerance primitives:
- CircuitBreaker   – prevents cascade failures on external providers
- RateLimiter      – Redis token-bucket coordination across workers
- BackpressureGuard – queue-depth-aware ingestion control
- TimeoutManager   – bounded async operations with adaptive p95 timeouts
"""

from .circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState
from .rate_limiter import RateLimiter, RateLimitExceeded
from .backpressure import BackpressureGuard, QueueStatus
from .timeout_manager import TimeoutManager, TaskTimeout, TIMEOUT_DEFAULTS

__all__ = [
    "CircuitBreaker",
    "CircuitOpenError",
    "CircuitState",
    "RateLimiter",
    "RateLimitExceeded",
    "BackpressureGuard",
    "QueueStatus",
    "TimeoutManager",
    "TaskTimeout",
    "TIMEOUT_DEFAULTS",
]
