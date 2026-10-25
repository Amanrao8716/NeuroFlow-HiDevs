"""
Ingestion backpressure guard.

Reads the ARQ ingestion queue depth from Redis using LLEN queue:ingest,
then applies three-tier logic:

    depth > HIGH_THRESHOLD (100) → reject with HTTP 503
    depth > WARN_THRESHOLD (50)  → enqueue but return HTTP 202 with warning
    depth ≤ WARN_THRESHOLD       → normal success path

Estimated wait calculation
──────────────────────────
We do not have a reliable worker-count metric exposed from arq, so we fall
back to a conservative estimate:

    estimated_wait_minutes = ceil(queue_depth / ASSUMED_WORKERS_TPM)

where ASSUMED_WORKERS_TPM is a configured constant (default: 2 items/minute).
Division by zero is prevented because the denominator is capped at ≥1.
This is clearly documented and used only for the advisory warning field;
it does not affect correctness of the 503 gate.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Queue key used by ARQ for ingestion jobs
INGEST_QUEUE_KEY = "queue:ingest"

# Tier thresholds
HIGH_THRESHOLD = 100    # reject above this
WARN_THRESHOLD = 50     # warn above this

# Fallback throughput estimate when worker count is unknown.
# Conservative: 2 documents processed per minute.
ASSUMED_THROUGHPUT_PER_MINUTE: int = 2


class QueueStatus(str, Enum):
    OK = "ok"
    WARNING = "warning"
    FULL = "full"


@dataclass
class BackpressureResult:
    status: QueueStatus
    queue_depth: int
    should_reject: bool           # True → return 503 before enqueueing
    warning: str | None = None
    estimated_wait_minutes: int | None = None
    retry_after: int | None = None


class BackpressureGuard:
    """
    Check ingestion queue depth before accepting a new document.

    The caller is responsible for:
      1. Calling check() BEFORE enqueueing.
      2. If result.should_reject is True, returning 503 without enqueueing.
      3. If result.status == WARNING, enqueueing normally and returning 202
         with the warning payload.
    """

    def __init__(
        self,
        redis,
        *,
        high_threshold: int = HIGH_THRESHOLD,
        warn_threshold: int = WARN_THRESHOLD,
        assumed_throughput_per_minute: int = ASSUMED_THROUGHPUT_PER_MINUTE,
        queue_key: str = INGEST_QUEUE_KEY,
    ) -> None:
        self._redis = redis
        self._high = high_threshold
        self._warn = warn_threshold
        self._throughput = max(1, assumed_throughput_per_minute)  # prevent ÷0
        self._queue_key = queue_key

    async def check(self) -> BackpressureResult:
        """Read queue depth from Redis and return the appropriate action."""
        try:
            depth = await self._redis.llen(self._queue_key)
            depth = int(depth or 0)
        except Exception:
            logger.warning(
                "backpressure: failed to read queue depth; defaulting to 0",
                exc_info=True,
            )
            depth = 0

        if depth > self._high:
            logger.warning(
                "backpressure: queue full depth=%d threshold=%d",
                depth,
                self._high,
            )
            return BackpressureResult(
                status=QueueStatus.FULL,
                queue_depth=depth,
                should_reject=True,
                retry_after=30,
            )

        if depth > self._warn:
            estimated_wait = math.ceil(depth / self._throughput)
            logger.info(
                "backpressure: high queue depth=%d estimated_wait_minutes=%d",
                depth,
                estimated_wait,
            )
            return BackpressureResult(
                status=QueueStatus.WARNING,
                queue_depth=depth,
                should_reject=False,
                warning="high_queue_depth",
                estimated_wait_minutes=estimated_wait,
            )

        return BackpressureResult(
            status=QueueStatus.OK,
            queue_depth=depth,
            should_reject=False,
        )
