"""
Structural Protocol for async Redis clients.

Mypy uses structural subtyping (protocols) to validate redis method calls
without coupling the resilience layer to a specific redis client implementation
(redis-py, fakeredis, etc.).  Every method here matches the redis.asyncio.Redis
signature used in this codebase.

The Protocol only declares the operations that the resilience layer actually
calls; it does not need to cover the full Redis API.
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AsyncRedisProtocol(Protocol):
    """Minimal async Redis interface required by the resilience layer."""

    # String / generic commands
    async def get(self, name: str) -> bytes | str | None: ...
    async def set(self, name: str, value: Any, **kwargs: Any) -> bool | None: ...  # noqa: ANN401
    async def incr(self, name: str) -> int: ...

    # List commands
    async def llen(self, name: str) -> int: ...
    async def blpop(
        self,
        keys: str | list[str],
        timeout: float = 0,  # noqa: ASYNC109
    ) -> tuple[bytes, bytes] | None: ...
    async def lpush(self, name: str, *values: Any) -> int: ...  # noqa: ANN401
    async def rpush(self, name: str, *values: Any) -> int: ...  # noqa: ANN401

    # Sorted-set commands
    async def zrange(
        self,
        name: str,
        start: int,
        end: int,
    ) -> list[str]: ...

    # Hash commands
    async def hget(self, name: str, key: str) -> bytes | str | None: ...
    async def hset(
        self,
        name: str,
        key: str | None = None,
        value: Any = None,  # noqa: ANN401
        mapping: dict[str, Any] | None = None,
    ) -> int: ...

    # Pub/Sub
    async def publish(self, channel: str, message: str) -> int: ...

    # Scripting
    async def script_load(self, script: str) -> str: ...
    async def evalsha(
        self,
        sha: str,
        numkeys: int,
        *keys_and_args: Any,  # noqa: ANN401
    ) -> Any: ...  # noqa: ANN401

    # Pipeline / transaction
    def pipeline(
        self,
        transaction: bool = True,
    ) -> AbstractAsyncContextManager[AsyncPipelineProtocol]: ...


class AsyncPipelineProtocol(Protocol):
    """Minimal interface for a Redis pipeline object."""

    def zadd(self, name: str, mapping: dict[str, float]) -> None: ...
    def zremrangebyrank(self, name: str, min: int, max: int) -> None: ...
    async def execute(self) -> list[Any]: ...
