import json
import re
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import cast

from redis.asyncio import Redis

from .base import BaseLLMProvider


@dataclass
class RoutingCriteria:
    task_type: str = "general"
    max_cost_per_call: float | None = None
    require_vision: bool = False
    require_long_context: bool = False
    latency_budget_ms: float | None = None
    prefer_fine_tuned: bool = False


class ModelRouter:
    """Select a registered model that satisfies routing constraints."""

    def __init__(
        self,
        redis_client: Redis,
        providers: dict[str, BaseLLMProvider],
    ) -> None:
        self.redis = redis_client
        self.providers = providers

    async def _get_models(self) -> list[dict[str, object]]:
        """Read and validate model registrations from Redis."""
        raw_models = await cast(
            Awaitable[dict[str | bytes, str | bytes]],
            self.redis.hgetall("router:models"),
        )
        models: list[dict[str, object]] = []

        for raw_name, raw_config in raw_models.items():
            name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else raw_name
            config_text = (
                raw_config.decode("utf-8") if isinstance(raw_config, bytes) else raw_config
            )

            try:
                config = json.loads(config_text)
            except (json.JSONDecodeError, TypeError):
                fixed = re.sub(
                    r"([{,])\s*([a-zA-Z0-9_]+)\s*:",
                    r'\1"\2":',
                    config_text,
                )
                fixed = re.sub(
                    r":\s*([a-zA-Z_][a-zA-Z0-9_-]*)\s*([,}])",
                    lambda match: (
                        f':"{match.group(1)}"{match.group(2)}'
                        if match.group(1) not in ("true", "false", "null")
                        else match.group(0)
                    ),
                    fixed,
                )

                try:
                    config = json.loads(fixed)
                except (json.JSONDecodeError, TypeError):
                    config = {"name": name}

            if not isinstance(config, dict):
                config = {"name": name}

            model: dict[str, object] = {str(key): value for key, value in config.items()}
            model.setdefault("name", name)
            models.append(model)

        return models

    async def route(self, criteria: RoutingCriteria | None = None) -> str:
        criteria = criteria or RoutingCriteria()
        models = await self._get_models()

        if criteria.task_type == "evaluation":
            for model in models:
                name = model.get("name")
                if (
                    model.get("role") == "judge"
                    and not model.get("fine_tuned", False)
                    and isinstance(name, str)
                    and name in self.providers
                ):
                    return name

            raise ValueError("No eligible judge model is registered.")

        candidates: list[dict[str, object]] = []

        for model in models:
            name = model.get("name")

            if not isinstance(name, str) or name not in self.providers:
                continue

            if criteria.require_vision and not model.get("vision", False):
                continue

            context_window = model.get("context_window", 0)
            if criteria.require_long_context:
                if (
                    isinstance(context_window, bool)
                    or not isinstance(context_window, (int, float))
                    or context_window <= 100000
                ):
                    continue

            if criteria.max_cost_per_call is not None:
                estimated_cost = model.get("estimated_cost_per_call")
                if (
                    isinstance(estimated_cost, bool)
                    or not isinstance(estimated_cost, (int, float))
                    or estimated_cost > criteria.max_cost_per_call
                ):
                    continue

            if criteria.latency_budget_ms is not None:
                estimated_latency = model.get("estimated_latency_ms")
                if (
                    isinstance(estimated_latency, bool)
                    or not isinstance(estimated_latency, (int, float))
                    or estimated_latency > criteria.latency_budget_ms
                ):
                    continue

            candidates.append(model)

        if not candidates:
            raise ValueError("No registered model satisfies the routing criteria.")

        if criteria.prefer_fine_tuned:
            fine_tuned = [model for model in candidates if model.get("fine_tuned", False)]
            if fine_tuned:
                candidates = fine_tuned

        def cost_sort_key(model: dict[str, object]) -> float:
            value = model.get("estimated_cost_per_call")

            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)

            return float("inf")

        candidates.sort(key=cost_sort_key)

        selected_name = candidates[0].get("name")
        if not isinstance(selected_name, str):
            raise ValueError("Selected model has an invalid name.")

        return selected_name
