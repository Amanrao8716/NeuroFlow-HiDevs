
import json
from dataclasses import dataclass

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

    def __init__(self, redis_client, providers: dict[str, BaseLLMProvider]):
        self.redis = redis_client
        self.providers = providers

    async def _get_models(self) -> list[dict]:
        """Read model registrations from Redis."""
        raw_models = await self.redis.hgetall("router:models")
        models = []

        for name, raw_config in raw_models.items():
            if isinstance(name, bytes):
                name = name.decode()

            if isinstance(raw_config, bytes):
                raw_config = raw_config.decode()

            try:
                config = json.loads(raw_config)
            except Exception:
                # Fallback in case raw_config had unquoted keys like {name:gpt-4o}
                import re
                fixed = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1"\2":', raw_config)
                # quote unquoted string values that aren't true/false/null/numbers
                fixed = re.sub(r':\s*([a-zA-Z_][a-zA-Z0-9_-]*)\s*([,}])', lambda m: f':"{m.group(1)}"{m.group(2)}' if m.group(1) not in ('true', 'false', 'null') else m.group(0), fixed)
                try:
                    config = json.loads(fixed)
                except Exception:
                    config = {"name": name}
            config.setdefault("name", name)
            models.append(config)

        return models

    async def route(self, criteria: RoutingCriteria | None = None) -> str:
        criteria = criteria or RoutingCriteria()
        models = await self._get_models()

        # Evaluation must use a registered judge with an available provider.
        if criteria.task_type == "evaluation":
            for model in models:
                name = model["name"]
                if (
                    model.get("role") == "judge"
                    and not model.get("fine_tuned", False)
                    and name in self.providers
                ):
                    return name

            raise ValueError("No eligible judge model is registered.")

        candidates = []

        for model in models:
            name = model["name"]

            if name not in self.providers:
                continue

            if criteria.require_vision and not model.get("vision", False):
                continue

            if (
                criteria.require_long_context
                and model.get("context_window", 0) <= 100000
            ):
                continue

            if criteria.max_cost_per_call is not None:
                estimated_cost = model.get("estimated_cost_per_call")
                if (
                    estimated_cost is None
                    or estimated_cost > criteria.max_cost_per_call
                ):
                    continue

            if criteria.latency_budget_ms is not None:
                estimated_latency = model.get("estimated_latency_ms")
                if (
                    estimated_latency is None
                    or estimated_latency > criteria.latency_budget_ms
                ):
                    continue

            candidates.append(model)

        if not candidates:
            raise ValueError(
                "No registered model satisfies the routing criteria."
            )

        if criteria.prefer_fine_tuned:
            fine_tuned = [
                model for model in candidates
                if model.get("fine_tuned", False)
            ]
            if fine_tuned:
                candidates = fine_tuned

        # Prefer the lowest estimated cost among eligible models.
        candidates.sort(
            key=lambda model: model.get(
                "estimated_cost_per_call", float("inf")
            )
        )

        return candidates[0]["name"]
