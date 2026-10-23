
import json

import pytest

from backend.providers.router import ModelRouter, RoutingCriteria


class FakeRedis:
    def __init__(self, models):
        self.models = models

    async def hgetall(self, key):
        assert key == "router:models"
        return {
            name: json.dumps(config)
            for name, config in self.models.items()
        }


@pytest.fixture
def models():
    return {
        "gpt-4o": {
            "name": "gpt-4o",
            "role": "judge",
            "vision": True,
            "context_window": 128000,
            "estimated_cost_per_call": 0.01,
            "fine_tuned": False,
        },
        "gpt-4o-mini": {
            "name": "gpt-4o-mini",
            "role": "general",
            "vision": True,
            "context_window": 128000,
            "estimated_cost_per_call": 0.001,
            "fine_tuned": False,
        },
        "text-model": {
            "name": "text-model",
            "role": "general",
            "vision": False,
            "context_window": 32000,
            "estimated_cost_per_call": 0.0005,
            "fine_tuned": False,
        },
        "fine-tuned-model": {
            "name": "fine-tuned-model",
            "role": "general",
            "vision": False,
            "context_window": 32000,
            "estimated_cost_per_call": 0.002,
            "fine_tuned": True,
        },
    }


@pytest.fixture
def router(models):
    providers = {
        name: object()
        for name in models
    }
    return ModelRouter(FakeRedis(models), providers)


@pytest.mark.asyncio
async def test_default_route_chooses_cheapest(router):
    result = await router.route()
    assert result == "text-model"


@pytest.mark.asyncio
async def test_evaluation_uses_judge_model(router):
    result = await router.route(
        RoutingCriteria(task_type="evaluation")
    )
    assert result == "gpt-4o"


@pytest.mark.asyncio
async def test_vision_requirement(router):
    result = await router.route(
        RoutingCriteria(require_vision=True)
    )
    assert result == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_long_context_requirement(router):
    result = await router.route(
        RoutingCriteria(require_long_context=True)
    )
    assert result == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_max_cost_filters_models(router):
    result = await router.route(
        RoutingCriteria(max_cost_per_call=0.001)
    )
    assert result == "text-model"


@pytest.mark.asyncio
async def test_fine_tuned_preference(router):
    result = await router.route(
        RoutingCriteria(prefer_fine_tuned=True)
    )
    assert result == "fine-tuned-model"


@pytest.mark.asyncio

@pytest.mark.asyncio
async def test_no_eligible_model_raises_error(models):
    # The only available provider does not support vision.
    providers = {"text-model": object()}
    router = ModelRouter(FakeRedis(models), providers)

    with pytest.raises(ValueError, match="No registered model"):
        await router.route(
            RoutingCriteria(require_vision=True)
        )


@pytest.mark.asyncio
async def test_latency_budget_filters_models(models):
    models["gpt-4o"]["estimated_latency_ms"] = 500
    models["gpt-4o-mini"]["estimated_latency_ms"] = 100
    models["text-model"]["estimated_latency_ms"] = 50
    models["fine-tuned-model"]["estimated_latency_ms"] = 200

    providers = {
        name: object()
        for name in models
    }
    router = ModelRouter(FakeRedis(models), providers)

    result = await router.route(
        RoutingCriteria(latency_budget_ms=100)
    )

    assert result == "text-model"