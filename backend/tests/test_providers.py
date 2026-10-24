
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.providers.base import ChatMessage
from backend.providers.openai_provider import OpenAIProvider
from backend.providers.anthropic_provider import AnthropicProvider


@pytest.mark.asyncio
async def test_openai_complete_parses_response():
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(
                    return_value=SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(content="Hello!"),
                                finish_reason="stop",
                            )
                        ],
                        usage=SimpleNamespace(
                            prompt_tokens=10,
                            completion_tokens=5,
                        ),
                        model="gpt-4o-mini",
                    )
                )
            )
        )
    )

    provider = OpenAIProvider(client=client)
    result = await provider.complete(
        [ChatMessage(role="user", content="Hi")]
    )

    assert result.content == "Hello!"
    assert result.model == "gpt-4o-mini"
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.cost_usd == pytest.approx(0.0000045)
    assert result.finish_reason == "stop"


@pytest.mark.asyncio
async def test_openai_embed_returns_vectors():
    client = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=AsyncMock(
                return_value=SimpleNamespace(
                    data=[
                        SimpleNamespace(index=0, embedding=[0.1, 0.2]),
                        SimpleNamespace(index=1, embedding=[0.3, 0.4]),
                    ]
                )
            )
        )
    )

    provider = OpenAIProvider(client=client)
    result = await provider.embed(["hello", "world"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    client.embeddings.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_anthropic_complete_parses_response():
    client = SimpleNamespace(
        messages=SimpleNamespace(
            create=AsyncMock(
                return_value=SimpleNamespace(
                    content=[
                        SimpleNamespace(type="text", text="Hello "),
                        SimpleNamespace(type="text", text="there!"),
                    ],
                    usage=SimpleNamespace(
                        input_tokens=10,
                        output_tokens=5,
                    ),
                    model="claude-3-5-haiku-latest",
                    stop_reason="end_turn",
                )
            )
        )
    )

    provider = AnthropicProvider(client=client)
    result = await provider.complete(
        [
            ChatMessage(role="system", content="Be helpful"),
            ChatMessage(role="user", content="Hi"),
        ],
        max_tokens=20,
    )

    assert result.content == "Hello there!"
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.cost_usd == pytest.approx(0.000028)
    assert result.finish_reason == "end_turn"

    request = client.messages.create.call_args.kwargs
    assert request["system"] == "Be helpful"
    assert request["max_tokens"] == 20


@pytest.mark.asyncio
async def test_anthropic_embedding_is_explicitly_unsupported():
    provider = AnthropicProvider(client=object())

    with pytest.raises(NotImplementedError, match="does not provide"):
        await provider.embed(["hello"])


@pytest.mark.asyncio
async def test_openai_complete_forwards_multimodal_content():
    image_content = [
        {"type": "text", "text": "Describe this image."},
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/jpeg;base64,dGVzdA=="
            },
        },
    ]

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(
                    return_value=SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(content="A test image."),
                                finish_reason="stop",
                            )
                        ],
                        usage=SimpleNamespace(
                            prompt_tokens=10,
                            completion_tokens=5,
                        ),
                        model="gpt-4o",
                    )
                )
            )
        )
    )

    provider = OpenAIProvider(model="gpt-4o", client=client)
    result = await provider.complete(
        [ChatMessage(role="user", content=image_content)]
    )

    sent_messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert sent_messages[0]["content"] == image_content
    assert result.content == "A test image."


@pytest.mark.asyncio
async def test_neuroflow_client_embed_empty(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")

    from backend.providers.client import NeuroFlowClient

    fake_redis = AsyncMock()
    client = NeuroFlowClient(fake_redis)
    result = await client.embed([])
    assert result == []


@pytest.mark.asyncio
async def test_neuroflow_client_embed_calls_local_encoder(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")

    from backend.providers import client as client_module
    from backend.providers.client import NeuroFlowClient

    fake_vectors = [[0.1] * 384, [0.2] * 384]
    monkeypatch.setattr(
        client_module, "_encode_texts", lambda texts: fake_vectors
    )

    fake_redis = AsyncMock()
    client = NeuroFlowClient(fake_redis)
    result = await client.embed(["test sentence 1", "test sentence 2"])

    assert len(result) == 2
    assert len(result[0]) == 384
    assert result == fake_vectors
