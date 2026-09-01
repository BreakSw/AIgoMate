import asyncio
import json

import httpx
import pytest

from app.config import Settings
from app.core.model_client import IntentModelClient, ModelConfigurationError
from app.core.model_config_store import RuntimeModelConfig


RUNTIME_CONFIG = RuntimeModelConfig(
    api_key="test-key",
    model="deepseek-v4-pro",
    base_url="https://model.example.test",
)


def test_env_api_key_is_not_used_without_redis_runtime_config() -> None:
    settings = make_settings(DEEPSEEK_API_KEY="must-not-be-used")
    assert not hasattr(settings, "deepseek_api_key")
    client = IntentModelClient(settings)

    with pytest.raises(ModelConfigurationError, match="模型设置"):
        asyncio.run(client.complete_json("system", "user"))


def test_runtime_model_and_search_config_are_isolated_between_concurrent_requests() -> None:
    client = IntentModelClient(make_settings())

    async def current_after_yield(config: RuntimeModelConfig) -> tuple[str, str | None]:
        with client.activate(config):
            await asyncio.sleep(0)
            return client.current_model, client.current_serpapi_api_key

    async def scenario() -> tuple[tuple[str, str | None], tuple[str, str | None]]:
        first, second = await asyncio.gather(
            current_after_yield(RuntimeModelConfig(
                "key-one", "model-one", "https://one.test", "serp-one"
            )),
            current_after_yield(RuntimeModelConfig(
                "key-two", "model-two", "https://two.test", "serp-two"
            )),
        )
        return first, second

    first, second = asyncio.run(scenario())

    assert (first, second) == (
        ("model-one", "serp-one"),
        ("model-two", "serp-two"),
    )


def make_settings(**updates) -> Settings:
    values = {
        "model": "deepseek-v4-pro",
        "llm-timeout": 1,
        "llm-max-disconnect-retries": 5,
        "llm-retry-base-delay-seconds": 0.5,
    }
    values.update(updates)
    return Settings(_env_file=None, **values)


def test_disconnects_are_retried_until_request_succeeds() -> None:
    calls = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise httpx.ConnectError("connection dropped", request=request)
        return httpx.Response(
            200,
            request=request,
            json={"choices": [{"message": {"content": '{"ok": true}'}}]},
        )

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    client = IntentModelClient(
        make_settings(),
        transport=httpx.MockTransport(handler),
        sleep=fake_sleep,
    )
    content = asyncio.run(client._call_openai_compatible(
        "system", "user", runtime_config=RUNTIME_CONFIG
    ))

    assert content == '{"ok": true}'
    assert calls == 3
    assert delays == [0.5, 1.0]


def test_disconnect_retry_stops_after_five_retries() -> None:
    calls = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.RemoteProtocolError("connection closed", request=request)

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    client = IntentModelClient(
        make_settings(),
        transport=httpx.MockTransport(handler),
        sleep=fake_sleep,
    )

    with pytest.raises(httpx.RemoteProtocolError):
        asyncio.run(client._call_openai_compatible(
            "system", "user", runtime_config=RUNTIME_CONFIG
        ))

    assert calls == 6
    assert delays == [0.5, 1.0, 2.0, 4.0, 8.0]


def test_http_errors_are_not_retried_as_disconnects() -> None:
    calls = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, request=request, json={"error": "unauthorized"})

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    client = IntentModelClient(
        make_settings(),
        transport=httpx.MockTransport(handler),
        sleep=fake_sleep,
    )

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(client._call_openai_compatible(
            "system", "user", runtime_config=RUNTIME_CONFIG
        ))

    assert calls == 1
    assert delays == []


def test_deepseek_runtime_disables_thinking_for_structured_output() -> None:
    captured_payload: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payload.update(json.loads(request.content))
        return httpx.Response(
            200,
            request=request,
            json={"choices": [{"message": {"content": '{"ok": true}'}}]},
        )

    client = IntentModelClient(
        make_settings(),
        transport=httpx.MockTransport(handler),
    )
    asyncio.run(client._call_openai_compatible(
        "system", "user", runtime_config=RUNTIME_CONFIG
    ))

    assert captured_payload["thinking"] == {"type": "disabled"}


def test_non_deepseek_runtime_does_not_receive_provider_specific_thinking() -> None:
    captured_payload: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payload.update(json.loads(request.content))
        return httpx.Response(
            200,
            request=request,
            json={"choices": [{"message": {"content": '{"ok": true}'}}]},
        )

    client = IntentModelClient(
        make_settings(),
        transport=httpx.MockTransport(handler),
    )
    asyncio.run(client._call_openai_compatible(
        "system",
        "user",
        runtime_config=RuntimeModelConfig(
            api_key="test-key",
            model="qwen-plus",
            base_url="https://model.example.test",
        ),
    ))

    assert "thinking" not in captured_payload
