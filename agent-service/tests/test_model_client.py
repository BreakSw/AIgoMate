import asyncio

import httpx
import pytest

from app.config import Settings
from app.core.model_client import IntentModelClient


def make_settings(**updates) -> Settings:
    values = {
        "url": "https://model.example.test",
        "model": "deepseek-v4-pro",
        "api-key": "test-key",
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
    content = asyncio.run(client._call_openai_compatible("system", "user"))

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
        asyncio.run(client._call_openai_compatible("system", "user"))

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
        asyncio.run(client._call_openai_compatible("system", "user"))

    assert calls == 1
    assert delays == []


def test_anthropic_compatible_calls_share_disconnect_retry_policy() -> None:
    calls = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadError("connection reset", request=request)
        return httpx.Response(
            200,
            request=request,
            json={"content": [{"type": "text", "text": '{"ok": true}'}]},
        )

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    settings = make_settings(
        **{
            "api-key": None,
            "anthropic_base_url": "http://127.0.0.1:15721",
            "anthropic_auth_token": "test-token",
            "anthropic_proxy_model": "proxy-model",
        }
    )
    client = IntentModelClient(
        settings,
        transport=httpx.MockTransport(handler),
        sleep=fake_sleep,
    )
    content = asyncio.run(client._call_anthropic_compatible("system", "user"))

    assert content == '{"ok": true}'
    assert calls == 2
    assert delays == [0.5]
