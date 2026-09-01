import asyncio
import logging
from collections.abc import Awaitable, Callable
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from urllib.parse import urlparse

import httpx

from app.config import Settings
from app.core.model_config_store import RuntimeModelConfig


logger = logging.getLogger(__name__)
RetryCallback = Callable[[int, int, float], Awaitable[None]]


class ModelConfigurationError(RuntimeError):
    pass


class IntentModelClient:
    def __init__(
        self,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.settings = settings
        self.transport = transport
        self.sleep = sleep
        self._runtime_config: ContextVar[RuntimeModelConfig | None] = ContextVar(
            "algomate_runtime_model_config",
            default=None,
        )

    @contextmanager
    def activate(self, runtime_config: RuntimeModelConfig) -> Iterator[None]:
        token = self._runtime_config.set(runtime_config)
        try:
            yield
        finally:
            self._runtime_config.reset(token)

    @property
    def current_model(self) -> str:
        runtime = self._runtime_config.get()
        return runtime.model if runtime is not None else self.settings.model

    @property
    def current_serpapi_api_key(self) -> str | None:
        runtime = self._runtime_config.get()
        return runtime.serpapi_api_key if runtime is not None else None

    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        on_retry: RetryCallback | None = None,
        max_tokens: int = 1400,
    ) -> tuple[str, str]:
        runtime = self._require_runtime_config()
        return await self._call_openai_compatible(
            system_prompt,
            user_prompt,
            on_retry,
            max_tokens,
            runtime,
        ), "redis-openai-compatible"

    def _require_runtime_config(self) -> RuntimeModelConfig:
        runtime = self._runtime_config.get()
        if runtime is None:
            raise ModelConfigurationError(
                "尚未配置可用的大模型。请先打开前端“模型设置”，保存 API URL、模型名称和 API Key。"
            )
        return runtime

    async def _call_openai_compatible(
        self,
        system_prompt: str,
        user_prompt: str,
        on_retry: RetryCallback | None = None,
        max_tokens: int = 1400,
        runtime_config: RuntimeModelConfig | None = None,
    ) -> str:
        runtime = runtime_config or self._require_runtime_config()
        base_url = runtime.base_url.rstrip("/")
        endpoint = (
            base_url
            if base_url.endswith("/chat/completions")
            else f"{base_url}/chat/completions"
        )
        headers = {
            "Authorization": f"Bearer {runtime.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": runtime.model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        # DeepSeek thinking mode can make strict JSON protocol output unstable.
        # The former .env-based client explicitly disabled it; retain that
        # behavior after moving runtime model configuration into Redis.
        if self._is_deepseek_runtime(runtime):
            payload["thinking"] = {"type": "disabled"}
        response = await self._post_with_disconnect_retry(endpoint, headers, payload, on_retry)
        response.raise_for_status()
        body = response.json()
        return body["choices"][0]["message"]["content"]

    @staticmethod
    def _is_deepseek_runtime(runtime: RuntimeModelConfig) -> bool:
        model = runtime.model.casefold()
        hostname = (urlparse(runtime.base_url).hostname or "").casefold()
        return "deepseek" in model or "deepseek" in hostname

    async def _post_with_disconnect_retry(
        self,
        endpoint: str,
        headers: dict[str, str],
        payload: dict,
        on_retry: RetryCallback | None = None,
    ) -> httpx.Response:
        max_retries = self.settings.llm_max_disconnect_retries
        for retry_number in range(max_retries + 1):
            try:
                async with httpx.AsyncClient(
                    timeout=self.settings.llm_timeout_seconds,
                    transport=self.transport,
                ) as client:
                    return await client.post(endpoint, headers=headers, json=payload)
            except httpx.TransportError:
                if retry_number >= max_retries:
                    raise
                delay = min(
                    self.settings.llm_retry_base_delay_seconds * (2**retry_number),
                    8.0,
                )
                logger.warning(
                    "LLM connection interrupted; retrying %s/%s in %.2f seconds",
                    retry_number + 1,
                    max_retries,
                    delay,
                )
                if on_retry is not None:
                    await on_retry(retry_number + 1, max_retries, delay)
                await self.sleep(delay)

        raise RuntimeError("模型请求重试循环意外结束")
