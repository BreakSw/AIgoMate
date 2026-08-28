import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from urllib.parse import urlparse

import httpx

from app.config import Settings


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

    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        on_retry: RetryCallback | None = None,
        max_tokens: int = 1400,
    ) -> tuple[str, str]:
        if self.settings.deepseek_api_key is not None:
            return await self._call_openai_compatible(
                system_prompt, user_prompt, on_retry, max_tokens
            ), "deepseek"

        if self._can_use_local_anthropic_proxy():
            return await self._call_anthropic_compatible(
                system_prompt, user_prompt, on_retry, max_tokens
            ), "local-anthropic-proxy"

        raise ModelConfigurationError(
            "未找到 DeepSeek API 密钥。请在项目根目录 .env 中配置 DEEPSEEK_API_KEY。"
        )

    def _can_use_local_anthropic_proxy(self) -> bool:
        if not self.settings.anthropic_base_url or not self.settings.anthropic_auth_token:
            return False
        hostname = urlparse(self.settings.anthropic_base_url).hostname
        return hostname in {"127.0.0.1", "localhost", "::1"}

    async def _call_openai_compatible(
        self,
        system_prompt: str,
        user_prompt: str,
        on_retry: RetryCallback | None = None,
        max_tokens: int = 1400,
    ) -> str:
        endpoint = f"{self.settings.deepseek_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.deepseek_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        response = await self._post_with_disconnect_retry(endpoint, headers, payload, on_retry)
        response.raise_for_status()
        body = response.json()
        return body["choices"][0]["message"]["content"]

    async def _call_anthropic_compatible(
        self,
        system_prompt: str,
        user_prompt: str,
        on_retry: RetryCallback | None = None,
        max_tokens: int = 1800,
    ) -> str:
        endpoint = f"{self.settings.anthropic_base_url.rstrip('/')}/v1/messages"
        headers = {
            "x-api-key": self.settings.anthropic_auth_token.get_secret_value(),
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.anthropic_proxy_model or self.settings.model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        response = await self._post_with_disconnect_retry(endpoint, headers, payload, on_retry)
        response.raise_for_status()
        body = response.json()
        text_blocks = [block["text"] for block in body.get("content", []) if block.get("type") == "text"]
        if not text_blocks:
            raise ValueError(f"模型未返回文本内容：{json.dumps(body, ensure_ascii=False)[:300]}")
        return "\n".join(text_blocks)

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
