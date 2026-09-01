import base64
import hashlib
import ipaddress
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit, urlunsplit

from cryptography.fernet import Fernet, InvalidToken
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from redis.asyncio import Redis
from redis.exceptions import RedisError


_GLOBAL_REDIS_KEY = "algomate:model-config:global"
_MODEL_CONFIG_KEY_PATTERN = "algomate:model-config:*"


class ModelConfigStoreUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeModelConfig:
    api_key: str
    model: str
    base_url: str
    serpapi_api_key: str | None = None


@dataclass(frozen=True)
class StoredServiceConfig:
    api_key: str | None = None
    model: str | None = None
    base_url: str | None = None
    serpapi_api_key: str | None = None


class ModelConfigUpsertRequest(BaseModel):
    model_config = ConfigDict(alias_generator=lambda value: {
        "api_key": "apiKey",
        "serpapi_api_key": "serpapiApiKey",
        "base_url": "baseUrl",
        "ttl_seconds": "ttlSeconds",
        "update_model": "updateModel",
        "update_search": "updateSearch",
    }.get(value, value), populate_by_name=True)

    api_key: SecretStr | None = None
    serpapi_api_key: SecretStr | None = None
    model: str | None = Field(default=None, min_length=1, max_length=200)
    base_url: str | None = Field(default=None, min_length=8, max_length=500)
    ttl_seconds: int = Field(ge=300, le=31_536_000)
    update_model: bool = False
    update_search: bool = False

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None:
            return None
        if len(value.get_secret_value().strip()) < 8:
            raise ValueError("API Key 至少需要 8 个字符")
        return SecretStr(value.get_secret_value().strip())

    @field_validator("serpapi_api_key", mode="before")
    @classmethod
    def validate_serpapi_api_key(cls, value: object) -> object:
        if value is None:
            return None
        raw = (
            value.get_secret_value()
            if isinstance(value, SecretStr)
            else str(value)
        ).strip()
        if not raw:
            return None
        if len(raw) < 8:
            raise ValueError("SerpAPI Key 至少需要 8 个字符")
        return SecretStr(raw)

    @field_validator("model", "base_url")
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None

    @model_validator(mode="after")
    def validate_update_sections(self) -> "ModelConfigUpsertRequest":
        # Backward compatibility: old clients sent both fields without flags.
        if not self.update_model and not self.update_search:
            self.update_model = self.api_key is not None
            self.update_search = self.serpapi_api_key is not None
        if not self.update_model and not self.update_search:
            raise ValueError("必须选择更新模型配置或 SerpAPI 配置")
        if self.update_model and (
            self.api_key is None or not self.model or not self.base_url
        ):
            raise ValueError("更新模型配置需要 API Key、模型名称和 API URL")
        if self.update_search and self.serpapi_api_key is None:
            raise ValueError("更新搜索配置需要 SerpAPI Key")
        return self


class ModelConfigStatus(BaseModel):
    model_config = ConfigDict(alias_generator=lambda value: {
        "base_url": "baseUrl",
        "masked_api_key": "maskedApiKey",
        "search_configured": "searchConfigured",
        "masked_serpapi_api_key": "maskedSerpapiApiKey",
        "ttl_seconds": "ttlSeconds",
        "expires_at": "expiresAt",
    }.get(value, value), populate_by_name=True)

    configured: bool
    model: str | None = None
    base_url: str | None = None
    masked_api_key: str | None = None
    search_configured: bool = False
    masked_serpapi_api_key: str | None = None
    ttl_seconds: int | None = None
    expires_at: str | None = None


class ModelConfigStore:
    def __init__(
        self,
        redis_url: str,
        redis_password: str | None,
        encryption_secret: str | None,
        max_ttl_seconds: int,
        allowed_hosts: str = "",
        redis_client: Redis | None = None,
    ) -> None:
        self.max_ttl_seconds = max_ttl_seconds
        self.allowed_hosts = {
            host.strip().lower()
            for host in allowed_hosts.split(",")
            if host.strip()
        }
        self._fernet = self._build_fernet(encryption_secret)
        self.redis = redis_client or Redis.from_url(
            redis_url,
            password=redis_password or None,
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
        )

    async def save(
        self,
        request: ModelConfigUpsertRequest,
    ) -> ModelConfigStatus:
        self._ensure_ready()
        if request.ttl_seconds > self.max_ttl_seconds:
            raise ValueError(
                f"保存时间不能超过 {self.max_ttl_seconds} 秒"
            )
        ciphertext, _ttl = await self._read_global_or_migrate_legacy()
        stored = (
            self._decrypt_stored(ciphertext)
            if ciphertext
            else StoredServiceConfig()
        )
        if request.update_model:
            stored = StoredServiceConfig(
                api_key=request.api_key.get_secret_value(),
                model=request.model,
                base_url=self.normalize_base_url(request.base_url),
                serpapi_api_key=stored.serpapi_api_key,
            )
        if request.update_search:
            stored = StoredServiceConfig(
                api_key=stored.api_key,
                model=stored.model,
                base_url=stored.base_url,
                serpapi_api_key=request.serpapi_api_key.get_secret_value(),
            )
        ciphertext = self._encrypt(stored)
        try:
            await self.redis.set(
                _GLOBAL_REDIS_KEY,
                ciphertext,
                ex=request.ttl_seconds,
            )
        except RedisError as exc:
            raise ModelConfigStoreUnavailable("Redis 模型配置服务暂时不可用") from exc
        return self._status(stored, request.ttl_seconds)

    async def get(self) -> RuntimeModelConfig | None:
        self._ensure_ready()
        ciphertext, _ttl = await self._read_global_or_migrate_legacy()
        if not ciphertext:
            return None
        stored = self._decrypt_stored(ciphertext)
        if not self._has_model(stored):
            return None
        return RuntimeModelConfig(
            api_key=stored.api_key,
            model=stored.model,
            base_url=stored.base_url,
            serpapi_api_key=stored.serpapi_api_key,
        )

    async def status(self) -> ModelConfigStatus:
        self._ensure_ready()
        ciphertext, ttl = await self._read_global_or_migrate_legacy()
        if not ciphertext or ttl <= 0:
            return ModelConfigStatus(configured=False)
        return self._status(self._decrypt_stored(ciphertext), ttl)

    async def delete(self) -> None:
        self._ensure_ready()
        try:
            await self.redis.delete(_GLOBAL_REDIS_KEY)
        except RedisError as exc:
            raise ModelConfigStoreUnavailable("Redis 模型配置服务暂时不可用") from exc

    async def clear_model(self) -> None:
        await self._clear_section("model")

    async def clear_search(self) -> None:
        await self._clear_section("search")

    async def _clear_section(self, section: str) -> None:
        self._ensure_ready()
        ciphertext, ttl = await self._read_global_or_migrate_legacy()
        if not ciphertext or ttl <= 0:
            return
        stored = self._decrypt_stored(ciphertext)
        if section == "model":
            updated = StoredServiceConfig(
                serpapi_api_key=stored.serpapi_api_key,
            )
        else:
            updated = StoredServiceConfig(
                api_key=stored.api_key,
                model=stored.model,
                base_url=stored.base_url,
            )
        try:
            if not self._has_model(updated) and not updated.serpapi_api_key:
                await self.redis.delete(_GLOBAL_REDIS_KEY)
            else:
                await self.redis.set(
                    _GLOBAL_REDIS_KEY,
                    self._encrypt(updated),
                    ex=ttl,
                )
        except RedisError as exc:
            raise ModelConfigStoreUnavailable("Redis 模型配置服务暂时不可用") from exc

    async def _read_global_or_migrate_legacy(self) -> tuple[str | None, int]:
        """Read the single deployment-wide config and adopt one legacy token key.

        Older builds scoped configuration by an anonymous browser token. A
        no-login deployment has one shared configuration, so the only legacy
        entry can be copied to the stable global key without asking the user to
        re-enter secrets. Multiple legacy entries are intentionally left
        untouched because choosing one would be ambiguous.
        """
        try:
            ciphertext = await self.redis.get(_GLOBAL_REDIS_KEY)
            ttl = await self.redis.ttl(_GLOBAL_REDIS_KEY)
            if ciphertext and ttl > 0:
                return ciphertext, ttl

            legacy_keys: list[str] = []
            async for key in self.redis.scan_iter(
                match=_MODEL_CONFIG_KEY_PATTERN,
                count=20,
            ):
                if key == _GLOBAL_REDIS_KEY:
                    continue
                legacy_keys.append(key)
                if len(legacy_keys) > 1:
                    return None, -2
            if len(legacy_keys) != 1:
                return None, -2

            legacy_key = legacy_keys[0]
            legacy_ciphertext = await self.redis.get(legacy_key)
            legacy_ttl = await self.redis.ttl(legacy_key)
            if not legacy_ciphertext or legacy_ttl <= 0:
                return None, -2
            await self.redis.set(
                _GLOBAL_REDIS_KEY,
                legacy_ciphertext,
                ex=legacy_ttl,
            )
            return legacy_ciphertext, legacy_ttl
        except RedisError as exc:
            raise ModelConfigStoreUnavailable("Redis 模型配置服务暂时不可用") from exc

    async def close(self) -> None:
        await self.redis.aclose()

    def normalize_base_url(self, value: str) -> str:
        parsed = urlsplit(value.strip().rstrip("/"))
        if parsed.scheme.lower() != "https":
            raise ValueError("模型 API URL 必须使用 HTTPS")
        if not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("模型 API URL 格式无效")
        if parsed.query or parsed.fragment:
            raise ValueError("模型 API URL 不能包含查询参数或片段")
        hostname = parsed.hostname.lower().rstrip(".")
        if hostname == "localhost" or hostname.endswith(".localhost"):
            raise ValueError("模型 API URL 不能指向本机地址")
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            address = None
        if address is not None and not address.is_global:
            raise ValueError("模型 API URL 不能指向内网或保留地址")
        if self.allowed_hosts and hostname not in self.allowed_hosts:
            raise ValueError("该模型 API 域名不在服务器允许列表中")
        return urlunsplit(("https", parsed.netloc, parsed.path.rstrip("/"), "", ""))

    @staticmethod
    def _build_fernet(secret: str | None) -> Fernet | None:
        if (
            not secret
            or len(secret.strip()) < 24
            or secret.strip().startswith("replace_with_")
        ):
            return None
        digest = hashlib.sha256(secret.strip().encode("utf-8")).digest()
        return Fernet(base64.urlsafe_b64encode(digest))

    def _ensure_ready(self) -> None:
        if self._fernet is None:
            raise ModelConfigStoreUnavailable(
                "服务器尚未配置 MODEL_CONFIG_ENCRYPTION_KEY"
            )

    def _encrypt(self, stored: StoredServiceConfig) -> str:
        return self._fernet.encrypt(
            json.dumps(asdict(stored), ensure_ascii=False).encode("utf-8")
        ).decode("ascii")

    def _decrypt_stored(self, ciphertext: str) -> StoredServiceConfig:
        try:
            payload = json.loads(self._fernet.decrypt(ciphertext.encode("ascii")))
            return StoredServiceConfig(
                api_key=(str(payload["api_key"]) if payload.get("api_key") else None),
                model=(str(payload["model"]) if payload.get("model") else None),
                base_url=(str(payload["base_url"]) if payload.get("base_url") else None),
                serpapi_api_key=(
                    str(payload["serpapi_api_key"])
                    if payload.get("serpapi_api_key")
                    else None
                ),
            )
        except (InvalidToken, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ModelConfigStoreUnavailable("Redis 中的模型配置无法解密") from exc

    @staticmethod
    def _has_model(stored: StoredServiceConfig) -> bool:
        return bool(stored.api_key and stored.model and stored.base_url)

    @staticmethod
    def _mask(api_key: str) -> str:
        if len(api_key) <= 8:
            return "••••••••"
        return f"{api_key[:3]}••••••{api_key[-4:]}"

    def _status(
        self,
        stored: StoredServiceConfig,
        ttl_seconds: int,
    ) -> ModelConfigStatus:
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
        configured = self._has_model(stored)
        return ModelConfigStatus(
            configured=configured,
            model=stored.model if configured else None,
            base_url=stored.base_url if configured else None,
            masked_api_key=(self._mask(stored.api_key) if configured else None),
            search_configured=bool(stored.serpapi_api_key),
            masked_serpapi_api_key=(
                self._mask(stored.serpapi_api_key)
                if stored.serpapi_api_key
                else None
            ),
            ttl_seconds=ttl_seconds,
            expires_at=expires_at.isoformat(),
        )
