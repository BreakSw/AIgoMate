import asyncio

import pytest
from pydantic import SecretStr

from app.core.model_config_store import ModelConfigStore, ModelConfigUpsertRequest


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.closed = False

    async def set(self, key: str, value: str, ex: int) -> None:
        self.values[key] = value
        self.ttls[key] = ex

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def ttl(self, key: str) -> int:
        return self.ttls.get(key, -2)

    async def delete(self, key: str) -> None:
        self.values.pop(key, None)
        self.ttls.pop(key, None)

    async def scan_iter(self, match: str, count: int = 10):
        prefix = match.removesuffix("*")
        for key in list(self.values):
            if key.startswith(prefix):
                yield key

    async def aclose(self) -> None:
        self.closed = True


def make_store(redis: FakeRedis) -> ModelConfigStore:
    return ModelConfigStore(
        "redis://unused",
        None,
        "a-long-random-test-encryption-secret",
        2_592_000,
        redis_client=redis,
    )


def test_global_config_is_encrypted_readable_and_deleted() -> None:
    async def scenario() -> None:
        redis = FakeRedis()
        store = make_store(redis)
        request = ModelConfigUpsertRequest(
            apiKey=SecretStr("example-user-secret-value"),
            serpapiApiKey=SecretStr("example-serpapi-secret-value"),
            model="deepseek-chat",
            baseUrl="https://api.deepseek.com/",
            ttlSeconds=3_600,
        )

        saved = await store.save(request)
        assert saved.configured is True
        assert saved.base_url == "https://api.deepseek.com"
        assert saved.masked_api_key == "exa••••••alue"
        assert saved.search_configured is True
        assert saved.masked_serpapi_api_key == "exa••••••alue"
        assert list(redis.values) == ["algomate:model-config:global"]
        assert all("example-user-secret-value" not in value for value in redis.values.values())
        assert all("example-serpapi-secret-value" not in value for value in redis.values.values())

        runtime = await store.get()
        assert runtime is not None
        assert runtime.api_key == "example-user-secret-value"
        assert runtime.model == "deepseek-chat"
        assert runtime.serpapi_api_key == "example-serpapi-secret-value"
        assert (await store.status()).ttl_seconds == 3_600

        await store.delete()
        assert await store.get() is None

    asyncio.run(scenario())


def test_config_without_serpapi_key_keeps_search_disabled() -> None:
    async def scenario() -> None:
        redis = FakeRedis()
        store = make_store(redis)
        request = ModelConfigUpsertRequest(
            apiKey=SecretStr("example-user-secret-value"),
            model="deepseek-chat",
            baseUrl="https://api.deepseek.com",
            ttlSeconds=3_600,
        )

        saved = await store.save(request)
        runtime = await store.get()

        assert saved.search_configured is False
        assert saved.masked_serpapi_api_key is None
        assert runtime is not None
        assert runtime.serpapi_api_key is None

    asyncio.run(scenario())


def test_model_and_search_keys_can_be_saved_and_removed_independently() -> None:
    async def scenario() -> None:
        redis = FakeRedis()
        store = make_store(redis)

        search_only = await store.save(ModelConfigUpsertRequest(
            serpapiApiKey=SecretStr("first-search-secret-value"),
            ttlSeconds=3_600,
            updateSearch=True,
        ))
        assert search_only.configured is False
        assert search_only.search_configured is True
        assert await store.get() is None

        both = await store.save(ModelConfigUpsertRequest(
            apiKey=SecretStr("later-model-secret-value"),
            model="deepseek-chat",
            baseUrl="https://api.deepseek.com",
            ttlSeconds=7_200,
            updateModel=True,
        ))
        runtime = await store.get()
        assert both.configured is True
        assert both.search_configured is True
        assert runtime is not None
        assert runtime.api_key == "later-model-secret-value"
        assert runtime.serpapi_api_key == "first-search-secret-value"

        updated_search = await store.save(ModelConfigUpsertRequest(
            serpapiApiKey=SecretStr("replacement-search-secret"),
            ttlSeconds=1_800,
            updateSearch=True,
        ))
        runtime = await store.get()
        assert updated_search.model == "deepseek-chat"
        assert runtime is not None
        assert runtime.api_key == "later-model-secret-value"
        assert runtime.serpapi_api_key == "replacement-search-secret"

        await store.clear_search()
        runtime = await store.get()
        assert runtime is not None
        assert runtime.api_key == "later-model-secret-value"
        assert runtime.serpapi_api_key is None

        await store.save(ModelConfigUpsertRequest(
            serpapiApiKey=SecretStr("final-search-secret-value"),
            ttlSeconds=1_800,
            updateSearch=True,
        ))
        await store.clear_model()
        final_status = await store.status()
        assert final_status.configured is False
        assert final_status.search_configured is True
        assert await store.get() is None

    asyncio.run(scenario())


def test_single_legacy_browser_config_is_migrated_to_global_key() -> None:
    async def scenario() -> None:
        redis = FakeRedis()
        store = make_store(redis)
        request = ModelConfigUpsertRequest(
            apiKey=SecretStr("example-user-secret-value"),
            model="deepseek-chat",
            baseUrl="https://api.deepseek.com",
            ttlSeconds=3_600,
        )
        await store.save(request)
        ciphertext = redis.values.pop("algomate:model-config:global")
        redis.ttls.pop("algomate:model-config:global")
        legacy_key = "algomate:model-config:legacy-token-hash"
        redis.values[legacy_key] = ciphertext
        redis.ttls[legacy_key] = 1_800

        status = await store.status()

        assert status.configured is True
        assert status.ttl_seconds == 1_800
        assert redis.values["algomate:model-config:global"] == ciphertext
        assert redis.ttls["algomate:model-config:global"] == 1_800

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "base_url",
    [
        "http://api.example.com/v1",
        "https://127.0.0.1/v1",
        "https://localhost/v1",
        "https://user@example.com/v1",
    ],
)
def test_unsafe_model_urls_are_rejected(base_url: str) -> None:
    store = make_store(FakeRedis())
    with pytest.raises(ValueError):
        store.normalize_base_url(base_url)
