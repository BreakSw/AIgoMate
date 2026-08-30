from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "AlgoMate Agent Service"
    app_env: str = "development"
    app_timezone: str = Field(
        default="Asia/Shanghai",
        validation_alias=AliasChoices("APP_TIMEZONE", "app-timezone"),
    )
    max_history_messages: int = 20
    context_window_tokens: int = Field(default=32_768, ge=8_192)
    context_soft_limit_tokens: int = Field(default=24_576, ge=4_096)
    context_hard_limit_tokens: int = Field(default=28_672, ge=8_192)
    context_output_reserve_tokens: int = Field(default=8_192, ge=1_024)
    context_recent_messages: int = Field(default=8, ge=2, le=30)
    context_recent_token_budget: int = Field(default=6_144, ge=1_024)
    # Kept as a compatibility setting. Context compaction is now budget-driven;
    # a true value no longer forces a model call on every request.
    context_compress_every_request: bool = False
    rag_excerpt_chars: int = Field(default=3_500, ge=500, le=12_000)
    rag_total_context_chars: int = Field(default=12_000, ge=2_000, le=40_000)
    embedding_base_url: str = Field(
        default="https://api.voyageai.com/v1",
        validation_alias=AliasChoices("embedding-base-url", "EMBEDDING_BASE_URL"),
    )
    embedding_general_model: str = Field(
        default="voyage-4",
        validation_alias=AliasChoices("embedding-general-model", "EMBEDDING_GENERAL_MODEL"),
    )
    embedding_code_model: str = Field(
        default="voyage-code-4",
        validation_alias=AliasChoices("embedding-code-model", "EMBEDDING_CODE_MODEL"),
    )
    embedding_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("embedding-api-key", "EMBEDDING_API_KEY"),
    )
    embedding_dimension: int = Field(
        default=1_024,
        ge=256,
        validation_alias=AliasChoices("embedding-dimension", "EMBEDDING_DIMENSION"),
    )
    milvus_uri: str = Field(
        default="rag-data/vector/algomate-milvus.db",
        validation_alias=AliasChoices("milvus-uri", "MILVUS_URI"),
    )
    milvus_token: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("milvus-token", "MILVUS_TOKEN"),
    )
    milvus_concept_collection: str = Field(
        default="algomate_algorithm_concepts_v1",
        validation_alias=AliasChoices(
            "milvus-concept-collection", "MILVUS_CONCEPT_COLLECTION"
        ),
    )
    milvus_problem_collection: str = Field(
        default="algomate_problem_bank_v1",
        validation_alias=AliasChoices(
            "milvus-problem-collection", "MILVUS_PROBLEM_COLLECTION"
        ),
    )
    milvus_code_collection: str = Field(
        default="algomate_code_cases_v1",
        validation_alias=AliasChoices("milvus-code-collection", "MILVUS_CODE_COLLECTION"),
    )
    agent_max_decision_iterations: int = Field(default=8, ge=2, le=12)
    agent_reflection_max_rounds: int = Field(default=10, ge=1, le=10)
    user_memory_dir: str = "agent-service/data/user-memory"
    durable_memory_recall_limit: int = Field(default=12, ge=3, le=30)
    web_search_enabled: bool = True
    web_search_max_results: int = Field(default=5, ge=1, le=10)
    serpapi_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("serpapi-key", "SERPAPI_API_KEY"),
    )
    deepseek_url: str = Field(
        default="https://api.deepseek.com",
        validation_alias=AliasChoices("url", "DEEPSEEK_BASE_URL"),
    )
    model: str = Field(
        default="deepseek-v4-pro",
        validation_alias=AliasChoices("model", "AGENT_MODEL"),
    )
    deepseek_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("DEEPSEEK_API_KEY", "api_key", "api-key"),
    )
    llm_timeout_seconds: float = Field(
        default=60,
        gt=0,
        validation_alias=AliasChoices("LLM_TIMEOUT", "llm_timeout", "llm-timeout"),
    )
    llm_max_disconnect_retries: int = Field(
        default=5,
        ge=0,
        le=10,
        validation_alias=AliasChoices(
            "LLM_MAX_DISCONNECT_RETRIES",
            "llm_max_disconnect_retries",
            "llm-max-disconnect-retries",
        ),
    )
    llm_retry_base_delay_seconds: float = Field(
        default=0.5,
        gt=0,
        validation_alias=AliasChoices(
            "LLM_RETRY_BASE_DELAY_SECONDS",
            "llm_retry_base_delay_seconds",
            "llm-retry-base-delay-seconds",
        ),
    )
    anthropic_base_url: str | None = None
    anthropic_auth_token: SecretStr | None = None
    anthropic_proxy_model: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_PROXY_MODEL"),
    )

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", PROJECT_ROOT / "agent-service" / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


settings = Settings()
