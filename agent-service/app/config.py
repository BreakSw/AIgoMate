from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "AlgoMate Agent Service"
    app_env: str = "development"
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
    )


settings = Settings()
