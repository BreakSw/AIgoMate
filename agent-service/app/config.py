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
    learning_profile_dir: str = "data/learning-profiles"
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
    rag_dense_candidate_k: int = Field(
        default=20,
        ge=5,
        le=100,
        validation_alias=AliasChoices("rag-dense-candidate-k", "RAG_DENSE_CANDIDATE_K"),
    )
    rag_bm25_candidate_k: int = Field(
        default=20,
        ge=5,
        le=100,
        validation_alias=AliasChoices("rag-bm25-candidate-k", "RAG_BM25_CANDIDATE_K"),
    )
    rag_fusion_candidate_k: int = Field(
        default=20,
        ge=5,
        le=100,
        validation_alias=AliasChoices("rag-fusion-candidate-k", "RAG_FUSION_CANDIDATE_K"),
    )
    rag_rrf_k: int = Field(
        default=60,
        ge=1,
        le=1_000,
        validation_alias=AliasChoices("rag-rrf-k", "RAG_RRF_K"),
    )
    rag_rerank_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("rag-rerank-enabled", "RAG_RERANK_ENABLED"),
    )
    voyage_rerank_model: str = Field(
        default="rerank-2.5",
        validation_alias=AliasChoices("voyage-rerank-model", "VOYAGE_RERANK_MODEL"),
    )
    rag_rerank_max_chars: int = Field(
        default=1_600,
        ge=200,
        le=8_000,
        validation_alias=AliasChoices("rag-rerank-max-chars", "RAG_RERANK_MAX_CHARS"),
    )
    agent_max_decision_iterations: int = Field(default=8, ge=2, le=12)
    agent_reflection_max_rounds: int = Field(default=10, ge=1, le=10)
    user_memory_dir: str = "agent-service/data/user-memory"
    durable_memory_recall_limit: int = Field(default=12, ge=3, le=30)
    web_search_enabled: bool = True
    web_search_max_results: int = Field(default=5, ge=1, le=10)
    redis_url: str = Field(
        default="redis://127.0.0.1:6379/0",
        validation_alias=AliasChoices("REDIS_URL", "redis-url"),
    )
    redis_password: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("REDIS_PASSWORD", "redis-password"),
    )
    model_config_encryption_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "MODEL_CONFIG_ENCRYPTION_KEY",
            "model-config-encryption-key",
        ),
    )
    model_config_max_ttl_seconds: int = Field(
        default=2_592_000,
        ge=300,
        le=31_536_000,
        validation_alias=AliasChoices(
            "MODEL_CONFIG_MAX_TTL_SECONDS",
            "model-config-max-ttl-seconds",
        ),
    )
    model_base_url_allowed_hosts: str = Field(
        default="",
        validation_alias=AliasChoices(
            "MODEL_BASE_URL_ALLOWED_HOSTS",
            "model-base-url-allowed-hosts",
        ),
    )
    langsmith_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("LANGSMITH_API_KEY", "X-API-Key"),
    )
    langsmith_project: str = Field(
        default="algomate-langgraph",
        validation_alias=AliasChoices("LANGSMITH_PROJECT", "langsmith-project"),
    )
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        validation_alias=AliasChoices("LANGSMITH_ENDPOINT", "langsmith-endpoint"),
    )
    # This is only a metadata fallback before a request binds its Redis model.
    # Chat completions never read an API key or base URL from Settings.
    model: str = Field(
        default="deepseek-v4-pro",
        validation_alias=AliasChoices("model", "AGENT_MODEL"),
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
    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", PROJECT_ROOT / "agent-service" / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


settings = Settings()
