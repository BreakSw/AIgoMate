import json

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from app.config import PROJECT_ROOT, settings
from app.core.adaptive_runtime import AdaptiveAgentRuntime
from app.core.context_manager import ContextManager
from app.core.context_compressor import ContextCompressionAgent
from app.core.coordinator_agent import CoordinatorAgent
from app.core.current_time_tool import CurrentTimeTool
from app.core.intent_recognizer import IntentRecognizer
from app.core.input_rewriter import UserInputRewriteAgent
from app.core.input_organizer import InputOrganizerAgent
from app.core.memory_agent import MemoryObserverAgent
from app.core.memory_store import DynamicSystemPromptBuilder, UserMemoryRepository
from app.core.model_client import IntentModelClient, ModelConfigurationError
from app.core.orchestrator import AgentOrchestrator
from app.core.output_format_agent import OutputFormattingAgent
from app.core.polish_agent import LanguagePolishAgent
from app.core.rag_retriever import MilvusRagRetriever
from app.core.rag_overview import RagOverviewService
from app.core.reflection import AgentProtocolExhaustedError
from app.core.response_agent import ResponseAgent
from app.core.progress_status import ProgressStatusStore
from app.core.web_search_agent import WebSearchAgent
from app.core.retry_status import RetryStatusStore
from app.models import AgentRequest, AgentResponse

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Agent orchestration boundary for AlgoMate.",
)
model_client = IntentModelClient(settings)
context_compressor = ContextCompressionAgent(
    model_client,
    settings.model,
    settings.agent_reflection_max_rounds,
)
memory_repository = UserMemoryRepository(PROJECT_ROOT, settings.user_memory_dir)
prompt_builder = DynamicSystemPromptBuilder()
current_time_tool = CurrentTimeTool(settings.app_timezone)
rag_retriever = MilvusRagRetriever(
    PROJECT_ROOT,
    excerpt_chars=settings.rag_excerpt_chars,
    total_context_chars=settings.rag_total_context_chars,
    embedding_base_url=settings.embedding_base_url,
    embedding_api_key=(
        settings.embedding_api_key.get_secret_value()
        if settings.embedding_api_key
        else None
    ),
    embedding_general_model=settings.embedding_general_model,
    embedding_code_model=settings.embedding_code_model,
    embedding_dimension=settings.embedding_dimension,
    milvus_uri=settings.milvus_uri,
    milvus_token=(settings.milvus_token.get_secret_value() if settings.milvus_token else None),
    collections={
        "algorithm_concepts": settings.milvus_concept_collection,
        "problem_bank": settings.milvus_problem_collection,
        "code_cases": settings.milvus_code_collection,
    },
)
coordinator = CoordinatorAgent(model_client, settings.agent_reflection_max_rounds)
response_agent = ResponseAgent(model_client, settings.agent_reflection_max_rounds)
web_search_agent = WebSearchAgent(
    model_client,
    settings,
    settings.agent_reflection_max_rounds,
    current_time_tool,
)
adaptive_runtime = AdaptiveAgentRuntime(
    coordinator,
    response_agent,
    rag_retriever,
    web_search_agent,
    current_time_tool,
    memory_repository,
    prompt_builder,
    settings.agent_max_decision_iterations,
)
orchestrator = AgentOrchestrator(
    ContextManager(settings, context_compressor),
    InputOrganizerAgent(
        model_client,
        settings.model,
        settings.agent_reflection_max_rounds,
    ),
    UserInputRewriteAgent(
        model_client,
        settings.model,
        settings.agent_reflection_max_rounds,
    ),
    IntentRecognizer(model_client, settings.agent_reflection_max_rounds),
    MemoryObserverAgent(model_client, settings.agent_reflection_max_rounds),
    memory_repository,
    adaptive_runtime,
    LanguagePolishAgent(model_client, settings.agent_reflection_max_rounds),
    OutputFormattingAgent(
        model_client,
        min(settings.agent_reflection_max_rounds, 2),
    ),
    settings.model,
)
retry_status_store = RetryStatusStore(settings.llm_max_disconnect_retries)
progress_status_store = ProgressStatusStore()
rag_overview_service = RagOverviewService(PROJECT_ROOT, memory_repository.root)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "algomate-agent"}


@app.post("/api/agent/respond", response_model=AgentResponse)
async def respond(request: AgentRequest) -> AgentResponse:
    return await run_intent_recognition(request)


@app.post("/api/agent/analyze-intent", response_model=AgentResponse)
async def analyze_intent(request: AgentRequest) -> AgentResponse:
    return await run_intent_recognition(request)


@app.get("/api/agent/sessions/{session_id}/retry-status")
async def retry_status(session_id: int) -> dict:
    return retry_status_store.get(session_id) or {
        "phase": "requesting",
        "retry_count": 0,
        "max_retries": settings.llm_max_disconnect_retries,
        "retry_delay_seconds": None,
        "updated_at": None,
    }


@app.get("/api/agent/sessions/{session_id}/progress-status")
async def progress_status(session_id: int) -> dict:
    return progress_status_store.get(session_id) or {
        "sequence": 0,
        "generation": 0,
        "phase": "idle",
        "agent": None,
        "message": "等待任务开始",
        "detail": None,
        "state": "completed",
        "updated_at": None,
    }


@app.get("/api/rag/overview")
async def rag_overview(user_id: int = 1) -> dict:
    return rag_overview_service.overview(user_id)


async def run_intent_recognition(request: AgentRequest) -> AgentResponse:
    retry_status_store.start(request.session_id)
    progress_generation = progress_status_store.start(request.session_id)

    async def on_retry(retry_count: int, max_retries: int, delay: float) -> None:
        retry_status_store.retry(request.session_id, retry_count, max_retries, delay)

    async def on_progress(
        phase: str,
        message: str,
        agent: str | None,
        detail: str | None,
    ) -> None:
        progress_status_store.update(
            request.session_id,
            phase,
            message,
            agent,
            detail,
            generation=progress_generation,
        )

    try:
        response = await orchestrator.respond(request, on_retry, on_progress)
        retry_status_store.complete(request.session_id)
        progress_status_store.complete(request.session_id, progress_generation)
        return response
    except ModelConfigurationError as exception:
        retry_status_store.fail(request.session_id)
        progress_status_store.fail(request.session_id, progress_generation)
        raise HTTPException(status_code=503, detail=str(exception)) from exception
    except httpx.HTTPStatusError as exception:
        retry_status_store.fail(request.session_id)
        progress_status_store.fail(request.session_id, progress_generation)
        status = exception.response.status_code
        raise HTTPException(
            status_code=502,
            detail=f"模型上游请求失败（HTTP {status}），请检查 DeepSeek 配置或稍后重试。",
        ) from exception
    except httpx.RequestError as exception:
        retry_status_store.fail(request.session_id)
        progress_status_store.fail(request.session_id, progress_generation)
        raise HTTPException(
            status_code=502,
            detail="无法连接模型上游，请检查网络、代理或 DeepSeek 服务状态。",
        ) from exception
    except AgentProtocolExhaustedError as exception:
        retry_status_store.fail(request.session_id)
        progress_status_store.fail(request.session_id, progress_generation)
        raise HTTPException(
            status_code=502,
            detail=(
                f"{exception.agent_name} 在 {exception.reflection_rounds} 轮 Reflection 后"
                "仍未生成合格结果，本轮已安全停止。"
            ),
        ) from exception
    except (json.JSONDecodeError, ValidationError, ValueError, KeyError, TypeError) as exception:
        retry_status_store.fail(request.session_id)
        progress_status_store.fail(request.session_id, progress_generation)
        raise HTTPException(
            status_code=502,
            detail=f"模型返回结果不符合 Agent 协议 v1：{str(exception)[:240]}",
        ) from exception
    except Exception:
        retry_status_store.fail(request.session_id)
        progress_status_store.fail(request.session_id, progress_generation)
        raise
