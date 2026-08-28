import json

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from app.config import settings
from app.core.context_manager import ContextManager
from app.core.context_compressor import ContextCompressionAgent
from app.core.intent_recognizer import IntentRecognizer
from app.core.input_rewriter import UserInputRewriteAgent
from app.core.model_client import IntentModelClient, ModelConfigurationError
from app.core.orchestrator import AgentOrchestrator
from app.core.retry_status import RetryStatusStore
from app.models import AgentRequest, AgentResponse

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Agent orchestration boundary for AlgoMate.",
)
model_client = IntentModelClient(settings)
context_compressor = ContextCompressionAgent(model_client, settings.model)
orchestrator = AgentOrchestrator(
    ContextManager(settings, context_compressor),
    UserInputRewriteAgent(model_client, settings.model),
    IntentRecognizer(model_client),
    settings.model,
)
retry_status_store = RetryStatusStore(settings.llm_max_disconnect_retries)


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


async def run_intent_recognition(request: AgentRequest) -> AgentResponse:
    retry_status_store.start(request.session_id)

    async def on_retry(retry_count: int, max_retries: int, delay: float) -> None:
        retry_status_store.retry(request.session_id, retry_count, max_retries, delay)

    try:
        response = await orchestrator.respond(request, on_retry)
        retry_status_store.complete(request.session_id)
        return response
    except ModelConfigurationError as exception:
        retry_status_store.fail(request.session_id)
        raise HTTPException(status_code=503, detail=str(exception)) from exception
    except httpx.HTTPStatusError as exception:
        retry_status_store.fail(request.session_id)
        status = exception.response.status_code
        raise HTTPException(
            status_code=502,
            detail=f"模型上游请求失败（HTTP {status}），请检查 DeepSeek 配置或稍后重试。",
        ) from exception
    except httpx.RequestError as exception:
        retry_status_store.fail(request.session_id)
        raise HTTPException(
            status_code=502,
            detail="无法连接模型上游，请检查网络、代理或 DeepSeek 服务状态。",
        ) from exception
    except (json.JSONDecodeError, ValidationError, ValueError, KeyError, TypeError) as exception:
        retry_status_store.fail(request.session_id)
        raise HTTPException(
            status_code=502,
            detail=f"模型返回的意图结果不符合 TaskSpec v1：{str(exception)[:240]}",
        ) from exception
    except Exception:
        retry_status_store.fail(request.session_id)
        raise
