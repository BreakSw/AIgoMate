from fastapi.testclient import TestClient

from app.main import app, model_config_store, orchestrator
from app.core.intent_recognizer import IntentRecognizer
from app.core.model_config_store import ModelConfigStatus, RuntimeModelConfig
from app.models import (
    AgentWorkResult,
    CoordinatorPlan,
    DeliverySpec,
    InputOrganizationResult,
    InputRewriteResult,
    LearningProfileSnapshot,
    MemoryUpdateBatch,
    OutputFormatResult,
    PolishResult,
    RoutingPlan,
    TaskSpec,
)


client = TestClient(app)


def stub_downstream_agents(monkeypatch) -> None:
    async def fake_model_config():
        return RuntimeModelConfig(
            api_key="test-key",
            model="test-runtime-model",
            base_url="https://model.example.test",
        )

    async def fake_organize(message, on_retry=None):
        return InputOrganizationResult(
            organized_input=message,
            input_shape="text",
            organization_summary="原样保留",
            organizer_model="test-model",
            organizer_provider="test-organizer",
        )

    async def fake_observe(user_message, task_spec, snapshot, existing_memory, on_retry=None):
        return MemoryUpdateBatch(), "test-memory"

    async def fake_load(user_id, session_id):
        return []

    async def fake_upsert(user_id, session_id, updates, source="memory_agent"):
        return []

    async def fake_recall(user_id, session_id, query, limit=12):
        return []

    async def fake_reset_session(user_id, session_id):
        return None

    async def fake_learning_profile(user_id, session_id, message, task_spec):
        return LearningProfileSnapshot(
            active=False,
            user_id=user_id,
            session_id=session_id,
            summary="本轮未触发学习画像。",
        )

    async def fake_run(**kwargs):
        task_spec = kwargs["task_spec"]
        plan = CoordinatorPlan(
            objective=task_spec.user_goal,
            selected_agent="tutoring_agent",
            task_instruction=task_spec.normalized_request,
            planned_steps=["生成回答"],
        )
        result = AgentWorkResult(
            agent="tutoring_agent",
            draft_answer="这是未经润色的测试回答。",
        )
        return plan, [], result, [], [], ["head#1:test", "worker#2:test"]

    async def fake_polish(work_result, task_spec, on_retry=None):
        return PolishResult(
            final_answer="这是润色后的测试回答。",
            style_changes=["精简表达"],
        ), "test-polish"

    async def fake_format(polished, task_spec, rag_evidence, on_retry=None):
        return OutputFormatResult(
            formatted_answer="## 这是格式整理后的测试回答。",
            formatting_changes=["增加标题层级"],
        ), "test-format"

    monkeypatch.setattr(orchestrator.input_organizer, "organize", fake_organize)
    monkeypatch.setattr(orchestrator.memory_observer, "observe", fake_observe)
    monkeypatch.setattr(orchestrator.memory_repository, "load", fake_load)
    monkeypatch.setattr(orchestrator.memory_repository, "upsert", fake_upsert)
    monkeypatch.setattr(orchestrator.memory_repository, "recall", fake_recall)
    monkeypatch.setattr(
        orchestrator.memory_repository,
        "reset_session",
        fake_reset_session,
    )
    monkeypatch.setattr(orchestrator.adaptive_runtime, "run", fake_run)
    monkeypatch.setattr(
        orchestrator.learning_profile_service,
        "process_turn",
        fake_learning_profile,
    )
    monkeypatch.setattr(orchestrator.polish_agent, "polish", fake_polish)
    monkeypatch.setattr(orchestrator.output_format_agent, "format", fake_format)
    monkeypatch.setattr(model_config_store, "get", fake_model_config)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "ok"


def test_clear_session_memory_endpoint(monkeypatch) -> None:
    calls = []

    async def fake_reset_session(user_id, session_id):
        calls.append((user_id, session_id))

    monkeypatch.setattr(
        orchestrator.memory_repository,
        "reset_session",
        fake_reset_session,
    )

    response = client.delete("/api/agent/users/3/sessions/19/memory")

    assert response.status_code == 204
    assert calls == [(3, 19)]


def test_model_config_status_only_returns_masked_key(monkeypatch) -> None:
    async def fake_status():
        return ModelConfigStatus(
            configured=True,
            model="test-model",
            baseUrl="https://model.example.test",
            maskedApiKey="tes••••••-key",
            searchConfigured=True,
            maskedSerpapiApiKey="ser••••••-key",
            ttlSeconds=3_600,
            expiresAt="2026-08-31T12:00:00+00:00",
        )

    monkeypatch.setattr(model_config_store, "status", fake_status)
    response = client.get("/api/model-config")

    assert response.status_code == 200
    assert response.json()["maskedApiKey"] == "tes••••••-key"
    assert response.json()["maskedSerpapiApiKey"] == "ser••••••-key"
    assert "apiKey" not in response.json()
    assert "serpapiApiKey" not in response.json()


def test_agent_rejects_expired_model_config_without_env_fallback(monkeypatch) -> None:
    async def missing_config():
        return None

    monkeypatch.setattr(model_config_store, "get", missing_config)
    response = client.post(
        "/api/agent/respond",
        json={
            "sessionId": 98,
            "message": "解释二分查找",
            "history": [],
        },
    )

    assert response.status_code == 503
    assert "模型配置不存在或已经过期" in response.json()["detail"]


def test_agent_response(monkeypatch) -> None:
    stub_downstream_agents(monkeypatch)
    async def fake_rewrite(message, history, on_retry=None):
        return InputRewriteResult(
            input_type="text",
            formatted_input="分析二分查找复杂度",
            explicit_request="分析复杂度",
            requested_operations=["analyze_complexity"],
            request_is_actionable=True,
            contains_code=False,
            rewrite_summary="规范化口语输入",
            rewrite_model="test-model",
            rewrite_provider="test-provider",
        )

    async def fake_recognize(message, history, on_retry=None, code_artifact=None, input_rewrite=None):
        return TaskSpec(
            primary_intent="complexity_analysis",
            normalized_request="分析二分查找复杂度",
            user_goal="理解二分查找复杂度",
            recognition_summary="用户希望分析二分查找的时间复杂度",
            response_mode="step_by_step_explanation",
            delivery=DeliverySpec(assistance_level="explanation_only"),
            routing=RoutingPlan(primary_capability="algorithm_tutoring"),
            confidence=0.96,
        ), "test-provider"

    monkeypatch.setattr(orchestrator.input_rewriter, "rewrite", fake_rewrite)
    monkeypatch.setattr(orchestrator.intent_recognizer, "recognize", fake_recognize)
    response = client.post(
        "/api/agent/respond",
        json={
            "sessionId": 1,
            "message": "帮我分析二分查找复杂度",
            "history": [],
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["intent"] == "complexity_analysis"
    assert response.json()["task_spec"]["confidence"] == 0.96
    assert response.json()["content"] == "## 这是格式整理后的测试回答。"
    assert response.json()["context_snapshot"]["agent_execution"]["coordinator_plan"]["selected_agent"] == "tutoring_agent"
    assert response.json()["context_snapshot"]["agent_execution"]["model_call_trace"] == [
        "input-organizer:test-organizer",
        "input-rewrite:test-provider",
        "intent:test-provider",
        "memory:test-memory",
        "head#1:test",
        "worker#2:test",
        "polish:test-polish",
        "format:test-format",
    ]


def test_learning_profile_is_visible_in_response_and_context(monkeypatch) -> None:
    stub_downstream_agents(monkeypatch)

    async def fake_rewrite(message, history, on_retry=None):
        return InputRewriteResult(
            input_type="text",
            formatted_input=message,
            explicit_request="记录学习结果",
            requested_operations=["plan_learning"],
            request_is_actionable=True,
            contains_code=False,
            rewrite_summary="保留明确学习反馈",
            rewrite_model="test-model",
            rewrite_provider="test-provider",
        )

    async def fake_recognize(message, history, on_retry=None, code_artifact=None, input_rewrite=None):
        return TaskSpec(
            primary_intent="review_planning",
            normalized_request=message,
            user_goal="记录二分查找学习结果",
            recognition_summary="用户明确表示独立做对二分查找题",
            response_mode="direct_answer",
            delivery=DeliverySpec(assistance_level="explanation_only"),
            routing=RoutingPlan(primary_capability="review_planning"),
            confidence=0.98,
        ), "test-provider"

    async def fake_learning_profile(user_id, session_id, message, task_spec):
        return LearningProfileSnapshot(
            active=True,
            updated=True,
            user_id=user_id,
            session_id=session_id,
            ability_theta=0.2,
            target_difficulty="medium",
            summary="已更新 BKT、IRT 与 FSRS-style 学习状态。",
            recommended_concepts=["二分查找"],
        )

    monkeypatch.setattr(orchestrator.input_rewriter, "rewrite", fake_rewrite)
    monkeypatch.setattr(orchestrator.intent_recognizer, "recognize", fake_recognize)
    monkeypatch.setattr(
        orchestrator.learning_profile_service,
        "process_turn",
        fake_learning_profile,
    )

    response = client.post(
        "/api/agent/respond",
        json={
            "sessionId": 101,
            "message": "我独立做对了一道中等二分查找题。",
            "history": [],
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert "### 个性化学习画像" in body["content"]
    assert "IRT 能力值" in body["content"]
    assert body["context_snapshot"]["learning_profile"]["active"] is True
    assert "learning-profile:local-bkt-irt-fsrs" in body["context_snapshot"]["agent_execution"]["model_call_trace"]


def test_recognizer_builds_agent_ready_task_spec() -> None:
    class FakeModelClient:
        async def complete_json(self, system_prompt, user_prompt, on_retry=None, max_tokens=None):
            assert "不回答问题、不解题、不生成代码" in system_prompt
            assert "只给我分步提示" in user_prompt
            return """{
              "schema_version": "1.0",
              "primary_intent": "guided_hint",
              "secondary_intents": ["problem_solving"],
              "normalized_request": "为两数之和提供分步提示，后续代码使用 Python",
              "user_goal": "在不直接获得完整答案的情况下解出两数之和",
              "recognition_summary": "用户需要 Python 方向的渐进式解题提示",
              "entities": [
                {"type": "problem", "value": "两数之和", "role": "target"},
                {"type": "programming_language", "value": "Python", "role": "preferred"}
              ],
              "input_artifacts": {
                "problem_statement": "力扣两数之和",
                "code": null,
                "error_message": null,
                "test_cases": [],
                "programming_language": "Python"
              },
              "constraints": ["只提供分步提示", "不要直接给完整答案"],
              "response_mode": "progressive_hint",
              "delivery": {
                "assistance_level": "hint_only",
                "explanation_depth": "standard",
                "response_language": "zh-CN",
                "expected_outputs": ["下一步提示"],
                "include_code": false
              },
              "routing": {
                "primary_capability": "algorithm_tutoring",
                "supporting_capabilities": [],
                "execution_mode": "single",
                "recommended_sequence": ["algorithm_tutoring"],
                "tool_requirements": ["none"]
              },
              "context_plan": {
                "recent_messages": true,
                "task_state": true,
                "long_term_memory": false,
                "user_learning_profile": true,
                "algorithm_knowledge": true
              },
              "success_criteria": ["不泄露完整解法", "给出可执行的下一步思考方向"],
              "ambiguities": [],
              "risk_flags": [],
              "confidence": 94,
              "clarifying_question": null
            }""", "fake-deepseek"

    import asyncio

    recognizer = IntentRecognizer(FakeModelClient())
    task_spec, provider = asyncio.run(recognizer.recognize(
        "我在做力扣两数之和，请只给我分步提示，不要直接给完整答案，后续代码使用 Python",
        [],
    ))

    assert provider == "fake-deepseek"
    assert task_spec.primary_intent == "guided_hint"
    assert task_spec.delivery.assistance_level == "hint_only"
    assert task_spec.routing.primary_capability == "algorithm_tutoring"
    assert task_spec.input_artifacts.programming_language == "Python"
    assert task_spec.confidence == 0.94


def test_retry_progress_is_exposed_for_sse_gateway(monkeypatch) -> None:
    stub_downstream_agents(monkeypatch)
    async def fake_rewrite(message, history, on_retry=None):
        return InputRewriteResult(
            input_type="text",
            formatted_input="解释二分查找",
            explicit_request="解释概念",
            requested_operations=["explain_concept"],
            request_is_actionable=True,
            contains_code=False,
            rewrite_summary="规范化口语输入",
            rewrite_model="test-model",
            rewrite_provider="test-provider",
        )

    async def fake_recognize(message, history, on_retry=None, code_artifact=None, input_rewrite=None):
        assert on_retry is not None
        await on_retry(1, 5, 0.5)
        return TaskSpec(
            primary_intent="concept_explanation",
            normalized_request="解释二分查找",
            user_goal="理解二分查找",
            recognition_summary="用户希望理解二分查找",
            response_mode="direct_answer",
            delivery=DeliverySpec(assistance_level="explanation_only"),
            routing=RoutingPlan(primary_capability="algorithm_tutoring"),
            confidence=0.95,
        ), "test-provider"

    monkeypatch.setattr(orchestrator.input_rewriter, "rewrite", fake_rewrite)
    monkeypatch.setattr(orchestrator.intent_recognizer, "recognize", fake_recognize)
    response = client.post(
        "/api/agent/analyze-intent",
        json={
            "sessionId": 77,
            "message": "解释二分查找",
            "history": [],
        },
    )
    retry_status = client.get("/api/agent/sessions/77/retry-status")

    assert response.status_code == 200
    assert retry_status.status_code == 200
    assert retry_status.json()["phase"] == "completed"
    assert retry_status.json()["retry_count"] == 1
    assert retry_status.json()["max_retries"] == 5
