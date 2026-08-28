from fastapi.testclient import TestClient

from app.main import app, orchestrator
from app.core.intent_recognizer import IntentRecognizer
from app.models import DeliverySpec, InputRewriteResult, RoutingPlan, TaskSpec


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "ok"


def test_agent_response(monkeypatch) -> None:
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
        json={"sessionId": 1, "message": "帮我分析二分查找复杂度", "history": []},
    )
    assert response.status_code == 200, response.text
    assert response.json()["intent"] == "complexity_analysis"
    assert response.json()["task_spec"]["confidence"] == 0.96


def test_recognizer_builds_agent_ready_task_spec() -> None:
    class FakeModelClient:
        async def complete_json(self, system_prompt, user_prompt, on_retry=None):
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
        json={"sessionId": 77, "message": "解释二分查找", "history": []},
    )
    retry_status = client.get("/api/agent/sessions/77/retry-status")

    assert response.status_code == 200
    assert retry_status.status_code == 200
    assert retry_status.json()["phase"] == "completed"
    assert retry_status.json()["retry_count"] == 1
    assert retry_status.json()["max_retries"] == 5
