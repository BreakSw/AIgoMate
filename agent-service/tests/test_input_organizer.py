import asyncio
import json

from app.core.input_organizer import InputOrganizerAgent


def organization_json(organized_input: str) -> str:
    return json.dumps({
        "schema_version": "1.0",
        "organized_input": organized_input,
        "input_shape": "mixed",
        "organization_summary": "仅调整代码与说明之间的空行",
        "preserved_meaning": True,
        "performed_intent_analysis": False,
    }, ensure_ascii=False)


def test_input_organizer_runs_before_semantic_agents_without_intent_fields() -> None:
    raw_input = "int answer = 42;\n\n帮我看看"

    class InspectingModelClient:
        async def complete_json(self, system_prompt, user_prompt, on_retry=None, max_tokens=None):
            assert "所有其他模型 Agent 之前的第一层" in system_prompt
            assert "不负责理解用户想做什么" in system_prompt
            assert max_tokens == 8_000
            assert json.loads(user_prompt)["raw_user_input"] == raw_input
            return organization_json(raw_input), "test-deepseek"

    result = asyncio.run(InputOrganizerAgent(
        InspectingModelClient(),
        "deepseek-v4-pro",
    ).organize(raw_input))

    assert result.organized_input == raw_input
    assert result.organizer_provider == "test-deepseek"


def test_input_organizer_reflects_when_model_adds_intent_output() -> None:
    raw_input = "解释二分查找"

    class ReflectingModelClient:
        def __init__(self) -> None:
            self.calls = 0

        async def complete_json(self, system_prompt, user_prompt, on_retry=None, max_tokens=None):
            self.calls += 1
            if self.calls == 1:
                invalid = json.loads(organization_json(raw_input))
                invalid["intent"] = "concept_explanation"
                return json.dumps(invalid, ensure_ascii=False), "test-deepseek"
            assert "Reflection" in system_prompt
            assert "Extra inputs are not permitted" in user_prompt
            return organization_json(raw_input), "test-deepseek"

    client = ReflectingModelClient()
    result = asyncio.run(InputOrganizerAgent(
        client,
        "deepseek-v4-pro",
    ).organize(raw_input))

    assert client.calls == 2
    assert result.organizer_provider == "test-deepseek+reflection:1"
