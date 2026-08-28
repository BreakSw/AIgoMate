import asyncio
import json

from app.core.code_artifact import extract_code_artifact
from app.core.intent_recognizer import IntentRecognizer


CPP_CODE = """class Solution {
public:
    string lexGreaterPermutation(string s, string target) {
        int left[26]{};
        for (int i = 0; i < s.size(); i++) {
            left[s[i] - 'a']++;
            left[target[i] - 'a']--;
        }
        return s;
    }
};"""


def valid_code_diagnosis_json() -> str:
    return json.dumps({
        "schema_version": "1.0",
        "primary_intent": "code_diagnosis",
        "secondary_intents": [],
        "normalized_request": "检查所提供的 C++ 代码为什么结果不正确",
        "user_goal": "定位代码问题",
        "recognition_summary": "用户希望诊断 C++ 代码",
        "entities": [{"type": "programming_language", "value": "C++", "role": "代码语言"}],
        "input_artifacts": {
            "problem_statement": None,
            "code": None,
            "error_message": None,
            "test_cases": [],
            "programming_language": "C++",
        },
        "constraints": [],
        "response_mode": "code_review",
        "delivery": {
            "assistance_level": "review_only",
            "explanation_depth": "standard",
            "response_language": "zh-CN",
            "expected_outputs": ["问题定位"],
            "include_code": False,
        },
        "routing": {
            "primary_capability": "code_diagnosis",
            "supporting_capabilities": [],
            "execution_mode": "single",
            "recommended_sequence": ["code_diagnosis"],
            "tool_requirements": ["none"],
        },
        "context_plan": {
            "recent_messages": True,
            "task_state": True,
            "long_term_memory": False,
            "user_learning_profile": False,
            "algorithm_knowledge": False,
        },
        "success_criteria": ["定位代码问题"],
        "ambiguities": [],
        "risk_flags": [],
        "confidence": 0.95,
        "clarifying_question": None,
    }, ensure_ascii=False)


def test_plain_cpp_code_only_is_still_analyzed_by_the_model() -> None:
    class InspectingModel:
        def __init__(self) -> None:
            self.calls = 0

        async def complete_json(self, system_prompt, user_prompt, on_retry=None):
            self.calls += 1
            assert "left[target[i]" not in user_prompt
            assert "current_code_artifact" in user_prompt
            return valid_code_diagnosis_json(), "test-model"

    model = InspectingModel()
    recognizer = IntentRecognizer(model)
    task_spec, provider = asyncio.run(recognizer.recognize(CPP_CODE, []))

    assert provider == "test-model"
    assert model.calls == 1
    assert task_spec.primary_intent == "code_diagnosis"
    assert task_spec.input_artifacts.code == CPP_CODE
    assert task_spec.input_artifacts.programming_language == "C++"


def test_code_with_instruction_is_redacted_from_prompt_and_injected_after_validation() -> None:
    class InspectingModelClient:
        async def complete_json(self, system_prompt, user_prompt, on_retry=None):
            assert "left[target[i]" not in user_prompt
            assert "current_code_artifact" in user_prompt
            assert "为什么结果不正确" in user_prompt
            return valid_code_diagnosis_json(), "test-model"

    message = f"为什么结果不正确？\n```cpp\n{CPP_CODE}\n```"
    recognizer = IntentRecognizer(InspectingModelClient())
    task_spec, provider = asyncio.run(recognizer.recognize(message, []))

    assert provider == "test-model"
    assert task_spec.input_artifacts.code == CPP_CODE
    assert task_spec.input_artifacts.programming_language == "C++"


def test_malformed_model_json_is_regenerated_once() -> None:
    class RepairingModelClient:
        def __init__(self) -> None:
            self.calls = 0

        async def complete_json(self, system_prompt, user_prompt, on_retry=None):
            self.calls += 1
            if self.calls == 1:
                return '{"primary_intent":"code_diagnosis","input_artifacts":{"code":"unterminated', "test-model"
            assert "previous_validation_error" in user_prompt
            assert "上一次输出未通过" in system_prompt
            return valid_code_diagnosis_json(), "test-model"

    model_client = RepairingModelClient()
    recognizer = IntentRecognizer(model_client)
    task_spec, provider = asyncio.run(recognizer.recognize(
        f"请检查为什么结果不正确\n```cpp\n{CPP_CODE}\n```",
        [],
    ))

    assert model_client.calls == 2
    assert provider == "test-model+schema-repair"
    assert task_spec.primary_intent == "code_diagnosis"
    assert task_spec.input_artifacts.code == CPP_CODE


def test_plain_python_code_is_detected_as_code_only() -> None:
    artifact = extract_code_artifact("""def binary_search(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
    return -1""")

    assert artifact is not None
    assert artifact.is_code_only is True
    assert artifact.programming_language == "Python"
