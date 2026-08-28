import asyncio
import json

from app.core.input_rewriter import UserInputRewriteAgent
from app.models import ConversationMessage


CPP_CODE = """class Solution {
public:
    int search(vector<int>& nums, int target) {
        for (int i = 0; i < nums.size(); i++) {
            if (nums[i] == target) return i;
        }
        return -1;
    }
};"""


def rewrite_json() -> str:
    return json.dumps({
        "schema_version": "1.0",
        "input_type": "mixed",
        "formatted_input": "请对所附 C++ 代码进行通用代码审查，解释代码逻辑并指出潜在问题。",
        "explicit_request": "查看并审查这段代码",
        "requested_operations": ["general_code_review"],
        "request_is_actionable": True,
        "instruction_verbatim": "帮我看看这段代码",
        "contextual_references": ["这段代码指当前输入中的 C++ 代码工件"],
        "constraints": [],
        "ambiguities": ["未明确要求仅查错、优化还是复杂度分析"],
        "contains_code": True,
        "programming_language": "C++",
        "rewrite_summary": "分离了粘连在代码结尾的自然语言要求，并改写为通用代码审查请求",
    }, ensure_ascii=False)


def test_model_rewriter_formats_mixed_code_input_before_intent_analysis() -> None:
    class FakeModelClient:
        async def complete_json(self, system_prompt, user_prompt, on_retry=None, max_tokens=None):
            assert "用户输入改写 Agent" in system_prompt
            assert "帮我看看这段代码" in user_prompt
            assert max_tokens == 1100
            return rewrite_json(), "test-deepseek"

    agent = UserInputRewriteAgent(FakeModelClient(), "deepseek-v4-pro")
    result = asyncio.run(agent.rewrite(
        CPP_CODE + "\n帮我看看这段代码",
        [ConversationMessage(role="user", content="这是一个查找函数")],
    ))

    assert result.formatted_input.startswith("请对所附 C++ 代码")
    assert result.explicit_request == "查看并审查这段代码"
    assert result.rewrite_provider == "test-deepseek"
    assert result.contains_code is True


def test_model_identified_trailing_instruction_is_removed_from_code_artifact() -> None:
    class FakeModelClient:
        async def complete_json(self, *args, **kwargs):
            return rewrite_json(), "test-deepseek"

    raw_input = CPP_CODE + "\n帮我看看这段代码"
    agent = UserInputRewriteAgent(FakeModelClient(), "deepseek-v4-pro")
    rewrite = asyncio.run(agent.rewrite(raw_input, []))
    artifact = agent.reconcile_code_artifact(raw_input, rewrite)

    assert artifact is not None
    assert artifact.code == CPP_CODE
    assert "帮我看看" not in artifact.code
    assert artifact.instruction == rewrite.formatted_input
    assert artifact.programming_language == "C++"
