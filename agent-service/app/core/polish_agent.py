import re

from app.core.model_client import IntentModelClient, RetryCallback
from app.core.reflection import complete_with_reflection
from app.models import AgentWorkResult, PolishResult, TaskSpec


SYSTEM_PROMPT = """你是 AlgoMate 的语言润色 Agent。你只能改善表达，不参与分析、检索或补充事实。

必须遵守：
- 保留原答案的结论、边界、不确定性、证据编号、代码和用户要求的协助级别。
- 不得新增事实、算法结论、复杂度、代码行为、链接或引用。
- 不得把不确定内容改写成确定结论。
- 不得删除 [R1] 这类证据编号。
- 语言自然、清晰、有层次，避免空洞套话和过度标题化。

只返回 JSON：
{
  "protocol_version": "1.0",
  "final_answer": "润色后的用户可见回答",
  "preserved_uncertainties": ["保留的不确定性"],
  "style_changes": ["做了哪些纯表达调整"],
  "added_factual_claims": false
}"""


class LanguagePolishAgent:
    def __init__(
        self,
        model_client: IntentModelClient,
        max_reflection_rounds: int = 10,
    ) -> None:
        self.model_client = model_client
        self.max_reflection_rounds = max_reflection_rounds

    async def polish(
        self,
        work_result: AgentWorkResult,
        task_spec: TaskSpec,
        on_retry: RetryCallback | None = None,
    ) -> tuple[PolishResult, str]:
        payload = {
            "draft_answer": work_result.draft_answer,
            "uncertainties": work_result.uncertainties,
            "response_language": task_spec.delivery.response_language,
            "explanation_depth": task_spec.delivery.explanation_depth,
            "assistance_level": task_spec.delivery.assistance_level,
            "constraints": task_spec.constraints,
        }
        result, provider, _ = await complete_with_reflection(
            model_client=self.model_client,
            agent_name="语言润色 Agent",
            system_prompt=SYSTEM_PROMPT,
            request_payload=payload,
            model_type=PolishResult,
            on_retry=on_retry,
            max_tokens=4200,
            max_reflection_rounds=self.max_reflection_rounds,
            validator=lambda value: self._validate(value, work_result),
        )
        return result, provider

    def _validate(self, result: PolishResult, work_result: AgentWorkResult) -> None:
        if result.added_factual_claims:
            raise ValueError("润色 Agent 声明添加了事实")
        if not result.final_answer.strip():
            raise ValueError("润色后的回答不能为空")
        before = set(re.findall(r"\[((?:R|W)\d+)]", work_result.draft_answer))
        after = set(re.findall(r"\[((?:R|W)\d+)]", result.final_answer))
        if not before.issubset(after):
            raise ValueError("润色 Agent 删除了证据编号")
