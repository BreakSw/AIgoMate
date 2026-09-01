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
- 用户要求简短时压缩重复表达但保留必要结论与限制；用户要求详细时只能展开原稿已有解释，不能补充新事实或新步骤。
- 保持术语、变量名、函数名、题号、日期、版本、复杂度和中英文专有名词一致，不为“通顺”而替换成不同概念。
- 代码块、行内代码、命令、路径、URL、数学公式、Markdown 表格单元格和引用文本不得改写；发现明显疑点也只能保留，
  不能在润色阶段偷偷修复。
- 原稿含多个方案、条件分支或正反结论时，保留它们之间的逻辑关系；不得把“可能/通常/在该题设下”删成绝对陈述。
- 原稿明确“未测试”“未核验”“国际版”“上下文推荐”或“工具不可用”时必须原样保留其含义和醒目程度。
- 不新增开场寒暄、道歉、身份说明、营销措辞或“希望对你有帮助”等套话；直接从用户关心的结果开始。
- 使用 task_spec 指定语言；原稿中的代码标识符、官方名称和必要英文术语可保留，不进行机械全量翻译。
- 证据编号必须留在其支持的句子附近，不得集中移动到无关段落；链接的 URL 与显示文本不得被替换。
- 如果原稿本身存在事实矛盾、缺少答案或格式损坏，只在 preserved_uncertainties 中保留问题，不得自行分析修补。

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
