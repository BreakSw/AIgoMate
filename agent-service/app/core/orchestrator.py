from app.core.context_manager import ContextManager
from app.core.intent_recognizer import IntentRecognizer
from app.core.input_rewriter import UserInputRewriteAgent
from app.core.model_client import RetryCallback
from app.models import AgentRequest, AgentResponse, ContextSnapshot, TaskSpec


class AgentOrchestrator:
    def __init__(
        self,
        context_manager: ContextManager,
        input_rewriter: UserInputRewriteAgent,
        intent_recognizer: IntentRecognizer,
        model: str,
    ) -> None:
        self.context_manager = context_manager
        self.input_rewriter = input_rewriter
        self.intent_recognizer = intent_recognizer
        self.model = model

    async def respond(
        self,
        request: AgentRequest,
        on_retry: RetryCallback | None = None,
    ) -> AgentResponse:
        context, context_snapshot = await self.context_manager.prepare(
            request.message,
            request.history,
            request.previous_context_snapshot,
            on_retry,
        )
        input_rewrite = await self.input_rewriter.rewrite(
            request.message,
            context,
            on_retry,
        )
        code_artifact = self.input_rewriter.reconcile_code_artifact(
            request.message,
            input_rewrite,
        )
        task_spec, provider = await self.intent_recognizer.recognize(
            input_rewrite.formatted_input,
            context,
            on_retry,
            code_artifact,
            input_rewrite,
        )
        context_snapshot = await self.context_manager.finalize_turn(
            request.message,
            request.history,
            context,
            context_snapshot,
            task_spec,
            provider,
            on_retry,
        )
        context_snapshot = context_snapshot.model_copy(update={"input_rewrite": input_rewrite})
        content = (
            self._format_input_rewrite(input_rewrite)
            + self._format_task_spec(task_spec, provider)
            + self._format_context_snapshot(context_snapshot)
        )
        return AgentResponse(
            content=content,
            intent=task_spec.primary_intent,
            context_messages_used=len(context),
            task_spec=task_spec,
            context_snapshot=context_snapshot,
            model=self.model,
            provider=provider,
        )

    def _format_input_rewrite(self, rewrite) -> str:
        constraints = "、".join(rewrite.constraints) or "无"
        ambiguities = "、".join(rewrite.ambiguities) or "无"
        return (
            "输入改写 Agent v1.0\n\n"
            f"输入类型：{rewrite.input_type}\n"
            f"规范输入：{rewrite.formatted_input}\n"
            f"明确请求：{rewrite.explicit_request}\n"
            f"保留约束：{constraints}\n"
            f"剩余歧义：{ambiguities}\n"
            f"改写模型：{rewrite.rewrite_model}（{rewrite.rewrite_provider}）\n\n"
        )

    def _format_context_snapshot(self, snapshot: ContextSnapshot) -> str:
        memory = snapshot.memory
        window = snapshot.window

        def lines(values: list[str]) -> str:
            return "\n".join(f"- {value}" for value in values) or "- 无"

        goal = memory.current_goal or "未形成稳定目标"
        compression_state = {
            "preflight_budget_exceeded": "试装超出预算，本轮已统一压缩历史",
            "turn_commit_budget_exceeded": "插入本轮意图后超出预算，本轮已统一压缩",
            "reused_checkpoint": "复用已有压缩检查点，本轮未重复压缩",
            "not_required": "试装成功，保留原始上下文",
        }[window.compression_trigger_reason]
        turn = snapshot.turn_context
        turn_summary = (
            f"{turn.primary_intent} · {turn.normalized_request}"
            if turn is not None
            else "尚未写入"
        )
        return (
            "\n\n========== 上下文审查 ==========\n\n"
            f"当前目标：{goal}\n"
            f"工作记忆：\n{lines(memory.working_memory)}\n"
            f"长期记忆：\n{lines(memory.long_term_memory)}\n"
            f"用户偏好：\n{lines(memory.user_preferences)}\n"
            f"固定约束：\n{lines(memory.pinned_constraints)}\n"
            f"待解决问题：\n{lines(memory.open_questions)}\n\n"
            f"本轮意图上下文：{turn_summary}\n\n"
            f"压缩状态：{compression_state}\n"
            f"压缩后内容：{snapshot.compressed_context.summary}\n"
            f"压缩来源：{snapshot.compressed_context.source_message_count} 条历史消息\n\n"
            f"试装占用：{window.candidate_input_tokens:,} / {window.safe_input_budget_tokens:,} tokens\n"
            f"上下文窗口：{window.estimated_input_tokens:,} / {window.window_size_tokens:,} tokens\n"
            f"剩余窗口：{window.remaining_tokens:,} tokens\n"
            f"安全预算剩余：{window.safe_remaining_tokens:,} tokens\n"
            f"使用比例：{window.usage_ratio:.1%}\n"
            f"窗口状态：{window.state}\n"
            f"原始历史估算：{window.raw_history_tokens:,} tokens\n"
            f"压缩上下文估算：{window.compressed_context_tokens:,} tokens"
        )

    def _format_task_spec(self, task_spec: TaskSpec, provider: str) -> str:
        entities = "\n".join(
            f"- {item.type}: {item.value}" + (f"（{item.role}）" if item.role else "")
            for item in task_spec.entities
        ) or "- 无明确实体"
        constraints = "\n".join(f"- {item}" for item in task_spec.constraints) or "- 无明确约束"
        criteria = "\n".join(f"- {item}" for item in task_spec.success_criteria) or "- 未指定"
        risks = "\n".join(f"- {item}" for item in task_spec.risk_flags) or "- 暂未发现"
        routing = " → ".join(task_spec.routing.recommended_sequence) or task_spec.routing.primary_capability
        clarification = task_spec.clarifying_question or "无需追问"
        return (
            f"TaskSpec v{task_spec.schema_version}\n\n"
            f"主要意图：{task_spec.primary_intent}\n"
            f"次要意图：{'、'.join(task_spec.secondary_intents) or '无'}\n"
            f"规范化请求：{task_spec.normalized_request}\n"
            f"用户目标：{task_spec.user_goal}\n"
            f"识别摘要：{task_spec.recognition_summary}\n"
            f"响应方式：{task_spec.response_mode}\n"
            f"协助级别：{task_spec.delivery.assistance_level}\n"
            f"置信度：{task_spec.confidence:.0%}\n\n"
            f"关键实体：\n{entities}\n\n"
            f"明确约束：\n{constraints}\n\n"
            f"能力路由：{routing}\n"
            f"执行模式：{task_spec.routing.execution_mode}\n"
            f"工具需求：{'、'.join(task_spec.routing.tool_requirements) or 'none'}\n\n"
            f"完成条件：\n{criteria}\n\n"
            f"风险提示：\n{risks}\n\n"
            f"澄清问题：{clarification}\n"
            f"识别模型：{self.model}（{provider}）"
        )
