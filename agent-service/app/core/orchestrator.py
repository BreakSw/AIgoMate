import os
from typing import Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from app.core.adaptive_runtime import AdaptiveAgentRuntime
from app.core.context_manager import ContextManager
from app.core.intent_recognizer import IntentRecognizer
from app.core.input_rewriter import UserInputRewriteAgent
from app.core.input_organizer import InputOrganizerAgent
from app.core.learning_profile import LearningProfileService
from app.core.memory_agent import MemoryObserverAgent
from app.core.memory_store import UserMemoryRepository
from app.core.model_client import IntentModelClient, RetryCallback
from app.core.output_format_agent import OutputFormattingAgent
from app.core.polish_agent import LanguagePolishAgent
from app.core.progress_status import ProgressCallback
from app.core.reflection import AgentProtocolExhaustedError
from app.models import (
    AgentExecutionTrace,
    AgentRequest,
    AgentResponse,
    AgentWorkResult,
    CodeExecutionReport,
    CoordinatorPlan,
    ContextSnapshot,
    DurableMemoryItem,
    InputOrganizationResult,
    InputRewriteResult,
    LearningProfileSnapshot,
    MemoryScope,
    MemorySnapshot,
    MemoryUpdate,
    MemoryUpdateBatch,
    OutputFormatResult,
    PolishResult,
    RagEvidence,
    TaskSpec,
)


class AgentWorkflowState(TypedDict, total=False):
    """State shared by the top-level LangGraph workflow for one request."""

    request: AgentRequest
    on_retry: RetryCallback | None
    on_progress: ProgressCallback | None
    previous_snapshot: ContextSnapshot | None
    input_organization: InputOrganizationResult
    organized_input: str
    context: list
    context_snapshot: ContextSnapshot
    input_rewrite: InputRewriteResult
    code_artifact: Any
    task_spec: TaskSpec
    intent_provider: str
    learning_profile: LearningProfileSnapshot
    memory_batch: MemoryUpdateBatch
    memory_provider: str
    durable_memory: list[DurableMemoryItem]
    coordinator_plan: CoordinatorPlan
    rag_evidence: list[RagEvidence]
    work_result: AgentWorkResult
    runtime_memory_updates: list[MemoryUpdate]
    runtime_providers: list[str]
    code_execution_reports: list[CodeExecutionReport]
    polish_result: PolishResult
    polish_provider: str
    format_result: OutputFormatResult
    format_provider: str
    response: AgentResponse


class AgentOrchestrator:
    def __init__(
        self,
        context_manager: ContextManager,
        input_organizer: InputOrganizerAgent,
        input_rewriter: UserInputRewriteAgent,
        intent_recognizer: IntentRecognizer,
        memory_observer: MemoryObserverAgent,
        memory_repository: UserMemoryRepository,
        learning_profile_service: LearningProfileService,
        adaptive_runtime: AdaptiveAgentRuntime,
        polish_agent: LanguagePolishAgent,
        output_format_agent: OutputFormattingAgent,
        model: str,
        model_client: IntentModelClient | None = None,
        langsmith_tracer: Any | None = None,
    ) -> None:
        self.context_manager = context_manager
        self.input_organizer = input_organizer
        self.input_rewriter = input_rewriter
        self.intent_recognizer = intent_recognizer
        self.memory_observer = memory_observer
        self.memory_repository = memory_repository
        self.learning_profile_service = learning_profile_service
        self.adaptive_runtime = adaptive_runtime
        self.polish_agent = polish_agent
        self.output_format_agent = output_format_agent
        self.model = model
        self.model_client = model_client
        self.langsmith_tracer = langsmith_tracer
        self.graph = self._compile_graph()

    @property
    def current_model(self) -> str:
        if self.model_client is None:
            return self.model
        return self.model_client.current_model

    async def respond(
        self,
        request: AgentRequest,
        on_retry: RetryCallback | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> AgentResponse:
        graph_config: RunnableConfig = {
            "recursion_limit": 50,
            "run_name": "algomate-agent-request",
            "tags": ["algomate", "langgraph", "multi-agent"],
            "metadata": {
                "user_id": request.user_id,
                "session_id": request.session_id,
            },
        }
        if self.langsmith_tracer is not None and not os.getenv("PYTEST_CURRENT_TEST"):
            graph_config["callbacks"] = [self.langsmith_tracer]
        state = await self.graph.ainvoke(
            AgentWorkflowState(
                request=request,
                on_retry=on_retry,
                on_progress=on_progress,
            ),
            config=graph_config,
        )
        return state["response"]

    def _compile_graph(self):
        workflow = StateGraph(AgentWorkflowState)
        workflow.add_node("session_scope", self._session_scope_node)
        workflow.add_node("input_organization", self._input_organization_node)
        workflow.add_node("context_prepare", self._context_prepare_node)
        workflow.add_node("input_rewrite", self._input_rewrite_node)
        workflow.add_node("intent_recognition", self._intent_recognition_node)
        workflow.add_node("context_finalize", self._context_finalize_node)
        workflow.add_node("learning_profile", self._learning_profile_node)
        workflow.add_node("memory_observation", self._memory_observation_node)
        workflow.add_node("agent_runtime", self._agent_runtime_node)
        workflow.add_node("polishing", self._polishing_node)
        workflow.add_node("formatting", self._formatting_node)
        workflow.add_node("response_assembly", self._response_assembly_node)
        workflow.add_edge(START, "session_scope")
        workflow.add_edge("session_scope", "input_organization")
        workflow.add_edge("input_organization", "context_prepare")
        workflow.add_edge("context_prepare", "input_rewrite")
        workflow.add_edge("input_rewrite", "intent_recognition")
        workflow.add_edge("intent_recognition", "context_finalize")
        workflow.add_edge("context_finalize", "learning_profile")
        workflow.add_edge("learning_profile", "memory_observation")
        workflow.add_edge("memory_observation", "agent_runtime")
        workflow.add_edge("agent_runtime", "polishing")
        workflow.add_edge("polishing", "formatting")
        workflow.add_edge("formatting", "response_assembly")
        workflow.add_edge("response_assembly", END)
        return workflow.compile(name="algomate-complete-agent-workflow")

    async def _session_scope_node(self, state: AgentWorkflowState) -> dict:
        request = state["request"]
        if not request.history and request.previous_context_snapshot is None:
            # SQLite session ids can be reused after a local database reset.
            # A first turn always starts a fresh memory scope, so an old JSON
            # file with the same numeric id can never leak into the new chat.
            await self.memory_repository.reset_session(
                request.user_id,
                request.session_id,
            )
        return {"previous_snapshot": self._scoped_previous_snapshot(request)}

    async def _input_organization_node(self, state: AgentWorkflowState) -> dict:
        request = state["request"]
        await self._progress(
            state.get("on_progress"),
            "input_organization",
            "正在整理用户输入",
            "输入整理 Agent",
            "识别题面、代码、约束与用户要求",
        )
        input_organization = await self.input_organizer.organize(
            request.message,
            state.get("on_retry"),
        )
        return {
            "input_organization": input_organization,
            "organized_input": input_organization.organized_input,
        }

    async def _context_prepare_node(self, state: AgentWorkflowState) -> dict:
        request = state["request"]
        await self._progress(
            state.get("on_progress"),
            "context_prepare",
            "正在准备会话上下文",
            "上下文管理 Agent",
            "装载当前会话记忆并检查上下文窗口",
        )
        checkpoint_hook = self._checkpoint_hook(request)
        context, context_snapshot = await self.context_manager.prepare(
            state["organized_input"],
            request.history,
            state.get("previous_snapshot"),
            state.get("on_retry"),
            checkpoint_hook,
        )
        return {"context": context, "context_snapshot": context_snapshot}

    async def _input_rewrite_node(self, state: AgentWorkflowState) -> dict:
        request = state["request"]
        await self._progress(
            state.get("on_progress"),
            "input_rewrite",
            "正在规范化本轮任务",
            "请求改写 Agent",
            "保留原始题面和代码，整理可执行目标",
        )
        input_rewrite = await self.input_rewriter.rewrite(
            state["organized_input"],
            state["context"],
            state.get("on_retry"),
        )
        code_artifact = self.input_rewriter.reconcile_code_artifact(
            request.message,
            input_rewrite,
        )
        return {"input_rewrite": input_rewrite, "code_artifact": code_artifact}

    async def _intent_recognition_node(self, state: AgentWorkflowState) -> dict:
        await self._progress(
            state.get("on_progress"),
            "intent_recognition",
            "正在识别意图与交付要求",
            "意图识别 Agent",
            "确定解题、提示、代码分析或学习规划模式",
        )
        task_spec, provider = await self.intent_recognizer.recognize(
            state["input_rewrite"].formatted_input,
            state["context"],
            state.get("on_retry"),
            state.get("code_artifact"),
            state["input_rewrite"],
        )
        return {"task_spec": task_spec, "intent_provider": provider}

    async def _context_finalize_node(self, state: AgentWorkflowState) -> dict:
        request = state["request"]
        context_snapshot = await self.context_manager.finalize_turn(
            state["organized_input"],
            request.history,
            state["context"],
            state["context_snapshot"],
            state["task_spec"],
            state["intent_provider"],
            state.get("on_retry"),
            self._checkpoint_hook(request),
        )
        context_snapshot = context_snapshot.model_copy(update={
            "input_organization": state["input_organization"],
            "input_rewrite": state["input_rewrite"],
            "memory_scope": MemoryScope(
                user_id=request.user_id,
                session_id=request.session_id,
            ),
        })
        return {"context_snapshot": context_snapshot}

    async def _learning_profile_node(self, state: AgentWorkflowState) -> dict:
        request = state["request"]
        await self._progress(
            state.get("on_progress"),
            "learning_profile",
            "正在分析个性化学习状态",
            "个性化学习建模 Agent",
            "仅依据明确学习反馈更新 BKT、IRT 与 FSRS-style 状态",
        )
        learning_profile = await self.learning_profile_service.process_turn(
            request.user_id,
            request.session_id,
            state["organized_input"],
            state["task_spec"],
        )
        context_snapshot = state["context_snapshot"].model_copy(
            update={"learning_profile": learning_profile}
        )
        return {
            "learning_profile": learning_profile,
            "context_snapshot": context_snapshot,
        }

    async def _memory_observation_node(self, state: AgentWorkflowState) -> dict:
        request = state["request"]
        await self._progress(
            state.get("on_progress"),
            "memory_observation",
            "正在检查本轮是否需要更新记忆",
            "记忆观察 Agent",
            "仅保存当前会话后续仍有价值的信息",
        )
        existing_memory = await self.memory_repository.load(
            request.user_id,
            request.session_id,
        )
        memory_batch, memory_provider = await self.memory_observer.observe(
            state["organized_input"],
            state["task_spec"],
            state["context_snapshot"],
            existing_memory,
            state.get("on_retry"),
        )
        if memory_batch.updates:
            await self.memory_repository.upsert(
                request.user_id,
                request.session_id,
                memory_batch.updates,
                source="memory_agent",
            )
        durable_memory = await self.memory_repository.recall(
            request.user_id,
            request.session_id,
            state["task_spec"].normalized_request,
        )
        context_snapshot = state["context_snapshot"].model_copy(
            update={
                "memory": self._merge_durable_memory(
                    state["context_snapshot"].memory,
                    durable_memory,
                )
            }
        )
        return {
            "memory_batch": memory_batch,
            "memory_provider": memory_provider,
            "durable_memory": durable_memory,
            "context_snapshot": context_snapshot,
        }

    async def _agent_runtime_node(
        self,
        state: AgentWorkflowState,
        config: RunnableConfig,
    ) -> dict:
        request = state["request"]
        previous_turn_evidence = self._continuation_evidence(
            state.get("previous_snapshot"),
            state["task_spec"],
        )
        runtime_result = await self.adaptive_runtime.run(
            user_id=request.user_id,
            session_id=request.session_id,
            task_spec=state["task_spec"],
            snapshot=state["context_snapshot"],
            conversation_context=state["context"],
            durable_memory=state["durable_memory"],
            previous_turn_evidence=previous_turn_evidence,
            on_retry=state.get("on_retry"),
            on_progress=state.get("on_progress"),
            runnable_config=config,
        )
        (
            coordinator_plan,
            rag_evidence,
            work_result,
            durable_memory,
            runtime_memory_updates,
            runtime_providers,
        ) = runtime_result[:6]
        code_execution_reports = (
            runtime_result[6] if len(runtime_result) > 6 else []
        )
        return {
            "coordinator_plan": coordinator_plan,
            "rag_evidence": rag_evidence,
            "work_result": work_result,
            "durable_memory": durable_memory,
            "runtime_memory_updates": runtime_memory_updates,
            "runtime_providers": runtime_providers,
            "code_execution_reports": code_execution_reports,
        }

    async def _polishing_node(self, state: AgentWorkflowState) -> dict:
        await self._progress(
            state.get("on_progress"),
            "polishing",
            "正在润色已验证的回答",
            "语言润色 Agent",
            "保持事实与代码不变，改善表达清晰度",
        )
        polish_result, polish_provider = await self.polish_agent.polish(
            state["work_result"],
            state["task_spec"],
            state.get("on_retry"),
        )
        return {"polish_result": polish_result, "polish_provider": polish_provider}

    async def _formatting_node(self, state: AgentWorkflowState) -> dict:
        try:
            await self._progress(
                state.get("on_progress"),
                "formatting",
                "正在整理最终展示格式",
                "输出格式 Agent",
                "整理标题、代码块、表格和参考链接",
            )
            format_result, format_provider = await self.output_format_agent.format(
                state["polish_result"],
                state["task_spec"],
                state["rag_evidence"],
                state.get("on_retry"),
            )
        except AgentProtocolExhaustedError:
            # Formatting is presentation-only. A cosmetic protocol failure must
            # never discard an otherwise valid, polished answer.
            from app.models import OutputFormatResult

            format_result = OutputFormatResult(
                formatted_answer=state["polish_result"].final_answer,
                formatting_changes=["格式整理未通过校验，安全回退到润色结果"],
            )
            format_provider = "local-safe-format-fallback"
        return {"format_result": format_result, "format_provider": format_provider}

    async def _response_assembly_node(self, state: AgentWorkflowState) -> dict:
        task_spec = state["task_spec"]
        learning_profile = state["learning_profile"]
        model_call_trace = [
            f"input-organizer:{state['input_organization'].organizer_provider}",
            f"input-rewrite:{state['input_rewrite'].rewrite_provider}",
            f"intent:{state['intent_provider']}",
            f"memory:{state['memory_provider']}",
            *state["runtime_providers"],
            f"polish:{state['polish_provider']}",
            f"format:{state['format_provider']}",
        ]
        if learning_profile.active:
            model_call_trace.insert(4, "learning-profile:local-bkt-irt-fsrs")
        execution = AgentExecutionTrace(
            task_spec=task_spec,
            coordinator_plan=state["coordinator_plan"],
            rag_evidence=state["rag_evidence"],
            work_result=state["work_result"],
            polish_result=state["polish_result"],
            format_result=state["format_result"],
            durable_memory=state["durable_memory"],
            memory_updates=[
                *state["memory_batch"].updates,
                *state["runtime_memory_updates"],
            ],
            model_call_trace=model_call_trace,
            code_execution_reports=state.get("code_execution_reports", []),
        )
        context_snapshot = state["context_snapshot"].model_copy(
            update={"agent_execution": execution}
        )
        combined_provider = " | ".join(model_call_trace)
        learning_report = self.learning_profile_service.render_markdown(
            learning_profile
        )
        final_content = state["format_result"].formatted_answer
        if learning_report:
            final_content = f"{final_content.rstrip()}\n\n{learning_report}"
        response = AgentResponse(
            content=final_content,
            intent=task_spec.primary_intent,
            context_messages_used=len(state["context"]),
            task_spec=task_spec,
            context_snapshot=context_snapshot,
            model=self.current_model,
            provider=combined_provider,
        )
        return {"context_snapshot": context_snapshot, "response": response}

    def _checkpoint_hook(self, request: AgentRequest):
        async def checkpoint_hook(memory: MemorySnapshot, stage: str) -> None:
            del stage
            await self.memory_repository.checkpoint_snapshot(
                request.user_id,
                request.session_id,
                memory,
            )

        return checkpoint_hook

    @staticmethod
    async def _progress(
        callback: ProgressCallback | None,
        phase: str,
        message: str,
        agent: str | None,
        detail: str | None,
    ) -> None:
        if callback is not None:
            await callback(phase, message, agent, detail)

    @staticmethod
    def _scoped_previous_snapshot(request: AgentRequest) -> ContextSnapshot | None:
        snapshot = request.previous_context_snapshot
        if snapshot is None or snapshot.memory_scope is None:
            return None
        scope = snapshot.memory_scope
        if scope.user_id != request.user_id or scope.session_id != request.session_id:
            return None
        return snapshot

    @staticmethod
    def _continuation_evidence(
        previous_snapshot: ContextSnapshot | None,
        task_spec: TaskSpec,
    ) -> list:
        """Reuse actual prior tool evidence for an explicit continuation only.

        The rendered assistant answer is never promoted to evidence. The source
        entries below came from the previous execution trace and retain their
        URLs, metadata, and original content for downstream re-validation.
        """
        if not task_spec.context_plan.task_state:
            return []
        if previous_snapshot is None or previous_snapshot.agent_execution is None:
            return []
        return list(previous_snapshot.agent_execution.rag_evidence)

    def _merge_durable_memory(
        self,
        active: MemorySnapshot,
        durable: list[DurableMemoryItem],
    ) -> MemorySnapshot:
        preferences = [item.content for item in durable if item.kind == "preference"]
        constraints = [item.content for item in durable if item.kind == "constraint"]
        long_term = [
            item.content
            for item in durable
            if item.kind in {"user_profile", "long_term_goal", "decision", "learned_fact"}
        ]
        open_questions = [
            item.content for item in durable if item.kind == "unfinished_task"
        ]
        return active.model_copy(update={
            "long_term_memory": self.context_manager._merge_unique(
                active.long_term_memory,
                long_term,
                limit=16,
            ),
            "user_preferences": self.context_manager._merge_unique(
                active.user_preferences,
                preferences,
                limit=12,
            ),
            "pinned_constraints": self.context_manager._merge_unique(
                active.pinned_constraints,
                constraints,
                limit=12,
            ),
            "open_questions": self.context_manager._merge_unique(
                active.open_questions,
                open_questions,
                limit=10,
            ),
        })
    def _select_memory(self, snapshot: ContextSnapshot, selection) -> MemorySnapshot:
        memory = snapshot.memory
        return MemorySnapshot(
            current_goal=memory.current_goal if selection.working_memory else None,
            working_memory=memory.working_memory if selection.working_memory else [],
            long_term_memory=memory.long_term_memory if selection.long_term_memory else [],
            user_preferences=memory.user_preferences if selection.user_preferences else [],
            pinned_constraints=(
                memory.pinned_constraints if selection.pinned_constraints else []
            ),
            open_questions=memory.open_questions if selection.working_memory else [],
            resolved_items=memory.resolved_items if selection.working_memory else [],
        )

    def _format_input_rewrite(self, rewrite) -> str:
        constraints = "、".join(rewrite.constraints) or "无"
        ambiguities = "、".join(rewrite.ambiguities) or "无"
        return (
            "输入改写 Agent v1.0\n\n"
            f"输入类型：{rewrite.input_type}\n"
            f"规范输入：{rewrite.formatted_input}\n"
            f"明确请求：{rewrite.explicit_request or '未提供'}\n"
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
            f"识别模型：{self.current_model}（{provider}）"
        )
