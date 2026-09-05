import asyncio
import hashlib
from typing import Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from app.core.code_artifact import extract_code_artifact, normalize_language
from app.core.code_test_generation_agent import CodeTestGenerationAgent
from app.core.coordinator_agent import CoordinatorAgent
from app.core.current_time_tool import CurrentTimeTool
from app.core.judge0_code_runner import Judge0CodeRunner
from app.core.memory_store import DynamicSystemPromptBuilder, UserMemoryRepository
from app.core.model_client import RetryCallback
from app.core.progress_status import ProgressCallback
from app.core.rag_retriever import LocalRagRetriever
from app.core.response_agent import ResponseAgent
from app.core.web_search_agent import WebSearchAgent
from app.models import (
    AgentWorkRequest,
    AgentWorkResult,
    CodeExecutionReport,
    ContextSnapshot,
    ConversationMessage,
    CoordinatorPlan,
    DurableMemoryItem,
    HeadDecision,
    MemorySelection,
    MemoryUpdate,
    RagEvidence,
    RagQuery,
    TaskSpec,
)


class AdaptiveRuntimeState(TypedDict):
    """In-memory LangGraph state for one isolated chat turn.

    The graph deliberately has no persistent checkpointer: conversation and
    durable-memory persistence remain owned by the existing context and memory
    services, preserving the public behavior and session-isolation guarantees.
    """

    user_id: int
    session_id: int
    task_spec: TaskSpec
    snapshot: ContextSnapshot
    conversation_context: list[ConversationMessage]
    durable_memory: list[DurableMemoryItem]
    on_retry: RetryCallback | None
    on_progress: ProgressCallback | None
    iteration: int
    decisions: list[HeadDecision]
    evidence: list[RagEvidence]
    observations: list[str]
    rag_queries: list[RagQuery]
    web_queries: list[str]
    known_limits: list[str]
    memory_updates: list[MemoryUpdate]
    providers: list[str]
    code_execution_reports: list[CodeExecutionReport]
    latest_work_result: AgentWorkResult | None
    work_history: list[AgentWorkResult]
    execution_mode: Literal["rag_assisted", "native_reasoning"]
    rag_status: Literal[
        "not_checked",
        "candidate_found",
        "hit",
        "miss",
        "insufficient",
        "unavailable",
        "error",
    ]
    last_instruction: str
    dynamic_prompt: str
    current_decision: HeadDecision | None


class AdaptiveAgentRuntime:
    """LangGraph-directed agent runtime. The head model chooses every action."""

    def __init__(
        self,
        coordinator: CoordinatorAgent,
        response_agent: ResponseAgent,
        rag_retriever: LocalRagRetriever,
        web_search_agent: WebSearchAgent,
        current_time_tool: CurrentTimeTool,
        memory_repository: UserMemoryRepository,
        prompt_builder: DynamicSystemPromptBuilder,
        max_iterations: int = 6,
        code_test_generation_agent: CodeTestGenerationAgent | None = None,
        judge0_code_runner: Judge0CodeRunner | None = None,
    ) -> None:
        self.coordinator = coordinator
        self.response_agent = response_agent
        self.rag_retriever = rag_retriever
        self.web_search_agent = web_search_agent
        self.current_time_tool = current_time_tool
        self.memory_repository = memory_repository
        self.prompt_builder = prompt_builder
        coordinator_model_client = getattr(coordinator, "model_client", None)
        self.code_test_generation_agent = code_test_generation_agent
        if self.code_test_generation_agent is None and coordinator_model_client is not None:
            self.code_test_generation_agent = CodeTestGenerationAgent(
                coordinator_model_client,
                3,
            )
        self.judge0_code_runner = judge0_code_runner or Judge0CodeRunner()
        self.max_iterations = max_iterations
        self.graph = self._compile_graph()

    async def run(
        self,
        *,
        user_id: int,
        session_id: int,
        task_spec: TaskSpec,
        snapshot: ContextSnapshot,
        conversation_context: list[ConversationMessage],
        durable_memory: list[DurableMemoryItem],
        previous_turn_evidence: list[RagEvidence] | None = None,
        on_retry: RetryCallback | None = None,
        on_progress: ProgressCallback | None = None,
        runnable_config: RunnableConfig | None = None,
    ) -> tuple[
        CoordinatorPlan,
        list[RagEvidence],
        AgentWorkResult,
        list[DurableMemoryItem],
        list[MemoryUpdate],
        list[str],
        list[CodeExecutionReport],
    ]:
        evidence: list[RagEvidence] = [
            item.model_copy(update={
                "metadata": {
                    **item.metadata,
                    "carried_from_previous_turn": True,
                }
            })
            for item in (previous_turn_evidence or [])
        ]
        observations: list[str] = []
        if evidence:
            observations.append(
                f"已载入上一轮 {len(evidence)} 条真实检索证据供续问复用；"
                "旧 assistant 文本仍不是证据，新增交付要求需要检查现有证据是否足够。"
            )
        native_only = CoordinatorAgent._requests_native_only(task_spec)
        if native_only:
            observations.append(
                "用户明确要求本轮跳过 RAG/网页检索，已直接进入算法自主推理模式。"
            )
        graph_config: RunnableConfig = dict(runnable_config or {})
        graph_config["recursion_limit"] = max(25, self.max_iterations * 3 + 5)
        graph_state = await self.graph.ainvoke(
            AdaptiveRuntimeState(
                user_id=user_id,
                session_id=session_id,
                task_spec=task_spec,
                snapshot=snapshot,
                conversation_context=conversation_context,
                durable_memory=durable_memory,
                on_retry=on_retry,
                on_progress=on_progress,
                iteration=0,
                decisions=[],
                evidence=evidence,
                observations=observations,
                rag_queries=[],
                web_queries=[],
                known_limits=[],
                memory_updates=[],
                providers=[],
                code_execution_reports=[],
                latest_work_result=None,
                work_history=[],
                execution_mode="native_reasoning" if native_only else "rag_assisted",
                rag_status="not_checked",
                last_instruction=task_spec.normalized_request,
                dynamic_prompt="",
                current_decision=None,
            ),
            config=graph_config,
        )
        decisions = graph_state["decisions"]
        evidence = graph_state["evidence"]
        observations = graph_state["observations"]
        rag_queries = graph_state["rag_queries"]
        web_queries = graph_state["web_queries"]
        known_limits = graph_state["known_limits"]
        memory_updates = graph_state["memory_updates"]
        providers = graph_state["providers"]
        code_execution_reports = graph_state["code_execution_reports"]
        latest_work_result = graph_state["latest_work_result"]
        work_history = graph_state["work_history"]
        execution_mode = graph_state["execution_mode"]
        rag_status = graph_state["rag_status"]
        last_instruction = graph_state["last_instruction"]

        if latest_work_result is None:
            user_material_available = self._has_user_problem_material(task_spec)
            if evidence or user_material_available:
                if user_material_available and rag_status != "hit":
                    execution_mode = "native_reasoning"
                    if rag_status == "candidate_found":
                        rag_status = "insufficient"
                    elif rag_status == "not_checked":
                        rag_status = "unavailable"
                selected_agent = self._fallback_execution_agent(task_spec)
                if execution_mode == "native_reasoning":
                    last_instruction = (
                        "决策轮次已经耗尽。不要再等待 RAG 或网页资料，直接根据用户已经提供的题面、"
                        "代码、样例和约束独立推导；给出算法思路、正确性说明、复杂度和符合用户要求的实现，"
                        "并自行检查边界条件。不要要求用户重复提供上下文中已有的信息。"
                    )
                else:
                    last_instruction = (
                        "阅读现有网页或知识库证据，直接完成用户任务；证据不足的部分明确说明，"
                        "不要要求用户重复提供已经请求联网搜索的信息。"
                    )
                decisions.append(HeadDecision(
                    iteration=len(decisions) + 1,
                    rationale="已有检索证据但首脑达到轮次上限，转交执行 Agent 形成可交付回答。",
                    action="delegate",
                    selected_agent=selected_agent,
                    task_instruction=last_instruction,
                ))
                dynamic_prompt = self.prompt_builder.build(
                    user_id,
                    session_id,
                    snapshot.memory,
                    durable_memory,
                    snapshot.learning_profile,
                )
                current_plan = self._build_plan(
                    task_spec,
                    decisions,
                    rag_queries,
                    web_queries,
                    known_limits,
                    None,
                    last_instruction,
                    execution_mode,
                    rag_status,
                )
                work_request = AgentWorkRequest(
                    task_spec=task_spec,
                    coordinator_plan=current_plan,
                    conversation_context=self._conversation_context_for_task(
                        task_spec,
                        conversation_context,
                    ),
                    memory=snapshot.memory,
                    durable_memory=durable_memory,
                    dynamic_system_prompt=dynamic_prompt,
                    rag_evidence=evidence,
                    prior_work_results=work_history,
                    code_execution_reports=code_execution_reports,
                )
                agent_label, activity = self._agent_activity(selected_agent)
                await self._progress(
                    on_progress,
                    "fallback_work",
                    activity,
                    agent_label,
                    "首脑决策预算即将结束，正在形成可交付答案",
                )
                latest_work_result, answer_provider = await self.response_agent.execute(
                    work_request,
                    on_retry,
                )
                providers.append(f"fallback-worker:{answer_provider}")
                work_history.append(latest_work_result)
            else:
                latest_work_result = AgentWorkResult(
                    agent="clarification_agent",
                    draft_answer=(
                        "我还没有获得足够可靠的信息来完成回答。请补充更具体的题目、代码、错误现象或目标，我再继续分析。"
                    ),
                    uncertainties=list(dict.fromkeys([
                        *known_limits,
                        "首脑在决策轮次上限内未形成可交付答案",
                    ])),
                    needs_follow_up=True,
                )

        if (
            execution_mode == "native_reasoning"
            and self._has_user_problem_material(task_spec)
            and latest_work_result.agent != "clarification_agent"
            and not self._has_solution_work(work_history)
        ):
            selected_agent = self._fallback_execution_agent(task_spec)
            last_instruction = (
                "决策轮次已结束，但当前只有题面整理或评审结果。阅读 prior_work_results，"
                "直接完成算法设计与实现，给出正确性依据、复杂度、边界条件以及符合用户要求的代码；"
                "不得因为没有 RAG 或网页资料而拒绝回答。"
            )
            decisions.append(HeadDecision(
                iteration=len(decisions) + 1,
                rationale="自主推理尚缺少实际解题或实现产物，执行安全收尾求解。",
                action="delegate",
                selected_agent=selected_agent,
                task_instruction=last_instruction,
            ))
            dynamic_prompt = self.prompt_builder.build(
                user_id,
                session_id,
                snapshot.memory,
                durable_memory,
                snapshot.learning_profile,
            )
            solution_plan = self._build_plan(
                task_spec,
                decisions,
                rag_queries,
                web_queries,
                known_limits,
                latest_work_result,
                last_instruction,
                execution_mode,
                rag_status,
            )
            solution_request = AgentWorkRequest(
                task_spec=task_spec,
                coordinator_plan=solution_plan,
                conversation_context=self._conversation_context_for_task(
                    task_spec,
                    conversation_context,
                ),
                memory=snapshot.memory,
                durable_memory=durable_memory,
                dynamic_system_prompt=dynamic_prompt,
                rag_evidence=evidence,
                prior_work_results=work_history,
                code_execution_reports=code_execution_reports,
            )
            agent_label, activity = self._agent_activity(selected_agent)
            await self._progress(
                on_progress,
                "fallback_solution",
                activity,
                agent_label,
                "补全自主推理链中的解题或实现阶段",
            )
            latest_work_result, answer_provider = await self.response_agent.execute(
                solution_request,
                on_retry,
            )
            providers.append(f"fallback-solver:{answer_provider}")
            work_history.append(latest_work_result)

        if (
            self._requires_code_execution(task_spec, work_history)
            and latest_work_result.agent != "clarification_agent"
            and self._latest_code_report(work_history, code_execution_reports) is None
        ):
            decisions.append(HeadDecision(
                iteration=len(decisions) + 1,
                rationale="决策轮次结束时发现最新代码尚未执行，启动强制真实验证收尾。",
                action="execute_code_tests",
            ))
            execution_state = dict(graph_state)
            execution_state.update({
                "decisions": decisions,
                "current_decision": decisions[-1],
                "iteration": decisions[-1].iteration,
                "latest_work_result": latest_work_result,
                "work_history": work_history,
                "observations": observations,
                "known_limits": known_limits,
                "providers": providers,
                "code_execution_reports": code_execution_reports,
                "on_retry": on_retry,
                "on_progress": on_progress,
            })
            execution_update = await self._code_test_node(execution_state)
            observations = execution_update.get("observations", observations)
            known_limits = execution_update.get("known_limits", known_limits)
            providers = execution_update.get("providers", providers)
            code_execution_reports = execution_update.get(
                "code_execution_reports",
                code_execution_reports,
            )

        latest_execution_report = self._latest_code_report(
            work_history,
            code_execution_reports,
        )
        execution_tool_blocked = bool(
            latest_execution_report
            and latest_execution_report.overall_status in {
                "failed",
                "unavailable",
                "unsupported",
                "error",
            }
        )
        if execution_tool_blocked:
            latest_work_result = latest_work_result.model_copy(update={
                "uncertainties": list(dict.fromkeys([
                    *latest_work_result.uncertainties,
                    "测试集生成或 Judge0 工具未完成，当前代码没有真实运行通过证明",
                ])),
                "needs_follow_up": True,
            })
            work_history[-1] = latest_work_result

        if (
            self._requires_verification(task_spec)
            and latest_work_result.agent != "clarification_agent"
            and self._needs_verification(work_history)
            and not execution_tool_blocked
        ):
            last_instruction = (
                "决策轮次已结束，但最新解题或代码产物必须完成独立验证。阅读 prior_work_results，"
                "检查题意一致性、样例、边界条件、正确性、复杂度以及代码与算法是否一致；"
                "修正发现的问题，并在 draft_answer 中返回可直接交付给用户的完整最终答案。"
            )
            decisions.append(HeadDecision(
                iteration=len(decisions) + 1,
                rationale="自主推理结果尚未经过独立验证，执行安全收尾验证。",
                action="delegate",
                selected_agent="verification_agent",
                task_instruction=last_instruction,
            ))
            dynamic_prompt = self.prompt_builder.build(
                user_id,
                session_id,
                snapshot.memory,
                durable_memory,
                snapshot.learning_profile,
            )
            verification_plan = self._build_plan(
                task_spec,
                decisions,
                rag_queries,
                web_queries,
                known_limits,
                latest_work_result,
                last_instruction,
                execution_mode,
                rag_status,
            )
            verification_request = AgentWorkRequest(
                task_spec=task_spec,
                coordinator_plan=verification_plan,
                conversation_context=self._conversation_context_for_task(
                    task_spec,
                    conversation_context,
                ),
                memory=snapshot.memory,
                durable_memory=durable_memory,
                dynamic_system_prompt=dynamic_prompt,
                rag_evidence=evidence,
                prior_work_results=work_history,
                code_execution_reports=code_execution_reports,
            )
            await self._progress(
                on_progress,
                "fallback_verification",
                "正在进行最终独立验证",
                "验证 Agent",
                "检查题意、样例、边界、复杂度以及代码一致性",
            )
            latest_work_result, answer_provider = await self.response_agent.execute(
                verification_request,
                on_retry,
            )
            providers.append(f"fallback-verifier:{answer_provider}")
            work_history.append(latest_work_result)

        plan = self._build_plan(
            task_spec,
            decisions,
            rag_queries,
            web_queries,
            known_limits,
            latest_work_result,
            last_instruction,
            execution_mode,
            rag_status,
        )
        return (
            plan,
            evidence,
            latest_work_result,
            durable_memory,
            memory_updates,
            providers,
            code_execution_reports,
        )

    def _compile_graph(self):
        workflow = StateGraph(AdaptiveRuntimeState)
        workflow.add_node("head", self._head_node)
        workflow.add_node("get_current_time", self._time_node)
        workflow.add_node("retrieve_rag", self._rag_node)
        workflow.add_node("switch_to_native_reasoning", self._native_reasoning_node)
        workflow.add_node("search_web", self._web_search_node)
        workflow.add_node("execute_code_tests", self._code_test_node)
        workflow.add_node("persist_memory", self._memory_node)
        workflow.add_node("ask_clarification", self._clarification_node)
        workflow.add_node("delegate", self._delegate_node)
        workflow.add_node("finish", self._finish_node)
        workflow.add_edge(START, "head")
        workflow.add_conditional_edges(
            "head",
            self._route_head_action,
            {
                "get_current_time": "get_current_time",
                "retrieve_rag": "retrieve_rag",
                "switch_to_native_reasoning": "switch_to_native_reasoning",
                "search_web": "search_web",
                "execute_code_tests": "execute_code_tests",
                "persist_memory": "persist_memory",
                "ask_clarification": "ask_clarification",
                "delegate": "delegate",
                "finish": "finish",
            },
        )
        for node in (
            "get_current_time",
            "retrieve_rag",
            "switch_to_native_reasoning",
            "search_web",
            "execute_code_tests",
            "persist_memory",
            "delegate",
        ):
            workflow.add_conditional_edges(
                node,
                self._route_after_action,
                {"continue": "head", "end": END},
            )
        workflow.add_edge("ask_clarification", END)
        workflow.add_edge("finish", END)
        return workflow.compile(name="algomate-adaptive-agent-runtime")

    async def _head_node(self, state: AdaptiveRuntimeState) -> dict:
        iteration = state["iteration"] + 1
        head_message, head_detail = self._head_progress(
            state["execution_mode"],
            state["rag_status"],
            state["work_history"],
        )
        await self._progress(
            state["on_progress"],
            "head_decision",
            head_message,
            "首脑智能体",
            head_detail,
        )
        dynamic_prompt = self.prompt_builder.build(
            state["user_id"],
            state["session_id"],
            state["snapshot"].memory,
            state["durable_memory"],
            state["snapshot"].learning_profile,
        )
        runtime_state = self._runtime_state(
            state["decisions"],
            state["observations"],
            state["evidence"],
            state["latest_work_result"],
            state["work_history"],
            state["execution_mode"],
            state["rag_status"],
            state["code_execution_reports"],
        )
        runtime_state["iteration_budget"] = {
            "current": iteration,
            "maximum": self.max_iterations,
            "remaining_after_this_decision": self.max_iterations - iteration,
        }
        decision, provider = await self.coordinator.decide(
            task_spec=state["task_spec"],
            snapshot=state["snapshot"],
            runtime_state=runtime_state,
            knowledge_availability=self.rag_retriever.availability(),
            web_search_available=self.web_search_agent.available(),
            dynamic_system_prompt=dynamic_prompt,
            iteration=iteration,
            conversation_context=self._conversation_context_for_task(
                state["task_spec"],
                state["conversation_context"],
            ),
            on_retry=state["on_retry"],
        )
        return {
            "iteration": iteration,
            "current_decision": decision,
            "dynamic_prompt": dynamic_prompt,
            "decisions": [*state["decisions"], decision],
            "providers": [*state["providers"], f"head#{iteration}:{provider}"],
        }

    @staticmethod
    def _route_head_action(state: AdaptiveRuntimeState) -> str:
        decision = state["current_decision"]
        if decision is None:
            raise RuntimeError("LangGraph head node did not produce a decision")
        return decision.action

    def _route_after_action(self, state: AdaptiveRuntimeState) -> str:
        return "end" if state["iteration"] >= self.max_iterations else "continue"

    async def _time_node(self, state: AdaptiveRuntimeState) -> dict:
        await self._progress(
            state["on_progress"],
            "time_tool",
            "正在读取应用当前时间",
            "时间工具",
            "把相对日期转换为可检索的绝对日期",
        )
        time_result = self.current_time_tool.read()
        observation = (
            "时间工具返回："
            f"当前日期时间 {time_result['local_datetime']}，"
            f"日期 {time_result['chinese_date']}，"
            f"星期 {time_result['weekday']}，"
            f"时区 {time_result['timezone']}（UTC{time_result['utc_offset']}）。"
            "后续涉及相对日期的检索必须使用该绝对日期。"
        )
        return {
            "observations": [*state["observations"], observation],
            "providers": [*state["providers"], f"time#{state['iteration']}:local"],
        }

    async def _rag_node(self, state: AdaptiveRuntimeState) -> dict:
        decision = state["current_decision"]
        query = decision.rag_query if decision is not None else None
        if query is None:
            return {
                "observations": [
                    *state["observations"],
                    "首脑请求 RAG，但没有给出合法查询。",
                ]
            }
        rag_queries = [*state["rag_queries"], query]
        collection_label = {
            "algorithm_concepts": "算法概念库",
            "problem_bank": "题库",
            "code_cases": "代码案例库",
        }.get(query.collection, query.collection)
        await self._progress(
            state["on_progress"],
            "rag_retrieval",
            f"正在对{collection_label}执行混合检索",
            "混合 RAG 检索 Agent",
            "并行 Dense/BM25 召回，经 RRF 融合与 Voyage Rerank 后返回候选",
        )
        if not self.rag_retriever.availability().get(query.collection, False):
            limit = f"{query.collection} 当前不可用"
            return {
                "rag_queries": rag_queries,
                "observations": [*state["observations"], limit],
                "known_limits": [*state["known_limits"], limit],
                "rag_status": "unavailable",
            }
        try:
            found = await asyncio.to_thread(self.rag_retriever.retrieve, query)
        except Exception:
            limit = f"{query.collection} 检索失败"
            return {
                "rag_queries": rag_queries,
                "observations": [*state["observations"], limit],
                "known_limits": [*state["known_limits"], limit],
                "rag_status": "error",
            }
        evidence = list(state["evidence"])
        added = self._merge_evidence(evidence, found, prefix="R")
        existing_rag_count = sum(
            1 for item in evidence if item.collection != "web_search"
        )
        if added:
            retrieval_provider = next(
                (
                    str(item.metadata.get("retrieval_provider"))
                    for item in found
                    if item.metadata.get("retrieval_provider")
                ),
                "unknown",
            )
            observation = (
                f"{query.collection} 检索完成，新增 {added} 条候选证据；"
                f"检索链路={retrieval_provider}；"
                "首脑仍需判断其是否真正覆盖用户题面，候选不等于有效命中。"
            )
        elif found and existing_rag_count:
            observation = (
                f"{query.collection} 检索结果与已有候选重复，没有新增证据；"
                f"继续保留现有 {existing_rag_count} 条候选供首脑判断。"
            )
        else:
            observation = f"{query.collection} 检索完成，但没有找到可用候选。"
        return {
            "rag_queries": rag_queries,
            "evidence": evidence,
            "observations": [*state["observations"], observation],
            "rag_status": (
                "candidate_found" if added or existing_rag_count else "miss"
            ),
        }

    async def _native_reasoning_node(self, state: AdaptiveRuntimeState) -> dict:
        await self._progress(
            state["on_progress"],
            "native_reasoning",
            "知识库证据不足，切换自主推理",
            "首脑智能体",
            "后续由专业 Agent 基于用户题面协作解题，仍可按需联网",
        )
        retained_rag_count = sum(
            1 for item in state["evidence"] if item.collection != "web_search"
        )
        return {
            "execution_mode": "native_reasoning",
            "rag_status": "insufficient" if retained_rag_count else state["rag_status"],
            "observations": [
                *state["observations"],
                "已切换到自主推理模式：不再继续依赖 RAG 扩展结论，"
                f"但保留 {retained_rag_count} 条候选作为低信任参考；"
                "执行 Agent 必须逐条核对适用范围，不能把候选冒充已确认事实。",
            ],
            "providers": [
                *state["providers"],
                f"mode#{state['iteration']}:native-reasoning",
            ],
        }

    async def _web_search_node(self, state: AdaptiveRuntimeState) -> dict:
        decision = state["current_decision"]
        query = (decision.web_query if decision is not None else None) or ""
        reason = (
            decision.web_search_reason if decision is not None else None
        ) or "补充外部信息"
        web_queries = [*state["web_queries"], query]
        await self._progress(
            state["on_progress"],
            "web_search",
            "正在搜索可核验的外部资料",
            "网页搜索 Agent",
            reason,
        )
        if not self.web_search_agent.available():
            limit = "网页搜索 Agent 当前不可用"
            return {
                "web_queries": web_queries,
                "observations": [*state["observations"], limit],
                "known_limits": [*state["known_limits"], limit],
            }
        found, web_provider = await self.web_search_agent.search(
            query,
            reason,
            state["on_retry"],
        )
        evidence = list(state["evidence"])
        added = self._merge_evidence(evidence, found, prefix="W")
        known_limits = list(state["known_limits"])
        if added:
            observation = f"网页搜索完成，新增 {added} 条证据。"
        elif "-error:" in web_provider:
            observation = "网页搜索请求失败，本轮未能补充外部证据。"
            known_limits.append(observation)
        else:
            observation = "网页搜索完成，但没有返回可用结果。"
        return {
            "web_queries": web_queries,
            "evidence": evidence,
            "observations": [*state["observations"], observation],
            "known_limits": known_limits,
            "providers": [
                *state["providers"],
                f"web#{state['iteration']}:{web_provider}",
            ],
        }

    async def _code_test_node(self, state: AdaptiveRuntimeState) -> dict:
        await self._progress(
            state["on_progress"],
            "code_execution",
            "正在生成测试并调用 Judge0 编译运行",
            "算法测试执行 Agent",
            "为当前代码生成边界/对抗测试 Harness，并用真实运行结果验证",
        )
        artifact = None
        solution_context = ""
        for item in reversed(state["work_history"]):
            candidate = extract_code_artifact(item.draft_answer)
            if candidate is not None:
                artifact = candidate
                solution_context = item.draft_answer
                break
        if artifact is None and state["task_spec"].input_artifacts.code:
            raw_code = state["task_spec"].input_artifacts.code or ""
            artifact = extract_code_artifact(raw_code)
            solution_context = state["task_spec"].normalized_request

        if artifact is None:
            return {
                "observations": [
                    *state["observations"],
                    "代码执行工具未找到可提取的候选代码块，无法生成 Harness。",
                ],
                "known_limits": [
                    *state["known_limits"],
                    "当前执行结果中没有可识别的完整代码",
                ],
            }

        source_hash = hashlib.sha256(artifact.code.encode("utf-8")).hexdigest()
        previous = next(
            (
                report
                for report in reversed(state["code_execution_reports"])
                if report.source_code_hash == source_hash
            ),
            None,
        )
        if previous is not None:
            return {
                "observations": [
                    *state["observations"],
                    f"当前代码版本已由 Judge0 执行，复用 {previous.verdict} 报告。",
                ]
            }

        language = normalize_language(
            artifact.programming_language
            or state["task_spec"].input_artifacts.programming_language
            or ""
        ) or "Unknown"
        if language not in {"Python", "Java", "C++"}:
            report = CodeExecutionReport(
                source_code_hash=source_hash,
                language=language,
                overall_status="unsupported",
                verdict="Unsupported Language",
                failure_reason="目前真实执行仅支持 Python、Java 和 C++。",
            )
            return {
                "code_execution_reports": [*state["code_execution_reports"], report],
                "observations": [
                    *state["observations"],
                    f"Judge0 尚未接入 {language} 的 Harness 生成协议。",
                ],
                "providers": [
                    *state["providers"],
                    f"judge0#{state['iteration']}:unsupported",
                ],
            }

        try:
            if self.code_test_generation_agent is None:
                raise RuntimeError("算法测试生成 Agent 未配置")
            plan, generation_provider = await self.code_test_generation_agent.generate(
                task_spec=state["task_spec"],
                candidate_code=artifact.code,
                language=language,
                solution_context=solution_context,
                on_retry=state["on_retry"],
            )
            report = await self.judge0_code_runner.run(plan)
            providers = [
                *state["providers"],
                f"test-generator#{state['iteration']}:{generation_provider}",
                f"judge0#{state['iteration']}:judge0-sdk",
            ]
        except Exception as error:
            report = CodeExecutionReport(
                source_code_hash=source_hash,
                language=language,
                overall_status="error",
                verdict="Test Generation Error",
                failure_reason=f"{type(error).__name__}: {error}"[:2_000],
            )
            providers = [
                *state["providers"],
                f"judge0#{state['iteration']}:test-generation-error",
            ]

        if report.overall_status == "passed":
            observation = (
                f"Judge0 对源码 {source_hash[:12]}… 的真实执行通过："
                f"{report.passed_tests}/{report.total_tests}，verdict={report.verdict}。"
            )
        else:
            observation = (
                f"Judge0 对源码 {source_hash[:12]}… 的执行未通过："
                f"status={report.overall_status}，verdict={report.verdict}，"
                f"原因={report.failure_reason or '未提供'}。"
            )
        return {
            "code_execution_reports": [*state["code_execution_reports"], report],
            "observations": [*state["observations"], observation],
            "providers": providers,
        }

    async def _memory_node(self, state: AdaptiveRuntimeState) -> dict:
        decision = state["current_decision"]
        updates = decision.memory_updates if decision is not None else []
        await self._progress(
            state["on_progress"],
            "memory_persist",
            "正在更新当前会话的私有记忆",
            "记忆 Agent",
            "保存对后续轮次仍有价值的目标、偏好或约束",
        )
        await self.memory_repository.upsert(
            state["user_id"],
            state["session_id"],
            updates,
            source="memory_agent",
        )
        durable_memory = await self.memory_repository.recall(
            state["user_id"],
            state["session_id"],
            state["task_spec"].normalized_request,
        )
        return {
            "durable_memory": durable_memory,
            "memory_updates": [*state["memory_updates"], *updates],
            "observations": [
                *state["observations"],
                f"已写入 {len(updates)} 条用户私有记忆，动态系统上下文将在下一步刷新。",
            ],
        }

    async def _clarification_node(self, state: AdaptiveRuntimeState) -> dict:
        decision = state["current_decision"]
        await self._progress(
            state["on_progress"],
            "clarification",
            "发现缺少会影响答案的关键信息",
            "澄清 Agent",
            "正在形成最小必要追问",
        )
        result = AgentWorkResult(
            agent="clarification_agent",
            draft_answer=(
                (decision.clarification_question if decision is not None else None)
                or state["task_spec"].clarifying_question
                or "请补充完成任务所需的信息。"
            ),
            uncertainties=state["known_limits"],
            needs_follow_up=True,
        )
        return {
            "latest_work_result": result,
            "observations": [
                *state["observations"],
                "首脑判断必须先澄清信息，本轮停止继续执行。",
            ],
        }

    async def _delegate_node(self, state: AdaptiveRuntimeState) -> dict:
        decision = state["current_decision"]
        if decision is None:
            raise RuntimeError("LangGraph delegate node has no head decision")
        last_instruction = decision.task_instruction or state["task_spec"].normalized_request
        selected_agent = decision.selected_agent or "problem_solving_agent"
        agent_label, activity = self._agent_activity(selected_agent)
        await self._progress(
            state["on_progress"],
            "agent_work",
            activity,
            agent_label,
            "正在读取题面、上下文及前序 Agent 的阶段产物",
        )
        rag_status = state["rag_status"]
        if state["execution_mode"] == "rag_assisted" and any(
            item.collection != "web_search" for item in state["evidence"]
        ):
            rag_status = "hit"
        current_plan = self._build_plan(
            state["task_spec"],
            state["decisions"],
            state["rag_queries"],
            state["web_queries"],
            state["known_limits"],
            state["latest_work_result"],
            last_instruction,
            state["execution_mode"],
            rag_status,
        )
        work_request = AgentWorkRequest(
            task_spec=state["task_spec"],
            coordinator_plan=current_plan,
            conversation_context=self._conversation_context_for_task(
                state["task_spec"],
                state["conversation_context"],
            ),
            memory=state["snapshot"].memory,
            durable_memory=state["durable_memory"],
            dynamic_system_prompt=state["dynamic_prompt"],
            rag_evidence=state["evidence"],
            prior_work_results=state["work_history"],
            code_execution_reports=state["code_execution_reports"],
        )
        result, answer_provider = await self.response_agent.execute(
            work_request,
            state["on_retry"],
        )
        return {
            "last_instruction": last_instruction,
            "latest_work_result": result,
            "work_history": [*state["work_history"], result],
            "rag_status": rag_status,
            "providers": [
                *state["providers"],
                f"worker#{state['iteration']}:{answer_provider}",
            ],
            "observations": [
                *state["observations"],
                "执行 Agent 已返回草稿；首脑必须在下一轮检查是否需要补证据、换 Agent 或结束。",
            ],
        }

    async def _finish_node(self, state: AdaptiveRuntimeState) -> dict:
        decision = state["current_decision"]
        await self._progress(
            state["on_progress"],
            "head_finish",
            "首脑已确认本轮结果可以交付",
            "首脑智能体",
            "准备进入语言润色与格式整理",
        )
        return {
            "observations": [
                *state["observations"],
                (decision.finish_reason if decision is not None else None)
                or "首脑结束本轮执行。",
            ]
        }

    def _build_plan(
        self,
        task_spec: TaskSpec,
        decisions: list[HeadDecision],
        rag_queries: list[RagQuery],
        web_queries: list[str],
        known_limits: list[str],
        work_result: AgentWorkResult | None,
        last_instruction: str,
        execution_mode: str,
        rag_status: str,
    ) -> CoordinatorPlan:
        current_delegation_agent = (
            decisions[-1].selected_agent
            if decisions
            and decisions[-1].action == "delegate"
            and decisions[-1].selected_agent is not None
            else None
        )
        selected_agent = (
            current_delegation_agent
            or (work_result.agent if work_result is not None else None)
            or next(
                (
                    decision.selected_agent
                    for decision in reversed(decisions)
                    if decision.selected_agent is not None
                ),
                "clarification_agent",
            )
        )
        requires_clarification = selected_agent == "clarification_agent"
        clarification = next(
            (
                decision.clarification_question
                for decision in reversed(decisions)
                if decision.clarification_question
            ),
            None,
        )
        if execution_mode == "native_reasoning":
            grounding = (
                "prefer_rag"
                if web_queries or rag_status in {"candidate_found", "insufficient", "hit"}
                else "no_rag"
            )
        else:
            grounding = (
                "require_rag"
                if any(query.required for query in rag_queries)
                else "prefer_rag" if rag_queries or web_queries else "no_rag"
            )
        return CoordinatorPlan(
            objective=task_spec.user_goal,
            selected_agent=selected_agent,
            task_instruction=last_instruction,
            planned_steps=[
                f"{decision.iteration}. {decision.action}：{decision.rationale}"
                for decision in decisions
            ],
            rag_queries=rag_queries,
            memory_selection=MemorySelection(
                working_memory=True,
                long_term_memory=task_spec.context_plan.long_term_memory,
                user_preferences=task_spec.context_plan.user_learning_profile,
                pinned_constraints=True,
                reason="首脑每步使用动态系统上下文，只召回当前任务相关的持久记忆。",
            ),
            execution_mode=execution_mode,
            rag_status=rag_status,
            grounding_policy=grounding,
            requires_clarification=requires_clarification,
            clarification_question=clarification,
            known_limits=list(dict.fromkeys(known_limits)),
            decision_trace=decisions,
            web_search_queries=web_queries,
        )

    @staticmethod
    def _runtime_state(
        decisions: list[HeadDecision],
        observations: list[str],
        evidence: list[RagEvidence],
        work_result: AgentWorkResult | None,
        work_history: list[AgentWorkResult],
        execution_mode: str,
        rag_status: str,
        code_execution_reports: list[CodeExecutionReport],
    ) -> dict:
        latest_code_hash = None
        latest_code_language = None
        for item in reversed(work_history):
            artifact = extract_code_artifact(item.draft_answer)
            if artifact is not None:
                latest_code_hash = hashlib.sha256(
                    artifact.code.encode("utf-8")
                ).hexdigest()
                latest_code_language = artifact.programming_language
                break
        return {
            "execution_mode": execution_mode,
            "rag_status": rag_status,
            "actions_taken": [
                {
                    "iteration": item.iteration,
                    "action": item.action,
                    "selected_agent": item.selected_agent,
                    "rationale": item.rationale,
                    "web_query": item.web_query,
                    "rag_query": (
                        item.rag_query.model_dump()
                        if item.rag_query is not None
                        else None
                    ),
                }
                for item in decisions
            ],
            "observations": observations[-12:],
            "evidence": [
                {
                    "evidence_id": item.evidence_id,
                    "collection": item.collection,
                    "title": item.title,
                    "source_url": item.source_url,
                    "content_excerpt": item.content[:1_200],
                    "metadata": {
                        key: value
                        for key, value in item.metadata.items()
                        if key in {
                            "rank",
                            "source_type",
                            "published_date",
                            "freshness_required",
                            "daily_challenge_date",
                            "verified_daily_challenge",
                            "problem_id",
                            "difficulty",
                            "carried_from_previous_turn",
                        }
                    },
                }
                for item in evidence
            ],
            "latest_work_result": (
                {
                    "agent": work_result.agent,
                    "draft_answer": work_result.draft_answer[:5_000],
                    "used_evidence_ids": work_result.used_evidence_ids,
                    "uncertainties": work_result.uncertainties,
                    "needs_follow_up": work_result.needs_follow_up,
                }
                if work_result is not None
                else None
            ),
            "work_history": [
                {
                    "agent": item.agent,
                    "draft_excerpt": item.draft_answer[:2_500],
                    "used_evidence_ids": item.used_evidence_ids,
                    "uncertainties": item.uncertainties,
                    "needs_follow_up": item.needs_follow_up,
                }
                for item in work_history[-5:]
            ],
            "code_execution_reports": [
                item.model_dump(exclude={"stdout", "stderr", "compile_output"})
                for item in code_execution_reports[-3:]
            ],
            "latest_code": {
                "detected": latest_code_hash is not None,
                "source_code_hash": latest_code_hash,
                "language": latest_code_language,
            },
        }

    @staticmethod
    def _has_user_problem_material(task_spec: TaskSpec) -> bool:
        artifacts = task_spec.input_artifacts
        return bool(
            (artifacts.problem_statement or "").strip()
            or (artifacts.code or "").strip()
            or (artifacts.error_message or "").strip()
            or artifacts.test_cases
        )

    @staticmethod
    def _needs_verification(work_history: list[AgentWorkResult]) -> bool:
        solution_agents = {
            "problem_solving_agent",
            "code_analysis_agent",
            "strategy_agent",
            "solution_review_agent",
            "implementation_agent",
        }
        last_solution = max(
            (
                index
                for index, item in enumerate(work_history)
                if item.agent in solution_agents
            ),
            default=-1,
        )
        last_verification = max(
            (
                index
                for index, item in enumerate(work_history)
                if item.agent == "verification_agent"
            ),
            default=-1,
        )
        return last_solution >= 0 and last_verification < last_solution

    @staticmethod
    def _requires_verification(task_spec: TaskSpec) -> bool:
        return CoordinatorAgent._requires_verification(task_spec)

    @staticmethod
    def _requires_code_execution(
        task_spec: TaskSpec,
        work_history: list[AgentWorkResult],
    ) -> bool:
        if task_spec.primary_intent not in {
            "problem_solving",
            "code_generation",
            "code_diagnosis",
        }:
            return False
        return any(
            extract_code_artifact(item.draft_answer) is not None
            for item in work_history
        )

    @staticmethod
    def _latest_code_report(
        work_history: list[AgentWorkResult],
        reports: list[CodeExecutionReport],
    ) -> CodeExecutionReport | None:
        for item in reversed(work_history):
            artifact = extract_code_artifact(item.draft_answer)
            if artifact is None:
                continue
            source_hash = hashlib.sha256(artifact.code.encode("utf-8")).hexdigest()
            return next(
                (
                    report
                    for report in reversed(reports)
                    if report.source_code_hash == source_hash
                ),
                None,
            )
        return None

    @staticmethod
    def _conversation_context_for_task(
        task_spec: TaskSpec,
        conversation_context: list,
    ) -> list:
        """Only continuation tasks receive old assistant drafts.

        A self-contained request already carries its complete meaning in TaskSpec.
        Supplying prior assistant answers in that case lets an earlier hallucination
        become an apparent fact source for a new task.
        """
        if not task_spec.context_plan.recent_messages:
            return []
        if not task_spec.context_plan.task_state:
            return []
        return conversation_context

    @staticmethod
    def _has_solution_work(work_history: list[AgentWorkResult]) -> bool:
        return any(
            item.agent in {
                "problem_solving_agent",
                "code_analysis_agent",
                "strategy_agent",
                "solution_review_agent",
                "implementation_agent",
            }
            for item in work_history
        )

    @staticmethod
    def _head_progress(
        execution_mode: str,
        rag_status: str,
        work_history: list[AgentWorkResult],
    ) -> tuple[str, str]:
        if work_history:
            agent_label, _ = AdaptiveAgentRuntime._agent_activity(
                work_history[-1].agent
            )
            return (
                f"正在审查{agent_label}的阶段结果",
                "决定继续检索、交给其他专业 Agent、修订还是完成本轮任务",
            )
        if execution_mode == "native_reasoning":
            return (
                "正在规划自主推理解题流程",
                "知识库证据不足，改由专业 Agent 基于用户材料协作",
            )
        if rag_status == "candidate_found":
            return (
                "正在评估知识库候选是否真正相关",
                "核对题目条件、约束与用户目标，避免误用相似但不同的资料",
            )
        if rag_status in {"miss", "unavailable", "error"}:
            return (
                "正在处理知识库未命中",
                "准备切换自主推理，必要时再联网补充资料",
            )
        return (
            "正在决定本轮下一步行动",
            "根据意图、上下文、记忆和可用工具进行任务编排",
        )

    @staticmethod
    def _agent_activity(agent: str) -> tuple[str, str]:
        return {
            "problem_structuring_agent": (
                "题面结构化 Agent",
                "正在整理题意、输入输出、约束和样例",
            ),
            "strategy_agent": (
                "算法策略 Agent",
                "正在设计候选算法并分析正确性与复杂度",
            ),
            "solution_review_agent": (
                "方案评审 Agent",
                "正在比较候选方案并寻找反例或遗漏条件",
            ),
            "implementation_agent": (
                "代码实现 Agent",
                "正在把已选算法落实为可运行代码",
            ),
            "verification_agent": (
                "验证 Agent",
                "正在检查样例、边界条件、复杂度和代码一致性",
            ),
            "problem_solving_agent": (
                "解题 Agent",
                "正在推导算法、证明和实现方案",
            ),
            "code_analysis_agent": (
                "代码分析 Agent",
                "正在定位代码逻辑、错误与复杂度问题",
            ),
            "tutoring_agent": (
                "教学 Agent",
                "正在组织适合当前学习目标的讲解",
            ),
            "learning_planning_agent": (
                "学习规划 Agent",
                "正在制定学习路径与练习安排",
            ),
            "conversation_agent": (
                "对话 Agent",
                "正在结合上下文形成直接回答",
            ),
            "clarification_agent": (
                "澄清 Agent",
                "正在确认完成任务所需的关键信息",
            ),
        }.get(agent, ("专业 Agent", "正在处理当前阶段任务"))

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
    def _fallback_execution_agent(task_spec: TaskSpec) -> str:
        if task_spec.primary_intent in {
            "problem_solving",
            "guided_hint",
            "solution_comparison",
            "mock_interview",
        }:
            return "problem_solving_agent"
        if task_spec.primary_intent == "code_generation":
            return "implementation_agent"
        if task_spec.primary_intent in {"code_diagnosis", "complexity_analysis"}:
            return "code_analysis_agent"
        if task_spec.primary_intent in {"review_planning", "learning_consultation"}:
            return "learning_planning_agent"
        if task_spec.primary_intent in {"concept_explanation", "visual_explanation"}:
            return "tutoring_agent"
        return "conversation_agent"

    @staticmethod
    def _merge_evidence(
        target: list[RagEvidence],
        incoming: list[RagEvidence],
        prefix: str,
    ) -> int:
        existing = {
            (item.collection, item.source_url, item.title)
            for item in target
        }
        added = 0
        for item in incoming:
            identity = (item.collection, item.source_url, item.title)
            if identity in existing:
                continue
            existing.add(identity)
            number = 1 + sum(
                1 for current in target if current.evidence_id.startswith(prefix)
            )
            target.append(item.model_copy(update={"evidence_id": f"{prefix}{number}"}))
            added += 1
        return added
