import asyncio

from app.core.coordinator_agent import CoordinatorAgent
from app.core.current_time_tool import CurrentTimeTool
from app.core.memory_store import DynamicSystemPromptBuilder, UserMemoryRepository
from app.core.model_client import RetryCallback
from app.core.progress_status import ProgressCallback
from app.core.rag_retriever import LocalRagRetriever
from app.core.response_agent import ResponseAgent
from app.core.web_search_agent import WebSearchAgent
from app.models import (
    AgentWorkRequest,
    AgentWorkResult,
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


class AdaptiveAgentRuntime:
    """Model-directed agent loop. The head model chooses every next action."""

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
    ) -> None:
        self.coordinator = coordinator
        self.response_agent = response_agent
        self.rag_retriever = rag_retriever
        self.web_search_agent = web_search_agent
        self.current_time_tool = current_time_tool
        self.memory_repository = memory_repository
        self.prompt_builder = prompt_builder
        self.max_iterations = max_iterations

    async def run(
        self,
        *,
        user_id: int,
        session_id: int,
        task_spec: TaskSpec,
        snapshot: ContextSnapshot,
        conversation_context: list[ConversationMessage],
        durable_memory: list[DurableMemoryItem],
        on_retry: RetryCallback | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> tuple[
        CoordinatorPlan,
        list[RagEvidence],
        AgentWorkResult,
        list[DurableMemoryItem],
        list[MemoryUpdate],
        list[str],
    ]:
        decisions: list[HeadDecision] = []
        observations: list[str] = []
        evidence: list[RagEvidence] = []
        rag_queries: list[RagQuery] = []
        web_queries: list[str] = []
        known_limits: list[str] = []
        memory_updates: list[MemoryUpdate] = []
        providers: list[str] = []
        latest_work_result: AgentWorkResult | None = None
        work_history: list[AgentWorkResult] = []
        execution_mode = "rag_assisted"
        rag_status = "not_checked"
        last_instruction = task_spec.normalized_request

        for iteration in range(1, self.max_iterations + 1):
            head_message, head_detail = self._head_progress(
                execution_mode,
                rag_status,
                work_history,
            )
            await self._progress(
                on_progress,
                "head_decision",
                head_message,
                "首脑智能体",
                head_detail,
            )
            dynamic_prompt = self.prompt_builder.build(
                user_id,
                session_id,
                snapshot.memory,
                durable_memory,
            )
            runtime_state = self._runtime_state(
                decisions,
                observations,
                evidence,
                latest_work_result,
                work_history,
                execution_mode,
                rag_status,
            )
            runtime_state["iteration_budget"] = {
                "current": iteration,
                "maximum": self.max_iterations,
                "remaining_after_this_decision": self.max_iterations - iteration,
            }
            decision, provider = await self.coordinator.decide(
                task_spec,
                snapshot,
                runtime_state,
                self.rag_retriever.availability(),
                self.web_search_agent.available(),
                dynamic_prompt,
                iteration,
                on_retry,
            )
            decisions.append(decision)
            providers.append(f"head#{iteration}:{provider}")

            if decision.action == "get_current_time":
                await self._progress(
                    on_progress,
                    "time_tool",
                    "正在读取应用当前时间",
                    "时间工具",
                    "把相对日期转换为可检索的绝对日期",
                )
                time_result = self.current_time_tool.read()
                observations.append(
                    "时间工具返回："
                    f"当前日期时间 {time_result['local_datetime']}，"
                    f"日期 {time_result['chinese_date']}，"
                    f"星期 {time_result['weekday']}，"
                    f"时区 {time_result['timezone']}（UTC{time_result['utc_offset']}）。"
                    "后续涉及相对日期的检索必须使用该绝对日期。"
                )
                providers.append(f"time#{iteration}:local")
                continue

            if decision.action == "retrieve_rag":
                query = decision.rag_query
                if query is None:
                    observations.append("首脑请求 RAG，但没有给出合法查询。")
                    continue
                rag_queries.append(query)
                collection_label = {
                    "algorithm_concepts": "算法概念库",
                    "problem_bank": "题库",
                    "code_cases": "代码案例库",
                }.get(query.collection, query.collection)
                await self._progress(
                    on_progress,
                    "rag_retrieval",
                    f"正在检索{collection_label}",
                    "RAG 检索 Agent",
                    "查找与本轮题目或算法最相关的候选内容",
                )
                if not self.rag_retriever.availability().get(query.collection, False):
                    limit = f"{query.collection} 当前不可用"
                    observations.append(limit)
                    known_limits.append(limit)
                    rag_status = "unavailable"
                    continue
                try:
                    found = await asyncio.to_thread(
                        self.rag_retriever.retrieve,
                        query,
                    )
                except Exception:
                    limit = f"{query.collection} 检索失败"
                    observations.append(limit)
                    known_limits.append(limit)
                    rag_status = "error"
                    continue
                added = self._merge_evidence(evidence, found, prefix="R")
                rag_status = "candidate_found" if added else "miss"
                if added:
                    observations.append(
                        f"{query.collection} 检索完成，新增 {added} 条候选证据；"
                        "首脑仍需判断其是否真正覆盖用户题面，候选不等于有效命中。"
                    )
                else:
                    observations.append(
                        f"{query.collection} 检索完成，但没有找到可用候选。"
                    )
                continue

            if decision.action == "switch_to_native_reasoning":
                await self._progress(
                    on_progress,
                    "native_reasoning",
                    "知识库证据不足，切换自主推理",
                    "首脑智能体",
                    "后续由专业 Agent 基于用户题面协作解题，仍可按需联网",
                )
                ignored_rag_count = sum(
                    1 for item in evidence if item.collection != "web_search"
                )
                evidence[:] = [
                    item for item in evidence if item.collection == "web_search"
                ]
                execution_mode = "native_reasoning"
                if rag_status == "candidate_found":
                    rag_status = "insufficient"
                observations.append(
                    "已切换到自主推理模式：后续不依赖 RAG，"
                    f"已隔离 {ignored_rag_count} 条未被首脑认可的候选；"
                    "可以联网补充资料，联网失败时仍必须根据用户题面继续推理。"
                )
                providers.append(f"mode#{iteration}:native-reasoning")
                continue

            if decision.action == "search_web":
                query = decision.web_query or ""
                reason = decision.web_search_reason or "补充外部信息"
                web_queries.append(query)
                await self._progress(
                    on_progress,
                    "web_search",
                    "正在搜索可核验的外部资料",
                    "网页搜索 Agent",
                    reason,
                )
                if not self.web_search_agent.available():
                    limit = "网页搜索 Agent 当前不可用"
                    observations.append(limit)
                    known_limits.append(limit)
                    continue
                found, web_provider = await self.web_search_agent.search(
                    query,
                    reason,
                    on_retry,
                )
                providers.append(f"web#{iteration}:{web_provider}")
                added = self._merge_evidence(evidence, found, prefix="W")
                if added:
                    observations.append(f"网页搜索完成，新增 {added} 条证据。")
                elif "-error:" in web_provider:
                    limit = "网页搜索请求失败，本轮未能补充外部证据。"
                    observations.append(limit)
                    known_limits.append(limit)
                else:
                    observations.append("网页搜索完成，但没有返回可用结果。")
                continue

            if decision.action == "persist_memory":
                await self._progress(
                    on_progress,
                    "memory_persist",
                    "正在更新当前会话的私有记忆",
                    "记忆 Agent",
                    "保存对后续轮次仍有价值的目标、偏好或约束",
                )
                memory_updates.extend(decision.memory_updates)
                await self.memory_repository.upsert(
                    user_id,
                    session_id,
                    decision.memory_updates,
                    source="memory_agent",
                )
                durable_memory = await self.memory_repository.recall(
                    user_id,
                    session_id,
                    task_spec.normalized_request,
                )
                observations.append(
                    f"已写入 {len(decision.memory_updates)} 条用户私有记忆，动态系统上下文将在下一步刷新。"
                )
                continue

            if decision.action == "ask_clarification":
                await self._progress(
                    on_progress,
                    "clarification",
                    "发现缺少会影响答案的关键信息",
                    "澄清 Agent",
                    "正在形成最小必要追问",
                )
                latest_work_result = AgentWorkResult(
                    agent="clarification_agent",
                    draft_answer=(
                        decision.clarification_question
                        or task_spec.clarifying_question
                        or "请补充完成任务所需的信息。"
                    ),
                    uncertainties=known_limits,
                    needs_follow_up=True,
                )
                observations.append("首脑判断必须先澄清信息，本轮停止继续执行。")
                break

            if decision.action == "delegate":
                last_instruction = decision.task_instruction or task_spec.normalized_request
                agent_label, activity = self._agent_activity(
                    decision.selected_agent or "problem_solving_agent"
                )
                await self._progress(
                    on_progress,
                    "agent_work",
                    activity,
                    agent_label,
                    "正在读取题面、上下文及前序 Agent 的阶段产物",
                )
                if execution_mode == "rag_assisted" and any(
                    item.collection != "web_search" for item in evidence
                ):
                    rag_status = "hit"
                current_plan = self._build_plan(
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
                work_request = AgentWorkRequest(
                    task_spec=task_spec,
                    coordinator_plan=current_plan,
                    conversation_context=(
                        conversation_context
                        if task_spec.context_plan.recent_messages
                        else []
                    ),
                    memory=snapshot.memory,
                    durable_memory=durable_memory,
                    dynamic_system_prompt=dynamic_prompt,
                    rag_evidence=evidence,
                    prior_work_results=work_history,
                )
                latest_work_result, answer_provider = await self.response_agent.execute(
                    work_request,
                    on_retry,
                )
                providers.append(f"worker#{iteration}:{answer_provider}")
                work_history.append(latest_work_result)
                observations.append(
                    "执行 Agent 已返回草稿；首脑必须在下一轮检查是否需要补证据、换 Agent 或结束。"
                )
                continue

            if decision.action == "finish":
                await self._progress(
                    on_progress,
                    "head_finish",
                    "首脑已确认本轮结果可以交付",
                    "首脑智能体",
                    "准备进入语言润色与格式整理",
                )
                observations.append(decision.finish_reason or "首脑结束本轮执行。")
                break

        if latest_work_result is None:
            user_material_available = self._has_user_problem_material(task_spec)
            if evidence or user_material_available:
                if user_material_available and rag_status != "hit":
                    execution_mode = "native_reasoning"
                    evidence[:] = [
                        item for item in evidence if item.collection == "web_search"
                    ]
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
                    conversation_context=(
                        conversation_context
                        if task_spec.context_plan.recent_messages
                        else []
                    ),
                    memory=snapshot.memory,
                    durable_memory=durable_memory,
                    dynamic_system_prompt=dynamic_prompt,
                    rag_evidence=evidence,
                    prior_work_results=work_history,
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
                conversation_context=(
                    conversation_context
                    if task_spec.context_plan.recent_messages
                    else []
                ),
                memory=snapshot.memory,
                durable_memory=durable_memory,
                dynamic_system_prompt=dynamic_prompt,
                rag_evidence=evidence,
                prior_work_results=work_history,
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
            execution_mode == "native_reasoning"
            and self._has_user_problem_material(task_spec)
            and latest_work_result.agent != "clarification_agent"
            and self._needs_verification(work_history)
        ):
            last_instruction = (
                "决策轮次已结束，但自主推理解题必须完成独立验证。阅读 prior_work_results，"
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
                conversation_context=(
                    conversation_context
                    if task_spec.context_plan.recent_messages
                    else []
                ),
                memory=snapshot.memory,
                durable_memory=durable_memory,
                dynamic_system_prompt=dynamic_prompt,
                rag_evidence=evidence,
                prior_work_results=work_history,
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
        )

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
            grounding = "prefer_rag" if web_queries else "no_rag"
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
    ) -> dict:
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
        if task_spec.primary_intent in {
            "code_generation",
            "code_diagnosis",
            "complexity_analysis",
        }:
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
