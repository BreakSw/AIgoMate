import asyncio
import json

import pytest

from app.core.coordinator_agent import CoordinatorAgent
from app.core.adaptive_runtime import AdaptiveAgentRuntime
from app.core.rag_retriever import LocalRagRetriever
from app.core.response_agent import ResponseAgent
from app.models import (
    AgentWorkResult,
    AgentWorkRequest,
    ConversationMessage,
    ContextSnapshot,
    ContextPlan,
    CoordinatorPlan,
    DeliverySpec,
    HeadDecision,
    InputArtifacts,
    MemorySnapshot,
    RagEvidence,
    RagQuery,
    RoutingPlan,
    TaskSpec,
)


def make_task_spec() -> TaskSpec:
    return TaskSpec(
        primary_intent="concept_explanation",
        normalized_request="解释二分查找",
        user_goal="理解二分查找",
        recognition_summary="用户需要概念解释",
        response_mode="step_by_step_explanation",
        delivery=DeliverySpec(assistance_level="explanation_only"),
        routing=RoutingPlan(primary_capability="algorithm_tutoring"),
        context_plan=ContextPlan(algorithm_knowledge=True),
        confidence=0.95,
    )


def test_local_rag_retriever_reads_only_selected_library(tmp_path) -> None:
    manifest = tmp_path / "rag-data/raw/algorithm-concepts/programmercarl/manifest.jsonl"
    markdown = tmp_path / "rag-data/raw/algorithm-concepts/programmercarl/markdown/binary-search.md"
    manifest.parent.mkdir(parents=True)
    markdown.parent.mkdir(parents=True)
    markdown.write_text("# 二分查找\n\n二分查找适用于有序区间。", encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "source_url": "https://example.test/binary-search",
                "title": "二分查找",
                "category": "array",
                "markdown_file": str(markdown.relative_to(tmp_path)).replace("\\", "/"),
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    retriever = LocalRagRetriever(tmp_path)
    plan = CoordinatorPlan(
        objective="解释二分查找",
        selected_agent="tutoring_agent",
        task_instruction="基于概念库解释",
        rag_queries=[
            RagQuery(
                collection="algorithm_concepts",
                query="二分查找",
                reason="需要概念依据",
                required=True,
            )
        ],
        grounding_policy="require_rag",
    )
    evidence = retriever.retrieve_for_plan(plan)

    assert len(evidence) == 1
    assert evidence[0].evidence_id == "R1"
    assert evidence[0].collection == "algorithm_concepts"
    assert "适用于有序区间" in evidence[0].content


def test_required_rag_without_evidence_returns_guarded_answer() -> None:
    class ModelMustNotBeCalled:
        async def complete_json(self, *args, **kwargs):
            raise AssertionError("required-RAG guard should stop before the model call")

    plan = CoordinatorPlan(
        objective="核实一个算法结论",
        selected_agent="tutoring_agent",
        task_instruction="只在证据充分时回答",
        grounding_policy="require_rag",
    )
    request = AgentWorkRequest(
        task_spec=make_task_spec(),
        coordinator_plan=plan,
        memory=MemorySnapshot(),
        rag_evidence=[],
    )
    result, provider = asyncio.run(ResponseAgent(ModelMustNotBeCalled()).execute(request))

    assert provider == "local-grounding-guard"
    assert result.needs_follow_up is True
    assert "不会凭空给出结论" in result.draft_answer


def make_unseen_problem_task() -> TaskSpec:
    return TaskSpec(
        primary_intent="problem_solving",
        normalized_request="解决用户提供的一道未收录算法题",
        user_goal="得到算法思路、正确性证明、复杂度和 Python 实现",
        recognition_summary="用户提供了完整题面，知识库可能尚未收录",
        input_artifacts=InputArtifacts(
            problem_statement=(
                "给定一个整数数组 nums，统计只在一个连续区间中出现的不同整数个数。"
                "如果某个值在数组中的所有出现位置构成一个连续区间，则它是特殊整数。"
                "返回特殊整数的数量。数组长度在 1 到 100 之间，每个元素在 1 到 100 之间。"
            ),
            test_cases=["nums=[3,3,1,2,2] => 2"],
            programming_language="Python",
        ),
        constraints=["1 <= nums.length <= 100", "1 <= nums[i] <= 100"],
        response_mode="step_by_step_explanation",
        delivery=DeliverySpec(assistance_level="direct_solution", include_code=True),
        routing=RoutingPlan(primary_capability="algorithm_tutoring"),
        context_plan=ContextPlan(recent_messages=True, task_state=True),
        confidence=0.98,
    )


def test_required_rag_miss_with_complete_user_problem_calls_model() -> None:
    class FakeModelClient:
        def __init__(self) -> None:
            self.calls = 0

        async def complete_json(self, *_args, **_kwargs):
            self.calls += 1
            return json.dumps({
                "protocol_version": "1.0",
                "agent": "strategy_agent",
                "draft_answer": "可以扫描每个值的首尾位置，并验证中间区间是否全部等于该值。",
                "used_evidence_ids": [],
                "uncertainties": [],
                "needs_follow_up": False,
            }, ensure_ascii=False), "fake-model"

    client = FakeModelClient()
    request = AgentWorkRequest(
        task_spec=make_unseen_problem_task(),
        coordinator_plan=CoordinatorPlan(
            objective="解决未收录题目",
            selected_agent="strategy_agent",
            task_instruction="基于用户题面独立设计算法",
            execution_mode="native_reasoning",
            rag_status="miss",
            grounding_policy="require_rag",
        ),
        memory=MemorySnapshot(),
        rag_evidence=[],
    )

    result, provider = asyncio.run(ResponseAgent(client).execute(request))

    assert client.calls == 1
    assert provider == "fake-model"
    assert result.agent == "strategy_agent"
    assert result.used_evidence_ids == []


def test_required_rag_miss_with_user_code_calls_model() -> None:
    class FakeModelClient:
        async def complete_json(self, *_args, **_kwargs):
            return json.dumps({
                "protocol_version": "1.0",
                "agent": "code_analysis_agent",
                "draft_answer": "循环边界需要改为 range(len(nums))。",
                "used_evidence_ids": [],
                "uncertainties": [],
                "needs_follow_up": False,
            }, ensure_ascii=False), "fake-model"

    task = make_unseen_problem_task().model_copy(update={
        "input_artifacts": InputArtifacts(
            code="def solve(nums):\n    for i in range(len(nums) + 1):\n        print(nums[i])",
            error_message="IndexError: list index out of range",
        )
    })
    request = AgentWorkRequest(
        task_spec=task,
        coordinator_plan=CoordinatorPlan(
            objective="修复用户代码",
            selected_agent="code_analysis_agent",
            task_instruction="仅依据用户代码诊断",
            execution_mode="native_reasoning",
            rag_status="miss",
            grounding_policy="require_rag",
        ),
        memory=MemorySnapshot(),
    )

    result, provider = asyncio.run(ResponseAgent(FakeModelClient()).execute(request))

    assert provider == "fake-model"
    assert result.needs_follow_up is False


def test_only_problem_number_without_statement_still_uses_grounding_guard() -> None:
    class ModelMustNotBeCalled:
        async def complete_json(self, *args, **kwargs):
            raise AssertionError("题号本身不是完整用户题面")

    task = make_unseen_problem_task().model_copy(update={
        "input_artifacts": InputArtifacts(problem_statement="LeetCode 4038")
    })
    request = AgentWorkRequest(
        task_spec=task,
        coordinator_plan=CoordinatorPlan(
            objective="解释题号对应题目",
            selected_agent="problem_solving_agent",
            task_instruction="证据充分时回答",
            grounding_policy="require_rag",
        ),
        memory=MemorySnapshot(),
    )

    result, provider = asyncio.run(ResponseAgent(ModelMustNotBeCalled()).execute(request))

    assert provider == "local-grounding-guard"
    assert result.needs_follow_up is True


def test_native_answer_cannot_forge_rag_evidence_id() -> None:
    request = AgentWorkRequest(
        task_spec=make_unseen_problem_task(),
        coordinator_plan=CoordinatorPlan(
            objective="解决未收录题目",
            selected_agent="strategy_agent",
            task_instruction="独立推理",
            execution_mode="native_reasoning",
            rag_status="miss",
        ),
        memory=MemorySnapshot(),
    )
    forged = AgentWorkResult(
        agent="strategy_agent",
        draft_answer="这是没有来源的伪造结论 [R1]。",
        used_evidence_ids=["R1"],
    )

    with pytest.raises(ValueError, match="不存在的 RAG 证据编号"):
        ResponseAgent(object())._validate_result(forged, request)


def test_head_requires_rag_then_allows_native_reasoning_switch() -> None:
    task = make_unseen_problem_task()
    delegate = HeadDecision(
        iteration=1,
        rationale="直接求解",
        action="delegate",
        selected_agent="strategy_agent",
        task_instruction="生成算法",
    )

    with pytest.raises(ValueError, match="必须先检索 RAG"):
        CoordinatorAgent._validate(delegate, {
            "actions_taken": [],
            "execution_mode": "rag_assisted",
            "rag_status": "not_checked",
        }, task, True)

    after_miss = {
        "actions_taken": [{"iteration": 1, "action": "retrieve_rag"}],
        "execution_mode": "rag_assisted",
        "rag_status": "miss",
    }
    with pytest.raises(ValueError, match="必须先切换到自主推理"):
        CoordinatorAgent._validate(delegate, after_miss, task, True)

    switch = HeadDecision(
        iteration=2,
        rationale="RAG 未命中，使用用户题面继续推理",
        action="switch_to_native_reasoning",
    )
    CoordinatorAgent._validate(switch, after_miss, task, True)

    native_state = {
        "actions_taken": [
            {"iteration": 1, "action": "retrieve_rag"},
            {"iteration": 2, "action": "switch_to_native_reasoning"},
        ],
        "execution_mode": "native_reasoning",
        "rag_status": "miss",
    }
    CoordinatorAgent._validate(delegate, native_state, task, True)

    finish = HeadDecision(
        iteration=5,
        rationale="结束",
        action="finish",
        finish_reason="已经验证",
    )
    verified_state = {
        **native_state,
        "latest_work_result": {"needs_follow_up": False},
        "work_history": [
            {"agent": "strategy_agent"},
            {"agent": "verification_agent"},
        ],
    }
    CoordinatorAgent._validate(finish, verified_state, task, True)

    stale_verification_state = {
        **verified_state,
        "work_history": [
            {"agent": "strategy_agent"},
            {"agent": "verification_agent"},
            {"agent": "implementation_agent"},
        ],
    }
    with pytest.raises(ValueError, match="必须重新调用 verification_agent"):
        CoordinatorAgent._validate(finish, stale_verification_state, task, True)


def test_runtime_rag_miss_switches_to_head_controlled_multi_agent_reasoning() -> None:
    class FakeCoordinator:
        def __init__(self) -> None:
            self.decisions = [
                HeadDecision(
                    iteration=1,
                    rationale="先检查题库",
                    action="retrieve_rag",
                    rag_query=RagQuery(
                        collection="problem_bank",
                        query="统计只在连续区间出现的整数",
                        reason="默认优先检查题库",
                        required=True,
                    ),
                ),
                HeadDecision(
                    iteration=2,
                    rationale="题库没有命中",
                    action="switch_to_native_reasoning",
                ),
                HeadDecision(
                    iteration=3,
                    rationale="独立设计算法",
                    action="delegate",
                    selected_agent="strategy_agent",
                    task_instruction="根据用户题面设计并证明算法",
                ),
                HeadDecision(
                    iteration=4,
                    rationale="独立检查方案",
                    action="delegate",
                    selected_agent="verification_agent",
                    task_instruction="审查前序方案并返回完整答案",
                ),
                HeadDecision(
                    iteration=5,
                    rationale="已经完成验证",
                    action="finish",
                    finish_reason="解法已经通过独立检查",
                ),
            ]

        async def decide(self, *_args, **_kwargs):
            return self.decisions.pop(0), "fake-head"

    class EmptyRetriever:
        @staticmethod
        def availability():
            return {"problem_bank": True, "algorithm_concepts": True, "code_cases": True}

        @staticmethod
        def retrieve(_query):
            return []

    class NoWeb:
        @staticmethod
        def available():
            return False

    class FakeResponseAgent:
        def __init__(self) -> None:
            self.requests = []

        async def execute(self, request, _on_retry=None):
            self.requests.append(request)
            agent = request.coordinator_plan.selected_agent
            answer = (
                "候选方案：记录每个值的首尾位置。"
                if agent == "strategy_agent"
                else "验证通过。完整解法、复杂度与 Python 代码均已整理。"
            )
            return AgentWorkResult(agent=agent, draft_answer=answer), "fake-worker"

    class NoopMemoryRepository:
        pass

    class PromptBuilder:
        @staticmethod
        def build(*_args):
            return ""

    response_agent = FakeResponseAgent()
    runtime = AdaptiveAgentRuntime(
        FakeCoordinator(),
        response_agent,
        EmptyRetriever(),
        NoWeb(),
        object(),
        NoopMemoryRepository(),
        PromptBuilder(),
        max_iterations=5,
    )
    result = asyncio.run(runtime.run(
        user_id=1,
        session_id=99,
        task_spec=make_unseen_problem_task(),
        snapshot=ContextSnapshot.model_construct(memory=MemorySnapshot()),
        conversation_context=[],
        durable_memory=[],
    ))
    plan, evidence, work_result, *_rest = result

    assert plan.execution_mode == "native_reasoning"
    assert plan.rag_status == "miss"
    assert plan.grounding_policy == "no_rag"
    assert evidence == []
    assert work_result.agent == "verification_agent"
    assert [item.coordinator_plan.selected_agent for item in response_agent.requests] == [
        "strategy_agent",
        "verification_agent",
    ]
    assert response_agent.requests[0].prior_work_results == []
    assert response_agent.requests[1].prior_work_results[0].agent == "strategy_agent"


def test_duplicate_rag_query_is_rejected() -> None:
    task = make_task_spec()
    duplicate = HeadDecision(
        iteration=2,
        rationale="重复检索",
        action="retrieve_rag",
        rag_query=RagQuery(
            collection="problem_bank",
            query="  动态规划   困难  ",
            reason="再次查找",
        ),
    )
    state = {
        "actions_taken": [{
            "iteration": 1,
            "action": "retrieve_rag",
            "rag_query": {
                "collection": "problem_bank",
                "query": "动态规划 困难",
            },
        }],
        "execution_mode": "rag_assisted",
        "rag_status": "candidate_found",
    }

    with pytest.raises(ValueError, match="禁止重复相同 RAG 查询"):
        CoordinatorAgent._validate(duplicate, state, task, True)


def test_self_contained_task_does_not_receive_old_assistant_drafts() -> None:
    context = [
        {"role": "user", "content": "上一轮用户请求"},
        {"role": "assistant", "content": "未经核验的旧题目"},
    ]
    self_contained = make_task_spec().model_copy(update={
        "context_plan": ContextPlan(recent_messages=True, task_state=False),
    })
    continuation = make_task_spec().model_copy(update={
        "context_plan": ContextPlan(recent_messages=True, task_state=True),
    })

    assert AdaptiveAgentRuntime._conversation_context_for_task(
        self_contained,
        context,
    ) == []
    assert AdaptiveAgentRuntime._conversation_context_for_task(
        continuation,
        context,
    ) == context


def test_head_protocol_exhaustion_falls_back_to_reverify_continuation() -> None:
    class InvalidHeadModel:
        async def complete_json(self, *_args, **_kwargs):
            return "{}", "fake-head"

    task = make_task_spec().model_copy(update={
        "primary_intent": "code_generation",
        "normalized_request": "为上一轮列出的每道题提供 C++ 代码",
        "user_goal": "获得上一轮四道周赛题的 C++ 实现",
        "delivery": DeliverySpec(
            assistance_level="direct_solution",
            include_code=True,
        ),
        "context_plan": ContextPlan(recent_messages=True, task_state=True),
    })
    context = [
        ConversationMessage(role="assistant", content="上一轮列出了周赛题号与标题"),
        ConversationMessage(role="user", content="我需要每道题的 C++ 代码"),
    ]
    coordinator = CoordinatorAgent(InvalidHeadModel(), max_reflection_rounds=1)

    decision, provider = asyncio.run(coordinator.decide(
        task_spec=task,
        snapshot=ContextSnapshot.model_construct(memory=MemorySnapshot()),
        runtime_state={
            "actions_taken": [],
            "execution_mode": "rag_assisted",
            "rag_status": "not_checked",
            "evidence": [],
            "work_history": [],
        },
        knowledge_availability={},
        web_search_available=True,
        dynamic_system_prompt="",
        iteration=1,
        conversation_context=context,
    ))

    assert decision.action == "search_web"
    assert "上一轮列出了周赛题号" in (decision.web_query or "")
    assert provider.endswith("+protocol-fallback")


def test_continuation_reuses_previous_tool_evidence_not_assistant_text() -> None:
    continuation = make_task_spec().model_copy(update={
        "primary_intent": "code_generation",
        "context_plan": ContextPlan(recent_messages=True, task_state=True),
    })
    previous = RagEvidence(
        evidence_id="W1",
        collection="web_search",
        title="LeetCode 官方题面",
        content="官方题面、约束和函数签名",
        source_url="https://leetcode.cn/problems/example/",
        score=1.0,
        metadata={"source_type": "leetcode_official"},
    )

    class FakeCoordinator:
        def __init__(self) -> None:
            self.decisions = [
                HeadDecision(
                    iteration=1,
                    rationale="已有官方题面",
                    action="delegate",
                    selected_agent="implementation_agent",
                    task_instruction="依据官方题面生成 C++",
                ),
                HeadDecision(
                    iteration=2,
                    rationale="实现完成",
                    action="finish",
                    finish_reason="完成",
                ),
            ]

        async def decide(self, *_args, **_kwargs):
            return self.decisions.pop(0), "fake-head"

    class Retriever:
        @staticmethod
        def availability():
            return {}

    class NoWeb:
        @staticmethod
        def available():
            return False

    class CapturingResponseAgent:
        def __init__(self) -> None:
            self.request = None

        async def execute(self, request, _on_retry=None):
            self.request = request
            return AgentWorkResult(
                agent="implementation_agent",
                draft_answer="C++ 实现",
                used_evidence_ids=["W1"],
            ), "fake-worker"

    class PromptBuilder:
        @staticmethod
        def build(*_args):
            return ""

    response = CapturingResponseAgent()
    runtime = AdaptiveAgentRuntime(
        FakeCoordinator(),
        response,
        Retriever(),
        NoWeb(),
        object(),
        object(),
        PromptBuilder(),
        max_iterations=2,
    )
    result = asyncio.run(runtime.run(
        user_id=1,
        session_id=1,
        task_spec=continuation,
        snapshot=ContextSnapshot.model_construct(memory=MemorySnapshot()),
        conversation_context=[
            ConversationMessage(role="assistant", content="未经核验的旧回答"),
        ],
        durable_memory=[],
        previous_turn_evidence=[previous],
    ))

    assert result[1][0].metadata["carried_from_previous_turn"] is True
    assert response.request.rag_evidence[0].content == "官方题面、约束和函数签名"
    assert response.request.conversation_context[0].content == "未经核验的旧回答"


def test_code_generation_fallback_routes_to_implementation_agent() -> None:
    task = make_task_spec().model_copy(update={"primary_intent": "code_generation"})

    assert AdaptiveAgentRuntime._fallback_execution_agent(task) == "implementation_agent"
    assert CoordinatorAgent._fallback_execution_agent(task) == "implementation_agent"


def test_contest_code_continuation_requires_solution_source_search() -> None:
    task = make_task_spec().model_copy(update={
        "primary_intent": "code_generation",
        "context_plan": ContextPlan(recent_messages=True, task_state=True),
    })
    delegate = HeadDecision(
        iteration=1,
        rationale="官方题面足够",
        action="delegate",
        selected_agent="implementation_agent",
        task_instruction="直接生成代码",
    )
    state = {
        "actions_taken": [],
        "execution_mode": "rag_assisted",
        "rag_status": "not_checked",
        "evidence": [{
            "collection": "web_search",
            "title": "LeetCode 官方题面",
            "metadata": {"source_type": "leetcode_official"},
        }],
    }

    with pytest.raises(ValueError, match="请先 search_web"):
        CoordinatorAgent._validate(delegate, state, task, True)

    searched = {
        **state,
        "actions_taken": [{
            "iteration": 1,
            "action": "search_web",
            "web_query": "题号 灵茶山艾府 题解 C++",
        }],
    }
    CoordinatorAgent._validate(delegate, searched, task, True)


def test_rag_assisted_code_must_be_verified_after_latest_implementation() -> None:
    task = make_task_spec().model_copy(update={
        "primary_intent": "code_generation",
        "context_plan": ContextPlan(recent_messages=True, task_state=False),
    })
    finish = HeadDecision(
        iteration=3,
        rationale="代码已生成",
        action="finish",
        finish_reason="准备交付",
    )
    stale = {
        "actions_taken": [],
        "execution_mode": "rag_assisted",
        "rag_status": "hit",
        "latest_work_result": {"needs_follow_up": False},
        "work_history": [{"agent": "implementation_agent"}],
        "evidence": [{"collection": "code_cases", "metadata": {}}],
    }

    with pytest.raises(ValueError, match="必须先由 verification_agent"):
        CoordinatorAgent._validate(finish, stale, task, True)

    verified = {
        **stale,
        "work_history": [
            {"agent": "implementation_agent"},
            {"agent": "verification_agent"},
        ],
    }
    CoordinatorAgent._validate(finish, verified, task, True)


def test_native_generated_problem_is_forced_through_verification() -> None:
    generated_task = TaskSpec(
        primary_intent="problem_solving",
        normalized_request="出两道动态规划难题",
        user_goal="获得两道动态规划练习题",
        recognition_summary="用户请求生成算法练习题",
        response_mode="direct_answer",
        delivery=DeliverySpec(assistance_level="direct_solution"),
        routing=RoutingPlan(primary_capability="algorithm_tutoring"),
        context_plan=ContextPlan(recent_messages=True, task_state=False),
        confidence=0.95,
    )

    class FakeCoordinator:
        def __init__(self) -> None:
            self.decisions = [
                HeadDecision(
                    iteration=1,
                    rationale="题库不足，转为自拟题",
                    action="retrieve_rag",
                    rag_query=RagQuery(
                        collection="problem_bank",
                        query="动态规划 困难",
                        reason="寻找真实题目",
                    ),
                ),
                HeadDecision(
                    iteration=2,
                    rationale="题库未命中",
                    action="switch_to_native_reasoning",
                ),
                HeadDecision(
                    iteration=3,
                    rationale="生成自拟题",
                    action="delegate",
                    selected_agent="problem_solving_agent",
                    task_instruction="生成并标注两道自拟题",
                ),
                HeadDecision(
                    iteration=4,
                    rationale="草稿已生成",
                    action="finish",
                    finish_reason="准备交付",
                ),
            ]

        async def decide(self, *_args, **_kwargs):
            return self.decisions.pop(0), "fake-head"

    class EmptyRetriever:
        @staticmethod
        def availability():
            return {"problem_bank": True, "algorithm_concepts": True, "code_cases": True}

        @staticmethod
        def retrieve(_query):
            return []

    class NoWeb:
        @staticmethod
        def available():
            return False

    class FakeResponseAgent:
        def __init__(self) -> None:
            self.requests = []

        async def execute(self, request, _on_retry=None):
            self.requests.append(request)
            agent = request.coordinator_plan.selected_agent
            return AgentWorkResult(
                agent=agent,
                draft_answer=(
                    "两道自拟题草稿"
                    if agent == "problem_solving_agent"
                    else "已重算样例并返回修正后的两道自拟题"
                ),
            ), "fake-worker"

    class PromptBuilder:
        @staticmethod
        def build(*_args):
            return ""

    response_agent = FakeResponseAgent()
    runtime = AdaptiveAgentRuntime(
        FakeCoordinator(),
        response_agent,
        EmptyRetriever(),
        NoWeb(),
        object(),
        object(),
        PromptBuilder(),
        max_iterations=4,
    )

    result = asyncio.run(runtime.run(
        user_id=1,
        session_id=100,
        task_spec=generated_task,
        snapshot=ContextSnapshot.model_construct(memory=MemorySnapshot()),
        conversation_context=[{"role": "assistant", "content": "旧的错误题目"}],
        durable_memory=[],
    ))

    work_result = result[2]
    assert work_result.agent == "verification_agent"
    assert [item.coordinator_plan.selected_agent for item in response_agent.requests] == [
        "problem_solving_agent",
        "verification_agent",
    ]
    assert response_agent.requests[0].conversation_context == []


def test_duplicate_rag_results_remain_available_after_native_switch() -> None:
    from app.models import RagEvidence

    target: list[RagEvidence] = []
    candidate = RagEvidence(
        evidence_id="source",
        collection="problem_bank",
        title="真实动态规划难题",
        content="题目正文与约束",
        score=0.91,
    )

    first_added = AdaptiveAgentRuntime._merge_evidence(target, [candidate], "R")
    second_added = AdaptiveAgentRuntime._merge_evidence(target, [candidate], "R")

    assert first_added == 1
    assert second_added == 0
    assert len(target) == 1
    assert target[0].evidence_id == "R1"
