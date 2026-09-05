import asyncio
import hashlib
import json

import judge0

from app.core.code_test_generation_agent import CodeTestGenerationAgent
from app.core.coordinator_agent import CoordinatorAgent
from app.core.judge0_code_runner import Judge0CodeRunner
from app.models import (
    CodeExecutionReport,
    CodeTestPlan,
    ContextPlan,
    DeliverySpec,
    HeadDecision,
    RoutingPlan,
    TaskSpec,
)


CANDIDATE = "def add(a, b):\n    return a + b"


def make_plan() -> CodeTestPlan:
    source_hash = hashlib.sha256(CANDIDATE.encode("utf-8")).hexdigest()
    return CodeTestPlan(
        language="Python",
        source_code_hash=source_hash,
        executable_source=(
            CANDIDATE
            + "\nassert add(20, 22) == 42\n"
            + "print('ALL_TESTS_PASSED 2/2')\n"
        ),
        expected_output="ALL_TESTS_PASSED 2/2\n",
        test_count=2,
        test_categories=["典型值", "负数边界"],
        oracle_strategy="人工可确认的整数加法期望值",
    )


def make_task() -> TaskSpec:
    return TaskSpec(
        primary_intent="code_generation",
        normalized_request="实现加法函数",
        user_goal="获得经过执行验证的实现",
        recognition_summary="生成并验证代码",
        response_mode="direct_answer",
        delivery=DeliverySpec(assistance_level="direct_solution", include_code=True),
        routing=RoutingPlan(primary_capability="code_sandbox"),
        context_plan=ContextPlan(task_state=False),
        confidence=0.99,
    )


def test_test_plan_is_bound_to_exact_candidate_source() -> None:
    plan = make_plan()
    CodeTestGenerationAgent._validate(
        plan,
        CANDIDATE,
        hashlib.sha256(CANDIDATE.encode("utf-8")).hexdigest(),
    )


def test_test_generation_uses_semantic_reflection_before_approval() -> None:
    plan = make_plan()
    dimensions = [
        "candidate_integrity",
        "candidate_invocation",
        "oracle_independence",
        "oracle_correctness",
        "constraint_compliance",
        "edge_coverage",
        "language_compilability",
        "output_protocol",
    ]
    revise = {
        "protocol_version": "1.0",
        "verdict": "revise",
        "summary": "负数边界已声明但 Harness 中尚未实际执行",
        "checked_dimensions": dimensions,
        "issues": [{
            "dimension": "edge_coverage",
            "problem": "test_count 与实际覆盖意图不一致",
            "impact": "可能漏掉负数边界错误",
        }],
        "revision_instructions": [{
            "action": "增加负数加法断言并保持总测试数为2",
        }],
        "confidence": 0.91,
    }
    approved = {
        "protocol_version": "1.0",
        "verdict": "approved",
        "summary": "候选代码被原样调用，Oracle 独立且边界覆盖一致",
        "checked_dimensions": dimensions,
        "issues": [],
        "revision_instructions": [],
        "confidence": 0.96,
    }

    class FakeModelClient:
        def __init__(self):
            self.outputs = [
                plan.model_dump_json(),
                json.dumps(revise, ensure_ascii=False),
                plan.model_dump_json(),
                json.dumps(approved, ensure_ascii=False),
            ]

        async def complete_json(self, *_args, **_kwargs):
            return self.outputs.pop(0), "fake-model"

    agent = CodeTestGenerationAgent(FakeModelClient(), max_reflection_rounds=2)
    reflected_plan, provider = asyncio.run(agent.generate(
        task_spec=make_task(),
        candidate_code=CANDIDATE,
        language="Python",
        solution_context="实现整数加法并验证边界",
    ))

    assert reflected_plan.semantic_reflection_rounds == 1
    assert reflected_plan.review_confidence == 0.96
    assert "候选代码被原样调用" in reflected_plan.review_summary
    assert "+test-reflection:1" in provider


def test_judge0_runner_normalizes_real_tool_result(monkeypatch) -> None:
    result = judge0.Submission(
        stdout="ALL_TESTS_PASSED 2/2\n",
        status={"id": 3},
        exit_code=0,
        time=0.012,
        memory=1234,
    )

    def fake_run(**kwargs):
        assert kwargs["submissions"].enable_network is False
        assert kwargs["submissions"].cpu_time_limit == 5
        return result

    monkeypatch.setattr(judge0, "run", fake_run)
    report = asyncio.run(Judge0CodeRunner().run(make_plan()))

    assert report.overall_status == "passed"
    assert report.verdict == "Accepted"
    assert report.passed_tests == 2
    assert report.total_tests == 2
    assert report.source_code_hash == make_plan().source_code_hash


def test_head_requires_execution_for_unseen_code_revision() -> None:
    source_hash = make_plan().source_code_hash
    runtime_state = {
        "actions_taken": [],
        "execution_mode": "rag_assisted",
        "rag_status": "hit",
        "latest_code": {
            "detected": True,
            "source_code_hash": source_hash,
            "language": "Python",
        },
        "code_execution_reports": [],
        "latest_work_result": {"needs_follow_up": False},
        "work_history": [{"agent": "implementation_agent"}],
        "evidence": [{"collection": "code_cases", "metadata": {}}],
    }
    execute = HeadDecision(
        iteration=2,
        rationale="执行最新代码",
        action="execute_code_tests",
    )
    CoordinatorAgent._validate(execute, runtime_state, make_task(), True)

    finish = HeadDecision(
        iteration=2,
        rationale="直接完成",
        action="finish",
        finish_reason="已有代码",
    )
    try:
        CoordinatorAgent._validate(finish, runtime_state, make_task(), True)
    except ValueError as error:
        assert "execute_code_tests" in str(error)
    else:
        raise AssertionError("未执行的代码不应允许直接交付")


def test_failed_execution_cannot_be_declared_finished() -> None:
    source_hash = make_plan().source_code_hash
    report = CodeExecutionReport(
        source_code_hash=source_hash,
        language="Python",
        overall_status="failed",
        verdict="Wrong Answer",
        total_tests=2,
    )
    runtime_state = {
        "actions_taken": [],
        "execution_mode": "rag_assisted",
        "rag_status": "hit",
        "latest_code": {"detected": True, "source_code_hash": source_hash},
        "code_execution_reports": [report.model_dump()],
        "latest_work_result": {"needs_follow_up": False},
        "work_history": [
            {"agent": "implementation_agent"},
            {"agent": "verification_agent"},
        ],
        "evidence": [{"collection": "code_cases", "metadata": {}}],
    }
    finish = HeadDecision(
        iteration=3,
        rationale="错误地完成",
        action="finish",
        finish_reason="完成",
    )
    try:
        CoordinatorAgent._validate(finish, runtime_state, make_task(), True)
    except ValueError as error:
        assert "Judge0" in str(error)
    else:
        raise AssertionError("Judge0 失败后不应允许 finish")


def test_explicit_no_rag_constraint_routes_directly_to_native_agents() -> None:
    task = make_task().model_copy(update={
        "normalized_request": "不走 RAG，直接用算法解题模块完成最长回文子串",
        "constraints": ["本轮不使用知识库或网页搜索"],
        "input_artifacts": {
            "problem_statement": "给定字符串 s，返回最长回文子串。",
            "programming_language": "Python",
        },
    })
    assert CoordinatorAgent._requests_native_only(task) is True
    assert CoordinatorAgent._requires_initial_rag(task) is False

    runtime_state = {
        "actions_taken": [],
        "execution_mode": "native_reasoning",
        "rag_status": "not_checked",
        "latest_code": {"detected": False, "source_code_hash": None},
        "code_execution_reports": [],
        "latest_work_result": None,
        "work_history": [],
        "evidence": [],
    }
    delegate = HeadDecision(
        iteration=1,
        rationale="直接进入算法推理",
        action="delegate",
        selected_agent="problem_structuring_agent",
        task_instruction="结构化题面",
    )
    CoordinatorAgent._validate(delegate, runtime_state, task, True)
