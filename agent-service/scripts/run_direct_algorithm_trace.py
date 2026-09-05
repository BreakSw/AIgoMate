"""Invoke the LangGraph algorithm runtime directly with an explicit TaskSpec.

This diagnostic entry point intentionally bypasses input rewriting and intent
recognition so the algorithm-solving graph can be evaluated in isolation.
It does not bypass code-test Reflection or Judge0 execution.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys

from app.main import (
    code_test_generation_agent,
    current_time_tool,
    judge0_code_runner,
    langsmith_tracer,
    memory_repository,
    model_client,
    model_config_store,
    prompt_builder,
    rag_retriever,
    response_agent,
    web_search_agent,
)
from app.core.adaptive_runtime import AdaptiveAgentRuntime
from app.models import (
    CompressedContext,
    ContextPlan,
    ContextSnapshot,
    ContextWindowStatus,
    DeliverySpec,
    InputArtifacts,
    HeadDecision,
    MemoryScope,
    MemorySnapshot,
    RoutingPlan,
    TaskSpec,
)


class DirectAlgorithmCoordinator:
    """Compact diagnostic router; all solution/test/verification work stays real."""

    def __init__(self) -> None:
        self.model_client = model_client

    async def decide(self, *, runtime_state: dict, iteration: int, **_kwargs):
        latest_code = runtime_state.get("latest_code") or {}
        source_hash = latest_code.get("source_code_hash")
        reports = runtime_state.get("code_execution_reports", [])
        matching_report = next(
            (
                item
                for item in reversed(reports)
                if item.get("source_code_hash") == source_hash
            ),
            None,
        )
        history = runtime_state.get("work_history", [])
        agents = [item.get("agent") for item in history]

        if not history:
            decision = HeadDecision(
                iteration=iteration,
                rationale="按用户要求跳过检索，直接生成完整算法与 C++ 实现",
                action="delegate",
                selected_agent="implementation_agent",
                task_instruction=(
                    "直接根据 TaskSpec 中的完整题面独立求解。给出中心扩展算法、正确性说明、"
                    "复杂度、边界条件和可提交的 C++17 class Solution 代码；不要使用 RAG 或网页事实。"
                ),
            )
        elif source_hash and matching_report is None:
            decision = HeadDecision(
                iteration=iteration,
                rationale="最新代码尚无对应源码哈希的真实执行报告",
                action="execute_code_tests",
            )
        elif matching_report and matching_report.get("overall_status") == "failed":
            decision = HeadDecision(
                iteration=iteration,
                rationale="Judge0 已发现失败，要求实现 Agent 根据报告修复",
                action="delegate",
                selected_agent="implementation_agent",
                task_instruction=(
                    "依据 code_execution_reports 的真实失败信息修复最新 C++ 代码，返回完整修订版；"
                    "不得自行声称已经通过，修订后必须重新执行。"
                ),
            )
        elif matching_report and matching_report.get("overall_status") in {
            "unavailable",
            "unsupported",
            "error",
        }:
            decision = HeadDecision(
                iteration=iteration,
                rationale="测试工具链未产生可用裁决，保留实现与错误报告供诊断",
                action="finish",
                finish_reason="直达评估在测试工具阶段停止，不把静态结果冒充真实通过",
            )
        elif "verification_agent" not in agents:
            decision = HeadDecision(
                iteration=iteration,
                rationale="实现已有执行报告，交由验证 Agent 综合审查",
                action="delegate",
                selected_agent="verification_agent",
                task_instruction=(
                    "综合题面、最新完整代码和匹配源码哈希的 Judge0 报告，核对中心扩展逻辑、"
                    "样例、边界与复杂度，返回包含完整 C++ 代码和真实执行结论的最终答案。"
                ),
            )
        else:
            decision = HeadDecision(
                iteration=iteration,
                rationale="算法、代码和真实执行报告均已完成验证",
                action="finish",
                finish_reason="直达算法评估链路完成",
            )
        return decision, "direct-evaluation-router"


def build_snapshot(session_id: int) -> ContextSnapshot:
    return ContextSnapshot(
        memory=MemorySnapshot(current_goal="独立求解并执行验证算法题"),
        compressed_context=CompressedContext(summary="直接算法模块诊断，无历史上下文。"),
        window=ContextWindowStatus(
            window_size_tokens=32_768,
            soft_limit_tokens=24_576,
            hard_limit_tokens=28_672,
            output_reserved_tokens=8_192,
            raw_history_tokens=0,
            current_input_tokens=0,
            compressed_context_tokens=0,
            recent_messages_tokens=0,
            estimated_input_tokens=0,
            remaining_tokens=24_576,
            usage_ratio=0,
            state="normal",
            compression_triggered=False,
            messages_before_compression=0,
            messages_after_compression=0,
        ),
        memory_scope=MemoryScope(user_id=1, session_id=session_id),
    )


async def run(problem: str, language: str, session_id: int) -> dict:
    task = TaskSpec(
        primary_intent="code_generation",
        normalized_request=(
            f"不走 RAG，不联网，直接用算法解题模块求解；给出 {language} 可提交代码并真实验证。"
        ),
        user_goal="独立完成算法设计、代码实现和真实执行验证",
        recognition_summary="显式诊断任务，直接调用算法解题模块",
        input_artifacts=InputArtifacts(
            problem_statement=problem,
            programming_language=language,
        ),
        constraints=[
            "本轮不使用 RAG",
            "本轮不使用网页搜索",
            f"使用 {language}",
            "必须经过测试集 Reflection 和 Judge0 真实执行",
        ],
        response_mode="step_by_step_explanation",
        delivery=DeliverySpec(
            assistance_level="direct_solution",
            explanation_depth="detailed",
            response_language="zh-CN",
            expected_outputs=[
                "算法思路",
                "正确性说明",
                "复杂度分析",
                f"可提交的 {language} 代码",
                "Judge0 真实执行结论",
            ],
            include_code=True,
        ),
        routing=RoutingPlan(
            primary_capability="code_sandbox",
            supporting_capabilities=["algorithm_tutoring"],
            execution_mode="sequential",
            recommended_sequence=["algorithm_tutoring", "code_sandbox"],
            tool_requirements=["code_sandbox"],
        ),
        context_plan=ContextPlan(
            recent_messages=False,
            task_state=False,
            long_term_memory=False,
            user_learning_profile=False,
            algorithm_knowledge=False,
        ),
        success_criteria=[
            "算法适用于全部给定约束",
            "代码能通过生成的样例、边界与对抗测试",
            "执行报告与最终代码源码哈希一致",
        ],
        confidence=1,
    )
    runtime_config = await model_config_store.get()
    if runtime_config is None:
        raise RuntimeError("Redis 中没有可用模型配置")

    progress: list[dict] = []

    async def on_progress(phase, message, agent=None, detail=None):
        progress.append({
            "phase": phase,
            "message": message,
            "agent": agent,
            "detail": detail,
        })
        print(
            json.dumps(progress[-1], ensure_ascii=False),
            flush=True,
        )

    original_generate = code_test_generation_agent.generate
    original_judge_run = judge0_code_runner.run

    async def traced_generate(**kwargs):
        try:
            plan, provider = await original_generate(**kwargs)
        except Exception as error:
            print(json.dumps({
                "trace_event": "test_plan_reflection_error",
                "error_type": type(error).__name__,
                "error": str(error),
                "provider": getattr(error, "provider", None),
                "validation_feedback": getattr(error, "validation_feedback", None),
            }, ensure_ascii=False), flush=True)
            raise
        print(json.dumps({
            "trace_event": "test_plan_reflection_complete",
            "source_code_hash": plan.source_code_hash,
            "test_count": plan.test_count,
            "test_categories": plan.test_categories,
            "oracle_strategy": plan.oracle_strategy,
            "semantic_reflection_rounds": plan.semantic_reflection_rounds,
            "review_summary": plan.review_summary,
            "review_confidence": plan.review_confidence,
            "provider": provider,
        }, ensure_ascii=False), flush=True)
        return plan, provider

    async def traced_judge_run(plan):
        report = await original_judge_run(plan)
        print(json.dumps({
            "trace_event": "judge0_complete",
            **report.model_dump(),
        }, ensure_ascii=False), flush=True)
        return report

    code_test_generation_agent.generate = traced_generate
    judge0_code_runner.run = traced_judge_run
    direct_runtime = AdaptiveAgentRuntime(
        DirectAlgorithmCoordinator(),
        response_agent,
        rag_retriever,
        web_search_agent,
        current_time_tool,
        memory_repository,
        prompt_builder,
        max_iterations=7,
        code_test_generation_agent=code_test_generation_agent,
        judge0_code_runner=judge0_code_runner,
    )

    graph_config = {
        "run_name": "algomate-direct-algorithm-evaluation",
        "tags": ["algomate", "direct-algorithm", "no-rag", "evaluation"],
        "metadata": {"session_id": session_id, "diagnostic": True},
    }
    if langsmith_tracer is not None:
        graph_config["callbacks"] = [langsmith_tracer]

    with model_client.activate(runtime_config):
        (
            plan,
            evidence,
            result,
            _memory,
            _memory_updates,
            providers,
            execution_reports,
        ) = await direct_runtime.run(
            user_id=1,
            session_id=session_id,
            task_spec=task,
            snapshot=build_snapshot(session_id),
            conversation_context=[],
            durable_memory=[],
            on_progress=on_progress,
            runnable_config=graph_config,
        )

    return {
        "task_spec": task.model_dump(),
        "progress": progress,
        "decision_trace": [item.model_dump() for item in plan.decision_trace],
        "execution_mode": plan.execution_mode,
        "rag_queries": [item.model_dump() for item in plan.rag_queries],
        "web_search_queries": plan.web_search_queries,
        "evidence_count": len(evidence),
        "providers": providers,
        "code_execution_reports": [item.model_dump() for item in execution_reports],
        "final_work_result": result.model_dump(),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem-base64", required=True)
    parser.add_argument("--language", default="C++")
    parser.add_argument("--session-id", type=int, default=905002)
    args = parser.parse_args()
    problem = base64.b64decode(args.problem_base64).decode("utf-8")
    output = asyncio.run(run(problem, args.language, args.session_id))
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
