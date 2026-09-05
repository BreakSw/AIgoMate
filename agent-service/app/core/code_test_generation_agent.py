from __future__ import annotations

import hashlib

from app.core.model_client import IntentModelClient, RetryCallback
from app.core.reflection import AgentProtocolExhaustedError, complete_with_reflection
from app.models import CodeTestPlan, CodeTestPlanReview, TaskSpec


SYSTEM_PROMPT = """你是 AlgoMate 的算法测试生成 Agent。你的唯一任务是把候选算法代码包装成可在 Judge0 中直接编译运行的测试 Harness。

必须遵守：
1. 只支持 Python、Java、C++。候选代码必须逐字出现在 executable_source 中，不得偷偷修复、重写或替换候选实现。
2. Harness 必须真实调用候选代码并比较实际结果，禁止无条件打印通过标志，禁止把候选实现本身当作 Oracle。
3. 优先在一个程序中执行全部测试，以节约远程判题额度。
4. 至少覆盖用户/题面样例、最小边界、典型用例和一个对抗性类别；条件允许时，用固定随机种子生成小规模随机数据，并用独立暴力解或可验证性质作为 Oracle。
5. Oracle 必须与候选算法实现独立。无法可靠构造 Oracle 时，使用人工可确认的输入输出，并在 oracle_strategy 中如实说明。
6. 程序通过时 stdout 只能输出 `ALL_TESTS_PASSED x/x` 加换行；失败时输出 `TESTS_FAILED x/y` 和首个失败用例的精简诊断，并以非零状态退出。
7. Java 必须能以 Main 作为入口；C++ 使用标准 main；Python 可直接执行。不得访问网络、文件系统或执行外部命令。
8. source_code_hash 必须原样返回输入给出的哈希；expected_output 必须与全部测试通过时的 stdout 完全一致。

只返回 JSON：
{
  "protocol_version": "1.0",
  "language": "Python | Java | C++",
  "source_code_hash": "输入给出的64位哈希",
  "executable_source": "包含候选代码原文及测试 Harness 的完整源码",
  "expected_output": "ALL_TESTS_PASSED x/x\\n",
  "test_count": 1,
  "test_categories": ["官方样例", "最小边界"],
  "oracle_strategy": "独立暴力解或人工期望值的说明"
}"""


REVIEW_SYSTEM_PROMPT = """你是 AlgoMate 的算法测试集 Reflection Critic。你不生成答案代码，只独立审查测试 Harness 是否可能误判候选算法。

逐项检查：
1. candidate_integrity：候选代码是否原样保留，测试是否意外修复了候选实现。
2. candidate_invocation：每条测试是否真正调用候选函数/类，而非只运行 Oracle 或固定打印 PASS。
3. oracle_independence 与 oracle_correctness：Oracle 是否独立于候选算法，人工期望值/暴力解本身是否正确。
4. constraint_compliance：测试数据、类型、返回顺序和比较方式是否符合题面；无序结果是否规范化，浮点结果是否采用合理误差。
5. edge_coverage：是否覆盖题面样例、最小规模、重复值、负数、溢出、空/不可达等真正适用的边界，禁止加入题面不允许的输入。
6. language_compilability：入口、类名、作用域、类型、导入和语言版本是否能在 Judge0 编译执行。
7. output_protocol：test_count、实际执行次数、通过标志和 expected_output 是否一致；失败是否非零退出并暴露首个反例。

发现任何可能造成假阳性、假阴性、编译失败或错误 Oracle 的问题，verdict 必须为 revise，并给出可直接执行的修订指令。
只有逐项检查完且没有实质问题时才能 approved。不要因为格式看起来完整就批准。

只返回 JSON：
{
  "protocol_version": "1.0",
  "verdict": "approved | revise",
  "summary": "审查结论",
  "checked_dimensions": ["candidate_integrity", "candidate_invocation"],
  "issues": [],
  "revision_instructions": [],
  "confidence": 0.0
}"""


REVISION_SYSTEM_PROMPT = SYSTEM_PROMPT + """

你现在处于 Reflection 修订阶段。根据独立 Critic 的 issues 和 revision_instructions 修正测试 Harness。
仍然不得修改候选代码本身。必须重新输出完整 CodeTestPlan JSON，不能只输出差异或解释。
不得为了让 Critic 满意而删除有效边界测试，也不得通过放宽比较条件掩盖候选代码错误。"""


REQUIRED_REVIEW_DIMENSIONS = {
    "candidate_integrity",
    "candidate_invocation",
    "oracle_independence",
    "oracle_correctness",
    "constraint_compliance",
    "edge_coverage",
    "language_compilability",
    "output_protocol",
}


class CodeTestGenerationAgent:
    def __init__(
        self,
        model_client: IntentModelClient,
        max_reflection_rounds: int = 3,
    ) -> None:
        self.model_client = model_client
        self.max_reflection_rounds = max_reflection_rounds

    async def generate(
        self,
        *,
        task_spec: TaskSpec,
        candidate_code: str,
        language: str,
        solution_context: str,
        on_retry: RetryCallback | None = None,
    ) -> tuple[CodeTestPlan, str]:
        source_hash = hashlib.sha256(candidate_code.encode("utf-8")).hexdigest()
        payload = {
            "source_code_hash": source_hash,
            "language": language,
            "candidate_code": candidate_code,
            "task": {
                "normalized_request": task_spec.normalized_request,
                "problem_statement": task_spec.input_artifacts.problem_statement,
                "provided_test_cases": task_spec.input_artifacts.test_cases,
                "constraints": task_spec.constraints,
                "success_criteria": task_spec.success_criteria,
            },
            "latest_solution_context": solution_context[-10_000:],
        }
        plan, generation_provider, _ = await complete_with_reflection(
            model_client=self.model_client,
            agent_name="算法测试生成 Agent",
            system_prompt=SYSTEM_PROMPT,
            request_payload=payload,
            model_type=CodeTestPlan,
            on_retry=on_retry,
            max_tokens=7000,
            max_reflection_rounds=self.max_reflection_rounds,
            validator=lambda value: self._validate(value, candidate_code, source_hash),
        )
        last_review: CodeTestPlanReview | None = None
        semantic_revisions = 0
        provider_trace = [f"generate:{generation_provider}"]

        for review_round in range(self.max_reflection_rounds + 1):
            review, review_provider, _ = await complete_with_reflection(
                model_client=self.model_client,
                agent_name="算法测试集 Reflection Critic",
                system_prompt=REVIEW_SYSTEM_PROMPT,
                request_payload={
                    "source_code_hash": source_hash,
                    "language": language,
                    "candidate_code": candidate_code,
                    "task": payload["task"],
                    "generated_test_plan": plan.model_dump(),
                    "reflection_round": review_round,
                },
                model_type=CodeTestPlanReview,
                on_retry=on_retry,
                max_tokens=2600,
                max_reflection_rounds=min(self.max_reflection_rounds, 2),
                validator=self._validate_review,
            )
            last_review = review
            provider_trace.append(f"critic#{review_round + 1}:{review_provider}")
            if review.verdict == "approved":
                approved_plan = plan.model_copy(update={
                    "semantic_reflection_rounds": semantic_revisions,
                    "review_summary": review.summary,
                    "review_confidence": review.confidence,
                })
                return (
                    approved_plan,
                    "|".join(provider_trace)
                    + f"+test-reflection:{semantic_revisions}",
                )
            if review_round >= self.max_reflection_rounds:
                break

            semantic_revisions += 1
            plan, revision_provider, _ = await complete_with_reflection(
                model_client=self.model_client,
                agent_name="算法测试生成 Agent Reflection 修订",
                system_prompt=REVISION_SYSTEM_PROMPT,
                request_payload={
                    "original_request": payload,
                    "previous_test_plan": plan.model_dump(),
                    "critic_review": review.model_dump(),
                    "semantic_revision_round": semantic_revisions,
                },
                model_type=CodeTestPlan,
                on_retry=on_retry,
                max_tokens=7000,
                max_reflection_rounds=min(self.max_reflection_rounds, 2),
                validator=lambda value: self._validate(
                    value,
                    candidate_code,
                    source_hash,
                ),
            )
            provider_trace.append(
                f"revise#{semantic_revisions}:{revision_provider}"
            )

        raise AgentProtocolExhaustedError(
            "算法测试集语义 Reflection",
            self.max_reflection_rounds,
            "|".join(provider_trace)
            + (f"|last-review:{last_review.summary}" if last_review else ""),
        )

    @staticmethod
    def _validate(plan: CodeTestPlan, candidate_code: str, source_hash: str) -> None:
        if plan.source_code_hash != source_hash:
            raise ValueError("测试计划绑定的源码哈希与候选代码不一致")
        if candidate_code.strip() not in plan.executable_source:
            raise ValueError("测试 Harness 未逐字包含候选代码，可能测试了被改写的实现")
        expected_marker = f"ALL_TESTS_PASSED {plan.test_count}/{plan.test_count}\n"
        if plan.expected_output != expected_marker:
            raise ValueError("expected_output 与测试数量或通过标志不一致")
        if len(plan.test_categories) < 2:
            raise ValueError("测试类别不足，至少需要样例/典型与边界类测试")

    @staticmethod
    def _validate_review(review: CodeTestPlanReview) -> None:
        checked = set(review.checked_dimensions)
        missing = REQUIRED_REVIEW_DIMENSIONS - checked
        if missing:
            raise ValueError(
                "Reflection Critic 未检查完整维度：" + "、".join(sorted(missing))
            )
        if review.verdict == "approved" and (
            review.issues or review.revision_instructions
        ):
            raise ValueError("approved 与仍存在的问题/修订要求相矛盾")
        if review.verdict == "revise" and not review.revision_instructions:
            raise ValueError("revise 必须给出具体修订指令")
