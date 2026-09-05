from __future__ import annotations

import asyncio
import re

import judge0

from app.models import CodeExecutionReport, CodeTestPlan


LANGUAGES = {
    "Python": judge0.PYTHON,
    "Java": judge0.JAVA,
    "C++": judge0.CPP,
}
PASS_RE = re.compile(r"ALL_TESTS_PASSED\s+(\d+)/(\d+)")
FAIL_RE = re.compile(r"TESTS_FAILED\s+(\d+)/(\d+)")


class Judge0CodeRunner:
    """Async tool boundary around the synchronous official Judge0 SDK."""

    async def run(self, plan: CodeTestPlan) -> CodeExecutionReport:
        judge_language = LANGUAGES.get(plan.language)
        if judge_language is None:
            return self._report(
                plan,
                overall_status="unsupported",
                verdict="Unsupported Language",
                failure_reason=f"Judge0 执行器暂不支持 {plan.language}",
            )

        submission = judge0.Submission(
            source_code=plan.executable_source,
            language=judge_language,
            expected_output=plan.expected_output,
            cpu_time_limit=5,
            wall_time_limit=15,
            memory_limit=256_000,
            enable_network=False,
        )
        try:
            result = await asyncio.to_thread(judge0.run, submissions=submission)
        except Exception as error:
            return self._report(
                plan,
                overall_status="unavailable",
                verdict="Judge0 Unavailable",
                failure_reason=f"{type(error).__name__}: {error}",
            )

        stdout = result.stdout or ""
        verdict = str(result.status) if result.status is not None else "Unknown"
        pass_match = PASS_RE.search(stdout)
        fail_match = FAIL_RE.search(stdout)
        passed = (
            verdict == "Accepted"
            and pass_match is not None
            and int(pass_match.group(1)) == plan.test_count
            and int(pass_match.group(2)) == plan.test_count
        )
        passed_tests = plan.test_count if passed else 0
        total_tests = plan.test_count
        if fail_match is not None:
            passed_tests = min(int(fail_match.group(1)), plan.test_count)
            total_tests = max(int(fail_match.group(2)), plan.test_count)

        return self._report(
            plan,
            overall_status="passed" if passed else "failed",
            verdict=verdict,
            passed_tests=passed_tests,
            total_tests=total_tests,
            stdout=stdout or None,
            stderr=result.stderr,
            compile_output=result.compile_output,
            exit_code=result.exit_code,
            time_seconds=float(result.time) if result.time is not None else None,
            memory_kb=int(result.memory) if result.memory is not None else None,
            failure_reason=None if passed else self._failure_reason(result, verdict),
        )

    @staticmethod
    def _failure_reason(result, verdict: str) -> str:
        detail = result.compile_output or result.stderr or result.message or ""
        return f"{verdict}: {detail}".strip()[:2_000]

    @staticmethod
    def _report(
        plan: CodeTestPlan,
        *,
        overall_status: str,
        verdict: str,
        passed_tests: int = 0,
        total_tests: int | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
        compile_output: str | None = None,
        exit_code: int | None = None,
        time_seconds: float | None = None,
        memory_kb: int | None = None,
        failure_reason: str | None = None,
    ) -> CodeExecutionReport:
        return CodeExecutionReport(
            source_code_hash=plan.source_code_hash,
            language=plan.language,
            overall_status=overall_status,
            verdict=verdict,
            passed_tests=passed_tests,
            total_tests=plan.test_count if total_tests is None else total_tests,
            test_categories=plan.test_categories,
            oracle_strategy=plan.oracle_strategy,
            semantic_reflection_rounds=plan.semantic_reflection_rounds,
            test_plan_review=plan.review_summary,
            test_plan_review_confidence=plan.review_confidence,
            stdout=stdout,
            stderr=stderr,
            compile_output=compile_output,
            exit_code=exit_code,
            time_seconds=time_seconds,
            memory_kb=memory_kb,
            failure_reason=failure_reason,
        )
