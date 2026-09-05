"""Minimal live smoke test for the official Judge0 Python SDK.

The SDK is only a client. With no Judge0-related environment variables set,
it uses Judge0's public free-tier CE endpoint and consumes one submission.
"""

from __future__ import annotations

import json
import sys

import judge0


SOURCE_CODE = """\
first, second = map(int, input().split())
print(first + second)
"""
STDIN = "20 22\n"
EXPECTED_STDOUT = "42\n"


def main() -> int:
    try:
        result = judge0.run(
            source_code=SOURCE_CODE,
            language=judge0.PYTHON,
            stdin=STDIN,
            expected_output=EXPECTED_STDOUT,
        )
    except Exception as error:  # The demo should expose remote/API failures clearly.
        print(json.dumps({
            "ok": False,
            "error_type": type(error).__name__,
            "message": str(error),
        }, ensure_ascii=False, indent=2))
        return 1

    actual_stdout = result.stdout or ""
    status = str(result.status) if result.status is not None else "Unknown"
    passed = status == "Accepted" and actual_stdout.strip() == EXPECTED_STDOUT.strip()
    print(json.dumps({
        "ok": passed,
        "sdk_version": judge0.__version__,
        "language": "Python",
        "stdin": STDIN,
        "expected_stdout": EXPECTED_STDOUT,
        "actual_stdout": actual_stdout,
        "status": status,
        "compile_output": result.compile_output,
        "stderr": result.stderr,
        "exit_code": result.exit_code,
        "time_seconds": result.time,
        "memory_kb": result.memory,
        "submission_token": str(result.token) if result.token is not None else None,
    }, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
