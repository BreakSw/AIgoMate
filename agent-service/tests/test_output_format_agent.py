import pytest

from app.core.output_format_agent import OutputFormattingAgent
from app.models import OutputFormatResult, PolishResult, RagEvidence


def test_format_validator_preserves_evidence_links_code_and_fallback_label() -> None:
    polished = PolishResult(
        final_answer=(
            "未能确认当天官方每日一题，以下是上下文推荐题。\n\n"
            "解法一：哈希表。 [R1]\n\n"
            "解法二：暴力枚举。\n\n"
            "```python\nreturn [0, 1]\n```"
        ),
        preserved_uncertainties=["官方每日一题未核实"],
    )
    formatted = OutputFormatResult(
        formatted_answer=(
            "> **未能确认当天官方每日一题，以下是上下文推荐题。**\n\n"
            "## 推荐题\n\n"
            "解法一：哈希表。 [R1]\n\n"
            "解法二：暴力枚举。\n\n"
            "| 方案 | 核心思路 | 时间 | 空间 |\n"
            "|---|---|---|---|\n"
            "| 哈希表 | 查找补数 | O(n) | O(n) |\n"
            "| 暴力 | 枚举数对 | O(n²) | O(1) |\n\n"
            "```python\nreturn [0, 1]\n```"
            "\n\n## 参考资料\n\n"
            "- [Two Sum](https://leetcode.cn/problems/two-sum/)"
        ),
        formatting_changes=["增加提示引用和标题"],
    )

    OutputFormattingAgent._validate(
        formatted,
        polished,
        [{
            "evidence_id": "R1",
            "title": "Two Sum",
            "url": "https://leetcode.cn/problems/two-sum/",
        }],
    )


def test_format_validator_rejects_relabeling_recommendation_as_daily() -> None:
    polished = PolishResult(
        final_answer="未能确认当天官方每日一题，以下是上下文推荐题。"
    )
    formatted = OutputFormatResult(
        formatted_answer="这是今天的官方每日一题。"
    )

    with pytest.raises(ValueError, match="必要提示"):
        OutputFormattingAgent._validate(formatted, polished, [])


def test_format_validator_rejects_unprovided_reference_url() -> None:
    polished = PolishResult(final_answer="结论见证据 [R1]。")
    formatted = OutputFormatResult(
        formatted_answer=(
            "结论见证据 [R1]。\n\n"
            "## 参考资料\n\n"
            "- [未知来源](https://example.com/invented)"
        )
    )

    with pytest.raises(ValueError, match="编造了参考链接"):
        OutputFormattingAgent._validate(formatted, polished, [])
