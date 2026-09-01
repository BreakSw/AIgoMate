import asyncio
import json
from app.core.coordinator_agent import CoordinatorAgent
from app.core.intent_recognizer import IntentRecognizer
from app.core.web_search_agent import WebSearchAgent
from app.models import (
    ContextPlan,
    ConversationMessage,
    DeliverySpec,
    HeadDecision,
    RoutingPlan,
    TaskSpec,
)


def weekly_contest_task() -> TaskSpec:
    return TaskSpec(
        primary_intent="problem_solving",
        normalized_request="分析这个周末的 LeetCode 周赛题目",
        user_goal="获取本周末 LeetCode Weekly Contest 的题目与解析",
        recognition_summary="用户需要本周周赛公开信息与题解",
        response_mode="step_by_step_explanation",
        delivery=DeliverySpec(assistance_level="direct_solution"),
        routing=RoutingPlan(primary_capability="knowledge_retrieval"),
        context_plan=ContextPlan(recent_messages=True),
        confidence=0.95,
    )


def test_intent_recognizer_reflects_instead_of_repeating_weekend_clarification() -> None:
    class FakeModelClient:
        def __init__(self) -> None:
            self.calls = 0

        async def complete_json(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                payload = weekly_contest_task().model_copy(update={
                    "normalized_request": "分析本周末周六或周日的 LeetCode 周赛",
                    "response_mode": "clarification_first",
                    "clarifying_question": "请问是周六还是周日的周赛？",
                }).model_dump()
            else:
                payload = weekly_contest_task().model_dump()
            return json.dumps(payload, ensure_ascii=False), "fake-deepseek"

    history = [
        ConversationMessage(role="user", content="帮我分析一下这个周末的 LeetCode 周赛题"),
        ConversationMessage(role="assistant", content="请问是周六还是周日的周赛？"),
    ]
    client = FakeModelClient()
    task_spec, provider = asyncio.run(IntentRecognizer(client).recognize(
        "周末的，LeetCode 周赛每周只有一次",
        history,
    ))

    assert client.calls == 2
    assert provider == "fake-deepseek+reflection:1"
    assert task_spec.response_mode != "clarification_first"
    assert task_spec.clarifying_question is None
    assert "周六或周日" not in task_spec.normalized_request


def test_intent_protocol_exhaustion_falls_back_for_contest_code_continuation() -> None:
    class InvalidModelClient:
        async def complete_json(self, *_args, **_kwargs):
            return "{}", "fake-deepseek"

    history = [
        ConversationMessage(
            role="assistant",
            content="上一轮列出了某场 LeetCode 周赛的四道题。",
        )
    ]
    task_spec, provider = asyncio.run(IntentRecognizer(
        InvalidModelClient(),
        max_reflection_rounds=1,
    ).recognize(
        "我需要每道题的 C++ 代码",
        history,
    ))

    assert task_spec.primary_intent == "code_generation"
    assert task_spec.delivery.include_code is True
    assert task_spec.input_artifacts.programming_language == "C++"
    assert task_spec.context_plan.task_state is True
    assert task_spec.routing.recommended_sequence == [
        "knowledge_retrieval",
        "code_sandbox",
    ]
    assert provider.endswith("+continuation-code-fallback")


def test_coordinator_does_not_hardcode_weekly_contest_workflow() -> None:
    task_spec = weekly_contest_task()
    ask = HeadDecision(
        iteration=1,
        rationale="需要日期",
        action="ask_clarification",
        clarification_question="请问是周六还是周日？",
    )

    # 这些业务选择由系统提示词引导，而不是 Python 校验器强制。
    CoordinatorAgent._validate(ask, {"actions_taken": []}, task_spec, True)

    runtime_state = {
        "actions_taken": [{"iteration": 1, "action": "get_current_time"}],
    }
    CoordinatorAgent._validate(ask, runtime_state, task_spec, True)

    official_first = HeadDecision(
        iteration=2,
        rationale="搜索官方",
        action="search_web",
        web_query="site:leetcode.cn 2026-08-30 LeetCode Weekly Contest",
        web_search_reason="定位赛事",
    )
    CoordinatorAgent._validate(official_first, runtime_state, task_spec, True)

    bilibili_first = official_first.model_copy(update={
        "web_query": "site:bilibili.com 灵茶山艾府 LeetCode 周赛 2026-08-30",
    })
    CoordinatorAgent._validate(bilibili_first, runtime_state, task_spec, True)


def test_bilibili_results_are_generic_and_expose_author_for_head_agent() -> None:
    payload = {
        "code": 0,
        "data": {
            "result": [
                {
                    "mid": 206214,
                    "author": "灵茶山艾府",
                    "title": "异或哈希【力扣周赛 516】",
                    "description": "讲解本场四道题，题号 3658-3661",
                    "bvid": "BV18p846TEwX",
                    "pubdate": 1787472946,
                },
                {
                    "mid": 123,
                    "author": "其他作者",
                    "title": "周赛搬运",
                    "description": "无关结果",
                    "bvid": "BV1OTHER",
                    "pubdate": 1787472946,
                },
            ]
        },
    }

    results = WebSearchAgent._extract_bilibili_results(payload)

    assert len(results) == 2
    assert results[0].title == "异或哈希【力扣周赛 516】"
    assert "3658-3661" in results[0].snippet
    assert "作者=灵茶山艾府" in results[0].snippet
    assert results[0].url == "https://www.bilibili.com/video/BV18p846TEwX"
    assert results[0].source_type == "bilibili_video"
    assert results[1].title == "周赛搬运"
    assert WebSearchAgent._bilibili_keyword(
        "site:bilibili.com 灵茶山艾府 LeetCode 周赛 2026-08-30"
    ) == "灵茶山艾府 LeetCode 周赛 2026-08-30"


def test_leetcode_graphql_resolves_target_contest_and_official_problem_urls() -> None:
    from datetime import date

    upcoming = {
        "data": {
            "upcomingContests": [
                {
                    "title": "Weekly Contest 518",
                    "titleSlug": "weekly-contest-518",
                    "startTime": 1788661800,
                    "duration": 5400,
                }
            ]
        }
    }
    contest_number = WebSearchAgent._infer_weekly_contest_number(
        upcoming,
        date(2026, 8, 30),
    )
    assert contest_number == 517
    results = WebSearchAgent._extract_leetcode_contest_results({
        "contest_number": contest_number,
        "contest_slug": "weekly-contest-517",
        "target_date": "2026-08-30",
        "questions": [
            {
                "credit": 3,
                "title": "Count Integers Appearing in a Single Block",
                "titleSlug": "count-integers-appearing-in-a-single-block",
                "questionId": "4410",
                "detail": {
                    "questionId": "4410",
                    "questionFrontendId": "4410",
                    "title": "Count Integers Appearing in a Single Block",
                    "titleSlug": "count-integers-appearing-in-a-single-block",
                    "content": "<p>Given an integer array <code>nums</code>, count values that appear in one contiguous block.</p>",
                    "difficulty": "Easy",
                    "topicTags": [{"name": "Array", "slug": "array"}],
                },
            }
        ],
    })

    assert len(results) == 1
    assert results[0].title.startswith("LeetCode 4410")
    assert results[0].source_type == "leetcode_official"
    assert results[0].url == (
        "https://leetcode.cn/problems/count-integers-appearing-in-a-single-block/"
    )
    assert "Weekly Contest 517" in results[0].snippet
    assert "difficulty=Easy" in results[0].snippet
    assert "Given an integer array" in results[0].snippet
