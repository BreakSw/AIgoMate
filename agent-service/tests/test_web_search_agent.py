"""WebSearchAgent（SerpAPI 调用）单元测试。

只覆盖 SerpAPI 调用相关逻辑：查询规划降级、请求参数规范、错误处理、
结果解析与 RagEvidence 格式一致性、证据在多 Agent 间的编号传递。
全部离线运行（httpx.MockTransport + 假模型客户端），不消耗搜索额度。
"""

import asyncio
from datetime import date
from typing import Any

import httpx
import pytest

import app.core.web_search_agent as wsa_module
from app.config import Settings
from app.core.reflection import AgentProtocolExhaustedError
from app.core.web_search_agent import WebSearchAgent, WebSearchPlan


def make_agent(**settings_overrides: Any) -> WebSearchAgent:
    serpapi_key = settings_overrides.pop(
        "serpapi_api_key",
        "test-key",
    )

    class RuntimeModelClient:
        current_serpapi_api_key = serpapi_key

    return WebSearchAgent(
        model_client=RuntimeModelClient(),
        settings=Settings(_env_file=None, **settings_overrides),
    )


def serpapi_payload() -> dict[str, Any]:
    return {
        "answer_box": {
            "title": "Two Sum 精选摘要",
            "link": "https://leetcode.com/problems/two-sum/",
            "answer": "哈希表一次遍历，时间复杂度 O(n)。",
        },
        "organic_results": [
            {
                "position": 1,
                "title": "Two Sum - LeetCode",
                "link": "https://leetcode.com/problems/two-sum/",
                "snippet": "Given an array of integers, return indices of the two numbers...",
            },
            {
                "position": 2,
                "title": "Two Sum 题解",
                "link": "https://example.com/two-sum-solution",
                "snippet": "用哈希表记录已遍历元素，查找 target - num。",
            },
        ],
    }


def install_mock_transport(monkeypatch, handler) -> list[httpx.Request]:
    """把 WebSearchAgent 内部创建的 AsyncClient 替换为注入 MockTransport 的版本。"""
    captured: list[httpx.Request] = []
    real_client = httpx.AsyncClient

    def capturing_handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return handler(request)

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs.pop("transport", None)
        kwargs.pop("app", None)
        return real_client(*args, transport=httpx.MockTransport(capturing_handler), **kwargs)

    monkeypatch.setattr(wsa_module.httpx, "AsyncClient", factory)
    return captured


def install_fake_planner(monkeypatch, plan: WebSearchPlan) -> None:
    async def fake_reflection(**_kwargs: Any):
        return plan, "fake-planner", None

    monkeypatch.setattr(wsa_module, "complete_with_reflection", fake_reflection)


# ---------------------------------------------------------------------------
# 1. 可用性判断
# ---------------------------------------------------------------------------

def test_blank_serpapi_key_does_not_disable_public_search_tools() -> None:
    agent = make_agent(serpapi_api_key="   ")
    assert agent.available() is True


def test_available_accepts_configured_key() -> None:
    agent = make_agent()
    assert agent.available() is True


def test_available_respects_disabled_flag() -> None:
    agent = make_agent(web_search_enabled=False)
    assert agent.available() is False


# ---------------------------------------------------------------------------
# 2. 端到端：规划 → SerpAPI 请求 → RagEvidence 格式
# ---------------------------------------------------------------------------

def test_model_selected_bilibili_provider_calls_only_bilibili(monkeypatch) -> None:
    plan = WebSearchPlan(
        query="灵茶山艾府 LeetCode 周赛 2026-08-30",
        provider="bilibili",
        target_date=date(2026, 8, 30),
    )
    install_fake_planner(monkeypatch, plan)
    agent = make_agent(serpapi_api_key="")
    calls = {"bilibili": 0, "leetcode": 0, "serpapi": 0}

    async def bilibili(_query: str):
        calls["bilibili"] += 1
        return {"data": {"result": [{
            "author": "灵茶山艾府",
            "title": "力扣周赛 516",
            "description": "本场题目解析",
            "bvid": "BV1TEST",
        }]}}

    async def forbidden_leetcode(*_args):
        calls["leetcode"] += 1
        raise AssertionError("单次计划不得自动串联 LeetCode")

    async def forbidden_serpapi(*_args, **_kwargs):
        calls["serpapi"] += 1
        raise AssertionError("B站计划不得调用 SerpAPI")

    monkeypatch.setattr(agent, "_bilibili_search", bilibili)
    monkeypatch.setattr(agent, "_leetcode_contest_search", forbidden_leetcode)
    monkeypatch.setattr(agent, "_serpapi_search", forbidden_serpapi)

    evidence, provider = asyncio.run(agent.search("定位上周周赛", "先定位场次"))

    assert provider == "fake-planner+bilibili"
    assert calls == {"bilibili": 1, "leetcode": 0, "serpapi": 0}
    assert evidence[0].metadata["source_type"] == "bilibili_video"


def test_model_selected_leetcode_provider_calls_only_graphql(monkeypatch) -> None:
    plan = WebSearchPlan(
        query="核验 LeetCode 中国站周赛 516 官方题单",
        provider="leetcode_graphql",
        contest_number=516,
    )
    install_fake_planner(monkeypatch, plan)
    agent = make_agent(serpapi_api_key="")
    calls = {"bilibili": 0, "leetcode": 0, "serpapi": 0}

    async def leetcode(contest_number, target_date):
        calls["leetcode"] += 1
        assert contest_number == 516
        assert target_date is None
        return {
            "contest_number": 516,
            "contest_slug": "weekly-contest-516",
            "target_date": None,
            "questions": [{
                "title": "Verified Problem",
                "titleSlug": "verified-problem",
                "questionId": "4401",
                "detail": {
                    "questionFrontendId": "4401",
                    "translatedTitle": "已核验赛题",
                    "titleSlug": "verified-problem",
                },
            }],
        }

    async def forbidden_bilibili(*_args):
        calls["bilibili"] += 1
        raise AssertionError("官方核验计划不得回退 B站")

    async def forbidden_serpapi(*_args, **_kwargs):
        calls["serpapi"] += 1
        raise AssertionError("GraphQL 计划不得调用 SerpAPI")

    monkeypatch.setattr(agent, "_leetcode_contest_search", leetcode)
    monkeypatch.setattr(agent, "_bilibili_search", forbidden_bilibili)
    monkeypatch.setattr(agent, "_serpapi_search", forbidden_serpapi)

    evidence, provider = asyncio.run(agent.search("核验周赛 516", "官方核验"))

    assert provider == "fake-planner+leetcode-cn-graphql"
    assert calls == {"bilibili": 0, "leetcode": 1, "serpapi": 0}
    assert evidence[0].source_url == "https://leetcode.cn/problems/verified-problem/"


def test_leetcode_plan_requires_structured_locator() -> None:
    with pytest.raises(ValueError, match="contest_number 或 target_date"):
        WebSearchAgent._validate_plan(WebSearchPlan(
            query="LeetCode 周赛",
            provider="leetcode_graphql",
        ))

def test_search_formats_evidence_and_request_params(monkeypatch) -> None:
    plan = WebSearchPlan(
        query="two sum 哈希表",
        relevance_criteria=["与 Two Sum 相关"],
        freshness_required=False,
    )
    install_fake_planner(monkeypatch, plan)
    captured = install_mock_transport(
        monkeypatch, lambda _request: httpx.Response(200, json=serpapi_payload())
    )
    agent = make_agent()

    evidence, provider = asyncio.run(agent.search("two sum", "补充外部资料"))

    assert provider == "fake-planner+serpapi"
    assert [item.evidence_id for item in evidence] == ["W1", "W2"]
    # answer_box 与 organic 第一条 URL 相同，去重后只保留 answer_box。
    assert evidence[0].collection == "web_search"
    assert evidence[0].title == "Two Sum 精选摘要"
    assert evidence[0].source_url == "https://leetcode.com/problems/two-sum/"
    assert "哈希表" in evidence[0].content
    assert evidence[0].score == pytest.approx(1.0)
    assert evidence[0].metadata["search_query"] == "two sum 哈希表"
    assert evidence[0].metadata["source_type"] == "answer_box"
    assert evidence[0].metadata["relevance_criteria"] == ["与 Two Sum 相关"]
    assert evidence[1].metadata["source_type"] == "organic_results"
    assert evidence[1].metadata["source_position"] == 2
    assert evidence[1].score < evidence[0].score

    # 请求参数符合 SerpAPI 规范
    assert len(captured) == 1
    url = captured[0].url
    assert str(url).startswith("https://serpapi.com/search.json")
    assert url.params["engine"] == "google"
    assert url.params["q"] == "two sum 哈希表"
    assert url.params["api_key"] == "test-key"
    assert int(url.params["num"]) == Settings().web_search_max_results
    assert url.params["hl"] == "zh-cn"
    assert "tbs" not in url.params  # freshness_required=False 不带时间过滤


def test_search_adds_freshness_param_when_required(monkeypatch) -> None:
    plan = WebSearchPlan(query="rust 1.88 release", freshness_required=True)
    install_fake_planner(monkeypatch, plan)
    captured = install_mock_transport(
        monkeypatch, lambda _request: httpx.Response(200, json={})
    )
    agent = make_agent()

    evidence, provider = asyncio.run(agent.search("rust 新版本", "时效性问题"))

    assert evidence == []
    assert provider == "fake-planner+serpapi"
    assert captured[0].url.params["tbs"] == "qdr:m"


def test_search_slices_results_to_max(monkeypatch) -> None:
    payload = {
        "organic_results": [
            {
                "position": index,
                "title": f"结果 {index}",
                "link": f"https://example.com/{index}",
                "snippet": f"摘要 {index}",
            }
            for index in range(1, 9)
        ]
    }
    install_fake_planner(monkeypatch, WebSearchPlan(query="测试"))
    install_mock_transport(monkeypatch, lambda _request: httpx.Response(200, json=payload))
    agent = make_agent()

    evidence, _ = asyncio.run(agent.search("测试", "测试原因"))

    assert len(evidence) == Settings().web_search_max_results
    assert [item.evidence_id for item in evidence] == [
        f"W{index}" for index in range(1, Settings().web_search_max_results + 1)
    ]


# ---------------------------------------------------------------------------
# 3. 错误处理与降级（回归测试：不得让整轮对话失败）
# ---------------------------------------------------------------------------

def test_search_degrades_when_planning_exhausts_reflection(monkeypatch) -> None:
    async def exhausted(**_kwargs: Any):
        raise AgentProtocolExhaustedError("网页搜索 Agent 查询规划器", 10, "fake")

    monkeypatch.setattr(wsa_module, "complete_with_reflection", exhausted)
    captured = install_mock_transport(monkeypatch, lambda _request: httpx.Response(200, json={}))
    agent = make_agent()

    evidence, provider = asyncio.run(agent.search("任意查询", "原因"))

    # 修复点：规划失败降级为空证据，而不是把异常抛给上层导致整轮 502。
    assert evidence == []
    assert provider == "serpapi-error:AgentProtocolExhaustedError"
    assert captured == []  # 规划失败时不应发起任何 SerpAPI 请求


def test_search_degrades_when_model_upstream_fails(monkeypatch) -> None:
    async def upstream_error(**_kwargs: Any):
        raise httpx.HTTPStatusError(
            "insufficient balance",
            request=httpx.Request("POST", "https://api.deepseek.com"),
            response=httpx.Response(402),
        )

    monkeypatch.setattr(wsa_module, "complete_with_reflection", upstream_error)
    agent = make_agent()

    evidence, provider = asyncio.run(agent.search("任意查询", "原因"))

    assert evidence == []
    assert provider == "serpapi-error:HTTPStatusError"


def test_search_surfaces_serpapi_api_error_without_retry(monkeypatch) -> None:
    plan = WebSearchPlan(query="测试")
    install_fake_planner(monkeypatch, plan)
    captured = install_mock_transport(
        monkeypatch,
        lambda _request: httpx.Response(401, json={"error": "Invalid API key."}),
    )
    agent = make_agent()

    evidence, provider = asyncio.run(agent.search("测试", "原因"))

    # 修复点：确定性 4xx 直接以 WebSearchError 结束，不重试、不吞掉错误详情。
    assert evidence == []
    assert provider == "serpapi-error:WebSearchError"
    assert len(captured) == 1  # 没有重试


def test_search_retries_5xx_then_succeeds(monkeypatch) -> None:
    plan = WebSearchPlan(query="测试")
    install_fake_planner(monkeypatch, plan)
    state = {"calls": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        if state["calls"] <= 2:
            return httpx.Response(500, json={"error": "internal error"})
        return httpx.Response(200, json=serpapi_payload())

    captured = install_mock_transport(monkeypatch, handler)

    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    agent = make_agent()

    evidence, provider = asyncio.run(agent.search("测试", "原因"))

    assert provider == "fake-planner+serpapi"
    assert len(evidence) == 2
    assert len(captured) == 3  # 两次 500 + 一次成功


def test_search_returns_empty_when_unavailable(monkeypatch) -> None:
    async def should_not_be_called(**_kwargs: Any):
        raise AssertionError("不可用时不应调用规划器")

    monkeypatch.setattr(wsa_module, "complete_with_reflection", should_not_be_called)
    agent = make_agent(web_search_enabled=False)

    evidence, provider = asyncio.run(agent.search("测试", "原因"))

    assert evidence == []
    assert provider == "web-search-unavailable"


# ---------------------------------------------------------------------------
# 4. 解析规则（脏数据 / 去重 / 无摘要条目）
# ---------------------------------------------------------------------------

def test_extract_results_handles_dirty_payload() -> None:
    payload = {
        "organic_results": [
            {"title": "A", "link": "https://a.com", "snippet": "摘要 A"},
            "not-a-dict",
            {"title": "B", "link": "https://b.com"},  # 无 snippet，应丢弃
            {"title": "A2", "link": "https://a.com", "snippet": "重复 URL"},  # 去重
            {"title": None, "link": "https://c.com", "snippet": "  摘要 C  "},
        ]
    }
    results = WebSearchAgent._extract_results(payload)

    assert [item.rank for item in results] == [1, 2]
    assert results[0].title == "A"
    assert results[1].title == "https://c.com"  # 缺标题回退为 URL
    assert results[1].snippet == "摘要 C"


# ---------------------------------------------------------------------------
# 5. 多 Agent 协作：W 编号证据合并传递
# ---------------------------------------------------------------------------

def test_merge_evidence_renumbers_web_ids() -> None:
    pytest.importorskip("app.core.adaptive_runtime", reason="依赖 pymilvus，跳过合并测试")
    from app.core.adaptive_runtime import AdaptiveAgentRuntime
    from app.models import RagEvidence

    first = [
        RagEvidence(
            evidence_id=f"W{index}",
            collection="web_search",
            title=f"结果 {index}",
            content=f"摘要 {index}",
            source_url=f"https://example.com/{index}",
            score=1.0,
        )
        for index in range(1, 4)
    ]
    second = [first[0]]  # 与第一次搜索完全相同（collection+URL+标题），应被去重
    second.append(
        RagEvidence(
            evidence_id="W1",
            collection="web_search",
            title="新结果",
            content="新摘要",
            source_url="https://example.com/new",
            score=1.0,
        )
    )

    target = list(first)
    added = AdaptiveAgentRuntime._merge_evidence(target, second, prefix="W")

    assert added == 1  # 第二次搜索中只有 1 条是新的
    assert [item.evidence_id for item in target] == ["W1", "W2", "W3", "W4"]
