import asyncio
import calendar
import html
import re
from datetime import date, datetime, timezone
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from app.config import Settings
from app.core.current_time_tool import CurrentTimeTool
from app.core.model_client import IntentModelClient, RetryCallback
from app.core.reflection import complete_with_reflection
from app.models import RagEvidence


class WebSearchPlan(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    relevance_criteria: list[str] = Field(default_factory=list)
    freshness_required: bool = False


WebSearchSource = Literal[
    "answer_box",
    "knowledge_graph",
    "organic_results",
    "bilibili_video",
    "leetcode_official",
]


class WebSearchResult(BaseModel):
    """SerpAPI 原始结果统一解析后的结构化条目，保证标题、链接、摘要、排名位置等关键字段齐全。"""

    rank: int = Field(ge=1)
    title: str
    url: str | None = None
    snippet: str = ""
    source_type: WebSearchSource
    source_position: int | None = None
    displayed_link: str | None = None
    published_date: str | None = None


class WebSearchError(RuntimeError):
    """SerpAPI 返回了明确错误（如无效密钥、配额耗尽），不应继续重试。"""


SYSTEM_PROMPT = """你是网页搜索 Agent 的查询规划器。根据首脑给出的任务和搜索原因，生成一个精确、单次可执行的网页搜索查询。
不要回答问题，不要假设搜索结果，不要添加用户未要求的主题。
请求载荷中的 current_date 是当前日期。遇到“今天、今日、每日一题、最新”等相对时间要求，查询中必须写入明确日期。
查询 LeetCode 每日一题时，同时包含 LeetCode Daily Challenge、明确日期和 solution，便于检索题目与解法来源。
查询 LeetCode 周赛时，“周赛”指 Weekly Contest，“双周赛”指 Biweekly Contest，不得混淆。
若首脑要定位目标周赛，优先保留或生成 `site:bilibili.com 灵茶山艾府 LeetCode 周赛 明确日期`，
用灵神最新解析的视频标题或摘要定位周赛编号和题号；若首脑要核验题号，则保留或生成
`site:leetcode.cn 周赛编号 题号`，以官方题目页为准。
requested_query 已包含 site:、作者名、周赛编号或题号时必须保留这些限定，不得泛化成普通搜索。
用户未指定版本时，LeetCode 搜索默认限定中国版 `site:leetcode.cn`，不得擅自替换成 `leetcode.com`。
只有 requested_query 明确要求国际版，或搜索原因明确说明中国版已无可用结果时，才允许查询 `.com`。
中国版和国际版的题号、标题与发布状态必须按来源分别记录，不得把国际版 questionId 映射成中国版题号；
若只能取得国际版资料，relevance_criteria 必须要求结果标注“国际版”，查询与证据链接保留 `.com` 域名。
只返回 JSON：
{
  "query": "优化后的搜索查询",
  "relevance_criteria": ["结果必须满足的条件"],
  "freshness_required": false
}"""


class WebSearchAgent:
    def __init__(
        self,
        model_client: IntentModelClient,
        settings: Settings,
        max_reflection_rounds: int = 10,
        current_time_tool: CurrentTimeTool | None = None,
    ) -> None:
        self.model_client = model_client
        self.settings = settings
        self.max_reflection_rounds = max_reflection_rounds
        self.current_time_tool = current_time_tool or CurrentTimeTool(
            settings.app_timezone
        )

    def available(self) -> bool:
        return bool(
            self.settings.web_search_enabled
            and self.settings.serpapi_api_key is not None
            # 空字符串密钥（如 .env 中写了 serpapi-key=）不算可用，
            # 否则每轮搜索都会触发 401。
            and self.settings.serpapi_api_key.get_secret_value().strip()
        )

    async def search(
        self,
        query: str,
        reason: str,
        on_retry: RetryCallback | None = None,
    ) -> tuple[list[RagEvidence], str]:
        if not self.available():
            return [], "web-search-unavailable"
        # 查询规划与 SerpAPI 请求都在同一层捕获异常并降级为空证据：
        # 网页搜索是辅助能力，任何一步失败都不应让整轮对话直接 502。
        search_backend = "serpapi"
        try:
            plan, provider, _ = await complete_with_reflection(
                model_client=self.model_client,
                agent_name="网页搜索 Agent 查询规划器",
                system_prompt=SYSTEM_PROMPT,
                request_payload={
                    "requested_query": query,
                    "reason": reason,
                    "current_date": self.current_time_tool.current_date().isoformat(),
                },
                model_type=WebSearchPlan,
                on_retry=on_retry,
                max_tokens=800,
                max_reflection_rounds=self.max_reflection_rounds,
            )
            search_query = self._make_date_aware(plan.query)
            if self._is_bilibili_query(search_query):
                search_backend = "bilibili"
                response = await self._bilibili_search(search_query)
                results = self._extract_bilibili_results(response)
            elif self._is_leetcode_contest_query(search_query):
                search_backend = "leetcode-graphql"
                response = await self._leetcode_contest_search(search_query)
                results = self._extract_leetcode_contest_results(response)
            else:
                response = await self._serpapi_search(
                    search_query,
                    freshness_required=plan.freshness_required,
                )
                results = self._extract_results(response)
        except (httpx.HTTPError, WebSearchError, ValueError, RuntimeError) as error:
            # ValueError 覆盖 AgentProtocolExhaustedError（规划器反思耗尽）；
            # httpx.HTTPError 覆盖模型上游断连/状态码错误与 SerpAPI 网络错误。
            # 返回 serpapi-error 前缀，供 adaptive_runtime 记入 known_limits 并继续执行。
            return [], f"{search_backend}-error:{type(error).__name__}"
        evidence: list[RagEvidence] = []
        for item in results[: self.settings.web_search_max_results]:
            evidence.append(RagEvidence(
                evidence_id=f"W{item.rank}",
                collection="web_search",
                title=item.title,
                content=item.snippet[:3_500],
                source_url=item.url,
                score=max(0.0, 1.0 - (item.rank - 1) * 0.1),
                metadata={
                    key: value
                    for key, value in {
                        "search_query": search_query,
                        "rank": item.rank,
                        "source_type": item.source_type,
                        "source_position": item.source_position,
                        "displayed_link": item.displayed_link,
                        "published_date": item.published_date,
                        "relevance_criteria": plan.relevance_criteria,
                        "freshness_required": plan.freshness_required,
                    }.items()
                    if value is not None
                },
            ))
        return evidence, f"{provider}+{search_backend}"

    @staticmethod
    def _is_bilibili_query(query: str) -> bool:
        lowered = query.casefold()
        return any(
            marker in lowered
            for marker in (
                "site:bilibili.com",
                "bilibili",
                "哔哩哔哩",
                "b站",
                "灵茶山艾府",
                "灵神",
            )
        )

    @staticmethod
    def _bilibili_keyword(query: str) -> str:
        contest_number = re.search(
            r"(?:周赛|weekly\s+contest)\s*#?\s*(\d{2,3})(?![-\d])",
            query,
            re.IGNORECASE,
        )
        if contest_number:
            return f"LeetCode 周赛 {contest_number.group(1)}"
        if "双周赛" in query or "biweekly contest" in query.casefold():
            return "LeetCode 双周赛"
        return "LeetCode 周赛"

    @staticmethod
    def _is_leetcode_contest_query(query: str) -> bool:
        lowered = query.casefold()
        return (
            ("leetcode" in lowered or "力扣" in lowered)
            and (
                "周赛" in lowered
                or "weekly contest" in lowered
                or "weekly-contest" in lowered
            )
            and "双周赛" not in lowered
            and "biweekly contest" not in lowered
            and "biweekly-contest" not in lowered
        )

    async def _bilibili_search(self, query: str) -> dict[str, Any]:
        """读取 B 站公开搜索结果；只保留灵茶山艾府投稿由解析阶段完成。"""
        params = {
            "search_type": "video",
            "keyword": self._bilibili_keyword(query),
            "order": "pubdate",
            "page": 1,
        }
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/127 Safari/537.36"
            ),
            "Referer": "https://www.bilibili.com/",
        }
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
                    response = await client.get(
                        "https://api.bilibili.com/x/web-interface/search/type",
                        params=params,
                    )
                response.raise_for_status()
                body = response.json()
                if not isinstance(body, dict):
                    raise ValueError("Bilibili search response must be an object")
                if int(body.get("code") or 0) != 0:
                    raise WebSearchError(str(body.get("message") or "Bilibili search failed"))
                return body
            except WebSearchError:
                raise
            except (httpx.TransportError, httpx.TimeoutException, httpx.HTTPStatusError) as error:
                last_error = error
                retryable = not isinstance(error, httpx.HTTPStatusError) or (
                    error.response.status_code in {412, 429}
                    or error.response.status_code >= 500
                )
                if attempt >= 4 or not retryable:
                    raise
                await asyncio.sleep(min(0.5 * (2**attempt), 8.0))
        raise RuntimeError("Bilibili search retry loop ended unexpectedly") from last_error

    async def _leetcode_contest_search(self, query: str) -> dict[str, Any]:
        contest_number = self._contest_number_from_query(query)
        target_date = self._date_from_query(query)
        if contest_number is None:
            upcoming = await self._leetcode_graphql(
                "query upcomingContests { upcomingContests { title titleSlug startTime duration } }",
                {},
            )
            contest_number = self._infer_weekly_contest_number(upcoming, target_date)
        if contest_number is None:
            raise WebSearchError("Unable to resolve the target Weekly Contest number")

        contest_slug = f"weekly-contest-{contest_number}"
        question_payload = await self._leetcode_graphql(
            """
            query contestQuestionList($contestSlug: String!) {
              contestQuestionList(contestSlug: $contestSlug) {
                credit
                title
                titleSlug
                questionId
              }
            }
            """,
            {"contestSlug": contest_slug},
        )
        questions = (question_payload.get("data") or {}).get("contestQuestionList") or []
        enriched_questions: list[dict[str, Any]] = []
        for item in questions:
            if not isinstance(item, dict):
                continue
            title_slug = str(item.get("titleSlug") or "").strip()
            detail = None
            if title_slug:
                try:
                    detail_payload = await self._leetcode_graphql(
                        """
                        query questionData($titleSlug: String!) {
                          question(titleSlug: $titleSlug) {
                            questionId
                            title
                            titleSlug
                            translatedTitle
                            content
                            translatedContent
                            difficulty
                            topicTags { name slug }
                          }
                        }
                        """,
                        {"titleSlug": title_slug},
                    )
                    detail = (detail_payload.get("data") or {}).get("question")
                except (httpx.HTTPError, WebSearchError, ValueError, RuntimeError):
                    # Contest list and official problem URLs are still useful if
                    # one problem-detail request is temporarily unavailable.
                    detail = None
            enriched_questions.append({**item, "detail": detail})
        return {
            "contest_number": contest_number,
            "contest_slug": contest_slug,
            "target_date": target_date.isoformat() if target_date else None,
            "questions": enriched_questions,
        }

    async def _leetcode_graphql(
        self,
        query: str,
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/127 Safari/537.36"
            ),
            "Referer": "https://leetcode.com/contest/",
            "Origin": "https://leetcode.com",
        }
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
                    response = await client.post(
                        "https://leetcode.com/graphql",
                        json={"query": query, "variables": variables},
                    )
                response.raise_for_status()
                body = response.json()
                if not isinstance(body, dict):
                    raise ValueError("LeetCode GraphQL response must be an object")
                if body.get("errors"):
                    raise WebSearchError(str(body["errors"])[:500])
                return body
            except WebSearchError:
                raise
            except (httpx.TransportError, httpx.TimeoutException, httpx.HTTPStatusError) as error:
                last_error = error
                retryable = not isinstance(error, httpx.HTTPStatusError) or (
                    error.response.status_code == 429
                    or error.response.status_code >= 500
                )
                if attempt >= 4 or not retryable:
                    raise
                await asyncio.sleep(min(0.5 * (2**attempt), 8.0))
        raise RuntimeError("LeetCode GraphQL retry loop ended unexpectedly") from last_error

    @staticmethod
    def _contest_number_from_query(query: str) -> int | None:
        match = re.search(
            r"(?:周赛|weekly\s+contest)\s*#?\s*(\d{2,3})(?![-\d])",
            query,
            re.IGNORECASE,
        )
        return int(match.group(1)) if match else None

    @staticmethod
    def _date_from_query(query: str) -> date | None:
        match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", query)
        if not match:
            return None
        try:
            return date.fromisoformat(match.group(1))
        except ValueError:
            return None

    @staticmethod
    def _infer_weekly_contest_number(
        upcoming_payload: dict[str, Any],
        target_date: date | None,
    ) -> int | None:
        contests = (upcoming_payload.get("data") or {}).get("upcomingContests") or []
        weekly = next(
            (
                item
                for item in contests
                if isinstance(item, dict)
                and str(item.get("title") or "").casefold().startswith("weekly contest")
            ),
            None,
        )
        if weekly is None:
            return None
        title_match = re.search(r"(\d{2,3})$", str(weekly.get("title") or ""))
        start_time = weekly.get("startTime")
        if not title_match or not isinstance(start_time, (int, float)):
            return None
        upcoming_number = int(title_match.group(1))
        if target_date is None:
            return upcoming_number
        upcoming_date = datetime.fromtimestamp(start_time, tz=timezone.utc).date()
        days_until_upcoming = (upcoming_date - target_date).days
        if days_until_upcoming < -1 or days_until_upcoming > 35:
            return None
        weeks_back = max(0, round(days_until_upcoming / 7))
        return upcoming_number - weeks_back

    def _make_date_aware(self, query: str) -> str:
        lowered = query.casefold()
        daily_terms = ("每日一题", "今天", "今日", "daily challenge", "today")
        contest_terms = (
            "周赛",
            "weekly contest",
            "weekly-contest",
            "双周赛",
            "biweekly contest",
            "biweekly-contest",
        )
        relative_contest_terms = (
            "周末",
            "这周",
            "本周",
            "上周",
            "最新",
            "最近",
            "weekend",
            "this week",
            "last week",
            "latest",
        )
        is_daily = any(term in lowered for term in daily_terms)
        is_relative_contest = (
            any(term in lowered for term in contest_terms)
            and any(term in lowered for term in relative_contest_terms)
        )
        if "leetcode" not in lowered or not (is_daily or is_relative_contest):
            return query
        today = self.current_time_tool.current_date()
        iso_date = today.isoformat()
        if iso_date in query:
            return query
        english_date = f"{calendar.month_name[today.month]} {today.day} {today.year}"
        chinese_date = f"{today.year}年{today.month}月{today.day}日"
        if is_relative_contest:
            return f"{query} {iso_date} {chinese_date}"
        return (
            f"{query} LeetCode Daily Challenge {iso_date} {english_date} "
            f"{chinese_date} solution"
        )

    async def _serpapi_search(
        self,
        query: str,
        *,
        freshness_required: bool,
    ) -> dict[str, Any]:
        params = {
            "engine": "google",
            "q": query,
            "api_key": self.settings.serpapi_api_key.get_secret_value(),
            "num": self.settings.web_search_max_results,
            "hl": "zh-cn",
        }
        if (
            freshness_required
            and self.current_time_tool.current_date().isoformat() not in query
        ):
            params["tbs"] = "qdr:m"
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(
                        "https://serpapi.com/search.json",
                        params=params,
                    )
                status_code = response.status_code
                if status_code >= 400:
                    # 429/5xx 保留原重试语义；其余 4xx 属于确定性失败，
                    # 优先提取 SerpAPI JSON 错误体（如 "Invalid API key"），
                    # 避免 raise_for_status 只留下无信息量的 HTTPStatusError。
                    retryable_status = status_code == 429 or status_code >= 500
                    api_error = self._extract_api_error(response)
                    if api_error and not retryable_status:
                        raise WebSearchError(api_error)
                response.raise_for_status()
                body = response.json()
                if not isinstance(body, dict):
                    raise ValueError("SerpAPI search response must be an object")
                api_error = body.get("error")
                if api_error:
                    raise WebSearchError(str(api_error))
                return body
            except WebSearchError:
                raise
            except (httpx.TransportError, httpx.TimeoutException, httpx.HTTPStatusError) as error:
                last_error = error
                retryable = not isinstance(error, httpx.HTTPStatusError) or (
                    error.response.status_code == 429
                    or error.response.status_code >= 500
                )
                if attempt >= 4 or not retryable:
                    raise
                await asyncio.sleep(min(0.5 * (2**attempt), 8.0))
        raise RuntimeError("SerpAPI search retry loop ended unexpectedly") from last_error

    @staticmethod
    def _extract_api_error(response: httpx.Response) -> str | None:
        """尽力从 SerpAPI 错误响应体中提取人类可读的 error 字段。"""
        try:
            body = response.json()
        except ValueError:
            return None
        if isinstance(body, dict) and body.get("error"):
            return str(body["error"])
        return None

    @staticmethod
    def _extract_results(payload: dict[str, Any]) -> list[WebSearchResult]:
        """把 SerpAPI 各类结果块统一解析为结构化条目，并按展示顺序赋予连续排名。"""
        results: list[WebSearchResult] = []
        seen_urls: set[str] = set()

        def append(
            *,
            title: Any,
            url: Any,
            snippet: Any,
            source_type: WebSearchSource,
            source_position: Any = None,
            displayed_link: Any = None,
            published_date: Any = None,
        ) -> None:
            url_value = str(url).strip() if url else None
            snippet_text = str(snippet or "").strip()
            title_text = str(title or "").strip() or url_value or "网页搜索结果"
            if not snippet_text:
                return
            if url_value is not None:
                if url_value in seen_urls:
                    return
                seen_urls.add(url_value)
            position = (
                source_position
                if isinstance(source_position, int) and source_position >= 1
                else None
            )
            results.append(WebSearchResult(
                rank=len(results) + 1,
                title=title_text,
                url=url_value,
                snippet=snippet_text,
                source_type=source_type,
                source_position=position,
                displayed_link=str(displayed_link).strip() if displayed_link else None,
                published_date=str(published_date).strip() if published_date else None,
            ))

        answer_boxes = payload.get("answer_box")
        if isinstance(answer_boxes, dict):
            answer_boxes = [answer_boxes]
        if isinstance(answer_boxes, list):
            for box in answer_boxes:
                if not isinstance(box, dict):
                    continue
                answer_content = (
                    box.get("answer")
                    or box.get("snippet")
                    or box.get("result")
                )
                if not answer_content:
                    continue
                append(
                    title=box.get("title") or "Google 精选摘要",
                    url=box.get("link"),
                    snippet=answer_content,
                    source_type="answer_box",
                )

        knowledge_graph = payload.get("knowledge_graph")
        if isinstance(knowledge_graph, dict):
            graph_parts = [
                str(knowledge_graph.get(key)).strip()
                for key in ("type", "description")
                if knowledge_graph.get(key)
            ]
            if graph_parts:
                append(
                    title=knowledge_graph.get("title") or "Google 知识面板",
                    url=knowledge_graph.get("website"),
                    snippet="；".join(graph_parts),
                    source_type="knowledge_graph",
                )

        organic_results = payload.get("organic_results")
        if isinstance(organic_results, list):
            for result in organic_results:
                if not isinstance(result, dict):
                    continue
                append(
                    title=result.get("title"),
                    url=result.get("link"),
                    # snippet 缺失时不能退回 snippet_highlighted_words：
                    # 它是关键词数组而不是摘要文本，直接拼接会产生乱码。
                    snippet=result.get("snippet"),
                    source_type="organic_results",
                    source_position=result.get("position"),
                    displayed_link=result.get("displayed_link"),
                    published_date=result.get("date"),
                )
        return results

    @staticmethod
    def _extract_bilibili_results(payload: dict[str, Any]) -> list[WebSearchResult]:
        data = payload.get("data")
        raw_results = data.get("result") if isinstance(data, dict) else None
        if not isinstance(raw_results, list):
            return []
        results: list[WebSearchResult] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            if item.get("mid") != 206214 and item.get("author") != "灵茶山艾府":
                continue
            title = html.unescape(re.sub(r"<[^>]+>", "", str(item.get("title") or ""))).strip()
            description = html.unescape(
                re.sub(r"<[^>]+>", "", str(item.get("description") or ""))
            ).strip()
            bvid = str(item.get("bvid") or "").strip()
            if not title or not bvid:
                continue
            published_date = None
            pubdate = item.get("pubdate")
            if isinstance(pubdate, (int, float)) and pubdate > 0:
                published_date = datetime.fromtimestamp(
                    pubdate,
                    tz=timezone.utc,
                ).date().isoformat()
            results.append(WebSearchResult(
                rank=len(results) + 1,
                title=title,
                url=f"https://www.bilibili.com/video/{bvid}",
                snippet=description or title,
                source_type="bilibili_video",
                source_position=len(results) + 1,
                displayed_link="bilibili.com",
                published_date=published_date,
            ))
        return results

    @staticmethod
    def _extract_leetcode_contest_results(payload: dict[str, Any]) -> list[WebSearchResult]:
        contest_number = payload.get("contest_number")
        contest_slug = str(payload.get("contest_slug") or "")
        target_date = payload.get("target_date")
        questions = payload.get("questions")
        if not contest_number or not contest_slug or not isinstance(questions, list):
            return []
        results: list[WebSearchResult] = []
        for item in questions:
            if not isinstance(item, dict):
                continue
            detail = item.get("detail") if isinstance(item.get("detail"), dict) else {}
            title = str(
                detail.get("translatedTitle")
                or detail.get("title")
                or item.get("title")
                or ""
            ).strip()
            title_slug = str(detail.get("titleSlug") or item.get("titleSlug") or "").strip()
            question_id = str(detail.get("questionId") or item.get("questionId") or "").strip()
            if not title or not title_slug:
                continue
            difficulty = str(detail.get("difficulty") or "unknown")
            tags = detail.get("topicTags") if isinstance(detail.get("topicTags"), list) else []
            tag_names = [
                str(tag.get("name") or "").strip()
                for tag in tags
                if isinstance(tag, dict) and tag.get("name")
            ]
            statement_html = str(
                detail.get("translatedContent") or detail.get("content") or ""
            )
            statement = html.unescape(
                re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", statement_html))
            ).strip()
            results.append(WebSearchResult(
                rank=len(results) + 1,
                title=f"LeetCode {question_id}: {title}" if question_id else title,
                url=f"https://leetcode.com/problems/{title_slug}/",
                snippet=(
                    f"LeetCode Weekly Contest {contest_number} official problem; "
                    f"questionId={question_id}; credit={item.get('credit')}; "
                    f"difficulty={difficulty}; tags={', '.join(tag_names) or 'unknown'}; "
                    f"contest=https://leetcode.com/contest/{contest_slug}/; "
                    f"statement={statement[:2_700]}"
                ),
                source_type="leetcode_official",
                source_position=len(results) + 1,
                displayed_link="leetcode.com",
                published_date=str(target_date) if target_date else None,
            ))
        return results
