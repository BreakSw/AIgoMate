import asyncio
import html
import json
import re
from datetime import date, datetime, timezone
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, Field

from app.config import Settings
from app.core.current_time_tool import CurrentTimeTool
from app.core.model_client import IntentModelClient, RetryCallback
from app.core.reflection import complete_with_reflection
from app.models import RagEvidence


WebSearchSource = Literal[
    "answer_box",
    "knowledge_graph",
    "organic_results",
    "bilibili_video",
    "leetcode_official",
]
WebSearchProvider = Literal["serpapi", "bilibili", "leetcode_graphql"]


class WebSearchPlan(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    provider: WebSearchProvider = "serpapi"
    target_date: date | None = None
    contest_number: int | None = Field(default=None, ge=1)
    relevance_criteria: list[str] = Field(default_factory=list)
    freshness_required: bool = False


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


SYSTEM_PROMPT = """你是 AlgoMate 的网页搜索查询规划 Agent。首脑只描述本轮要查明的信息；你负责选择最合适的
搜索工具并生成一次可执行的调用计划。你不回答用户问题、不虚构搜索结果、不执行第二个工具，也不把常见流程写死。

一、工具边界与选择
- bilibili：通过 B站公开搜索接口查找视频和作者投稿，适合从标题、摘要、作者与发布日期中定位线索。query 使用
  B站站内可理解的纯关键词，例如 `灵茶山艾府 LeetCode 周赛 2026-08-30`；不要使用 site: 操作符。无需 SerpAPI Key。
- leetcode_graphql：通过 LeetCode 官方 GraphQL 核验 Weekly Contest 官方题单，优先中国站；若中国站被访问保护拦截，
  可回退国际站官方接口，并在证据中保留实际域名。只适用于周赛官方题单，不适用于
  每日一题、普通题目题解、博客或任意 LeetCode 页面。必须提供 contest_number 或 target_date；query 仅用于审计。
- serpapi：检索普通 HTTPS 网页、官方文档、第三方教程、新闻、Daily Challenge、普通 LeetCode 页面以及其他站点。
  query 可使用 site:、引号、版本号和日期等搜索限定，需要用户在 Redis 中配置 SerpAPI Key。

每次只能选择一个 provider。选择依据是本轮搜索目的，而不是单个关键词：出现“LeetCode”不等于使用 GraphQL，
出现“视频”也不等于一定使用 B站。如果 requested_query 含站点、作者、题号、版本或日期限定，应保留其语义；只有
当 provider=bilibili 时移除不适用的 site:bilibili.com 操作符。禁止在一个 query 中混写两个工具的调用意图。

二、时间与版本
- current_date 是应用当前日期。把“今天、昨天、本周、上周、这个周末、最近、最新”等相对表达换算为可审计的
  绝对日期或日期范围；若首脑已给绝对日期，以首脑提供的日期为准。不得依据模型记忆猜赛事编号或发布日期。
- freshness_required 仅在结果会随时间变化时设为 true，例如当天题目、最新版本、新闻和近期赛事；稳定概念设为 false。
- 用户未指定版本时，“LeetCode/力扣”默认中国站。SerpAPI 查询优先加 `site:leetcode.cn`；GraphQL 优先中国站，只有
  中国站接口被 403 等访问保护拦截时才使用国际站同一官方接口，并在 relevance_criteria 中要求标记实际域名。
  禁止把两站的 questionId、标题、发布日期或 titleSlug 自行映射。

三、LeetCode 常见场景
- 周赛与双周赛严格区分：“周赛”是 Weekly Contest；只有用户明确说“双周赛”才按 Biweekly Contest。
- 用户询问“上周/本周/最新周赛”，且本轮目标是取得官方赛题列表时，只要目标日期已经明确，优先选择
  leetcode_graphql 并填写 target_date；官方工具可以由日期解析 Weekly Contest，不必先知道场次编号。
- 只有本轮目标明确是寻找灵茶山艾府解析、B站视频，或需要用第三方标题辅助定位场次时才选择 bilibili。B站站内
  query 使用稳定关键词 `灵茶山艾府 力扣周赛`，不要把 `YYYY-MM-DD` 当作必须命中的搜索词；目标日期写入
  relevance_criteria，通过结果的 published_date 判断哪条视频对应目标周赛。
- 当 requested_query 或 reason 已包含上一轮证据确认的周赛编号，且本轮目标是核验官方赛题时，选择
  leetcode_graphql 并填写 contest_number；若只有可靠目标日期，可填写 target_date 让官方接口解析对应 Weekly Contest。
- B站结果只作为定位线索，LeetCode GraphQL 才是当前周赛题单的官方核验来源。用户同时需要题目和第三方解析时，
  先取得官方题单，首脑读完证据后再决定是否搜索对应场次的解析；
  你只规划当前一轮，绝不能自行串联 bilibili → leetcode_graphql。
- 已有官方题单后，若本轮目标是寻找每题解析、C++ 实现或灵茶山艾府（EndlessCheng）的做法，不能再选择
  leetcode_graphql，因为该接口只提供题面而不提供第三方题解。优先选择 serpapi，查询应包含已确认的场次/题号/
  titleSlug、`灵茶山艾府` 或 `EndlessCheng`、`题解`、所需语言，并优先限定 `site:leetcode.cn/problems`；若目标明确是
  找灵神的周赛讲解视频才选择 bilibili。SerpAPI 结果必须能从标题、URL 或摘要确认作者与目标题号，不能只因关键词相似采用。
- B站视频摘要通常不含完整算法与代码。它可以确认“灵神讲过哪场/哪些题”，但用户要求可复制实现时还应由首脑根据
  已核验官方题面独立生成和验证代码，或继续寻找实际包含题解正文的 LeetCode/官方页面；不得把视频标题冒充代码证据。
- “每日一题”不是周赛。查询每日一题时选择 serpapi，加入 `LeetCode Daily Challenge`、明确日期、`solution` 和
  `site:leetcode.cn`。摘要若缺少明确日期、题号或标题及来源，应在 relevance_criteria 中判定为不足以核验。

四、其他常见搜索
- 用户指定 URL 或域名：保留域名和页面主题；用 SerpAPI 定位该页或同域官方资料。搜索摘要不等于读过网页正文，
  relevance_criteria 应要求标题、URL 和摘要确实支持目标事实。
- 框架/API/模型版本：查询中保留准确产品名、版本号、语言和发布日期，优先官方文档或发布说明。
- 报错诊断：保留最有辨识度的原始错误文本并用引号包裹，同时加入框架、版本和运行环境；不要把整段密钥或隐私数据放入查询。
- 算法题：已有完整题面或代码且只需推理时通常无需搜索；若首脑仍要求搜索，只查询缺失的官方约束、题号或权威资料，
  不要把问题泛化为宽泛的“算法教程”。
- 搜索第三方解析时，relevance_criteria 应要求结果与目标题号、场次、日期和作者相符；不得凭标题相似拼接事实。

五、计划质量
- query 要短而有区分度，通常包含主题实体、要核验的字段、日期/版本和必要站点限制；不要堆叠同义词。
- relevance_criteria 写 1 到 4 条可从返回结果验证的条件，不要写无法由摘要或该工具证明的要求。
- 若当前工具无法满足搜索目的，仍选择最接近的 provider，并通过 relevance_criteria 限定证据边界；不要伪造能力。
- provider=leetcode_graphql 时，contest_number 与 target_date 至少一个非空；已有已核验场次优先编号，未知时不得臆造。

只返回 JSON：
{
  "query": "优化后的搜索查询",
  "provider": "serpapi | bilibili | leetcode_graphql",
  "target_date": "YYYY-MM-DD 或 null",
  "contest_number": "已由上游证据确认的周赛编号，未知时为 null",
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
        # Bilibili public search and LeetCode CN GraphQL remain usable without
        # a SerpAPI credential. The model-selected provider is checked inside
        # search(), after the query planner has made an explicit tool choice.
        return self.settings.web_search_enabled

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
                validator=self._validate_plan,
            )
            search_query = plan.query.strip()
            if plan.provider == "bilibili":
                search_backend = "bilibili"
                response = await self._bilibili_search(search_query)
                results = self._extract_bilibili_results(response)
            elif plan.provider == "leetcode_graphql":
                response = await self._leetcode_contest_search(
                    plan.contest_number,
                    plan.target_date,
                )
                official_host = urlsplit(
                    str(response.get("official_base_url") or "https://leetcode.cn")
                ).hostname
                search_backend = (
                    "leetcode-com-graphql"
                    if official_host == "leetcode.com"
                    else "leetcode-cn-graphql"
                )
                results = self._extract_leetcode_contest_results(response)
            else:
                key = self.model_client.current_serpapi_api_key
                if not key or not key.strip():
                    raise WebSearchError("当前 Redis 配置未提供 SerpAPI Key")
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
        ranked_results = [
            item.model_copy(update={"rank": rank})
            for rank, item in enumerate(results, start=1)
        ]
        for item in ranked_results[: self.settings.web_search_max_results]:
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
    def _validate_plan(plan: WebSearchPlan) -> None:
        if plan.provider == "leetcode_graphql" and (
            plan.contest_number is None and plan.target_date is None
        ):
            raise ValueError(
                "leetcode_graphql 需要 contest_number 或 target_date"
            )

    @staticmethod
    def _bilibili_keyword(query: str) -> str:
        # Only remove a web-search operator that the Bilibili site search does
        # not understand. The model owns the semantic keywords.
        return re.sub(
            r"(?:^|\s)site:bilibili\.com(?=\s|$)",
            " ",
            query,
            flags=re.IGNORECASE,
        ).strip()

    async def _bilibili_search(self, query: str) -> dict[str, Any]:
        """读取 B 站公开搜索结果；语义筛选由模型查询和后续首脑判断完成。"""
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

    async def _leetcode_contest_search(
        self,
        contest_number: int | None,
        target_date: date | None,
    ) -> dict[str, Any]:
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
        official_base_url = str(
            question_payload.get("_official_base_url") or "https://leetcode.cn"
        ).rstrip("/")
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
                            questionFrontendId
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
            "official_base_url": official_base_url,
            "questions": enriched_questions,
        }

    async def _leetcode_graphql(
        self,
        query: str,
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for official_base_url in ("https://leetcode.cn", "https://leetcode.com"):
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/127 Safari/537.36"
                ),
                "Referer": f"{official_base_url}/contest/",
                "Origin": official_base_url,
            }
            for attempt in range(5):
                try:
                    async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
                        response = await client.post(
                            f"{official_base_url}/graphql/",
                            json={"query": query, "variables": variables},
                        )
                    response.raise_for_status()
                    body = response.json()
                    if not isinstance(body, dict):
                        raise ValueError("LeetCode GraphQL response must be an object")
                    if body.get("errors"):
                        raise WebSearchError(str(body["errors"])[:500])
                    body["_official_base_url"] = official_base_url
                    return body
                except WebSearchError as error:
                    last_error = error
                    break
                except (httpx.TransportError, httpx.TimeoutException, httpx.HTTPStatusError) as error:
                    last_error = error
                    retryable = not isinstance(error, httpx.HTTPStatusError) or (
                        error.response.status_code == 429
                        or error.response.status_code >= 500
                    )
                    if attempt >= 4 or not retryable:
                        break
                    await asyncio.sleep(min(0.5 * (2**attempt), 8.0))
                except (ValueError, json.JSONDecodeError) as error:
                    last_error = error
                    break
        raise WebSearchError("LeetCode official GraphQL endpoints are unavailable") from last_error

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

    async def _serpapi_search(
        self,
        query: str,
        *,
        freshness_required: bool,
    ) -> dict[str, Any]:
        params = {
            "engine": "google",
            "q": query,
            "api_key": self.model_client.current_serpapi_api_key,
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
                snippet="；".join(
                    part
                    for part in (
                        (
                            f"作者={str(item.get('author') or '').strip()}"
                            if item.get("author")
                            else ""
                        ),
                        description or title,
                    )
                    if part
                ),
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
        official_base_url = str(
            payload.get("official_base_url") or "https://leetcode.cn"
        ).rstrip("/")
        displayed_link = urlsplit(official_base_url).hostname or "leetcode.cn"
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
            question_id = str(
                detail.get("questionFrontendId")
                or detail.get("questionId")
                or item.get("questionId")
                or ""
            ).strip()
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
                url=f"{official_base_url}/problems/{title_slug}/",
                snippet=(
                    f"LeetCode Weekly Contest {contest_number} official problem; "
                    f"questionFrontendId={question_id}; credit={item.get('credit')}; "
                    f"difficulty={difficulty}; tags={', '.join(tag_names) or 'unknown'}; "
                    f"contest={official_base_url}/contest/{contest_slug}/; "
                    f"statement={statement[:2_700]}"
                ),
                source_type="leetcode_official",
                source_position=len(results) + 1,
                displayed_link=displayed_link,
                published_date=str(target_date) if target_date else None,
            ))
        return results
