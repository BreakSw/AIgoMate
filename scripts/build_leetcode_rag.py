"""Build the auditable LeetCode portions of the AlgoMate RAG staging set.

The script uses Firecrawl for every page body. LeetCode's public GraphQL API is
used only for study-plan membership, stable identifiers, tags, and public
solution-card counters. No embedding is performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from crawl_rag_sources import FirecrawlClient, PROJECT_ROOT, canonical_url, save_records


RAW_ROOT = PROJECT_ROOT / "rag-data" / "raw"
CONCEPT_ROOT = RAW_ROOT / "algorithm-concepts" / "programmercarl"
PROBLEM_ROOT = RAW_ROOT / "problem-bank" / "leetcode"
CASE_ROOT = RAW_ROOT / "code-cases" / "leetcode"
CATALOG_ROOT = RAW_ROOT / "catalog"

PLAN_SLUGS = {
    "leetcode_hot_100": "top-100-liked",
    "leetcode_interview_150": "top-interview-150",
}

CURRICULUM_CATEGORIES = (
    "array",
    "linked-list",
    "hash-table",
    "string",
    "two-pointers",
    "stack-queue",
    "binary-tree",
    "backtracking",
    "greedy",
    "dynamic-programming",
    "monotonic-stack",
    "graph",
)

CATEGORY_TAGS = {
    "array": {"array", "matrix"},
    "linked-list": {"linked-list"},
    "hash-table": {"hash-table"},
    "string": {"string", "string-matching"},
    "two-pointers": {"two-pointers", "sliding-window"},
    "stack-queue": {"stack", "queue", "monotonic-queue"},
    "binary-tree": {"binary-tree", "binary-search-tree", "tree"},
    "backtracking": {"backtracking"},
    "greedy": {"greedy"},
    "dynamic-programming": {"dynamic-programming", "memoization"},
    "monotonic-stack": {"monotonic-stack"},
    "graph": {
        "graph",
        "depth-first-search",
        "breadth-first-search",
        "topological-sort",
        "shortest-path",
        "minimum-spanning-tree",
        "union-find",
    },
}

LEETCODE_LINK = re.compile(
    r"https?://leetcode(?:\.cn|\.com)/problems/([a-z0-9-]+)(?:/|[?#]|\)|$)",
    re.IGNORECASE,
)
SOLUTION_LINK = re.compile(
    r"\[([^\]]+)\]\((https://leetcode\.cn/problems/([a-z0-9-]+)/solutions/\d+/[^)]+)\)",
    re.IGNORECASE,
)
AUTHOR_LINK = re.compile(
    r"\[([^\]\n]+)\]\(https://leetcode\.cn/u/([^/]+)/\)",
    re.IGNORECASE,
)
METRIC_LINE = re.compile(r"^\d+(?:\.\d+)?[KkMm]?$|^\d+(?:\.\d+)?万$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
        newline="\n",
    )


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class LeetCodePublicClient:
    """Read-only metadata client with the project's five-attempt retry cap."""

    endpoint = "https://leetcode.com/graphql/"

    def __init__(self) -> None:
        self.client = httpx.Client(
            headers={"Referer": "https://leetcode.com/", "User-Agent": "AlgoMate-RAG-Collector/1.0"},
            timeout=httpx.Timeout(45.0),
        )

    def graphql(self, query: str, variables: dict) -> dict:
        last_error: Exception | None = None
        for attempt in range(1, 6):
            try:
                response = self.client.post(
                    self.endpoint,
                    json={"query": query, "variables": variables},
                )
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
                response.raise_for_status()
                payload = response.json()
                if payload.get("errors"):
                    raise RuntimeError(json.dumps(payload["errors"], ensure_ascii=False))
                return payload["data"]
            except (httpx.HTTPError, ValueError, RuntimeError) as exc:
                last_error = exc
                if attempt == 5:
                    break
                time.sleep(min(2 ** (attempt - 1), 8))
        raise RuntimeError("LeetCode public metadata call failed after five attempts") from last_error

    def study_plan(self, slug: str) -> dict:
        query = """
        query studyPlanV2Detail($slug: String!) {
          studyPlanV2Detail(planSlug: $slug) {
            name
            planSubGroups {
              name
              questions { questionFrontendId titleSlug title translatedTitle }
            }
          }
        }
        """
        return self.graphql(query, {"slug": slug})["studyPlanV2Detail"]

    def question(self, slug: str) -> dict:
        query = """
        query questionData($titleSlug: String!) {
          question(titleSlug: $titleSlug) {
            questionId
            questionFrontendId
            title
            titleSlug
            difficulty
            topicTags { name slug translatedName }
          }
        }
        """
        return self.graphql(query, {"titleSlug": slug})["question"]

    def solution_articles(self, slug: str, first: int = 100) -> list[dict]:
        query = """
        query solutionArticles(
          $questionSlug: String!, $skip: Int!, $first: Int,
          $orderBy: ArticleOrderByEnum
        ) {
          ugcArticleSolutionArticles(
            questionSlug: $questionSlug, skip: $skip,
            first: $first, orderBy: $orderBy
          ) {
            edges {
              node {
                title slug topicId hitCount
                reactions { count reactionType }
                author { userName userSlug }
              }
            }
          }
        }
        """
        result = self.graphql(
            query,
            {
                "questionSlug": slug,
                "skip": 0,
                "first": first,
                "orderBy": "MOST_VOTES",
            },
        )["ugcArticleSolutionArticles"]
        return [edge["node"] for edge in (result or {}).get("edges") or []]


def concept_candidates() -> dict[str, Counter]:
    counters: dict[str, Counter] = {category: Counter() for category in CURRICULUM_CATEGORIES}
    manifest = read_jsonl(CONCEPT_ROOT / "manifest.jsonl")
    for item in manifest:
        category = item.get("category")
        if category not in counters:
            continue
        markdown = (PROJECT_ROOT / item["markdown_file"]).read_text(encoding="utf-8")
        for slug in set(LEETCODE_LINK.findall(markdown)):
            counters[category][slug.lower()] += 1
    return counters


def fetch_question_metadata(slugs: list[str]) -> tuple[dict[str, dict], dict[str, str]]:
    found: dict[str, dict] = {}
    failures: dict[str, str] = {}

    def fetch(slug: str) -> tuple[str, dict | None, str | None]:
        try:
            return slug, LeetCodePublicClient().question(slug), None
        except Exception as exc:  # failure is retained for audit, not hidden
            return slug, None, type(exc).__name__

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fetch, slug) for slug in slugs]
        for index, future in enumerate(as_completed(futures), start=1):
            slug, metadata, failure = future.result()
            if metadata:
                found[slug] = metadata
            else:
                failures[slug] = failure or "unknown"
            if index % 25 == 0 or index == len(futures):
                print(f"[metadata] {index}/{len(futures)}", flush=True)
    return found, failures


def build_catalog() -> None:
    CATALOG_ROOT.mkdir(parents=True, exist_ok=True)
    client = LeetCodePublicClient()
    plans: dict[str, dict] = {}
    plan_members: dict[str, dict] = defaultdict(lambda: {"source_sets": set(), "groups": defaultdict(list)})
    ordered_plan_slugs: list[str] = []

    for source_name, plan_slug in PLAN_SLUGS.items():
        plan = client.study_plan(plan_slug)
        plans[source_name] = plan
        for group in plan["planSubGroups"]:
            for position, question in enumerate(group["questions"]):
                slug = question["titleSlug"].lower()
                if slug not in ordered_plan_slugs:
                    ordered_plan_slugs.append(slug)
                plan_members[slug]["source_sets"].add(source_name)
                plan_members[slug]["groups"][source_name].append(
                    {"name": group["name"], "position": position}
                )
                plan_members[slug]["question"] = question
        write_json(CATALOG_ROOT / f"{source_name}.firecrawl-support-metadata.json", plan)

    direct = concept_candidates()
    tag_supplements: dict[str, list[str]] = defaultdict(list)
    # The two official study plans and Code随想录 only expose five distinct
    # monotonic-stack questions. Preserve the official LeetCode tag ordering as
    # the auditable supplementation source instead of inventing a hand list.
    firecrawl = FirecrawlClient()
    tag_url = "https://leetcode.cn/tag/monotonic-stack/"
    tag_response = firecrawl.request(
        "POST",
        f"{firecrawl.base_url}/v1/scrape",
        json={
            "url": tag_url,
            "formats": ["markdown", "links"],
            "onlyMainContent": True,
            "removeBase64Images": True,
            "waitFor": 3000,
        },
    )
    tag_response.raise_for_status()
    tag_payload = tag_response.json().get("data") or {}
    seen_tag_slugs: set[str] = set()
    for link in tag_payload.get("links") or []:
        match = LEETCODE_LINK.search(link)
        if match and match.group(1) not in seen_tag_slugs:
            seen_tag_slugs.add(match.group(1))
            tag_supplements["monotonic-stack"].append(match.group(1))
    (CATALOG_ROOT / "leetcode-tag-monotonic-stack.md").write_text(
        tag_payload.get("markdown") or "", encoding="utf-8", newline="\n"
    )
    write_json(
        CATALOG_ROOT / "leetcode-tag-monotonic-stack.firecrawl.json",
        {"source_url": tag_url, "fetched_at": utc_now(), "data": tag_payload},
    )

    direct_union = set().union(*(set(counter) for counter in direct.values()))
    metadata_slugs = sorted(
        set(ordered_plan_slugs)
        | direct_union
        | set().union(*(set(items[:20]) for items in tag_supplements.values()))
    )
    metadata, failures = fetch_question_metadata(metadata_slugs)
    write_json(
        CATALOG_ROOT / "leetcode-question-metadata.json",
        {"fetched_at": utc_now(), "questions": metadata, "failures": failures},
    )

    curriculum: dict[str, list[str]] = {}
    curriculum_reasons: dict[str, dict[str, str]] = defaultdict(dict)
    plan_rank = {slug: rank for rank, slug in enumerate(ordered_plan_slugs)}

    for category in CURRICULUM_CATEGORIES:
        selected = [slug for slug, _ in direct[category].most_common(10)]
        for slug in selected:
            curriculum_reasons[category][slug] = "代码随想录对应分类中出现频率靠前"
        if len(selected) < 10:
            tag_candidates = []
            for slug in ordered_plan_slugs:
                tags = {tag["slug"] for tag in (metadata.get(slug) or {}).get("topicTags") or []}
                if tags & CATEGORY_TAGS[category] and slug not in selected:
                    tag_candidates.append(slug)
            tag_candidates.sort(key=lambda slug: plan_rank[slug])
            for slug in tag_candidates[: 10 - len(selected)]:
                selected.append(slug)
                curriculum_reasons[category][slug] = "代码随想录该分类不足十题，由官方热门题单按标签补齐"
        if len(selected) < 10:
            for slug in tag_supplements.get(category, []):
                if slug in selected:
                    continue
                selected.append(slug)
                curriculum_reasons[category][slug] = "由力扣官方算法标签页按公开顺序补齐"
                if len(selected) == 10:
                    break
        curriculum[category] = selected

    included_slugs = set(ordered_plan_slugs) | set().union(*(set(items) for items in curriculum.values()))
    records: list[dict] = []
    for slug in sorted(included_slugs, key=lambda value: (plan_rank.get(value, 10**9), value)):
        member = plan_members.get(slug) or {}
        question = metadata.get(slug) or member.get("question") or {}
        curriculum_categories = [category for category, items in curriculum.items() if slug in items]
        source_sets = set(member.get("source_sets") or set())
        if curriculum_categories:
            source_sets.add("programmercarl_curated")
        records.append(
            {
                "problem_id": question.get("questionFrontendId"),
                "title": question.get("translatedTitle") or question.get("title"),
                "title_slug": slug,
                "difficulty": question.get("difficulty"),
                "topic_tags": (question.get("topicTags") or []),
                "problem_url": f"https://leetcode.cn/problems/{slug}/",
                "source_sets": sorted(source_sets),
                "study_plan_groups": {
                    key: value for key, value in (member.get("groups") or {}).items()
                },
                "curriculum_categories": curriculum_categories,
                "curriculum_selection_reasons": {
                    category: curriculum_reasons[category][slug]
                    for category in curriculum_categories
                },
                "cataloged_at": utc_now(),
            }
        )

    write_jsonl(PROBLEM_ROOT / "problem-manifest.jsonl", records)
    stats = {
        "total_unique_questions": len(records),
        "source_counts": dict(
            Counter(source for item in records for source in item["source_sets"])
        ),
        "curriculum_category_counts": {
            category: len(items) for category, items in curriculum.items()
        },
        "metadata_failures": len(failures),
        "generated_at": utc_now(),
    }
    write_json(PROBLEM_ROOT / "catalog-stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False), flush=True)


def scrape_in_batches(
    urls: list[str],
    output_dir: Path,
    label: str,
    batch_size: int,
    wait_for_ms: int | None = None,
) -> None:
    firecrawl = FirecrawlClient()
    # Rehydrate prior batches and checkpoint after every new batch. This makes
    # the collector resumable even when a provider quota is exhausted midway.
    records: list[dict] = []
    json_dir = output_dir / "firecrawl-json"
    if json_dir.exists():
        for path in json_dir.glob("*.json"):
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
    existing = {
        canonical_url(
            (record.get("metadata") or {}).get("sourceURL")
            or (record.get("metadata") or {}).get("url")
            or ""
        )
        for record in records
    }
    remaining = [url for url in urls if canonical_url(url) not in existing]
    for start in range(0, len(remaining), batch_size):
        batch = remaining[start : start + batch_size]
        new_records = firecrawl.batch_scrape(
            batch,
            f"{label} {len(urls) - len(remaining) + start + 1}-"
            f"{len(urls) - len(remaining) + start + len(batch)}/{len(urls)}",
            wait_for_ms=wait_for_ms,
        )
        records.extend(new_records)
        checkpoint = save_records(records, output_dir, urls)
        write_json(output_dir / "crawl-stats.json", checkpoint)
        print(f"[{label}] checkpoint saved={checkpoint['saved']}", flush=True)
    stats = save_records(records, output_dir, urls)
    write_json(output_dir / "crawl-stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False), flush=True)


def scrape_problem_pages(batch_size: int) -> None:
    questions = read_jsonl(PROBLEM_ROOT / "problem-manifest.jsonl")
    scrape_in_batches(
        [item["problem_url"] for item in questions],
        PROBLEM_ROOT / "pages",
        "problem pages",
        batch_size,
    )


def scrape_solution_lists(batch_size: int) -> None:
    questions = read_jsonl(PROBLEM_ROOT / "problem-manifest.jsonl")
    urls = [
        f"https://leetcode.cn/problems/{item['title_slug']}/solutions/?orderBy=most_votes"
        for item in questions
    ]
    scrape_in_batches(urls, CASE_ROOT / "candidate-pages", "solution lists", batch_size)


def scrape_missing_solution_lists(batch_size: int) -> None:
    selected = read_jsonl(CASE_ROOT / "selected-posts.jsonl")
    pending = [item for item in selected if item["status"] == "pending_metrics"]
    urls = [
        f"https://leetcode.cn/problems/{item['problem_slug']}/solutions/?orderBy=most_votes"
        for item in pending
    ]
    scrape_in_batches(
        urls,
        CASE_ROOT / "candidate-pages-rendered",
        "rendered solution lists",
        batch_size,
        wait_for_ms=3000,
    )


def extract_problem_pages() -> None:
    """Reuse the question section rendered above each solution list."""
    questions = read_jsonl(PROBLEM_ROOT / "problem-manifest.jsonl")
    page_manifest = read_jsonl(CASE_ROOT / "candidate-pages" / "manifest.jsonl")
    pages_by_slug: dict[str, dict] = {}
    for item in page_manifest:
        match = re.search(r"/problems/([a-z0-9-]+)/solutions", item["source_url"])
        if match:
            pages_by_slug[match.group(1)] = item

    output_dir = PROBLEM_ROOT / "pages-from-solution-lists"
    markdown_dir = output_dir / "markdown"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    missing: list[dict] = []
    for question in questions:
        slug = question["title_slug"]
        page = pages_by_slug.get(slug)
        if not page:
            missing.append({"title_slug": slug, "reason": "solution list was not captured"})
            continue
        raw = (PROJECT_ROOT / page["markdown_file"]).read_text(encoding="utf-8")
        content = raw.split("评论 (", 1)[0].strip()
        title_link = re.search(
            rf"\[[^\]]+\]\(https://leetcode\.cn/problems/{re.escape(slug)}/?\)",
            content,
        )
        if title_link:
            content = content[title_link.start() :]
        if len(content) < 200:
            missing.append({"title_slug": slug, "reason": "rendered question section was too short"})
            continue
        path = markdown_dir / f"{slug}.md"
        path.write_text(content + "\n", encoding="utf-8", newline="\n")
        manifest.append(
            {
                "problem_id": question.get("problem_id"),
                "title_slug": slug,
                "problem_url": question["problem_url"],
                "source_solution_list_url": page["source_url"],
                "markdown_file": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "markdown_chars": len(content),
                "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "fetched_at": page["fetched_at"],
                "extracted_at": utc_now(),
            }
        )
    write_jsonl(output_dir / "manifest.jsonl", manifest)
    write_json(output_dir / "missing.json", missing)
    stats = {
        "requested": len(questions),
        "saved": len(manifest),
        "missing": len(missing),
        "markdown_chars": sum(item["markdown_chars"] for item in manifest),
    }
    write_json(output_dir / "crawl-stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False), flush=True)


def metric_value(value: str) -> int:
    normalized = value.strip().replace(",", "")
    multiplier = 1
    if normalized.endswith(("K", "k")):
        multiplier = 1_000
        normalized = normalized[:-1]
    elif normalized.endswith(("M", "m")):
        multiplier = 1_000_000
        normalized = normalized[:-1]
    elif normalized.endswith("万"):
        multiplier = 10_000
        normalized = normalized[:-1]
    return int(float(normalized) * multiplier)


def solution_cards(markdown: str, expected_slug: str, source_url: str, fetched_at: str) -> list[dict]:
    if "登录并分享题解" not in markdown:
        return []
    listing = markdown.split("登录并分享题解", 1)[1]
    authors = list(AUTHOR_LINK.finditer(listing))
    cards: list[dict] = []
    seen_urls: set[str] = set()
    for index, author_match in enumerate(authors):
        end = authors[index + 1].start() if index + 1 < len(authors) else len(listing)
        segment = listing[author_match.end() : end]
        solution_match = SOLUTION_LINK.search(segment)
        if not solution_match or solution_match.group(3).lower() != expected_slug:
            continue
        post_url = canonical_url(solution_match.group(2))
        if post_url in seen_urls:
            continue
        seen_urls.add(post_url)
        metrics = [
            line.strip()
            for line in segment[solution_match.end() :].splitlines()
            if METRIC_LINE.fullmatch(line.strip())
        ]
        likes = views = comments = None
        if len(metrics) >= 3:
            likes, views, comments = (metric_value(value) for value in metrics[-3:])
        cards.append(
            {
                "problem_slug": expected_slug,
                "post_url": post_url,
                "title": solution_match.group(1).replace("\\.", "."),
                "author": author_match.group(1),
                "author_slug": author_match.group(2),
                "likes": likes,
                "views": views,
                "comments": comments,
                "metrics_source_url": source_url,
                "metrics_fetched_at": fetched_at,
            }
        )
    return cards


def choose_post(cards: list[dict]) -> tuple[dict | None, str, str]:
    measurable = [card for card in cards if card["likes"] is not None and card["views"] is not None]
    qualifying = [card for card in measurable if card["likes"] >= 100 and card["views"] >= 10_000]

    def best(items: list[dict]) -> dict:
        return max(items, key=lambda item: (item["likes"], item["views"]))

    endlesscheng = [
        card for card in qualifying if card["author_slug"].lower() == "endlesscheng"
    ]
    if endlesscheng:
        return best(endlesscheng), "selected", "优先选择灵茶山艾府，且点赞与浏览量通过质量门槛"
    official = [
        card
        for card in qualifying
        if card["author_slug"].lower() in {"leetcode-solution", "leetcode"}
        or "官方" in card["author"]
    ]
    if official:
        return best(official), "selected", "选择力扣官方题解，且点赞与浏览量通过质量门槛"
    if qualifying:
        return best(qualifying), "selected", "无合格的优先作者题解，选择可验证指标最高的高质量社区题解"
    if measurable:
        return best(measurable), "pending_quality_threshold", "最佳可见题解未同时达到100赞与1万浏览，保留映射并标记待补"
    return None, "pending_metrics", "Firecrawl渲染结果中没有可验证的点赞与浏览量"


def select_solution_posts() -> None:
    questions = {item["title_slug"]: item for item in read_jsonl(PROBLEM_ROOT / "problem-manifest.jsonl")}
    page_manifest = read_jsonl(CASE_ROOT / "candidate-pages" / "manifest.jsonl")
    rendered_manifest = CASE_ROOT / "candidate-pages-rendered" / "manifest.jsonl"
    if rendered_manifest.exists():
        page_manifest.extend(read_jsonl(rendered_manifest))
    all_cards: list[dict] = []
    mapping: list[dict] = []
    pages_by_slug: dict[str, dict] = {}
    for item in page_manifest:
        match = re.search(r"/problems/([a-z0-9-]+)/solutions", item["source_url"])
        if match:
            pages_by_slug[match.group(1)] = item

    for slug, question in questions.items():
        page = pages_by_slug.get(slug)
        cards: list[dict] = []
        if page:
            markdown = (PROJECT_ROOT / page["markdown_file"]).read_text(encoding="utf-8")
            cards = solution_cards(markdown, slug, page["source_url"], page["fetched_at"])
        all_cards.extend(cards)
        chosen, status, reason = choose_post(cards)
        mapping.append(
            {
                "problem_id": question.get("problem_id"),
                "problem_slug": slug,
                "problem_url": question["problem_url"],
                "post_url": chosen.get("post_url") if chosen else None,
                "post_title": chosen.get("title") if chosen else None,
                "author": chosen.get("author") if chosen else None,
                "author_slug": chosen.get("author_slug") if chosen else None,
                "likes": chosen.get("likes") if chosen else None,
                "views": chosen.get("views") if chosen else None,
                "comments": chosen.get("comments") if chosen else None,
                "status": status,
                "selection_reason": reason,
                "source_sets": question["source_sets"],
                "metrics_source_url": chosen.get("metrics_source_url") if chosen else (page or {}).get("source_url"),
                "metrics_fetched_at": chosen.get("metrics_fetched_at") if chosen else (page or {}).get("fetched_at"),
                "selected_at": utc_now(),
            }
        )

    write_jsonl(CASE_ROOT / "solution-candidates.jsonl", all_cards)
    write_jsonl(CASE_ROOT / "selected-posts.jsonl", mapping)
    stats = {
        "questions": len(mapping),
        "candidate_posts": len(all_cards),
        "status_counts": dict(Counter(item["status"] for item in mapping)),
        "author_counts": dict(Counter(item["author_slug"] or "missing" for item in mapping)),
        "generated_at": utc_now(),
    }
    write_json(CASE_ROOT / "selection-stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False), flush=True)


def fallback_public_solution_posts() -> None:
    mapping = read_jsonl(CASE_ROOT / "selected-posts.jsonl")
    pending = [item for item in mapping if item["status"] != "selected"]
    candidates_by_slug: dict[str, list[dict]] = {}
    failures: dict[str, str] = {}

    def fetch(item: dict) -> tuple[str, list[dict], str | None]:
        slug = item["problem_slug"]
        try:
            nodes = LeetCodePublicClient().solution_articles(slug)
            cards: list[dict] = []
            for node in nodes:
                author = node.get("author") or {}
                article_slug = node.get("slug")
                topic_id = node.get("topicId")
                if not article_slug or not topic_id:
                    continue
                reactions = {
                    reaction.get("reactionType"): reaction.get("count", 0)
                    for reaction in node.get("reactions") or []
                }
                cards.append(
                    {
                        "problem_slug": slug,
                        "post_url": (
                            f"https://leetcode.com/problems/{slug}/solutions/"
                            f"{topic_id}/{article_slug}/"
                        ),
                        "title": node.get("title"),
                        "author": author.get("userName"),
                        "author_slug": author.get("userSlug"),
                        "likes": reactions.get("UPVOTE", 0),
                        "views": node.get("hitCount", 0),
                        "comments": None,
                        "metrics_source_url": (
                            f"https://leetcode.com/problems/{slug}/solutions/"
                        ),
                        "metrics_fetched_at": utc_now(),
                    }
                )
            return slug, cards, None
        except Exception as exc:
            return slug, [], type(exc).__name__

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fetch, item) for item in pending]
        for index, future in enumerate(as_completed(futures), start=1):
            slug, cards, failure = future.result()
            candidates_by_slug[slug] = cards
            if failure:
                failures[slug] = failure
            if index % 20 == 0 or index == len(futures):
                print(f"[public solution metadata] {index}/{len(futures)}", flush=True)

    all_candidates = [
        card for cards in candidates_by_slug.values() for card in cards
    ]
    for item in mapping:
        if item["status"] == "selected":
            continue
        chosen, status, reason = choose_post(candidates_by_slug.get(item["problem_slug"], []))
        if not chosen:
            continue
        item.update(
            {
                "post_url": chosen["post_url"],
                "post_title": chosen["title"],
                "author": chosen["author"],
                "author_slug": chosen["author_slug"],
                "likes": chosen["likes"],
                "views": chosen["views"],
                "comments": None,
                "status": status,
                "selection_reason": (
                    "中文页未暴露可验证指标；使用力扣国际站公开指标补选。" + reason
                ),
                "metrics_source_url": chosen["metrics_source_url"],
                "metrics_fetched_at": chosen["metrics_fetched_at"],
                "selected_at": utc_now(),
            }
        )

    write_jsonl(CASE_ROOT / "solution-candidates-public.jsonl", all_candidates)
    write_json(CASE_ROOT / "solution-public-failures.json", failures)
    write_jsonl(CASE_ROOT / "selected-posts.jsonl", mapping)
    stats = {
        "questions": len(mapping),
        "status_counts": dict(Counter(item["status"] for item in mapping)),
        "author_counts": dict(Counter(item["author_slug"] or "missing" for item in mapping)),
        "public_metadata_failures": len(failures),
        "generated_at": utc_now(),
    }
    write_json(CASE_ROOT / "selection-stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False), flush=True)


def scrape_selected_posts(batch_size: int) -> None:
    selected = read_jsonl(CASE_ROOT / "selected-posts.jsonl")
    urls = [item["post_url"] for item in selected if item.get("post_url")]
    scrape_in_batches(urls, CASE_ROOT / "posts", "selected posts", batch_size)


COMMENT_BOUNDARIES = (
    re.compile(r"\n#{0,6}\s*评论\s*(?:\([^\n)]*\))?\s*\n"),
    re.compile(r"\n#{0,6}\s*Comments?\s*(?:\([^\n)]*\))?\s*\n", re.IGNORECASE),
    re.compile(r"\n💡\s*讨论区规则\s*\n"),
)


def clean_selected_posts() -> None:
    """Remove reader comments while retaining Firecrawl JSON as audit evidence."""
    manifest_path = CASE_ROOT / "posts" / "manifest.jsonl"
    manifest = read_jsonl(manifest_path)
    cleaned_count = 0
    removed_chars = 0
    for item in manifest:
        path = PROJECT_ROOT / item["markdown_file"]
        raw = path.read_text(encoding="utf-8")
        boundaries: list[tuple[int, str]] = []
        # Ignore navigation labels near the top; a real comment section follows
        # substantial article content.
        for pattern in COMMENT_BOUNDARIES:
            for match in pattern.finditer(raw):
                if match.start() >= 500:
                    boundaries.append((match.start(), match.group(0).strip()))
        cleaned = raw
        marker = None
        if boundaries:
            boundary, marker = min(boundaries, key=lambda value: value[0])
            cleaned = raw[:boundary].rstrip() + "\n"
        if cleaned != raw:
            cleaned_count += 1
            removed_chars += len(raw) - len(cleaned)
            path.write_text(cleaned, encoding="utf-8", newline="\n")
        item["raw_markdown_chars"] = len(raw)
        item["markdown_chars"] = len(cleaned)
        item["comments_removed"] = cleaned != raw
        item["comment_boundary_marker"] = marker
        item["content_sha256"] = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
        item["cleaned_at"] = utc_now()
    write_jsonl(manifest_path, manifest)
    stats_path = CASE_ROOT / "posts" / "crawl-stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats.update(
        {
            "markdown_chars": sum(item["markdown_chars"] for item in manifest),
            "raw_markdown_chars": sum(item["raw_markdown_chars"] for item in manifest),
            "comment_sections_removed": cleaned_count,
            "comment_chars_removed": removed_chars,
            "cleaned_at": utc_now(),
        }
    )
    write_json(stats_path, stats)
    print(json.dumps(stats, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=[
            "catalog",
            "problem-pages",
            "solution-lists",
            "solution-lists-rendered",
            "extract-problems",
            "select-posts",
            "fallback-posts",
            "posts",
            "clean-posts",
        ],
    )
    parser.add_argument("--batch-size", type=int, default=40)
    args = parser.parse_args()
    batch_size = max(1, min(args.batch_size, 100))

    if args.stage == "catalog":
        build_catalog()
    elif args.stage == "problem-pages":
        scrape_problem_pages(batch_size)
    elif args.stage == "solution-lists":
        scrape_solution_lists(batch_size)
    elif args.stage == "solution-lists-rendered":
        scrape_missing_solution_lists(batch_size)
    elif args.stage == "extract-problems":
        extract_problem_pages()
    elif args.stage == "select-posts":
        select_solution_posts()
    elif args.stage == "fallback-posts":
        fallback_public_solution_posts()
    elif args.stage == "posts":
        scrape_selected_posts(batch_size)
    elif args.stage == "clean-posts":
        clean_selected_posts()


if __name__ == "__main__":
    main()
