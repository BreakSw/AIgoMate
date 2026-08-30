"""Replace inaccessible leetcode.com editorials with auditable public articles.

LeetCode's public GraphQL endpoint supplies both the visible counters and the
article Markdown. This stage is used only for international-site editorials
that redirect anonymous browsers to the problem description.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = PROJECT_ROOT / "rag-data" / "raw" / "code-cases" / "leetcode"
MAPPING_PATH = CASE_ROOT / "selected-posts.jsonl"
POST_ROOT = CASE_ROOT / "posts"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
        newline="\n",
    )


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")


class PublicLeetCode:
    endpoint = "https://leetcode.com/graphql/"

    def request(self, query: str, variables: dict) -> dict:
        last: Exception | None = None
        for attempt in range(1, 6):
            try:
                response = httpx.post(
                    self.endpoint,
                    headers={"Referer": "https://leetcode.com/", "User-Agent": "AlgoMate-RAG-Collector/1.0"},
                    json={"query": query, "variables": variables},
                    timeout=45,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
                response.raise_for_status()
                payload = response.json()
                if payload.get("errors"):
                    raise RuntimeError(json.dumps(payload["errors"], ensure_ascii=False))
                return payload["data"]
            except (httpx.HTTPError, ValueError, RuntimeError) as exc:
                last = exc
                if attempt < 5:
                    time.sleep(min(2 ** (attempt - 1), 8))
        raise RuntimeError("LeetCode public request failed after five attempts") from last

    def candidates(self, problem_slug: str) -> list[dict]:
        query = """
        query candidates($slug: String!, $first: Int) {
          ugcArticleSolutionArticles(
            questionSlug: $slug, skip: 0, first: $first, orderBy: MOST_VOTES
          ) {
            edges {
              node {
                uuid title slug topicId hitCount
                reactions { count reactionType }
                author { userName userSlug }
              }
            }
          }
        }
        """
        result = self.request(query, {"slug": problem_slug, "first": 100})
        return [edge["node"] for edge in result["ugcArticleSolutionArticles"]["edges"]]

    def content(self, article_id: str) -> str:
        query = """
        query article($id: ID!) {
          ugcArticleSolutionArticle(articleId: $id) { content }
        }
        """
        result = self.request(query, {"id": article_id})
        return ((result.get("ugcArticleSolutionArticle") or {}).get("content") or "").strip()


def metric(node: dict, reaction_type: str) -> int:
    for reaction in node.get("reactions") or []:
        if reaction.get("reactionType") == reaction_type:
            return int(reaction.get("count") or 0)
    return 0


def fetch_one(item: dict) -> tuple[str, dict | None, str | None]:
    slug = item["problem_slug"]
    try:
        client = PublicLeetCode()
        candidates = []
        for node in client.candidates(slug):
            author = node.get("author") or {}
            likes = metric(node, "UPVOTE")
            views = int(node.get("hitCount") or 0)
            if author.get("userSlug", "").lower() == "leetcode":
                continue
            if likes < 100 or views < 10_000:
                continue
            candidates.append((likes, views, node))
        candidates.sort(reverse=True, key=lambda value: (value[0], value[1]))
        for likes, views, node in candidates:
            content = client.content(node["uuid"])
            if len(content) < 300:
                continue
            author = node.get("author") or {}
            url = (
                f"https://leetcode.com/problems/{slug}/solutions/"
                f"{node['topicId']}/{node['slug']}/"
            )
            return slug, {
                "node": node,
                "content": content,
                "post_url": url,
                "author": author.get("userName"),
                "author_slug": author.get("userSlug"),
                "likes": likes,
                "views": views,
            }, None
        return slug, None, "no qualifying public article with non-empty content"
    except Exception as exc:
        return slug, None, type(exc).__name__


def main() -> None:
    mapping = read_jsonl(MAPPING_PATH)
    targets = [item for item in mapping if item["post_url"].startswith("https://leetcode.com/")]
    results: dict[str, dict] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fetch_one, item) for item in targets]
        for index, future in enumerate(as_completed(futures), start=1):
            slug, result, error = future.result()
            if result:
                results[slug] = result
            else:
                failures[slug] = error or "unknown"
            if index % 10 == 0 or index == len(futures):
                print(f"[public contents] {index}/{len(futures)}", flush=True)

    markdown_dir = POST_ROOT / "markdown"
    metadata_dir = POST_ROOT / "provider-json"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = POST_ROOT / "manifest.jsonl"
    manifest = read_jsonl(manifest_path) if manifest_path.exists() else []
    manifest = [item for item in manifest if item.get("content_provider") != "leetcode_public_graphql"]
    mapping_by_slug = {item["problem_slug"]: item for item in mapping}

    for slug, result in results.items():
        selected = mapping_by_slug[slug]
        content = re.sub(r"\n{4,}", "\n\n\n", result["content"]).strip() + "\n"
        path = markdown_dir / f"{slug}.md"
        path.write_text(content, encoding="utf-8", newline="\n")
        metadata_path = metadata_dir / f"{slug}.json"
        write_json(
            metadata_path,
            {
                "problem_slug": slug,
                "post_url": result["post_url"],
                "article": result["node"],
                "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "fetched_at": now(),
            },
        )
        selected.update(
            {
                "post_url": result["post_url"],
                "post_title": result["node"].get("title"),
                "author": result["author"],
                "author_slug": result["author_slug"],
                "likes": result["likes"],
                "views": result["views"],
                "selection_reason": (
                    "优先官方帖在未登录国际站不提供正文；改选满足100赞、1万浏览门槛且正文公开的最高票社区题解"
                ),
                "metrics_source_url": f"https://leetcode.com/problems/{slug}/solutions/",
                "metrics_fetched_at": now(),
                "selected_at": now(),
            }
        )
        manifest.append(
            {
                "problem_id": selected.get("problem_id"),
                "problem_slug": slug,
                "problem_url": selected["problem_url"],
                "post_url": result["post_url"],
                "post_title": result["node"].get("title"),
                "author": result["author"],
                "author_slug": result["author_slug"],
                "likes": result["likes"],
                "views": result["views"],
                "source_sets": selected["source_sets"],
                "content_provider": "leetcode_public_graphql",
                "markdown_file": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "provider_json_file": str(metadata_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "markdown_chars": len(content),
                "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "comments_removed": True,
                "fetched_at": now(),
            }
        )

    manifest.sort(key=lambda item: item["problem_slug"])
    write_jsonl(MAPPING_PATH, mapping)
    write_jsonl(manifest_path, manifest)
    write_json(POST_ROOT / "public-content-failures.json", failures)
    stats = {
        "requested": len(targets),
        "saved": len(results),
        "failed": len(failures),
        "authors": dict(Counter(result["author_slug"] for result in results.values())),
        "generated_at": now(),
    }
    write_json(POST_ROOT / "public-content-stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
