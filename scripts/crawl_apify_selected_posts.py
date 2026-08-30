"""Crawl selected LeetCode CN posts through Apify with resumable checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify

from crawl_rag_sources import PROJECT_ROOT, canonical_url, load_env


CASE_ROOT = PROJECT_ROOT / "rag-data" / "raw" / "code-cases" / "leetcode"
MAPPING_PATH = CASE_ROOT / "selected-posts.jsonl"
CANDIDATE_PATH = CASE_ROOT / "solution-candidates.jsonl"
POST_ROOT = CASE_ROOT / "posts"
REQUIRED_CONTAINER_CLASSES = {"relative", "flex", "w-full", "flex-col", "p-4", "pb-8", "gap-4"}
OFFICIAL_CONTAINER_CLASSES = {"relative", "flex", "w-full", "flex-col"}


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


class ApifyClient:
    def __init__(self) -> None:
        env = load_env()
        token = env.get("apify-api-token")
        if not token:
            raise RuntimeError(".env 中缺少 apify-api-token")
        self.actor_id = env.get("apify-actor-id") or "apify~website-content-crawler"
        self.client = httpx.Client(
            base_url="https://api.apify.com/v2",
            headers={"Authorization": f"Bearer {token}"},
            timeout=httpx.Timeout(180.0),
        )

    def request(self, method: str, path: str, **kwargs) -> httpx.Response:
        last: Exception | None = None
        for attempt in range(1, 6):
            try:
                response = self.client.request(method, path, **kwargs)
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
                response.raise_for_status()
                return response
            except httpx.HTTPError as exc:
                last = exc
                if attempt < 5:
                    time.sleep(min(2 ** (attempt - 1), 8))
        raise RuntimeError("Apify request failed after five attempts") from last

    def run_batch(self, urls: list[str], concurrency: int) -> tuple[dict, list[dict]]:
        actor_input = {
            "startUrls": [{"url": url} for url in urls],
            "crawlerType": "playwright:firefox",
            "maxCrawlDepth": 0,
            "maxCrawlPages": len(urls),
            "maxResults": len(urls),
            "useSitemaps": False,
            "useLlmsTxt": False,
            "initialConcurrency": min(2, concurrency),
            "maxConcurrency": concurrency,
            "requestTimeoutSecs": 180,
            "maxRequestRetries": 5,
            "maxSessionRotations": 5,
            "dynamicContentWaitSecs": 30,
            "maxScrollHeightPixels": 50_000,
            "blockMedia": False,
            "clickElementsCssSelector": "",
            "htmlTransformer": "none",
            "aggressivePrune": False,
            "removeCookieWarnings": True,
            "removeElementsCssSelector": "script, style, noscript, svg",
            "saveMarkdown": False,
            "saveHtml": False,
            "saveHtmlAsFile": True,
            "proxyConfiguration": {"useApifyProxy": True},
        }
        run = self.request(
            "POST",
            f"/acts/{self.actor_id}/runs",
            params={"memory": 8192},
            json=actor_input,
        ).json()["data"]
        run_id = run["id"]
        deadline = time.monotonic() + 3_600
        last_status = None
        while time.monotonic() < deadline:
            run = self.request("GET", f"/actor-runs/{run_id}").json()["data"]
            status = run["status"]
            if status != last_status:
                print(f"[apify {run_id}] {status}", flush=True)
                last_status = status
            if status in {"SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"}:
                break
            time.sleep(5)
        if run.get("status") != "SUCCEEDED":
            raise RuntimeError(f"Apify run {run_id} ended as {run.get('status')}")
        dataset_id = run["defaultDatasetId"]
        items = self.request(
            "GET",
            f"/datasets/{dataset_id}/items",
            params={"clean": "true", "format": "json", "limit": 1000},
        ).json()
        return run, items

    def html(self, html_url: str) -> str:
        return self.request("GET", html_url).text


def normalize(value: str | None) -> str:
    return re.sub(r"\s+", "", value or "").lower()


def extract_post(html: str, selected: dict) -> tuple[str, bool]:
    soup = BeautifulSoup(html, "html.parser")
    wanted_title = normalize(selected.get("post_title"))
    wanted_author = normalize(selected.get("author"))
    candidates = []
    for div in soup.find_all("div"):
        classes = set(div.get("class") or [])
        if not REQUIRED_CONTAINER_CLASSES.issubset(classes):
            continue
        text = div.get_text(" ", strip=True)
        normalized = normalize(text)
        title_match = bool(wanted_title and wanted_title in normalized)
        author_match = bool(wanted_author and wanted_author in normalized)
        if len(text) >= 300 and (title_match or author_match):
            candidates.append(div)
    used_official_fallback = False
    if candidates:
        container = max(candidates, key=lambda node: len(node.get_text(" ", strip=True)))
    else:
        # Some LeetCode CN solution URLs render the requested post only as a
        # list entry and open the official solution in the article pane.  The
        # official pane is still an authoritative, high-engagement source, so
        # retain it instead of paying to crawl the same SPA route again.
        official_candidates = []
        for div in soup.find_all("div"):
            classes = set(div.get("class") or [])
            if not OFFICIAL_CONTAINER_CLASSES.issubset(classes):
                continue
            text = div.get_text(" ", strip=True)
            if len(text) >= 300 and ("力扣官方题解" in text or "官方" in text[:120]):
                official_candidates.append(div)
        if not official_candidates:
            raise ValueError("target solution container not found")
        container = max(official_candidates, key=lambda node: len(node.get_text(" ", strip=True)))
        used_official_fallback = True

    for node in container.find_all(["button", "script", "style", "noscript", "svg"]):
        node.decompose()
    for image in container.find_all("img"):
        source = image.get("src") or ""
        if source.startswith("data:"):
            image.decompose()
            continue
        image["src"] = urljoin(selected["post_url"], source)
    for anchor in list(container.find_all("a")):
        label = anchor.get_text(" ", strip=True)
        if any(marker in label for marker in ("上一篇题解", "下一篇题解", "Previous Solution", "Next Solution")):
            anchor.decompose()
            continue
        href = anchor.get("href")
        if href:
            anchor["href"] = urljoin(selected["post_url"], href)

    markdown = markdownify(
        str(container),
        heading_style="ATX",
        bullets="-",
        code_language="",
    )
    markdown = re.sub(r"(?m)^关注\s*$", "", markdown)
    markdown = re.sub(r"!\[[^\]]*\]\(data:[^)]+\)", "", markdown)
    markdown = re.sub(r"\n{4,}", "\n\n\n", markdown).strip() + "\n"
    if len(markdown) < 300:
        raise ValueError("extracted solution is too short")
    if re.search(r"(?:^|\n)(?:评论|Comments?)\s*\(", markdown, re.IGNORECASE):
        raise ValueError("comment section leaked into extracted solution")
    return markdown, used_official_fallback


def save_checkpoint(manifest: list[dict], failures: dict[str, str], runs: list[dict]) -> None:
    manifest.sort(key=lambda item: item["problem_slug"])
    write_jsonl(POST_ROOT / "manifest.jsonl", manifest)
    write_json(POST_ROOT / "apify-failures.json", failures)
    write_jsonl(POST_ROOT / "apify-runs.jsonl", runs)
    stats = {
        "saved_total": len(manifest),
        "provider_counts": dict(Counter(item["content_provider"] for item in manifest)),
        "failed_apify": len(failures),
        "markdown_chars": sum(item["markdown_chars"] for item in manifest),
        "updated_at": now(),
    }
    write_json(POST_ROOT / "crawl-stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()
    batch_size = max(1, min(args.batch_size, 50))
    concurrency = max(1, min(args.concurrency, 5))

    selected = read_jsonl(MAPPING_PATH)
    candidates = read_jsonl(CANDIDATE_PATH)
    official_by_slug = {
        item["problem_slug"]: item
        for item in candidates
        if item.get("author_slug") == "leetcode-solution"
        and int(item.get("likes") or 0) >= 100
        and int(item.get("views") or 0) >= 10_000
    }
    targets = [item for item in selected if item["post_url"].startswith("https://leetcode.cn/")]
    manifest_path = POST_ROOT / "manifest.jsonl"
    manifest = read_jsonl(manifest_path) if manifest_path.exists() else []
    existing_urls = {canonical_url(item["post_url"]) for item in manifest}
    remaining = [item for item in targets if canonical_url(item["post_url"]) not in existing_urls]
    failures: dict[str, str] = {}
    failure_path = POST_ROOT / "apify-failures.json"
    if failure_path.exists():
        failures = json.loads(failure_path.read_text(encoding="utf-8"))
    runs_path = POST_ROOT / "apify-runs.jsonl"
    runs = read_jsonl(runs_path) if runs_path.exists() else []
    selected_by_url = {canonical_url(item["post_url"]): item for item in targets}
    client = ApifyClient()
    markdown_dir = POST_ROOT / "markdown"
    metadata_dir = POST_ROOT / "provider-json"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    print(f"[apify] targets={len(targets)} existing={len(existing_urls)} remaining={len(remaining)}", flush=True)
    for start in range(0, len(remaining), batch_size):
        batch_items = remaining[start : start + batch_size]
        urls = [item["post_url"] for item in batch_items]
        run, output_items = client.run_batch(urls, concurrency)
        run_record = {
            "run_id": run["id"],
            "dataset_id": run.get("defaultDatasetId"),
            "status": run.get("status"),
            "requested": len(urls),
            "output_items": len(output_items),
            "usage_total_usd": run.get("usageTotalUsd"),
            "started_at": run.get("startedAt"),
            "finished_at": run.get("finishedAt"),
        }
        runs.append(run_record)
        seen: set[str] = set()
        for output in output_items:
            source_url = canonical_url(output.get("url") or "")
            selected_item = selected_by_url.get(source_url)
            if not selected_item:
                continue
            seen.add(source_url)
            try:
                html_url = output.get("htmlUrl")
                if not html_url:
                    raise ValueError("Apify output has no htmlUrl")
                html = client.html(html_url)
                markdown, used_official_fallback = extract_post(html, selected_item)
                slug = selected_item["problem_slug"]
                content_item = selected_item
                if used_official_fallback:
                    official = official_by_slug.get(slug)
                    if not official:
                        raise ValueError("official fallback has no audited candidate metadata")
                    selected_item.update(
                        {
                            "post_url": official["post_url"],
                            "post_title": official["title"],
                            "author": official["author"],
                            "author_slug": official["author_slug"],
                            "likes": official["likes"],
                            "views": official["views"],
                            "selection_reason": (
                                "指定高赞帖子未在力扣 SPA 正文窗中展开；采用同题满足门槛的力扣官方题解"
                            ),
                            "metrics_source_url": official["metrics_source_url"],
                            "metrics_fetched_at": official["metrics_fetched_at"],
                            "selected_at": now(),
                        }
                    )
                    content_item = selected_item
                markdown_path = markdown_dir / f"{slug}.md"
                markdown_path.write_text(markdown, encoding="utf-8", newline="\n")
                metadata_path = metadata_dir / f"{slug}.json"
                write_json(
                    metadata_path,
                    {
                        "problem_slug": slug,
                        "post_url": content_item["post_url"],
                        "requested_url": output.get("url"),
                        "loaded_url": (output.get("crawl") or {}).get("loadedUrl"),
                        "run_id": run["id"],
                        "dataset_id": run.get("defaultDatasetId"),
                        "used_official_fallback": used_official_fallback,
                        "html_sha256": hashlib.sha256(html.encode("utf-8")).hexdigest(),
                        "content_sha256": hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
                        "fetched_at": now(),
                    },
                )
                manifest = [item for item in manifest if item["problem_slug"] != slug]
                manifest.append(
                    {
                        "problem_id": selected_item.get("problem_id"),
                        "problem_slug": slug,
                        "problem_url": selected_item["problem_url"],
                        "post_url": content_item["post_url"],
                        "post_title": content_item.get("post_title"),
                        "author": content_item.get("author"),
                        "author_slug": content_item.get("author_slug"),
                        "likes": content_item.get("likes"),
                        "views": content_item.get("views"),
                        "source_sets": selected_item["source_sets"],
                        "content_provider": "apify_website_content_crawler",
                        "markdown_file": str(markdown_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                        "provider_json_file": str(metadata_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                        "markdown_chars": len(markdown),
                        "content_sha256": hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
                        "comments_removed": True,
                        "fetched_at": now(),
                    }
                )
                failures.pop(slug, None)
            except Exception as exc:
                failures[selected_item["problem_slug"]] = f"{type(exc).__name__}: {exc}"
        for item in batch_items:
            if canonical_url(item["post_url"]) not in seen:
                failures[item["problem_slug"]] = "Apify run returned no item for URL"
        write_jsonl(MAPPING_PATH, selected)
        save_checkpoint(manifest, failures, runs)
    save_checkpoint(manifest, failures, runs)


if __name__ == "__main__":
    main()
