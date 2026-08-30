"""Use a few Firecrawl credits to fill solution bodies absent from Apify HTML."""

from __future__ import annotations

import hashlib
import json

from crawl_apify_selected_posts import (
    CANDIDATE_PATH,
    MAPPING_PATH,
    POST_ROOT,
    PROJECT_ROOT,
    extract_post,
    now,
    read_jsonl,
    write_json,
    write_jsonl,
)
from crawl_rag_sources import FirecrawlClient


def main() -> None:
    mapping = read_jsonl(MAPPING_PATH)
    manifest_path = POST_ROOT / "manifest.jsonl"
    manifest = read_jsonl(manifest_path)
    saved = {item["problem_slug"] for item in manifest}
    targets = [
        item
        for item in mapping
        if item["problem_slug"] not in saved
        and item["post_url"].startswith("https://leetcode.cn/")
    ]
    candidates = read_jsonl(CANDIDATE_PATH)
    official_by_slug = {
        item["problem_slug"]: item
        for item in candidates
        if item.get("author_slug") == "leetcode-solution"
        and int(item.get("likes") or 0) >= 100
        and int(item.get("views") or 0) >= 10_000
    }
    client = FirecrawlClient()
    failures = {}

    for index, selected in enumerate(targets, start=1):
        slug = selected["problem_slug"]
        try:
            response = client.request(
                "POST",
                f"{client.base_url}/v1/scrape",
                json={
                    "url": selected["post_url"],
                    "formats": ["rawHtml"],
                    "onlyMainContent": False,
                    "waitFor": 30_000,
                    "timeout": 120_000,
                },
            ).json()
            data = response.get("data") or response
            html = data.get("rawHtml") or ""
            if len(html) < 1_000:
                raise ValueError("Firecrawl returned no usable raw HTML")
            markdown, used_official_fallback = extract_post(html, selected)
            if used_official_fallback:
                official = official_by_slug.get(slug)
                if not official:
                    raise ValueError("official fallback does not satisfy the engagement threshold")
                selected.update(
                    {
                        "post_url": official["post_url"],
                        "post_title": official["title"],
                        "author": official["author"],
                        "author_slug": official["author_slug"],
                        "likes": official["likes"],
                        "views": official["views"],
                        "selection_reason": (
                            "指定帖子未在正文窗中展开；采用同题满足门槛的力扣官方题解"
                        ),
                        "metrics_source_url": official["metrics_source_url"],
                        "metrics_fetched_at": official["metrics_fetched_at"],
                        "selected_at": now(),
                    }
                )

            markdown_path = POST_ROOT / "markdown" / f"{slug}.md"
            metadata_path = POST_ROOT / "provider-json" / f"{slug}.json"
            markdown_path.write_text(markdown, encoding="utf-8", newline="\n")
            content_hash = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
            write_json(
                metadata_path,
                {
                    "problem_slug": slug,
                    "post_url": selected["post_url"],
                    "requested_url": (data.get("metadata") or {}).get("sourceURL"),
                    "loaded_url": (data.get("metadata") or {}).get("url"),
                    "used_official_fallback": used_official_fallback,
                    "html_sha256": hashlib.sha256(html.encode("utf-8")).hexdigest(),
                    "content_sha256": content_hash,
                    "fetched_at": now(),
                },
            )
            manifest.append(
                {
                    "problem_id": selected.get("problem_id"),
                    "problem_slug": slug,
                    "problem_url": selected["problem_url"],
                    "post_url": selected["post_url"],
                    "post_title": selected.get("post_title"),
                    "author": selected.get("author"),
                    "author_slug": selected.get("author_slug"),
                    "likes": selected.get("likes"),
                    "views": selected.get("views"),
                    "source_sets": selected["source_sets"],
                    "content_provider": "firecrawl_raw_html",
                    "markdown_file": str(markdown_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "provider_json_file": str(metadata_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "markdown_chars": len(markdown),
                    "content_sha256": content_hash,
                    "comments_removed": True,
                    "fetched_at": now(),
                }
            )
            saved.add(slug)
            print(f"[firecrawl] {index}/{len(targets)} saved {slug}", flush=True)
        except Exception as exc:
            failures[slug] = f"{type(exc).__name__}: {exc}"
            print(f"[firecrawl] {index}/{len(targets)} failed {slug}: {type(exc).__name__}", flush=True)

    manifest.sort(key=lambda item: item["problem_slug"])
    write_jsonl(MAPPING_PATH, mapping)
    write_jsonl(manifest_path, manifest)
    write_json(POST_ROOT / "firecrawl-failures.json", failures)
    print(
        json.dumps(
            {"requested": len(targets), "saved": len(targets) - len(failures), "failed": len(failures)},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
