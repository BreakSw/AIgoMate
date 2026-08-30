"""Replace solution-listing captures with real single-post Markdown.

The original listing captures are intentionally left untouched. Repaired
single-post sources are written beside them and the active raw manifest is
updated to point at the repaired files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from fetch_public_solution_contents import fetch_one


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = PROJECT_ROOT / "rag-data" / "raw" / "code-cases" / "leetcode"
POST_ROOT = CASE_ROOT / "posts"
MANIFEST_PATH = POST_ROOT / "manifest.jsonl"
MAPPING_PATH = CASE_ROOT / "selected-posts.jsonl"
SOLUTION_LINK = re.compile(r"https://leetcode\.cn/problems/[^/]+/solutions/[^)\s]+")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
        newline="\n",
    )


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def listing_capture(item: dict) -> bool:
    path = PROJECT_ROOT / item["markdown_file"]
    text = path.read_text(encoding="utf-8")
    return len(set(SOLUTION_LINK.findall(text))) >= 5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="persist successful repairs")
    args = parser.parse_args()

    manifest = read_jsonl(MANIFEST_PATH)
    mapping = read_jsonl(MAPPING_PATH)
    targets = [item for item in manifest if listing_capture(item)]
    print(f"invalid listing captures: {len(targets)}", flush=True)

    results: dict[str, dict] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_one, item): item["problem_slug"] for item in targets}
        for index, future in enumerate(as_completed(futures), start=1):
            slug, result, error = future.result()
            if result:
                results[slug] = result
                print(
                    f"[{index}/{len(targets)}] {slug}: "
                    f"{result['likes']} likes, {result['views']} views, "
                    f"{len(result['content'])} chars",
                    flush=True,
                )
            else:
                failures[slug] = error or "unknown"
                print(f"[{index}/{len(targets)}] {slug}: FAILED ({failures[slug]})", flush=True)

    if not args.write:
        print(json.dumps({"repairable": len(results), "failures": failures}, ensure_ascii=False, indent=2))
        return
    if failures:
        raise RuntimeError(
            "Refusing partial write because some listing captures could not be repaired: "
            + json.dumps(failures, ensure_ascii=False)
        )

    markdown_dir = POST_ROOT / "single-post-repairs" / "markdown"
    metadata_dir = POST_ROOT / "single-post-repairs" / "provider-json"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest_by_slug = {item["problem_slug"]: item for item in manifest}
    mapping_by_slug = {item["problem_slug"]: item for item in mapping}
    repaired_at = now()

    for slug, result in sorted(results.items()):
        content = re.sub(r"\n{4,}", "\n\n\n", result["content"]).strip() + "\n"
        markdown_path = markdown_dir / f"{slug}.md"
        markdown_path.write_text(content, encoding="utf-8", newline="\n")
        metadata_path = metadata_dir / f"{slug}.json"
        write_json(
            metadata_path,
            {
                "problem_slug": slug,
                "post_url": result["post_url"],
                "article": result["node"],
                "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "repair_reason": "original provider returned a multi-post solution listing",
                "comments_removed": True,
                "fetched_at": repaired_at,
            },
        )

        active = manifest_by_slug[slug]
        original_source = active["markdown_file"]
        active.update(
            {
                "post_url": result["post_url"],
                "post_title": result["node"].get("title"),
                "author": result["author"],
                "author_slug": result["author_slug"],
                "likes": result["likes"],
                "views": result["views"],
                "content_provider": "leetcode_public_graphql_repair",
                "markdown_file": rel(markdown_path),
                "provider_json_file": rel(metadata_path),
                "markdown_chars": len(content),
                "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "comments_removed": True,
                "replaces_invalid_source": original_source,
                "fetched_at": repaired_at,
            }
        )

        selected = mapping_by_slug[slug]
        selected.update(
            {
                "post_url": result["post_url"],
                "post_title": result["node"].get("title"),
                "author": result["author"],
                "author_slug": result["author_slug"],
                "likes": result["likes"],
                "views": result["views"],
                "selection_reason": (
                    "原抓取结果是多帖列表页；改用力扣公开 GraphQL 中满足 "
                    "100 赞、1 万浏览门槛的最高票单篇题解"
                ),
                "metrics_source_url": f"https://leetcode.com/problems/{slug}/solutions/",
                "metrics_fetched_at": repaired_at,
                "selected_at": repaired_at,
            }
        )

    write_jsonl(MANIFEST_PATH, sorted(manifest, key=lambda item: item["problem_slug"]))
    write_jsonl(MAPPING_PATH, sorted(mapping, key=lambda item: item["problem_slug"]))
    write_json(
        POST_ROOT / "single-post-repairs" / "repair-summary.json",
        {
            "generated_at": repaired_at,
            "repaired": len(results),
            "failed": 0,
            "original_listing_files_preserved": True,
            "minimum_likes": min(result["likes"] for result in results.values()),
            "minimum_views": min(result["views"] for result in results.values()),
            "slugs": sorted(results),
        },
    )
    print(json.dumps({"repaired": len(results), "failed": 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
