"""Recover authoritative solution bodies from already-paid Apify run HTML."""

from __future__ import annotations

import hashlib
import json

from crawl_apify_selected_posts import (
    ApifyClient,
    CANDIDATE_PATH,
    MAPPING_PATH,
    POST_ROOT,
    PROJECT_ROOT,
    canonical_url,
    extract_post,
    now,
    read_jsonl,
    save_checkpoint,
    write_json,
    write_jsonl,
)


def main() -> None:
    mapping = read_jsonl(MAPPING_PATH)
    selected_by_requested_url = {canonical_url(item["post_url"]): item for item in mapping}
    candidates = read_jsonl(CANDIDATE_PATH)
    official_by_slug = {
        item["problem_slug"]: item
        for item in candidates
        if item.get("author_slug") == "leetcode-solution"
        and int(item.get("likes") or 0) >= 100
        and int(item.get("views") or 0) >= 10_000
    }

    manifest_path = POST_ROOT / "manifest.jsonl"
    manifest = read_jsonl(manifest_path) if manifest_path.exists() else []
    saved_slugs = {item["problem_slug"] for item in manifest}
    failures_path = POST_ROOT / "apify-failures.json"
    failures = json.loads(failures_path.read_text(encoding="utf-8")) if failures_path.exists() else {}
    runs_path = POST_ROOT / "apify-runs.jsonl"
    runs = read_jsonl(runs_path) if runs_path.exists() else []
    client = ApifyClient()
    recovered = 0

    for run in runs:
        dataset_id = run.get("dataset_id")
        if not dataset_id:
            continue
        outputs = client.request(
            "GET",
            f"/datasets/{dataset_id}/items",
            params={"clean": "true", "format": "json", "limit": 1000},
        ).json()
        for output in outputs:
            selected = selected_by_requested_url.get(canonical_url(output.get("url") or ""))
            if not selected or selected["problem_slug"] in saved_slugs:
                continue
            html_url = output.get("htmlUrl")
            if not html_url:
                continue
            slug = selected["problem_slug"]
            try:
                html = client.html(html_url)
                markdown, used_official_fallback = extract_post(html, selected)
                if used_official_fallback:
                    official = official_by_slug.get(slug)
                    if not official:
                        raise ValueError("official fallback has no audited candidate metadata")
                    selected.update(
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

                markdown_path = POST_ROOT / "markdown" / f"{slug}.md"
                metadata_path = POST_ROOT / "provider-json" / f"{slug}.json"
                markdown_path.write_text(markdown, encoding="utf-8", newline="\n")
                content_hash = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
                write_json(
                    metadata_path,
                    {
                        "problem_slug": slug,
                        "post_url": selected["post_url"],
                        "requested_url": output.get("url"),
                        "loaded_url": (output.get("crawl") or {}).get("loadedUrl"),
                        "run_id": run["run_id"],
                        "dataset_id": dataset_id,
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
                        "content_provider": "apify_website_content_crawler",
                        "markdown_file": str(markdown_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                        "provider_json_file": str(metadata_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                        "markdown_chars": len(markdown),
                        "content_sha256": content_hash,
                        "comments_removed": True,
                        "fetched_at": now(),
                    }
                )
                saved_slugs.add(slug)
                failures.pop(slug, None)
                recovered += 1
            except Exception as exc:
                failures[slug] = f"{type(exc).__name__}: {exc}"

    write_jsonl(MAPPING_PATH, mapping)
    save_checkpoint(manifest, failures, runs)
    print(json.dumps({"recovered": recovered, "saved_total": len(manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
