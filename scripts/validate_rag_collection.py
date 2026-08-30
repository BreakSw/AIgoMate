"""Validate the staged RAG corpus and write a secret-free collection report."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAG_ROOT = PROJECT_ROOT / "rag-data"
CONCEPT_ROOT = RAG_ROOT / "raw" / "algorithm-concepts" / "programmercarl"
PROBLEM_ROOT = RAG_ROOT / "raw" / "problem-bank" / "leetcode"
CASE_ROOT = RAG_ROOT / "raw" / "code-cases" / "leetcode"
WAIVED_CASES = {"verify-preorder-sequence-in-binary-search-tree"}
COMMENT_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:#{1,4}\s*)?(?:comments?|评论)\s*[（(]|Sort by:\s*Best|No comments yet",
    re.IGNORECASE,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def validate_manifest(
    values: list[dict], slug_key: str, file_key: str, minimum_chars: int
) -> dict:
    slugs = [item[slug_key] for item in values]
    duplicate_slugs = sorted(slug for slug, count in Counter(slugs).items() if count > 1)
    missing_files = []
    short_files = []
    hash_mismatches = []
    base64_images = []
    comment_leaks = []
    total_chars = 0
    total_bytes = 0
    for item in values:
        path = PROJECT_ROOT / item[file_key]
        if not path.is_file():
            missing_files.append(relative(path))
            continue
        content = path.read_text(encoding="utf-8")
        total_chars += len(content)
        total_bytes += path.stat().st_size
        if len(content) < minimum_chars:
            short_files.append(item[slug_key])
        expected_hash = item.get("content_sha256")
        actual_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        hash_without_terminal_newline = (
            hashlib.sha256(content[:-1].encode("utf-8")).hexdigest()
            if content.endswith("\n")
            else actual_hash
        )
        if expected_hash and expected_hash not in {actual_hash, hash_without_terminal_newline}:
            hash_mismatches.append(item[slug_key])
        if "data:image/" in content:
            base64_images.append(item[slug_key])
        if COMMENT_PATTERN.search(content):
            comment_leaks.append(item[slug_key])
    return {
        "manifest_records": len(values),
        "unique_slugs": len(set(slugs)),
        "markdown_chars": total_chars,
        "markdown_bytes": total_bytes,
        "duplicate_slugs": duplicate_slugs,
        "missing_files": missing_files,
        "short_files": short_files,
        "hash_mismatches": hash_mismatches,
        "base64_images": base64_images,
        "comment_leaks": comment_leaks,
    }


def main() -> None:
    concepts = read_jsonl(CONCEPT_ROOT / "manifest.jsonl")
    problem_catalog = read_jsonl(PROBLEM_ROOT / "problem-manifest.jsonl")
    problems = read_jsonl(PROBLEM_ROOT / "pages-from-solution-lists" / "manifest.jsonl")
    selected_posts = read_jsonl(CASE_ROOT / "selected-posts.jsonl")
    cases = read_jsonl(CASE_ROOT / "posts" / "manifest.jsonl")

    concept_check = validate_manifest(concepts, "source_url", "markdown_file", 100)
    problem_check = validate_manifest(problems, "title_slug", "markdown_file", 300)
    case_check = validate_manifest(cases, "problem_slug", "markdown_file", 300)
    catalog_slugs = {item["title_slug"] for item in problem_catalog}
    problem_slugs = {item["title_slug"] for item in problems}
    case_slugs = {item["problem_slug"] for item in cases}
    selected_slugs = {item["problem_slug"] for item in selected_posts}
    invalid_metrics = sorted(
        item["problem_slug"]
        for item in cases
        if int(item.get("likes") or 0) < 100 or int(item.get("views") or 0) < 10_000
    )
    comments_flags_invalid = sorted(
        item["problem_slug"] for item in cases if item.get("comments_removed") is not True
    )
    source_stats = json.loads((PROBLEM_ROOT / "catalog-stats.json").read_text(encoding="utf-8"))
    provider_counts = Counter(item["content_provider"] for item in cases)
    author_counts = Counter(item.get("author_slug") or "unknown" for item in cases)
    missing_cases = sorted(selected_slugs - case_slugs)
    unexpected_cases = sorted(case_slugs - selected_slugs)
    total_disk_bytes = sum(path.stat().st_size for path in RAG_ROOT.rglob("*") if path.is_file())

    errors = []
    for label, check in (
        ("concept", concept_check),
        ("problem", problem_check),
        ("case", case_check),
    ):
        for key in (
            "duplicate_slugs",
            "missing_files",
            "short_files",
            "hash_mismatches",
            "base64_images",
            "comment_leaks",
        ):
            if check[key]:
                errors.append(f"{label}.{key}")
    if catalog_slugs != problem_slugs:
        errors.append("problem_catalog_coverage")
    if set(missing_cases) != WAIVED_CASES:
        errors.append("case_coverage")
    if unexpected_cases:
        errors.append("unexpected_cases")
    if invalid_metrics:
        errors.append("case_metric_threshold")
    if comments_flags_invalid:
        errors.append("comments_removed_flags")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ready_with_user_waiver" if not errors else "validation_failed",
        "embedding_status": "not_started",
        "planned_vector_store": "Milvus",
        "concept_library": concept_check,
        "problem_bank": {
            **problem_check,
            "catalog_questions": len(problem_catalog),
            "source_counts": source_stats["source_counts"],
            "curriculum_category_counts": source_stats["curriculum_category_counts"],
        },
        "code_case_library": {
            **case_check,
            "selected_mappings": len(selected_posts),
            "missing_cases": missing_cases,
            "waived_cases": sorted(WAIVED_CASES),
            "provider_counts": dict(provider_counts),
            "official_cases": sum(
                count for author, count in author_counts.items() if author in {"leetcode", "leetcode-solution"}
            ),
            "endlesscheng_cases": author_counts.get("endlesscheng", 0),
            "minimum_likes": min(int(item["likes"]) for item in cases),
            "minimum_views": min(int(item["views"]) for item in cases),
            "invalid_metrics": invalid_metrics,
            "comments_removed_flags_invalid": comments_flags_invalid,
        },
        "total_rag_staging_bytes": total_disk_bytes,
        "validation_errors": errors,
    }
    (RAG_ROOT / "collection-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    final_case_stats = {
        "selected_total": len(selected_posts),
        "saved_total": len(cases),
        "waived_total": len(WAIVED_CASES),
        "waived_cases": sorted(WAIVED_CASES),
        "provider_counts": dict(provider_counts),
        "markdown_chars": case_check["markdown_chars"],
        "validation_errors": errors,
        "updated_at": summary["generated_at"],
    }
    (CASE_ROOT / "posts" / "crawl-stats.json").write_text(
        json.dumps(final_case_stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    report = f"""# AlgoMate RAG 原始数据集

本目录目前只包含抓取、清洗和审核后的原始数据，尚未执行 embedding。后续向量存储按项目决定使用 Milvus。

## 最终覆盖率

- 算法概念库（代码随想录）：{concept_check['manifest_records']} / 298 篇
- LeetCode 题库：{problem_check['manifest_records']} / {len(problem_catalog)} 题
- 一题一帖代码案例库：{case_check['manifest_records']} / {len(selected_posts)} 篇
- 用户明确跳过：`verify-preorder-sequence-in-binary-search-tree`（会员题匿名接口不返回正文）

## 质量约束

- 已落盘案例最低点赞数：{summary['code_case_library']['minimum_likes']}
- 已落盘案例最低浏览量：{summary['code_case_library']['minimum_views']}
- 力扣官方案例：{summary['code_case_library']['official_cases']} 篇
- 灵茶山艾府案例：{summary['code_case_library']['endlesscheng_cases']} 篇
- 评论区泄漏：{len(case_check['comment_leaks'])} 篇
- Base64 图片残留：{len(case_check['base64_images'])} 篇
- 哈希不一致：{len(case_check['hash_mismatches'])} 篇

## 目录

- `raw/algorithm-concepts/programmercarl/`：算法概念正文与来源清单
- `raw/problem-bank/leetcode/`：222 道题目录与题面正文
- `raw/code-cases/leetcode/selected-posts.jsonl`：题目到权威帖子的审计映射
- `raw/code-cases/leetcode/posts/markdown/`：剔除评论后的案例正文
- `collection-summary.json`：机器可读的最终验证结果

## 后续处理

Embedding 前应保留当前原始层不变，另建 processed/chunks 层进行切块、去重和元数据标准化，再写入 Milvus。抓取内容的使用仍需遵守各来源网站的条款与版权要求。
"""
    (RAG_ROOT / "README.md").write_text(report, encoding="utf-8", newline="\n")
    print(json.dumps({"status": summary["status"], "errors": errors}, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
