"""Validate processed RAG libraries before chunking and embedding."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from prepare_rag_processed import SOLUTION_LINK, balanced_fences


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "rag-data" / "processed"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    errors: list[str] = []
    warnings: list[str] = []
    manifests = {
        name: read_jsonl(PROCESSED_ROOT / name / "manifest.jsonl")
        for name in ("algorithm-concepts", "problem-bank", "code-cases")
    }
    expected = {"algorithm-concepts": 298, "problem-bank": 222, "code-cases": 221}
    library_reports: dict[str, dict] = {}

    forbidden = {
        "algorithm-concepts": ("### 评论", "登录后评论", "阅读更多", "粤ICP备"),
        "problem-bank": ("通过次数",),
        "code-cases": ("## 分类题单", "Sort by: Best", "No comments yet"),
    }

    for name, rows in manifests.items():
        if len(rows) != expected[name]:
            errors.append(f"{name}: expected {expected[name]} logical documents, got {len(rows)}")
        included = [row for row in rows if row["included_for_embedding"]]
        aliases = [row for row in rows if row.get("duplicate_of")]
        excluded = [row for row in rows if row.get("exclusion_reason")]
        lengths: list[int] = []
        providers: Counter = Counter()

        by_id = {row["document_id"]: row for row in rows}
        for row in rows:
            source_path = PROJECT_ROOT / row["source_file"]
            cleaned_path = PROJECT_ROOT / row["cleaned_file"]
            if not source_path.is_file():
                errors.append(f"{name}/{row['document_id']}: source file is missing")
                continue
            if not cleaned_path.is_file():
                errors.append(f"{name}/{row['document_id']}: cleaned file is missing")
                continue
            source = source_path.read_text(encoding="utf-8")
            cleaned = cleaned_path.read_text(encoding="utf-8")
            if sha256(source) != row["source_sha256"]:
                errors.append(f"{name}/{row['document_id']}: source hash changed after processing")
            canonical = by_id.get(row.get("duplicate_of"))
            expected_cleaned_hash = (
                canonical["cleaned_sha256"] if canonical is not None else row["cleaned_sha256"]
            )
            if sha256(cleaned) != expected_cleaned_hash:
                errors.append(f"{name}/{row['document_id']}: cleaned hash mismatch")
            if not balanced_fences(cleaned):
                errors.append(f"{name}/{row['document_id']}: unbalanced Markdown fences")
            if cleaned.count("\n") <= 2 and cleaned.count("\\n") >= 3:
                errors.append(f"{name}/{row['document_id']}: whole document remains JSON escaped")
            for marker in forbidden[name]:
                if marker in cleaned:
                    errors.append(f"{name}/{row['document_id']}: forbidden page noise remains ({marker})")
            if row["included_for_embedding"]:
                lengths.append(len(cleaned))
                if len(cleaned) < 200:
                    errors.append(f"{name}/{row['document_id']}: embedding document is shorter than 200 chars")
            if name == "code-cases":
                providers[row["source_metadata"].get("content_provider", "unknown")] += 1
                if len(set(SOLUTION_LINK.findall(source))) >= 5:
                    errors.append(f"{name}/{row['document_id']}: active source is a multi-post listing")
                if not row["source_metadata"].get("comments_removed"):
                    errors.append(f"{name}/{row['document_id']}: comments_removed audit flag is missing")

        library_reports[name] = {
            "logical_documents": len(rows),
            "included_for_embedding": len(included),
            "duplicate_aliases": len(aliases),
            "excluded_documents": len(excluded),
            "minimum_cleaned_chars": min(lengths) if lengths else 0,
            "maximum_cleaned_chars": max(lengths) if lengths else 0,
            "cleaned_markdown_files": len(list((PROCESSED_ROOT / name / "markdown").glob("*.md"))),
        }
        if providers:
            library_reports[name]["content_providers"] = dict(sorted(providers.items()))

    usable_problem_ids = {
        row["document_id"] for row in manifests["problem-bank"] if row["included_for_embedding"]
    }
    case_ids = {
        row["document_id"] for row in manifests["code-cases"] if row["included_for_embedding"]
    }
    if usable_problem_ids != case_ids:
        errors.append(
            "problem/case one-to-one mapping mismatch: "
            f"missing_cases={sorted(usable_problem_ids - case_ids)}, "
            f"orphan_cases={sorted(case_ids - usable_problem_ids)}"
        )

    case_rows = manifests["code-cases"]
    likes = [int(row["source_metadata"]["likes"]) for row in case_rows]
    views = [int(row["source_metadata"]["views"]) for row in case_rows]
    if min(likes) < 100 or min(views) < 10_000:
        errors.append(f"case engagement threshold failed: min_likes={min(likes)}, min_views={min(views)}")

    repair_summary_path = (
        PROJECT_ROOT
        / "rag-data/raw/code-cases/leetcode/posts/single-post-repairs/repair-summary.json"
    )
    repair_summary = json.loads(repair_summary_path.read_text(encoding="utf-8"))
    for slug in repair_summary["slugs"]:
        row = next(item for item in case_rows if item["document_id"] == slug)
        original = row["source_metadata"].get("replaces_invalid_source")
        if not original or not (PROJECT_ROOT / original).is_file():
            errors.append(f"code-cases/{slug}: original invalid listing was not preserved")

    near_candidate = json.loads(
        (PROCESSED_ROOT / "problem-bank" / "near-duplicate-candidates.json").read_text(encoding="utf-8")
    )
    if near_candidate:
        warnings.append(
            "The preorder/postorder traversal problem statements are structurally similar but semantically distinct; both remain included."
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not errors else "failed",
        "libraries": library_reports,
        "one_problem_one_case_pairs": len(case_ids),
        "case_minimum_likes": min(likes),
        "case_minimum_views": min(views),
        "single_post_repairs": repair_summary["repaired"],
        "original_invalid_listings_preserved": repair_summary["original_listing_files_preserved"],
        "errors": errors,
        "warnings": warnings,
    }
    output_path = PROCESSED_ROOT / "quality-report.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
