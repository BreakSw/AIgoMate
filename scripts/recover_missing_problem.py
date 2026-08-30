"""Recover non-premium problem statements missing from rendered list pages."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from markdownify import markdownify

from fetch_public_solution_contents import PublicLeetCode


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROBLEM_ROOT = PROJECT_ROOT / "rag-data" / "raw" / "problem-bank" / "leetcode"
PAGE_ROOT = PROBLEM_ROOT / "pages-from-solution-lists"


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


def main() -> None:
    slug = "ipo"
    query = """
    query questionContent($slug: String!) {
      question(titleSlug: $slug) {
        questionFrontendId title titleSlug difficulty content translatedContent
      }
    }
    """
    question = PublicLeetCode().request(query, {"slug": slug})["question"]
    html = question.get("translatedContent") or question.get("content") or ""
    if len(html) < 300:
        raise RuntimeError("public GraphQL returned no usable IPO statement")
    body = markdownify(html, heading_style="ATX", bullets="-")
    body = re.sub(r"\n{4,}", "\n\n\n", body).strip()
    markdown = (
        f"# {question['questionFrontendId']}. {question['title']}\n\n"
        f"Difficulty: {question['difficulty']}\n\n{body}\n"
    )

    path = PAGE_ROOT / "markdown" / f"{slug}.md"
    path.write_text(markdown, encoding="utf-8", newline="\n")
    content_hash = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
    catalog = read_jsonl(PROBLEM_ROOT / "problem-manifest.jsonl")
    selected = next(item for item in catalog if item["title_slug"] == slug)
    manifest_path = PAGE_ROOT / "manifest.jsonl"
    manifest = [item for item in read_jsonl(manifest_path) if item["title_slug"] != slug]
    manifest.append(
        {
            "problem_id": selected["problem_id"],
            "title_slug": slug,
            "problem_url": selected["problem_url"],
            "source_solution_list_url": f"https://leetcode.cn/problems/{slug}/solutions",
            "markdown_file": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "markdown_chars": len(markdown),
            "content_sha256": content_hash,
            "content_provider": "leetcode_public_graphql",
            "fetched_at": now(),
            "extracted_at": now(),
        }
    )
    manifest.sort(key=lambda item: item["title_slug"])
    write_jsonl(manifest_path, manifest)
    (PAGE_ROOT / "missing.json").write_text("[]\n", encoding="utf-8", newline="\n")
    stats = {
        "requested": len(catalog),
        "saved": len(manifest),
        "missing": len(catalog) - len(manifest),
        "markdown_chars": sum(item["markdown_chars"] for item in manifest),
    }
    (PAGE_ROOT / "crawl-stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
