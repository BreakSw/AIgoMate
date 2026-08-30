"""Create cleaned, file-deduplicated RAG libraries without changing raw data."""

from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAG_ROOT = PROJECT_ROOT / "rag-data"
PROCESSED_ROOT = RAG_ROOT / "processed"
CONCEPT_ROOT = RAG_ROOT / "raw" / "algorithm-concepts" / "programmercarl"
PROBLEM_ROOT = RAG_ROOT / "raw" / "problem-bank" / "leetcode"
CASE_ROOT = RAG_ROOT / "raw" / "code-cases" / "leetcode"

LANGUAGE_TAB = re.compile(
    r"^(?:C\+\+|C|C#|Java|JavaScript|TypeScript|Python|Python2|Python3|Go|Golang|"
    r"Rust|Kotlin|Scala|Swift|PHP|Ruby|Dart|Elixir|Erlang|Racket|MySQL|Bash)$",
    re.IGNORECASE,
)
DATE_LINE = re.compile(r"^20\d{2}[./-]\d{1,2}[./-]\d{1,2}$")
VIEW_LINE = re.compile(r"^\d{3,}(?:\.\d+)?[KkMm]?$|^\d+(?:\.\d+)?万$")
IMAGE = re.compile(r"!\[([^\]]*)\]\((?:\\.|[^)])*\)")
LINK = re.compile(r"\[([^\]]+)\]\((?:\\.|[^)])*\)")
HTML_TAG = re.compile(
    r"</?(?:div|span|p|br|img|a|table|thead|tbody|tr|td|th|ul|ol|li|em|strong|"
    r"pre|code|details|summary|figure|figcaption|section|article|video|source)\b[^>]*>",
    re.IGNORECASE,
)
COMMENT_HEADING = re.compile(r"^#{1,6}\s*(?:评论|comments?)\b", re.IGNORECASE)
FENCE_LINE = re.compile(r"^(`{3,})(.*)$")
SOLUTION_LINK = re.compile(r"https://leetcode\.cn/problems/[^/]+/solutions/[^)\s]+")


def fence_event(line: str, open_length: int | None) -> tuple[int | None, str]:
    """Track CommonMark backtick fences without confusing nested shorter runs."""
    stripped = line.strip()
    empty_pair = re.fullmatch(r"`{3,}\s+`{3,}", stripped)
    if empty_pair:
        return open_length, "artifact"
    match = FENCE_LINE.match(stripped)
    if not match:
        return open_length, "none"
    marker_length = len(match.group(1))
    remainder = match.group(2)
    if open_length is None:
        # Backticks are not allowed in a backtick fence's info string.
        if "`" in remainder:
            return open_length, "none"
        return marker_length, "open"
    if marker_length >= open_length and not remainder.strip():
        return None, "close"
    return open_length, "none"


def balanced_fences(text: str) -> bool:
    open_fence: int | None = None
    for line in text.splitlines():
        open_fence, event = fence_event(line, open_fence)
        if event == "artifact":
            return False
    return open_fence is None


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
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def decode_one_escape_layer(value: str) -> str:
    """Decode one JSON-style escape layer while preserving existing Unicode."""
    output: list[str] = []
    index = 0
    simple = {"n": "\n", "r": "\r", "t": "\t", "b": "\b", "f": "\f", '"': '"', "'": "'", "/": "/", "\\": "\\"}
    while index < len(value):
        char = value[index]
        if char != "\\" or index + 1 >= len(value):
            output.append(char)
            index += 1
            continue
        marker = value[index + 1]
        if marker in simple:
            output.append(simple[marker])
            index += 2
            continue
        if marker == "u" and index + 5 < len(value):
            code_text = value[index + 2 : index + 6]
            if re.fullmatch(r"[0-9a-fA-F]{4}", code_text):
                code = int(code_text, 16)
                if 0xD800 <= code <= 0xDBFF and index + 11 < len(value):
                    second = value[index + 6 : index + 12]
                    if re.fullmatch(r"\\u[0-9a-fA-F]{4}", second):
                        low = int(second[2:], 16)
                        if 0xDC00 <= low <= 0xDFFF:
                            output.append(chr(0x10000 + ((code - 0xD800) << 10) + low - 0xDC00))
                            index += 12
                            continue
                output.append(chr(code))
                index += 6
                continue
        output.append(char)
        output.append(marker)
        index += 2
    return "".join(output)


def meaningful_image_alt(alt: str) -> str:
    value = re.sub(r"\\([_#|])", r"\1", alt).strip()
    lower = value.lower()
    if not value or lower in {"icon", "premium lock icon", "image", "img", "图片"}:
        return ""
    if any(word in lower for word in ("勋章", "badge", "avatar", "二维码", "logo")):
        return ""
    if re.fullmatch(r"(?:figures?|slide)?\d+(?:\.png|\.jpg|\.gif)?", lower):
        return ""
    return value


def strip_markdown_markup(line: str, stats: Counter) -> str:
    def image_replacement(match: re.Match[str]) -> str:
        stats["images_removed"] += 1
        alt = meaningful_image_alt(match.group(1))
        return f"图示：{alt}" if alt else ""

    line = IMAGE.sub(image_replacement, line)

    def link_replacement(match: re.Match[str]) -> str:
        stats["links_unwrapped"] += 1
        return match.group(1)

    line = LINK.sub(link_replacement, line)
    line = line.replace("(opens new window)", "").replace("（opens new window）", "")
    line = re.sub(r"^(#{1,6})\s+\\?#\s*", r"\1 ", line)
    line = re.sub(r"^(#{1,6})\s*\[?\\?#\]?\s*", r"\1 ", line)
    line = HTML_TAG.sub("", line)
    return html.unescape(line)


def truncate_noise_sections(lines: list[str], library: str, stats: Counter) -> list[str]:
    output = []
    open_fence: int | None = None
    for line in lines:
        stripped = line.strip()
        # These are exact page-shell boundaries, not article prose. Treat them
        # as terminal even when malformed source fences would otherwise make
        # the parser believe the footer is still inside a code block.
        if COMMENT_HEADING.match(stripped) or re.match(r"^(?:Sort by:\s*Best|No comments yet)", stripped, re.I):
            stats["comment_section_truncated"] += 1
            break
        if library == "concepts" and re.match(r"^(?:阅读更多|@20\d{2}-\d{4}\s+代码随想录)", stripped):
            stats["footer_truncated"] += 1
            break
        if library == "cases" and re.match(r"^##\s*分类题单\s*$", stripped):
            stats["promotion_section_truncated"] += 1
            break
        next_fence, event = fence_event(line, open_fence)
        if event == "artifact":
            stats["empty_fence_artifacts_removed"] += 1
            continue
        output.append(line)
        open_fence = next_fence
    return output


def remove_number_runs(lines: list[str], stats: Counter) -> list[str]:
    in_code_flags = []
    open_fence: int | None = None
    for line in lines:
        in_code_flags.append(open_fence is not None)
        open_fence, _ = fence_event(line, open_fence)
    remove_indexes: set[int] = set()
    index = 0
    while index < len(lines):
        if in_code_flags[index] or not lines[index].strip().isdigit():
            index += 1
            continue
        cursor = index
        number_indexes = []
        numbers = []
        while cursor < len(lines) and not in_code_flags[cursor]:
            stripped = lines[cursor].strip()
            if not stripped:
                cursor += 1
                continue
            if not stripped.isdigit():
                break
            number_indexes.append(cursor)
            numbers.append(int(stripped))
            cursor += 1
        consecutive = len(numbers) >= 5 and all(b == a + 1 for a, b in zip(numbers, numbers[1:]))
        if consecutive:
            remove_indexes.update(range(index, cursor))
            stats["code_line_numbers_removed"] += len(number_indexes)
        index = max(cursor, index + 1)
    return [line for idx, line in enumerate(lines) if idx not in remove_indexes]


def collapse_blank_lines(lines: list[str]) -> list[str]:
    output = []
    blank_count = 0
    open_fence: int | None = None
    for raw in lines:
        line = raw.rstrip()
        next_fence, _ = fence_event(line, open_fence)
        if not line.strip() and open_fence is None:
            blank_count += 1
            if blank_count > 2:
                continue
        else:
            blank_count = 0
        output.append(line)
        open_fence = next_fence
    while output and not output[0].strip():
        output.pop(0)
    while output and not output[-1].strip():
        output.pop()
    return output


def common_clean(text: str, library: str, stats: Counter) -> list[str]:
    text = unicodedata.normalize("NFC", text).replace("\ufeff", "").replace("\u200b", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    actual_lines = text.count("\n") + 1
    escaped_lines = text.count("\\n")
    if actual_lines <= 4 and escaped_lines >= 3:
        text = decode_one_escape_layer(text)
        stats["escaped_document_decoded"] += 1
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", "", text, flags=re.S | re.I)
    lines = truncate_noise_sections(text.split("\n"), library, stats)
    output = []
    open_fence: int | None = None
    for line in lines:
        stripped = line.strip()
        next_fence, event = fence_event(line, open_fence)
        if event == "artifact":
            stats["empty_fence_artifacts_removed"] += 1
            continue
        if event in {"open", "close"}:
            output.append(stripped)
            open_fence = next_fence
            continue
        if open_fence is not None:
            output.append(line.rstrip())
            continue
        line = strip_markdown_markup(line, stats).strip()
        if re.fullmatch(r"(?:\*\s*){3,}|-{3,}|_{3,}|[←→]", line):
            stats["separators_removed"] += 1
            continue
        if re.search(r"粤ICP备|版权所有", line):
            stats["footer_lines_removed"] += 1
            continue
        if library == "cases" and LANGUAGE_TAB.fullmatch(line):
            stats["language_tabs_removed"] += 1
            continue
        if library == "cases" and re.fullmatch(r"\d+\s*/\s*\d+", line):
            stats["image_pager_lines_removed"] += 1
            continue
        output.append(line)
    if open_fence is not None:
        output.append("`" * open_fence)
        stats["unclosed_fences_repaired"] += 1
    output = remove_number_runs(output, stats)
    return collapse_blank_lines(output)


def clean_concept(text: str, item: dict) -> tuple[str, Counter]:
    stats: Counter = Counter()
    lines = common_clean(text, "concepts", stats)
    return "\n".join(lines).strip() + "\n", stats


def clean_problem(text: str, item: dict) -> tuple[str, Counter]:
    stats: Counter = Counter()
    lines = common_clean(text, "problems", stats)
    output = []
    skip_next_value = False
    difficulty_written = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "相似题目":
            stats["similar_questions_truncated"] += 1
            break
        if stripped in {"通过次数", "通过率"}:
            stats["statistics_labels_removed"] += 1
            skip_next_value = True
            continue
        if skip_next_value:
            if not stripped:
                continue
            stats["statistics_values_removed"] += 1
            skip_next_value = False
            continue
        if stripped in {"相关企业", "相关企业提示"}:
            stats["company_section_lines_removed"] += 1
            continue
        if stripped == "相关标签":
            output.append("## 相关标签")
            continue
        if index < 15 and stripped in {"简单", "中等", "困难", "Easy", "Medium", "Hard"}:
            if not difficulty_written:
                output.append(f"难度：{stripped}")
                difficulty_written = True
            continue
        output.append(line)
    output = collapse_blank_lines(output)
    if output and not output[0].lstrip().startswith("#"):
        output[0] = "# " + output[0].lstrip()
    return "\n".join(output).strip() + "\n", stats


def clean_case(text: str, item: dict) -> tuple[str, Counter]:
    stats: Counter = Counter()
    lines = common_clean(text, "cases", stats)
    title = (item.get("post_title") or item["problem_slug"]).strip()
    author_values = {
        re.sub(r"\s+", "", value).lower()
        for value in (item.get("author"), item.get("author_slug"))
        if value
    }
    output = []
    title_removed = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        normalized = re.sub(r"\s+", "", stripped).lower()
        if index < 8 and not title_removed and normalized == re.sub(r"\s+", "", title).lower():
            title_removed = True
            stats["duplicate_title_removed"] += 1
            continue
        if index < 30 and normalized in author_values:
            stats["author_profile_lines_removed"] += 1
            continue
        if index < 30 and (DATE_LINE.fullmatch(stripped) or VIEW_LINE.fullmatch(stripped)):
            stats["post_metadata_lines_removed"] += 1
            continue
        if stripped.startswith("发布于"):
            stats["location_lines_removed"] += 1
            continue
        if stripped in {"官方题解", "力扣官方题解"} and index < 30:
            stats["post_metadata_lines_removed"] += 1
            continue
        if stripped.startswith("欢迎关注") or stripped.startswith("我的题解精选"):
            stats["promotion_lines_removed"] += 1
            continue
        output.append(line)
    output = collapse_blank_lines(output)
    output.insert(0, f"# {title}")
    output.insert(1, "")
    return "\n".join(output).strip() + "\n", stats


def fingerprint_text(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].lstrip().startswith("#"):
        lines = lines[1:]
    value = "\n".join(lines)
    value = unicodedata.normalize("NFKC", value).lower()
    return re.sub(r"\s+", "", value)


def deduplicate(records: list[dict], cleaned: dict[str, str]) -> tuple[dict[str, str], list[dict], list[dict]]:
    duplicate_of: dict[str, str] = {}
    exact_groups = defaultdict(list)
    fingerprints = {key: fingerprint_text(value) for key, value in cleaned.items()}
    for key, value in fingerprints.items():
        exact_groups[hashlib.sha256(value.encode("utf-8")).hexdigest()].append(key)
    exact_duplicates = []
    for digest, keys in exact_groups.items():
        if len(keys) < 2:
            continue
        canonical = max(keys, key=lambda key: (len(cleaned[key]), key))
        for key in sorted(keys):
            if key == canonical:
                continue
            duplicate_of[key] = canonical
            exact_duplicates.append(
                {"duplicate": key, "canonical": canonical, "reason": "normalized_exact", "hash": digest}
            )

    near_candidates = []
    keys = sorted(fingerprints)
    for left_index, left_key in enumerate(keys):
        if left_key in duplicate_of:
            continue
        left = fingerprints[left_key]
        if not left:
            continue
        for right_key in keys[left_index + 1 :]:
            if right_key in duplicate_of:
                continue
            right = fingerprints[right_key]
            length_ratio = min(len(left), len(right)) / max(len(left), len(right), 1)
            if length_ratio < 0.94:
                continue
            prefix_ratio = SequenceMatcher(None, left[:600], right[:600], autojunk=False).ratio()
            if prefix_ratio < 0.82:
                continue
            similarity = SequenceMatcher(None, left, right, autojunk=False).ratio()
            if similarity < 0.92:
                continue
            decision = "candidate"
            if similarity >= 0.985:
                canonical = max((left_key, right_key), key=lambda key: (len(cleaned[key]), key))
                duplicate = right_key if canonical == left_key else left_key
                duplicate_of[duplicate] = canonical
                decision = "auto_alias"
            near_candidates.append(
                {
                    "left": left_key,
                    "right": right_key,
                    "length_ratio": round(length_ratio, 6),
                    "similarity": round(similarity, 6),
                    "decision": decision,
                }
            )
    return duplicate_of, exact_duplicates, near_candidates


def process_library(
    name: str,
    source_manifest: list[dict],
    id_key: str,
    cleaner,
    output_dir: Path,
) -> dict:
    cleaned: dict[str, str] = {}
    excluded: dict[str, str] = {}
    stats_by_id: dict[str, Counter] = {}
    source_by_id: dict[str, Path] = {}
    item_by_id: dict[str, dict] = {}
    forced_aliases: dict[str, str] = {}
    available_ids = {item[id_key] for item in source_manifest}
    for item in source_manifest:
        key = item[id_key]
        source_path = PROJECT_ROOT / item["markdown_file"]
        source = source_path.read_text(encoding="utf-8")
        if name == "cases" and len(set(SOLUTION_LINK.findall(source))) >= 5:
            raise RuntimeError(f"multi-post solution listing cannot be embedded: {key}")
        if name == "problems" and (
            "该题目是 Plus 会员专享题" in source
            or "需要升级为 Plus 会员来解锁该题目" in source
        ):
            excluded[key] = "plus_locked_without_problem_statement"
        if name == "concepts":
            redirect = re.search(
                r"(?m)^同：\s*\[[^\]]+\]\((https://programmercarl\.com/[^)#]+)\)",
                source,
            )
            if redirect:
                target = redirect.group(1).rstrip("/")
                if target in available_ids and target != key:
                    forced_aliases[key] = target
        result, stats = cleaner(source, item)
        if len(result) < 200 and key not in forced_aliases:
            raise RuntimeError(f"cleaned document is too short: {name}/{key}")
        if not balanced_fences(result):
            raise RuntimeError(f"unbalanced Markdown fences after cleaning: {name}/{key}")
        if name == "concepts" and any(
            marker in result for marker in ("### 评论", "登录后评论", "阅读更多", "粤ICP备")
        ):
            raise RuntimeError(f"concept page-shell noise remains after cleaning: {key}")
        if name == "cases" and "## 分类题单" in result:
            raise RuntimeError(f"case promotion section remains after cleaning: {key}")
        cleaned[key] = result
        stats_by_id[key] = stats
        source_by_id[key] = source_path
        item_by_id[key] = item

    deduplication_input = {key: value for key, value in cleaned.items() if key not in excluded}
    deduplication_manifest = [item for item in source_manifest if item[id_key] not in excluded]
    duplicate_of, exact_duplicates, near_candidates = deduplicate(
        deduplication_manifest, deduplication_input
    )
    for key, target in forced_aliases.items():
        duplicate_of[key] = target
        exact_duplicates.append(
            {"duplicate": key, "canonical": target, "reason": "source_redirect_alias"}
        )

    def resolve_alias(key: str) -> str:
        seen = set()
        while key in duplicate_of and key not in seen:
            seen.add(key)
            key = duplicate_of[key]
        return key

    for key in list(duplicate_of):
        duplicate_of[key] = resolve_alias(key)
    output_dir.mkdir(parents=True, exist_ok=True)
    path_by_id: dict[str, Path] = {}
    for key, content in cleaned.items():
        canonical = duplicate_of.get(key, key)
        if canonical != key:
            continue
        filename = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12] + ".md" if name == "concepts" else f"{key}.md"
        path = output_dir / filename
        path.write_text(content, encoding="utf-8", newline="\n")
        path_by_id[key] = path
    for key, canonical in duplicate_of.items():
        path_by_id[key] = path_by_id[canonical]

    processed_manifest = []
    aggregate_stats: Counter = Counter()
    for key in sorted(cleaned):
        source_path = source_by_id[key]
        source_text = source_path.read_text(encoding="utf-8")
        aggregate_stats.update(stats_by_id[key])
        processed_manifest.append(
            {
                "document_id": key,
                "library": name,
                "source_file": rel(source_path),
                "cleaned_file": rel(path_by_id[key]),
                "duplicate_of": duplicate_of.get(key),
                "included_for_embedding": key not in duplicate_of and key not in excluded,
                "exclusion_reason": excluded.get(key),
                "source_chars": len(source_text),
                "cleaned_chars": len(cleaned[key]),
                "removed_chars": len(source_text) - len(cleaned[key]),
                "source_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
                "cleaned_sha256": hashlib.sha256(cleaned[key].encode("utf-8")).hexdigest(),
                "cleanup_stats": dict(stats_by_id[key]),
                "source_metadata": item_by_id[key],
                "processed_at": now(),
            }
        )
    write_jsonl(output_dir.parent / "manifest.jsonl", processed_manifest)
    write_json(output_dir.parent / "exact-duplicates.json", exact_duplicates)
    write_json(output_dir.parent / "near-duplicate-candidates.json", near_candidates)
    return {
        "logical_documents": len(cleaned),
        "unique_embedding_documents": len(cleaned) - len(duplicate_of) - len(excluded),
        "duplicate_aliases": len(duplicate_of),
        "excluded_documents": len(excluded),
        "exclusions": excluded,
        "exact_duplicates": len(exact_duplicates),
        "near_duplicate_candidates": len(near_candidates),
        "source_chars": sum(item["source_chars"] for item in processed_manifest),
        "cleaned_chars": sum(
            len(cleaned[key]) for key in cleaned if key not in duplicate_of and key not in excluded
        ),
        "removed_chars_all_logical_documents": sum(item["removed_chars"] for item in processed_manifest),
        "cleanup_totals": dict(aggregate_stats),
    }


def main() -> None:
    concept_manifest = read_jsonl(CONCEPT_ROOT / "manifest.jsonl")
    problem_manifest = read_jsonl(PROBLEM_ROOT / "pages-from-solution-lists" / "manifest.jsonl")
    case_manifest = read_jsonl(CASE_ROOT / "posts" / "manifest.jsonl")
    reports = {
        "algorithm-concepts": process_library(
            "concepts",
            concept_manifest,
            "source_url",
            clean_concept,
            PROCESSED_ROOT / "algorithm-concepts" / "markdown",
        ),
        "problem-bank": process_library(
            "problems",
            problem_manifest,
            "title_slug",
            clean_problem,
            PROCESSED_ROOT / "problem-bank" / "markdown",
        ),
        "code-cases": process_library(
            "cases",
            case_manifest,
            "problem_slug",
            clean_case,
            PROCESSED_ROOT / "code-cases" / "markdown",
        ),
    }
    summary = {
        "generated_at": now(),
        "raw_layer_modified": False,
        "cross_library_deduplication": False,
        "near_duplicate_auto_alias_threshold": 0.985,
        "libraries": reports,
    }
    write_json(PROCESSED_ROOT / "processing-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
