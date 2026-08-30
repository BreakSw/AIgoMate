"""Firecrawl-backed source collector for the AlgoMate RAG staging area.

The script never prints or persists credentials. It stores raw Firecrawl JSON,
normalized Markdown, and JSONL manifests so later chunking/embedding is fully
reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
OUTPUT_ROOT = PROJECT_ROOT / "rag-data" / "raw"


def load_env() -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


class FirecrawlClient:
    def __init__(self) -> None:
        env = load_env()
        api_key = env.get("firecrawl-api-key")
        if not api_key:
            raise RuntimeError(".env 中缺少 firecrawl-api-key")
        self.base_url = env.get("firecrawl-base-url", "https://api.firecrawl.dev").rstrip("/")
        self.client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(180.0),
        )

    def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Call Firecrawl with the project-wide five-attempt disconnect limit."""
        last_error: Exception | None = None
        for attempt in range(1, 6):
            try:
                response = self.client.request(method, url, **kwargs)
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
                return response
            except (httpx.TransportError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                last_error = exc
                if attempt == 5:
                    break
                time.sleep(min(2 ** (attempt - 1), 8))
        raise RuntimeError("Firecrawl request failed after five attempts") from last_error

    def map(self, url: str, limit: int = 5000) -> list[str]:
        response = self.request("POST", f"{self.base_url}/v1/map", json={"url": url, "limit": limit})
        response.raise_for_status()
        payload = response.json()
        if not payload.get("success"):
            raise RuntimeError(f"Firecrawl map failed for {url}")
        return list(payload.get("links") or [])

    def batch_scrape(
        self,
        urls: list[str],
        label: str,
        wait_for_ms: int | None = None,
    ) -> list[dict]:
        payload = {
            "urls": urls,
            "formats": ["markdown", "links"],
            "onlyMainContent": True,
            "removeBase64Images": True,
        }
        if wait_for_ms:
            payload["waitFor"] = wait_for_ms
        response = self.request(
            "POST",
            f"{self.base_url}/v1/batch/scrape",
            json=payload,
        )
        response.raise_for_status()
        job = response.json()
        job_id = job.get("id")
        if not job.get("success") or not job_id:
            raise RuntimeError(f"Firecrawl batch could not start: {label}")

        last_progress: tuple[int, int, int] | None = None
        deadline = time.monotonic() + 1_800
        while time.monotonic() < deadline:
            status_response = self.request("GET", f"{self.base_url}/v1/batch/scrape/{job_id}")
            status_response.raise_for_status()
            status = status_response.json()
            progress = (
                int(status.get("completed") or 0),
                int(status.get("total") or len(urls)),
                int(status.get("creditsUsed") or 0),
            )
            if progress != last_progress:
                print(
                    f"[{label}] {status.get('status')} "
                    f"{progress[0]}/{progress[1]} credits={progress[2]}",
                    flush=True,
                )
                last_progress = progress
            if status.get("status") == "completed":
                return list(status.get("data") or [])
            if status.get("status") in {"failed", "cancelled"}:
                raise RuntimeError(f"Firecrawl batch {label} ended as {status.get('status')}")
            time.sleep(2)
        raise TimeoutError(f"Firecrawl batch timed out: {label}")


def canonical_url(url: str) -> str:
    return url.split("#", 1)[0].split("?", 1)[0].rstrip("/")


def source_url(record: dict) -> str:
    metadata = record.get("metadata") or {}
    return canonical_url(
        metadata.get("sourceURL")
        or metadata.get("url")
        or record.get("url")
        or ""
    )


def safe_relative_path(url: str) -> Path:
    parsed = urlsplit(url)
    path = unquote(parsed.path).strip("/") or "index"
    parts = []
    for raw in path.split("/"):
        cleaned = re.sub(r"[^0-9A-Za-z._-]+", "-", raw).strip("-.") or "page"
        parts.append(cleaned[:120])
    leaf = parts[-1]
    if leaf.endswith(".html"):
        leaf = leaf[:-5]
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
    parts[-1] = f"{leaf}-{digest}.md"
    return Path(*parts)


def save_records(records: list[dict], output_dir: Path, requested_urls: list[str]) -> dict:
    markdown_dir = output_dir / "markdown"
    json_dir = output_dir / "firecrawl-json"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    fetched_at = datetime.now(timezone.utc).isoformat()
    manifest: list[dict] = []
    seen_urls: set[str] = set()

    for record in records:
        url = source_url(record)
        if not url:
            continue
        seen_urls.add(url)
        markdown = record.get("markdown") or ""
        metadata = record.get("metadata") or {}
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        relative = safe_relative_path(url)
        markdown_path = markdown_dir / relative
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(markdown, encoding="utf-8", newline="\n")
        json_path = json_dir / f"{digest}.json"
        json_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
            newline="\n",
        )
        category_match = re.search(r"/algo/([^/]+)", url)
        manifest.append(
            {
                "source_url": url,
                "title": metadata.get("title"),
                "status_code": metadata.get("statusCode"),
                "category": category_match.group(1) if category_match else None,
                "markdown_file": str(markdown_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "firecrawl_json_file": str(json_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "markdown_chars": len(markdown),
                "content_sha256": hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
                "fetched_at": fetched_at,
            }
        )

    requested = {canonical_url(url) for url in requested_urls}
    missing = sorted(requested - seen_urls)
    manifest.sort(key=lambda item: item["source_url"])
    (output_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in manifest),
        encoding="utf-8",
        newline="\n",
    )
    (output_dir / "missing-urls.json").write_text(
        json.dumps(missing, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    return {
        "requested": len(requested),
        "saved": len(manifest),
        "missing": len(missing),
        "markdown_chars": sum(item["markdown_chars"] for item in manifest),
    }


def crawl_programmercarl(batch_size: int) -> None:
    firecrawl = FirecrawlClient()
    links = firecrawl.map("https://programmercarl.com", limit=5000)
    urls = sorted(
        {
            canonical_url(link)
            for link in links
            if re.match(r"^https://programmercarl\.com/algo(?:/|$)", canonical_url(link))
        }
    )
    output_dir = OUTPUT_ROOT / "algorithm-concepts" / "programmercarl"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mapped-urls.json").write_text(
        json.dumps(urls, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    all_records: list[dict] = []
    for start in range(0, len(urls), batch_size):
        batch = urls[start : start + batch_size]
        label = f"programmercarl {start + 1}-{start + len(batch)}/{len(urls)}"
        all_records.extend(firecrawl.batch_scrape(batch, label))
    stats = save_records(all_records, output_dir, urls)
    (output_dir / "crawl-stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(stats, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=["programmercarl"])
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()
    crawl_programmercarl(max(1, min(args.batch_size, 100)))


if __name__ == "__main__":
    main()
