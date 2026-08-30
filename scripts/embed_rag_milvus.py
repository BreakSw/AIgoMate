"""Chunk processed RAG documents, embed with Voyage, and load Milvus Lite.

The pipeline is deterministic and resumable. Voyage results are checkpointed
in SQLite before Milvus import, so a later import never needs to pay for the
same embeddings again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import httpx
import voyageai
from dotenv import dotenv_values
from pymilvus import MilvusClient
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "rag-data" / "processed"
VECTOR_ROOT = PROJECT_ROOT / "rag-data" / "vector"
CHUNK_ROOT = VECTOR_ROOT / "chunks"
CHECKPOINT_PATH = VECTOR_ROOT / "embedding-checkpoint.sqlite3"
PLAN_PATH = VECTOR_ROOT / "embedding-plan.json"
QUALITY_REPORT_PATH = PROCESSED_ROOT / "quality-report.json"
SELECTION_PLAN_PATH = VECTOR_ROOT / "selection-plan.json"
PROBLEM_CATALOG_PATH = (
    PROJECT_ROOT / "rag-data/raw/problem-bank/leetcode/problem-manifest.jsonl"
)
TOP_ITEMS_PER_GROUP = 3


@dataclass(frozen=True)
class LibrarySpec:
    key: str
    processed_dir: str
    model_env: str
    collection_env: str
    default_collection: str
    max_tokens: int
    overlap_tokens: int


SPECS = (
    LibrarySpec(
        key="algorithm_concepts",
        processed_dir="algorithm-concepts",
        model_env="embedding-general-model",
        collection_env="milvus-concept-collection",
        default_collection="algomate_algorithm_concepts_v1",
        max_tokens=900,
        overlap_tokens=120,
    ),
    LibrarySpec(
        key="problem_bank",
        processed_dir="problem-bank",
        model_env="embedding-general-model",
        collection_env="milvus-problem-collection",
        default_collection="algomate_problem_bank_v1",
        max_tokens=700,
        overlap_tokens=80,
    ),
    LibrarySpec(
        key="code_cases",
        processed_dir="code-cases",
        model_env="embedding-code-model",
        collection_env="milvus-code-collection",
        default_collection="algomate_code_cases_v1",
        max_tokens=1_000,
        overlap_tokens=140,
    ),
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
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


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def config() -> dict[str, str]:
    values = dotenv_values(PROJECT_ROOT / ".env")
    result = {key: str(value) for key, value in values.items() if value is not None}
    if result.get("embedding-provider", "voyage").lower() == "siliconflow":
        result["embedding-base-url"] = result.get(
            "SiliconFlow-Url", result.get("embedding-base-url", "")
        )
        result["embedding-api-key"] = result.get("SiliconFlow-Api-Key", "")
    return result


def validate_inputs(values: dict[str, str]) -> None:
    report = json.loads(QUALITY_REPORT_PATH.read_text(encoding="utf-8"))
    if report.get("status") != "passed":
        raise RuntimeError("processed RAG quality report is not passed")
    provider = values.get("embedding-provider", "voyage").lower()
    expected = (
        {
            "embedding-general-model": "Qwen/Qwen3-Embedding-8B",
            "embedding-code-model": "Qwen/Qwen3-Embedding-8B",
            "embedding-dimension": "4096",
        }
        if provider == "siliconflow"
        else {
            "embedding-general-model": "voyage-4",
            "embedding-code-model": "voyage-code-4",
            "embedding-dimension": "1024",
        }
    )
    for key, expected_value in expected.items():
        if values.get(key) != expected_value:
            raise RuntimeError(f"{key} must be {expected_value}, got {values.get(key)!r}")


@dataclass(frozen=True)
class EmbeddingResponse:
    embeddings: list[list[float]]
    total_tokens: int | None = None


class SiliconFlowEmbeddingClient:
    def __init__(self, api_key: str, base_url: str, timeout: float = 120) -> None:
        self.client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def embed(
        self,
        texts: list[str],
        *,
        model: str,
        input_type: str | None = None,
        truncation: bool = False,
        output_dtype: str = "float",
        output_dimension: int | None = None,
    ) -> EmbeddingResponse:
        del truncation, output_dtype
        inputs = texts
        if input_type == "query":
            inputs = [
                "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
                f"Query: {text}"
                for text in texts
            ]
        payload: dict = {
            "model": model,
            "input": inputs,
            "encoding_format": "float",
        }
        if output_dimension is not None:
            payload["dimensions"] = output_dimension
        last_error: Exception | None = None
        for attempt in range(6):
            try:
                response = self.client.post("/embeddings", json=payload)
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
                response.raise_for_status()
                body = response.json()
                rows = sorted(body["data"], key=lambda row: row.get("index", 0))
                usage = body.get("usage") or {}
                return EmbeddingResponse(
                    embeddings=[row["embedding"] for row in rows],
                    total_tokens=usage.get("total_tokens"),
                )
            except (httpx.TransportError, httpx.HTTPStatusError, ValueError, KeyError) as exc:
                last_error = exc
                retryable = not isinstance(exc, httpx.HTTPStatusError) or (
                    exc.response.status_code == 429 or exc.response.status_code >= 500
                )
                if attempt >= 5 or not retryable:
                    raise
                retry_after = 0.0
                if isinstance(exc, httpx.HTTPStatusError):
                    try:
                        retry_after = float(exc.response.headers.get("retry-after", "0"))
                    except ValueError:
                        retry_after = 0.0
                time.sleep(max(retry_after, min(2**attempt, 16)))
        raise RuntimeError("SiliconFlow embedding failed after five retries") from last_error


@dataclass(frozen=True)
class TokenEncoding:
    ids: list[int]

    def __len__(self) -> int:
        return len(self.ids)


class HuggingFaceTokenizerAdapter:
    def __init__(self, model: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    def encode(self, text: str) -> TokenEncoding:
        return TokenEncoding(
            ids=self.tokenizer.encode(text, add_special_tokens=False)
        )

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


def embedding_client(values: dict[str, str], require_key: bool = False):
    api_key = values.get("embedding-api-key")
    if require_key and not api_key:
        raise RuntimeError("embedding-api-key is missing")
    if values.get("embedding-provider", "voyage").lower() == "siliconflow":
        if not api_key:
            raise RuntimeError("SiliconFlow-Api-Key is missing")
        return SiliconFlowEmbeddingClient(
            api_key=api_key,
            base_url=values.get("embedding-base-url") or "https://api.siliconflow.com/v1",
            timeout=120,
        )
    return voyageai.Client(
        api_key=api_key,
        base_url=values.get("embedding-base-url") or "https://api.voyageai.com/v1",
        max_retries=5,
        timeout=120,
    )


def embedding_tokenizer(values: dict[str, str], model: str, client):
    if values.get("embedding-provider", "voyage").lower() == "siliconflow":
        return HuggingFaceTokenizerAdapter(model)
    return client.tokenizer(model)


def select_paired_problem_ids(
    catalog_rows: list[dict],
    problem_rows: list[dict],
    code_case_rows: list[dict],
    limit: int = TOP_ITEMS_PER_GROUP,
) -> tuple[set[str], dict]:
    """Select the first N paired items from each algorithm/study-plan group.

    Curriculum groups preserve the catalog's curated order. LeetCode study-plan
    groups use their explicit position. Taking the union keeps cross-listed
    canonical problems once, while the paired-id intersection guarantees that
    every retained problem has exactly one retained solution post.
    """
    selected: set[str] = set()
    curriculum_groups: dict[str, list[str]] = {}
    for row in catalog_rows:
        slug = row["title_slug"]
        for category in row.get("curriculum_categories") or []:
            curriculum_groups.setdefault(category, []).append(slug)
    curriculum_selected = {
        group: slugs[:limit] for group, slugs in curriculum_groups.items()
    }
    for slugs in curriculum_selected.values():
        selected.update(slugs)

    study_groups: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for row in catalog_rows:
        slug = row["title_slug"]
        for source_set, groups in (row.get("study_plan_groups") or {}).items():
            for group in groups:
                key = (source_set, group["name"])
                study_groups.setdefault(key, []).append((int(group["position"]), slug))
    study_selected: dict[str, list[str]] = {}
    for (source_set, group_name), positioned in sorted(study_groups.items()):
        slugs = [slug for _, slug in sorted(positioned, key=lambda item: (item[0], item[1]))]
        key = f"{source_set}/{group_name}"
        study_selected[key] = slugs[:limit]
        selected.update(study_selected[key])

    included_problem_ids = {
        row["document_id"] for row in problem_rows if row["included_for_embedding"]
    }
    included_case_ids = {
        row["document_id"] for row in code_case_rows if row["included_for_embedding"]
    }
    paired_ids = included_problem_ids & included_case_ids
    selected_paired = selected & paired_ids
    summary = {
        "strategy": "first_n_per_algorithm_and_study_plan_group",
        "limit_per_group": limit,
        "curriculum_groups": curriculum_selected,
        "study_plan_groups": study_selected,
        "selected_before_pair_check": len(selected),
        "selected_paired_documents": len(selected_paired),
        "selected_without_pair": sorted(selected - paired_ids),
        "paired_problem_ids": sorted(selected_paired),
    }
    return selected_paired, summary


def first_heading(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return None


def compact_title(value: str) -> str:
    return value.split(" | ", 1)[0].strip()[:500]


def metadata_for(
    spec: LibrarySpec,
    row: dict,
    text: str,
    problem_catalog: dict[str, dict],
) -> tuple[str, str | None, dict]:
    source = row["source_metadata"]
    if spec.key == "algorithm_concepts":
        title = compact_title(source.get("title") or first_heading(text) or "算法概念")
        source_url = source.get("source_url")
        metadata = {
            "category": source.get("category"),
            "source": "programmercarl",
        }
    elif spec.key == "problem_bank":
        catalog = problem_catalog.get(row["document_id"], {})
        title = compact_title(catalog.get("title") or first_heading(text) or row["document_id"])
        source_url = catalog.get("problem_url") or source.get("problem_url")
        metadata = {
            "problem_id": catalog.get("problem_id") or source.get("problem_id"),
            "problem_slug": row["document_id"],
            "difficulty": catalog.get("difficulty"),
            "topic_tags": catalog.get("topic_tags") or [],
            "source_sets": catalog.get("source_sets") or [],
            "curriculum_categories": catalog.get("curriculum_categories") or [],
        }
    else:
        title = compact_title(source.get("post_title") or first_heading(text) or row["document_id"])
        source_url = source.get("post_url") or source.get("problem_url")
        metadata = {
            "problem_id": source.get("problem_id"),
            "problem_slug": source.get("problem_slug"),
            "problem_url": source.get("problem_url"),
            "author": source.get("author"),
            "author_slug": source.get("author_slug"),
            "likes": source.get("likes"),
            "views": source.get("views"),
            "source_sets": source.get("source_sets") or [],
            "content_provider": source.get("content_provider"),
        }
    return title, source_url, {key: value for key, value in metadata.items() if value is not None}


def markdown_blocks(text: str) -> list[str]:
    """Split into paragraph/code blocks while keeping fenced code together."""
    blocks: list[str] = []
    current: list[str] = []
    fence_length: int | None = None
    for line in text.splitlines():
        stripped = line.strip()
        marker_length = 0
        if stripped.startswith("```"):
            marker_length = len(stripped) - len(stripped.lstrip("`"))
        if fence_length is not None:
            current.append(line)
            if marker_length >= fence_length and stripped == "`" * marker_length:
                fence_length = None
                blocks.append("\n".join(current).strip())
                current = []
            continue
        if marker_length >= 3:
            if current:
                blocks.append("\n".join(current).strip())
                current = []
            current.append(line)
            fence_length = marker_length
            continue
        if stripped.startswith("#") or not stripped:
            if current:
                blocks.append("\n".join(current).strip())
                current = []
            if stripped:
                blocks.append(stripped)
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current).strip())
    return [block for block in blocks if block]


def split_large_block(block: str, tokenizer, max_tokens: int, overlap_tokens: int) -> list[str]:
    token_ids = tokenizer.encode(block).ids
    if len(token_ids) <= max_tokens:
        return [block]
    step = max_tokens - overlap_tokens
    return [
        tokenizer.decode(token_ids[start : start + max_tokens]).strip()
        for start in range(0, len(token_ids), step)
        if token_ids[start : start + max_tokens]
    ]


def chunk_markdown(
    text: str,
    title: str,
    tokenizer,
    max_tokens: int,
    overlap_tokens: int,
) -> list[tuple[str, str, int]]:
    title_prefix = f"标题：{title}\n\n"
    prefix_tokens = len(tokenizer.encode(title_prefix))
    # Leave a small tokenizer boundary margin because encoding the concatenated
    # prefix/content can differ by a token from encoding both independently.
    content_limit = max(128, max_tokens - prefix_tokens - 8)
    blocks: list[str] = []
    for block in markdown_blocks(text):
        blocks.extend(split_large_block(block, tokenizer, content_limit, overlap_tokens))

    groups: list[list[str]] = []
    current: list[str] = []
    for block in blocks:
        candidate = "\n\n".join([*current, block])
        if current and len(tokenizer.encode(candidate)) > content_limit:
            groups.append(current)
            overlap: list[str] = []
            for previous in reversed(current):
                test = [previous, *overlap]
                if len(tokenizer.encode("\n\n".join(test))) > overlap_tokens:
                    break
                overlap = test
            current = [*overlap, block]
            if len(tokenizer.encode("\n\n".join(current))) > content_limit:
                current = [block]
        else:
            current.append(block)
    if current:
        groups.append(current)

    chunks: list[tuple[str, str, int]] = []
    for group in groups:
        content = "\n\n".join(group).strip()
        embedding_text = title_prefix + content
        token_count = len(tokenizer.encode(embedding_text))
        if token_count > max_tokens:
            raise RuntimeError(f"chunk exceeds limit: {token_count} > {max_tokens}")
        chunks.append((content, embedding_text, token_count))
    return chunks


def build_chunks(values: dict[str, str]) -> dict:
    validate_inputs(values)
    VECTOR_ROOT.mkdir(parents=True, exist_ok=True)
    CHUNK_ROOT.mkdir(parents=True, exist_ok=True)
    client = embedding_client(values)
    problem_catalog_rows = read_jsonl(PROBLEM_CATALOG_PATH)
    problem_catalog = {row["title_slug"]: row for row in problem_catalog_rows}
    processed_rows = {
        spec.key: read_jsonl(PROCESSED_ROOT / spec.processed_dir / "manifest.jsonl")
        for spec in SPECS
    }
    selected_pair_ids, selection_summary = select_paired_problem_ids(
        problem_catalog_rows,
        processed_rows["problem_bank"],
        processed_rows["code_cases"],
    )
    write_json(SELECTION_PLAN_PATH, selection_summary)
    print(
        f"[selection] paired problem/solution documents: {len(selected_pair_ids)} "
        f"(first {TOP_ITEMS_PER_GROUP} per group)",
        flush=True,
    )
    plan_libraries: dict[str, dict] = {}
    plan_fingerprint_parts: list[str] = []

    for spec in SPECS:
        model = values[spec.model_env]
        tokenizer = embedding_tokenizer(values, model, client)
        rows = [row for row in processed_rows[spec.key] if row["included_for_embedding"]]
        if spec.key in {"problem_bank", "code_cases"}:
            rows = [row for row in rows if row["document_id"] in selected_pair_ids]
        chunk_rows: list[dict] = []
        for row in rows:
            cleaned_path = PROJECT_ROOT / row["cleaned_file"]
            text = cleaned_path.read_text(encoding="utf-8")
            title, source_url, metadata = metadata_for(spec, row, text, problem_catalog)
            pieces = chunk_markdown(
                text,
                title,
                tokenizer,
                spec.max_tokens,
                spec.overlap_tokens,
            )
            for index, (content, embedding_text, token_count) in enumerate(pieces):
                content_hash = sha256(embedding_text)
                dimension = int(values["embedding-dimension"])
                chunk_id = sha256(
                    f"{spec.key}|{row['document_id']}|{index}|{content_hash}|{model}|{dimension}"
                )[:32]
                chunk_rows.append(
                    {
                        "chunk_id": chunk_id,
                        "library": spec.key,
                        "document_id": row["document_id"],
                        "chunk_index": index,
                        "chunk_count": len(pieces),
                        "title": title,
                        "content": content,
                        "embedding_text": embedding_text,
                        "source_url": source_url,
                        "source_file": row["source_file"],
                        "cleaned_file": row["cleaned_file"],
                        "content_sha256": content_hash,
                        "token_count": token_count,
                        "model": model,
                        "dimension": dimension,
                        "metadata": metadata,
                    }
                )
        chunk_path = CHUNK_ROOT / f"{spec.key}.jsonl"
        write_jsonl(chunk_path, chunk_rows)
        token_total = sum(row["token_count"] for row in chunk_rows)
        plan_libraries[spec.key] = {
            "documents": len(rows),
            "chunks": len(chunk_rows),
            "tokens": token_total,
            "model": model,
            "dimension": int(values["embedding-dimension"]),
            "max_chunk_tokens": spec.max_tokens,
            "overlap_tokens": spec.overlap_tokens,
            "collection": values.get(spec.collection_env) or spec.default_collection,
            "chunk_file": str(chunk_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        }
        plan_fingerprint_parts.extend(
            f"{row['chunk_id']}:{row['content_sha256']}" for row in chunk_rows
        )
        print(
            f"[plan] {spec.key}: {len(rows)} docs, {len(chunk_rows)} chunks, "
            f"{token_total:,} tokens",
            flush=True,
        )

    plan = {
        "generated_at": now(),
        "fingerprint": sha256("\n".join(plan_fingerprint_parts)),
        "milvus_uri": values.get("milvus-uri", "rag-data/vector/algomate-milvus.db"),
        "libraries": plan_libraries,
        "total_chunks": sum(value["chunks"] for value in plan_libraries.values()),
        "total_tokens": sum(value["tokens"] for value in plan_libraries.values()),
        "max_disconnect_retries": 5,
        "rate_limit_tpm": int(values.get("embedding-rate-limit-tpm", "10000")),
        "rate_limit_rpm": int(values.get("embedding-rate-limit-rpm", "3")),
        "batch_max_texts": 96,
        "batch_max_tokens": int(values.get("embedding-batch-token-budget", "9500")),
        "selection": selection_summary,
    }
    write_json(PLAN_PATH, plan)
    print(
        f"[plan] total: {plan['total_chunks']} chunks, {plan['total_tokens']:,} tokens",
        flush=True,
    )
    return plan


def checkpoint_connection() -> sqlite3.Connection:
    VECTOR_ROOT.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(CHECKPOINT_PATH)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            chunk_id TEXT PRIMARY KEY,
            library TEXT NOT NULL,
            model TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            content_sha256 TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            vector BLOB NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    connection.commit()
    return connection


def load_chunks(spec: LibrarySpec) -> list[dict]:
    return read_jsonl(CHUNK_ROOT / f"{spec.key}.jsonl")


def embedding_batches(
    chunks: list[dict], max_texts: int = 96, max_tokens: int = 90_000
) -> Iterable[list[dict]]:
    batch: list[dict] = []
    token_count = 0
    for chunk in chunks:
        if batch and (
            len(batch) >= max_texts or token_count + chunk["token_count"] > max_tokens
        ):
            yield batch
            batch = []
            token_count = 0
        batch.append(chunk)
        token_count += chunk["token_count"]
    if batch:
        yield batch


def embed(values: dict[str, str]) -> dict:
    validate_inputs(values)
    if not PLAN_PATH.is_file():
        raise RuntimeError("embedding plan is missing; run --mode plan first")
    client = embedding_client(values, require_key=True)
    dimension = int(values["embedding-dimension"])
    connection = checkpoint_connection()
    started = time.monotonic()
    total_embedded = 0
    total_pending = 0
    library_results: dict[str, dict] = {}
    rate_limit_tpm = int(values.get("embedding-rate-limit-tpm", "10000"))
    rate_limit_rpm = int(values.get("embedding-rate-limit-rpm", "3"))
    batch_token_budget = int(values.get("embedding-batch-token-budget", "9500"))
    if batch_token_budget > rate_limit_tpm:
        raise RuntimeError("embedding-batch-token-budget cannot exceed embedding-rate-limit-tpm")
    minimum_interval = max(
        60.0 / max(rate_limit_rpm, 1),
        60.0 * batch_token_budget / max(rate_limit_tpm, 1),
    ) + 4.0
    last_request_started: float | None = None

    try:
        for spec in SPECS:
            chunks = load_chunks(spec)
            cached = {
                row[0]
                for row in connection.execute(
                    "SELECT chunk_id FROM embeddings WHERE library = ? AND model = ? AND dimension = ?",
                    (spec.key, values[spec.model_env], dimension),
                )
            }
            pending = [chunk for chunk in chunks if chunk["chunk_id"] not in cached]
            total_pending += len(pending)
            done = len(chunks) - len(pending)
            print(
                f"[embed] {spec.key}: cached={done}, pending={len(pending)}, total={len(chunks)}",
                flush=True,
            )
            batches = list(embedding_batches(pending, max_tokens=batch_token_budget))
            library_start = time.monotonic()
            for batch_index, batch in enumerate(batches, start=1):
                if last_request_started is not None:
                    wait_seconds = minimum_interval - (time.monotonic() - last_request_started)
                    if wait_seconds > 0:
                        print(
                            f"[rate] waiting {wait_seconds:.1f}s for "
                            f"{rate_limit_tpm:,} TPM / {rate_limit_rpm} RPM limit",
                            flush=True,
                        )
                        time.sleep(wait_seconds)
                last_request_started = time.monotonic()
                response = client.embed(
                    [chunk["embedding_text"] for chunk in batch],
                    model=values[spec.model_env],
                    input_type="document",
                    truncation=False,
                    output_dtype="float",
                    output_dimension=dimension,
                )
                if len(response.embeddings) != len(batch):
                    raise RuntimeError("provider returned a different number of embeddings")
                timestamp = now()
                records = []
                for chunk, vector in zip(batch, response.embeddings):
                    if len(vector) != dimension:
                        raise RuntimeError(
                            f"unexpected vector dimension for {chunk['chunk_id']}: {len(vector)}"
                        )
                    records.append(
                        (
                            chunk["chunk_id"],
                            spec.key,
                            values[spec.model_env],
                            dimension,
                            chunk["content_sha256"],
                            chunk["token_count"],
                            np.asarray(vector, dtype=np.float32).tobytes(),
                            timestamp,
                        )
                    )
                connection.executemany(
                    """
                    INSERT OR REPLACE INTO embeddings
                    (chunk_id, library, model, dimension, content_sha256, token_count, vector, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    records,
                )
                connection.commit()
                done += len(batch)
                total_embedded += len(batch)
                elapsed = max(time.monotonic() - library_start, 0.001)
                rate = (done - (len(chunks) - len(pending))) / elapsed
                remaining = len(chunks) - done
                eta = remaining / rate if rate > 0 else 0
                print(
                    f"[embed] {spec.key} batch {batch_index}/{len(batches)}: "
                    f"{done}/{len(chunks)} ({done / len(chunks) * 100:.1f}%), "
                    f"ETA {eta:.1f}s",
                    flush=True,
                )
            library_results[spec.key] = {
                "chunks": len(chunks),
                "newly_embedded": len(pending),
                "cached_before_run": len(chunks) - len(pending),
                "elapsed_seconds": round(time.monotonic() - library_start, 3),
            }
    finally:
        connection.close()

    result = {
        "completed_at": now(),
        "newly_embedded": total_embedded,
        "pending_at_start": total_pending,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "libraries": library_results,
    }
    write_json(VECTOR_ROOT / "embedding-result.json", result)
    return result


def milvus_uri(values: dict[str, str]) -> str:
    configured = values.get("milvus-uri") or "rag-data/vector/algomate-milvus.db"
    if configured.startswith(("http://", "https://")):
        return configured
    return str((PROJECT_ROOT / configured).resolve())


def import_milvus(values: dict[str, str]) -> dict:
    validate_inputs(values)
    dimension = int(values["embedding-dimension"])
    connection = checkpoint_connection()
    client = MilvusClient(uri=milvus_uri(values), token=values.get("milvus-token"))
    results: dict[str, dict] = {}
    try:
        for spec in SPECS:
            collection = values.get(spec.collection_env) or spec.default_collection
            chunks = load_chunks(spec)
            vectors = {
                row[0]: np.frombuffer(row[1], dtype=np.float32).tolist()
                for row in connection.execute(
                    "SELECT chunk_id, vector FROM embeddings WHERE library = ? AND model = ? AND dimension = ?",
                    (spec.key, values[spec.model_env], dimension),
                )
            }
            missing = [chunk["chunk_id"] for chunk in chunks if chunk["chunk_id"] not in vectors]
            if missing:
                raise RuntimeError(f"{spec.key}: {len(missing)} embeddings are missing from checkpoint")
            if not client.has_collection(collection):
                client.create_collection(
                    collection_name=collection,
                    dimension=dimension,
                    primary_field_name="id",
                    id_type="string",
                    vector_field_name="vector",
                    metric_type="COSINE",
                    auto_id=False,
                    consistency_level="Strong",
                    enable_dynamic_field=True,
                )
            for start in range(0, len(chunks), 100):
                batch = chunks[start : start + 100]
                data = [
                    {
                        "id": chunk["chunk_id"],
                        "vector": vectors[chunk["chunk_id"]],
                        "library": spec.key,
                        "document_id": chunk["document_id"][:1_000],
                        "chunk_index": chunk["chunk_index"],
                        "chunk_count": chunk["chunk_count"],
                        "title": chunk["title"][:1_000],
                        "content": chunk["content"],
                        "source_url": (chunk["source_url"] or "")[:2_000],
                        "source_file": chunk["source_file"][:2_000],
                        "content_sha256": chunk["content_sha256"],
                        "token_count": chunk["token_count"],
                        "model": chunk["model"],
                        "metadata_json": json.dumps(chunk["metadata"], ensure_ascii=False),
                    }
                    for chunk in batch
                ]
                client.upsert(collection_name=collection, data=data)
                print(
                    f"[milvus] {spec.key}: {min(start + len(batch), len(chunks))}/{len(chunks)}",
                    flush=True,
                )
            client.flush(collection)
            stats = client.get_collection_stats(collection)
            row_count = int(stats.get("row_count", 0))
            if row_count != len(chunks):
                raise RuntimeError(
                    f"{collection}: expected {len(chunks)} rows after import, got {row_count}"
                )
            results[spec.key] = {"collection": collection, "rows": row_count}
    finally:
        client.close()
        connection.close()
    result = {"completed_at": now(), "milvus_uri": values.get("milvus-uri"), "libraries": results}
    write_json(VECTOR_ROOT / "milvus-import-result.json", result)
    return result


def verify(values: dict[str, str]) -> dict:
    dimension = int(values["embedding-dimension"])
    client = embedding_client(values, require_key=True)
    milvus = MilvusClient(uri=milvus_uri(values), token=values.get("milvus-token"))
    queries = {
        "algorithm_concepts": "动态规划背包问题的状态转移思路",
        "problem_bank": "使用哈希表求两数之和的题目",
        "code_cases": "two sum hash map implementation",
    }
    results: dict[str, dict] = {}
    try:
        for spec in SPECS:
            collection = values.get(spec.collection_env) or spec.default_collection
            response = client.embed(
                [queries[spec.key]],
                model=values[spec.model_env],
                input_type="query",
                truncation=False,
                output_dtype="float",
                output_dimension=dimension,
            )
            hits = milvus.search(
                collection_name=collection,
                data=[response.embeddings[0]],
                limit=3,
                output_fields=[
                    "document_id",
                    "chunk_index",
                    "title",
                    "source_url",
                    "token_count",
                    "model",
                ],
            )[0]
            results[spec.key] = {
                "query": queries[spec.key],
                "collection": collection,
                "hits": [
                    {
                        "id": hit["id"],
                        "score": round(float(hit["distance"]), 6),
                        **hit.get("entity", {}),
                    }
                    for hit in hits
                ],
            }
            if not hits:
                raise RuntimeError(f"{collection}: verification query returned no hits")
    finally:
        milvus.close()
    report = {"verified_at": now(), "status": "passed", "libraries": results}
    write_json(VECTOR_ROOT / "verification-report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("plan", "embed", "import", "verify", "all"),
        default="all",
    )
    args = parser.parse_args()
    values = config()
    if args.mode in {"plan", "all"}:
        build_chunks(values)
    if args.mode in {"embed", "all"}:
        print(json.dumps(embed(values), ensure_ascii=False, indent=2), flush=True)
    if args.mode in {"import", "all"}:
        print(json.dumps(import_milvus(values), ensure_ascii=False, indent=2), flush=True)
    if args.mode in {"verify", "all"}:
        verify(values)


if __name__ == "__main__":
    main()
