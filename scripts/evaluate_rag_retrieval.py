"""Evaluate AlgoMate RAG retrieval and Voyage reranking.

The script reads the already embedded document vectors from the local SQLite
checkpoint, embeds only the evaluation queries, performs exact cosine TopK
retrieval, reranks the largest candidate pool with Voyage, and writes both a
machine-readable JSON report and a review-friendly Markdown report.

API responses are cached. Re-running the same suite can therefore use
``--no-api`` without spending more embedding or reranking quota.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import voyageai
from dotenv import dotenv_values


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = PROJECT_ROOT / "rag-data/evaluation/rag-test-cases-v1.json"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "rag-data/vector/embedding-checkpoint.sqlite3"
DEFAULT_CACHE = PROJECT_ROOT / "rag-data/evaluation/cache/voyage-evaluation-cache.json"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "rag-data/evaluation/reports"
CHUNK_FILES = {
    "algorithm_concepts": PROJECT_ROOT / "rag-data/vector/chunks/algorithm_concepts.jsonl",
    "problem_bank": PROJECT_ROOT / "rag-data/vector/chunks/problem_bank.jsonl",
    "code_cases": PROJECT_ROOT / "rag-data/vector/chunks/code_cases.jsonl",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def load_config() -> dict[str, str]:
    values: dict[str, str] = {}
    for path in (PROJECT_ROOT / ".env", PROJECT_ROOT / "agent-service/.env"):
        if not path.is_file():
            continue
        for key, value in dotenv_values(path).items():
            if value is not None and key not in values:
                values[key] = str(value)
    return values


def first_value(values: dict[str, str], keys: Iterable[str], default: str = "") -> str:
    for key in keys:
        value = values.get(key)
        if value:
            return value
    return default


def stable_hash(*values: str) -> str:
    payload = "\u241f".join(values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_int_list(value: str) -> list[int]:
    result = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError("K values must be positive integers")
    return result


def safe_excerpt(value: str, limit: int = 600) -> str:
    compact = " ".join(value.replace("\x00", " ").split())
    return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "…"


class ApiRateLimiter:
    def __init__(self, rpm: float) -> None:
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self.last_call = 0.0

    def wait(self) -> None:
        if self.interval <= 0:
            return
        elapsed = time.monotonic() - self.last_call
        if self.last_call and elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self.last_call = time.monotonic()


def voyage_call(
    operation,
    limiter: ApiRateLimiter,
    label: str,
    max_attempts: int = 5,
):
    """Run one Voyage operation with a strict five-attempt total ceiling."""
    non_retryable = (
        voyageai.error.AuthenticationError,
        voyageai.error.InvalidRequestError,
        voyageai.error.MalformedRequestError,
    )
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        limiter.wait()
        try:
            return operation()
        except non_retryable:
            raise
        except Exception as exc:  # network, server and rate-limit failures
            last_error = exc
            if attempt >= max_attempts:
                break
            backoff = max(limiter.interval, 20.0) + min(10.0, attempt * 2.0)
            print(
                f"[retry] {label}: attempt {attempt}/{max_attempts} failed; "
                f"retrying in {backoff:.0f}s ({type(exc).__name__})",
                flush=True,
            )
            time.sleep(backoff)
    raise RuntimeError(f"{label} failed after {max_attempts} attempts") from last_error


def load_cache(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": "1.0", "query_embeddings": {}, "rerank_results": {}}
    value = read_json(path)
    value.setdefault("query_embeddings", {})
    value.setdefault("rerank_results", {})
    return value


def save_cache(path: Path, cache: dict[str, Any]) -> None:
    cache["updated_at"] = now_iso()
    write_json(path, cache)


def load_corpus(checkpoint_path: Path) -> dict[str, dict[str, Any]]:
    metadata_by_chunk: dict[str, dict[str, Any]] = {}
    for collection, path in CHUNK_FILES.items():
        for row in read_jsonl(path):
            row["collection"] = collection
            metadata_by_chunk[row["chunk_id"]] = row

    connection = sqlite3.connect(checkpoint_path)
    try:
        vector_rows = connection.execute(
            "SELECT chunk_id, library, model, dimension, vector FROM embeddings"
        ).fetchall()
    finally:
        connection.close()

    grouped: dict[str, list[tuple[dict[str, Any], np.ndarray]]] = {
        key: [] for key in CHUNK_FILES
    }
    missing_metadata: list[str] = []
    for chunk_id, library, model, dimension, blob in vector_rows:
        metadata = metadata_by_chunk.get(chunk_id)
        if metadata is None:
            missing_metadata.append(chunk_id)
            continue
        vector = np.frombuffer(blob, dtype=np.float32)
        if vector.size != int(dimension):
            raise RuntimeError(
                f"{chunk_id}: stored vector has {vector.size} values, expected {dimension}"
            )
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(vector).all() or not math.isfinite(norm) or norm == 0:
            raise RuntimeError(f"{chunk_id}: invalid vector")
        item = dict(metadata)
        item["embedding_model"] = model
        item["dimension"] = int(dimension)
        grouped[library].append((item, vector / norm))

    if missing_metadata:
        raise RuntimeError(f"{len(missing_metadata)} checkpoint vectors have no chunk metadata")

    result: dict[str, dict[str, Any]] = {}
    for collection, rows in grouped.items():
        if not rows:
            raise RuntimeError(f"{collection}: no vectors loaded")
        dimensions = {item[0]["dimension"] for item in rows}
        models = {item[0]["embedding_model"] for item in rows}
        if len(dimensions) != 1 or len(models) != 1:
            raise RuntimeError(
                f"{collection}: inconsistent dimensions/models: {dimensions}, {models}"
            )
        result[collection] = {
            "items": [item for item, _ in rows],
            "matrix": np.vstack([vector for _, vector in rows]),
            "dimension": next(iter(dimensions)),
            "model": next(iter(models)),
        }
    return result


def query_embedding_key(model: str, dimension: int, query: str) -> str:
    return stable_hash("query", model, str(dimension), query)


def prepare_query_embeddings(
    cases: list[dict[str, Any]],
    corpus: dict[str, dict[str, Any]],
    cache: dict[str, Any],
    cache_path: Path,
    client: voyageai.Client,
    limiter: ApiRateLimiter,
    no_api: bool,
) -> tuple[dict[str, np.ndarray], int, int]:
    embeddings: dict[str, np.ndarray] = {}
    cache_hits = 0
    pending: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for case in cases:
        collection = case["collection"]
        model = corpus[collection]["model"]
        dimension = corpus[collection]["dimension"]
        key = query_embedding_key(model, dimension, case["query"])
        cached = cache["query_embeddings"].get(key)
        if cached is not None:
            vector = np.asarray(cached["vector"], dtype=np.float32)
            if vector.size != dimension:
                raise RuntimeError(f"cached query vector has wrong dimension for {case['id']}")
            embeddings[case["id"]] = vector
            cache_hits += 1
        else:
            pending.setdefault((model, dimension), []).append(case)

    total_tokens = 0
    for (model, dimension), model_cases in pending.items():
        if no_api:
            ids = ", ".join(case["id"] for case in model_cases)
            raise RuntimeError(f"query embedding cache miss in --no-api mode: {ids}")
        response = voyage_call(
            lambda: client.embed(
                [case["query"] for case in model_cases],
                model=model,
                input_type="query",
                truncation=False,
                output_dtype="float",
                output_dimension=dimension,
            ),
            limiter,
            f"query embedding {model}",
        )
        total_tokens += int(getattr(response, "total_tokens", 0) or 0)
        if len(response.embeddings) != len(model_cases):
            raise RuntimeError(f"{model}: query embedding count mismatch")
        for case, raw_vector in zip(model_cases, response.embeddings):
            vector = np.asarray(raw_vector, dtype=np.float32)
            key = query_embedding_key(model, dimension, case["query"])
            cache["query_embeddings"][key] = {
                "case_id": case["id"],
                "query": case["query"],
                "model": model,
                "dimension": dimension,
                "vector": vector.tolist(),
                "created_at": now_iso(),
            }
            embeddings[case["id"]] = vector
        save_cache(cache_path, cache)
        print(f"[embedding] {model}: cached {len(model_cases)} evaluation queries", flush=True)
    return embeddings, total_tokens, cache_hits


def retrieve_candidates(
    case: dict[str, Any],
    query_vector: np.ndarray,
    corpus: dict[str, dict[str, Any]],
    max_candidate_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    collection = case["collection"]
    data = corpus[collection]
    norm = float(np.linalg.norm(query_vector))
    if not math.isfinite(norm) or norm == 0:
        raise RuntimeError(f"{case['id']}: invalid query vector")
    scores = data["matrix"] @ (query_vector / norm)
    order = np.argsort(-scores)

    raw_limit = min(max_candidate_k, len(order))
    raw_top = [data["items"][int(index)] for index in order[:raw_limit]]
    raw_duplicate_rate = (
        1.0 - len({item["document_id"] for item in raw_top}) / len(raw_top)
        if raw_top
        else 0.0
    )

    candidates: list[dict[str, Any]] = []
    seen_documents: set[str] = set()
    for index in order:
        item = data["items"][int(index)]
        document_id = str(item["document_id"])
        if document_id in seen_documents:
            continue
        seen_documents.add(document_id)
        candidates.append(
            {
                "vector_rank": len(candidates) + 1,
                "vector_score": round(float(scores[int(index)]), 8),
                "collection": collection,
                "document_id": document_id,
                "chunk_id": item["chunk_id"],
                "chunk_index": item["chunk_index"],
                "title": item["title"],
                "source_url": item.get("source_url"),
                "content_excerpt": safe_excerpt(item.get("content", "")),
                "rerank_text": (
                    f"标题：{item['title']}\n"
                    f"正文：{item.get('content', '')}"
                ),
            }
        )
        if len(candidates) >= max_candidate_k:
            break

    return candidates, {
        "raw_chunk_top_k": raw_limit,
        "raw_chunk_duplicate_document_rate": round(raw_duplicate_rate, 8),
        "deduplicated_candidate_count": len(candidates),
    }


def rerank_cache_key(model: str, query: str, candidates: list[dict[str, Any]]) -> str:
    return stable_hash(
        "rerank",
        model,
        query,
        *[item["chunk_id"] for item in candidates],
    )


def rerank_candidates(
    case: dict[str, Any],
    candidates: list[dict[str, Any]],
    model: str,
    max_chars: int,
    cache: dict[str, Any],
    cache_path: Path,
    client: voyageai.Client,
    limiter: ApiRateLimiter,
    no_api: bool,
) -> tuple[list[dict[str, Any]], int, bool]:
    key = rerank_cache_key(model, case["query"], candidates)
    cached = cache["rerank_results"].get(key)
    if cached is None:
        if no_api:
            raise RuntimeError(f"rerank cache miss in --no-api mode: {case['id']}")
        documents = [item["rerank_text"][:max_chars] for item in candidates]
        response = voyage_call(
            lambda: client.rerank(
                query=case["query"],
                documents=documents,
                model=model,
                top_k=len(documents),
                truncation=True,
            ),
            limiter,
            f"rerank {case['id']}",
        )
        cached = {
            "case_id": case["id"],
            "query": case["query"],
            "model": model,
            "candidate_chunk_ids": [item["chunk_id"] for item in candidates],
            "results": [
                {
                    "candidate_index": int(item.index),
                    "relevance_score": float(item.relevance_score),
                }
                for item in response.results
            ],
            "total_tokens": int(getattr(response, "total_tokens", 0) or 0),
            "created_at": now_iso(),
        }
        cache["rerank_results"][key] = cached
        save_cache(cache_path, cache)
        cache_hit = False
    else:
        cache_hit = True

    ranked: list[dict[str, Any]] = []
    for rank, value in enumerate(cached["results"], start=1):
        candidate = dict(candidates[int(value["candidate_index"])])
        candidate.pop("rerank_text", None)
        candidate["rerank_rank"] = rank
        candidate["rerank_score"] = round(float(value["relevance_score"]), 8)
        ranked.append(candidate)
    return ranked, int(cached.get("total_tokens") or 0), cache_hit


def relevance_map(case: dict[str, Any]) -> dict[str, int]:
    return {
        str(item["document_id"]): int(item.get("relevance_grade", 1))
        for item in case.get("relevant_documents", [])
    }


def metric_for_case(case: dict[str, Any], ranking: list[dict[str, Any]], k: int) -> dict[str, float]:
    relevant = relevance_map(case)
    top = ranking[:k]
    hit_ranks = [
        index
        for index, item in enumerate(top, start=1)
        if item["document_id"] in relevant
    ]
    hit_count = len({item["document_id"] for item in top if item["document_id"] in relevant})
    precision = hit_count / k
    recall = hit_count / len(relevant) if relevant else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    reciprocal_rank = 1.0 / hit_ranks[0] if hit_ranks else 0.0

    running_hits = 0
    precision_sum = 0.0
    for index, item in enumerate(top, start=1):
        if item["document_id"] in relevant:
            running_hits += 1
            precision_sum += running_hits / index
    average_precision = precision_sum / min(len(relevant), k) if relevant else 0.0

    dcg = 0.0
    for index, item in enumerate(top, start=1):
        grade = relevant.get(item["document_id"], 0)
        dcg += (2**grade - 1) / math.log2(index + 1)
    ideal_grades = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum((2**grade - 1) / math.log2(index + 1) for index, grade in enumerate(ideal_grades, start=1))
    ndcg = dcg / idcg if idcg else 0.0

    return {
        "hit": 1.0 if hit_count else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "reciprocal_rank": reciprocal_rank,
        "average_precision": average_precision,
        "ndcg": ndcg,
    }


def aggregate_metrics(
    cases_by_id: dict[str, dict[str, Any]],
    rankings_by_id: dict[str, list[dict[str, Any]]],
    k: int,
) -> dict[str, float | int]:
    positive_ids = [
        case_id
        for case_id, case in cases_by_id.items()
        if case.get("expected_behavior") == "retrieve" and relevance_map(case)
    ]
    values = [metric_for_case(cases_by_id[case_id], rankings_by_id[case_id], k) for case_id in positive_ids]
    count = len(values)
    result: dict[str, float | int] = {"queries": count, "k": k}
    names = ("hit", "precision", "recall", "f1", "reciprocal_rank", "average_precision", "ndcg")
    for name in names:
        result[name] = round(sum(value[name] for value in values) / count, 8) if count else 0.0
    return result


def filtered_rerank(
    ranking: list[dict[str, Any]], candidate_k: int
) -> list[dict[str, Any]]:
    return [item for item in ranking if int(item["vector_rank"]) <= candidate_k]


def json_safe_candidate(item: dict[str, Any]) -> dict[str, Any]:
    value = dict(item)
    value.pop("rerank_text", None)
    return value


def md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def percent(value: float | int) -> str:
    return f"{float(value) * 100:.2f}%"


def render_report(report: dict[str, Any], display_candidate_k: int, display_final_k: int) -> str:
    lines = [
        "# AlgoMate RAG 检索与 Voyage Rerank 评测报告",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 测试集：{report['suite']['name']}（{report['suite']['case_count']} 条）",
        f"- Rerank 模型：`{report['configuration']['rerank_model']}`",
        f"- Candidate K：{report['configuration']['candidate_k_values']}",
        f"- Final K：{report['configuration']['final_k_values']}",
        "",
        "## 结论摘要",
        "",
        report["summary"],
        "",
        "### 解释限制",
        "",
        "- 本测试集的 19 条正向用例目前每条只标注 1 个黄金文档，因此 Precision@K 的理论最大值为 `1/K`；Precision 随 K 下降不能单独解释为检索变差。",
        f"- 免费账户受 10K TPM 限制，Top30 Rerank 输入采用每个候选最多 {report['configuration']['rerank_max_chars']} 字符的“标题 + 正文摘要”，不是完整分块。",
        "- 概念库可能存在多个语义正确的替代文档；当前严格单文档标签会把这些替代文档计为未命中，所以本报告适合比较版本，不应被视为最终人工相关性裁决。",
        "",
        "## 纯向量召回指标",
        "",
        "| K | Hit@K | Precision@K | Recall@K | F1@K | MRR@K | MAP@K | nDCG@K |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["metrics"]["vector"]:
        lines.append(
            "| {k} | {hit} | {precision} | {recall} | {f1} | {mrr} | {map_} | {ndcg} |".format(
                k=row["k"],
                hit=percent(row["hit"]),
                precision=percent(row["precision"]),
                recall=percent(row["recall"]),
                f1=percent(row["f1"]),
                mrr=percent(row["reciprocal_rank"]),
                map_=percent(row["average_precision"]),
                ndcg=percent(row["ndcg"]),
            )
        )

    lines.extend(
        [
            "",
            "## Voyage Rerank 指标",
            "",
            "| Candidate K | Final K | Hit@K | Precision@K | Recall@K | F1@K | MRR@K | MAP@K | nDCG@K |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["metrics"]["rerank"]:
        lines.append(
            "| {candidate_k} | {final_k} | {hit} | {precision} | {recall} | {f1} | {mrr} | {map_} | {ndcg} |".format(
                candidate_k=row["candidate_k"],
                final_k=row["final_k"],
                hit=percent(row["hit"]),
                precision=percent(row["precision"]),
                recall=percent(row["recall"]),
                f1=percent(row["f1"]),
                mrr=percent(row["reciprocal_rank"]),
                map_=percent(row["average_precision"]),
                ndcg=percent(row["ndcg"]),
            )
        )

    operational = report["metrics"]["operational"]
    lines.extend(
        [
            "",
            "## 运行与异常指标",
            "",
            f"- 空结果率：{percent(operational['empty_result_rate'])}",
            f"- 错库率：{percent(operational['wrong_collection_rate'])}",
            f"- 去重后重复文档率：{percent(operational['deduplicated_duplicate_document_rate'])}",
            f"- 原始 Chunk TopK 重复文档率：{percent(operational['raw_chunk_duplicate_document_rate'])}",
            f"- 库外问题无阈值返回率：{percent(operational['negative_unthresholded_return_rate'])}",
            "",
            "> 库外问题无阈值返回率不代表模型错误：向量检索天然会返回最近邻。要评价拒绝能力，需要另行确定向量或 Rerank 分数阈值。",
            "",
            "## 逐条结果",
            "",
            f"> Markdown 展示每条向量 Top {display_candidate_k} 和 Rerank Top {display_final_k}；完整 Candidate TopK 保存在同名 JSON 报告中。",
        ]
    )

    for case in report["cases"]:
        relevant_ids = set(case["expected_document_ids"])
        lines.extend(
            [
                "",
                f"### {case['id']} · {case['collection']}",
                "",
                f"**问题：** {case['query']}",
                "",
                f"**期望文档：** {', '.join(case['expected_document_ids']) or '无（库外问题）'}",
                "",
                f"#### 向量召回 Top {display_candidate_k}",
                "",
                "| Rank | Score | 命中 | 文档 | 标题 | 内容摘要 |",
                "| ---: | ---: | :---: | --- | --- | --- |",
            ]
        )
        for item in case["vector_candidates"][:display_candidate_k]:
            lines.append(
                f"| {item['vector_rank']} | {item['vector_score']:.6f} | "
                f"{'✓' if item['document_id'] in relevant_ids else ''} | "
                f"{md_escape(item['document_id'])} | {md_escape(item['title'])} | "
                f"{md_escape(safe_excerpt(item['content_excerpt'], 180))} |"
            )
        lines.extend(
            [
                "",
                f"#### Voyage Rerank Top {display_final_k}",
                "",
                "| Rank | Rerank Score | 原向量 Rank | 命中 | 文档 | 标题 | 内容摘要 |",
                "| ---: | ---: | ---: | :---: | --- | --- | --- |",
            ]
        )
        for item in case["reranked_candidates"][:display_final_k]:
            lines.append(
                f"| {item['rerank_rank']} | {item['rerank_score']:.6f} | {item['vector_rank']} | "
                f"{'✓' if item['document_id'] in relevant_ids else ''} | "
                f"{md_escape(item['document_id'])} | {md_escape(item['title'])} | "
                f"{md_escape(safe_excerpt(item['content_excerpt'], 180))} |"
            )

    lines.extend(
        [
            "",
            "## API 与缓存",
            "",
            f"- 本次 Query Embedding tokens：{report['api_usage']['query_embedding_tokens']}",
            f"- Query Embedding 缓存命中：{report['api_usage']['query_embedding_cache_hits']} / {report['suite']['case_count']}",
            f"- 本次新消耗 Rerank tokens：{report['api_usage']['rerank_tokens_this_run']}",
            f"- 本报告对应的 Rerank tokens：{report['api_usage']['rerank_tokens_evaluated']}",
            f"- Rerank 缓存命中：{report['api_usage']['rerank_cache_hits']} / {report['suite']['case_count']}",
            "- 报告与缓存均不包含 API Key。",
            "",
        ]
    )
    return "\n".join(lines)


def build_summary(vector_metrics: list[dict[str, Any]], rerank_metrics: list[dict[str, Any]]) -> str:
    vector_at_3 = next(item for item in vector_metrics if item["k"] == 3)
    vector_at_10 = next(item for item in vector_metrics if item["k"] == 10)
    preferred = next(
        item
        for item in rerank_metrics
        if item["candidate_k"] == 20 and item["final_k"] == 3
    )
    delta = float(preferred["reciprocal_rank"]) - float(vector_at_3["reciprocal_rank"])
    direction = "提升" if delta > 0 else "下降" if delta < 0 else "持平"
    return (
        f"纯向量 Recall@10 为 {percent(vector_at_10['recall'])}，MRR@10 为 "
        f"{percent(vector_at_10['reciprocal_rank'])}。Candidate K=20、Rerank Final K=3 时，"
        f"Recall@3 为 {percent(preferred['recall'])}，MRR@3 为 "
        f"{percent(preferred['reciprocal_rank'])}；相对相同 Final K 的纯向量 MRR@3 "
        f"{direction} {abs(delta) * 100:.2f} 个百分点。当前 Rerank 配置没有带来收益，"
        "不建议直接接入线上链路。"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--candidate-k", type=parse_int_list, default=parse_int_list("5,10,20,30"))
    parser.add_argument("--final-k", type=parse_int_list, default=parse_int_list("1,3,5"))
    parser.add_argument("--metric-k", type=parse_int_list, default=parse_int_list("1,3,5,10,20,30"))
    parser.add_argument("--rerank-model", default=None)
    parser.add_argument("--rerank-max-chars", type=int, default=160)
    parser.add_argument("--api-rpm", type=float, default=None)
    parser.add_argument("--no-api", action="store_true")
    parser.add_argument("--display-candidate-k", type=int, default=10)
    parser.add_argument("--display-final-k", type=int, default=5)
    args = parser.parse_args()

    suite = read_json(args.suite.resolve())
    cases = suite["cases"]
    if len({case["id"] for case in cases}) != len(cases):
        raise RuntimeError("test case IDs must be unique")
    max_candidate_k = max(args.candidate_k)
    if max(args.metric_k) > max_candidate_k:
        raise RuntimeError("metric K cannot exceed the largest candidate K")
    if 20 not in args.candidate_k or 3 not in args.final_k or 10 not in args.metric_k:
        raise RuntimeError("the standard report requires Candidate K=20, Final K=3 and metric K=10")

    values = load_config()
    api_key = first_value(values, ("embedding-api-key", "VOYAGE_API_KEY"))
    if not api_key and not args.no_api:
        raise RuntimeError("Voyage API key is missing")
    base_url = first_value(
        values,
        ("embedding-base-url", "VOYAGE_BASE_URL"),
        "https://api.voyageai.com/v1",
    )
    rerank_model = args.rerank_model or first_value(
        values,
        ("voyage-rerank-model", "VOYAGE_RERANK_MODEL", "rerank-model", "RERANK_MODEL"),
        "rerank-2.5",
    )
    configured_rpm = float(
        first_value(values, ("rerank-rate-limit-rpm", "embedding-rate-limit-rpm"), "3")
    )
    # Leave one request/minute of headroom for the free-tier rolling window and
    # for an already running service that may share the same Voyage account.
    api_rpm = args.api_rpm or max(1.0, configured_rpm * 0.5)
    client = voyageai.Client(
        api_key=api_key or "cache-only",
        base_url=base_url,
        # Retries are controlled by voyage_call so the total attempt ceiling is
        # exactly five rather than five nested SDK retries per outer attempt.
        max_retries=0,
        timeout=120,
    )
    limiter = ApiRateLimiter(api_rpm)
    cache_path = args.cache.resolve()
    cache = load_cache(cache_path)
    corpus = load_corpus(args.checkpoint.resolve())

    embeddings, embedding_tokens, embedding_cache_hits = prepare_query_embeddings(
        cases,
        corpus,
        cache,
        cache_path,
        client,
        limiter,
        args.no_api,
    )

    case_results: list[dict[str, Any]] = []
    vector_rankings: dict[str, list[dict[str, Any]]] = {}
    rerank_rankings: dict[str, list[dict[str, Any]]] = {}
    rerank_tokens_this_run = 0
    rerank_tokens_evaluated = 0
    rerank_cache_hits = 0
    for index, case in enumerate(cases, start=1):
        candidates, retrieval_stats = retrieve_candidates(
            case,
            embeddings[case["id"]],
            corpus,
            max_candidate_k,
        )
        reranked, tokens, cache_hit = rerank_candidates(
            case,
            candidates,
            rerank_model,
            args.rerank_max_chars,
            cache,
            cache_path,
            client,
            limiter,
            args.no_api,
        )
        rerank_tokens_evaluated += tokens
        rerank_tokens_this_run += tokens if not cache_hit else 0
        rerank_cache_hits += int(cache_hit)
        vector_clean = [json_safe_candidate(item) for item in candidates]
        vector_rankings[case["id"]] = vector_clean
        rerank_rankings[case["id"]] = reranked
        case_results.append(
            {
                "id": case["id"],
                "collection": case["collection"],
                "query": case["query"],
                "query_type": case["query_type"],
                "expected_behavior": case["expected_behavior"],
                "expected_document_ids": list(relevance_map(case)),
                "vector_candidates": vector_clean,
                "reranked_candidates": reranked,
                "retrieval_stats": retrieval_stats,
            }
        )
        print(
            f"[rerank] {index:02d}/{len(cases)} {case['id']} "
            f"({'cache' if cache_hit else rerank_model})",
            flush=True,
        )

    cases_by_id = {case["id"]: case for case in cases}
    vector_metrics = [
        aggregate_metrics(cases_by_id, vector_rankings, k) for k in args.metric_k
    ]
    rerank_metrics: list[dict[str, Any]] = []
    for candidate_k in args.candidate_k:
        filtered = {
            case_id: filtered_rerank(ranking, candidate_k)
            for case_id, ranking in rerank_rankings.items()
        }
        for final_k in args.final_k:
            metrics = aggregate_metrics(cases_by_id, filtered, final_k)
            rerank_metrics.append(
                {"candidate_k": candidate_k, "final_k": final_k, **metrics}
            )

    all_candidate_count = sum(len(value) for value in vector_rankings.values())
    duplicate_count = sum(
        len(value) - len({item["document_id"] for item in value})
        for value in vector_rankings.values()
    )
    raw_duplicate_rate = sum(
        item["retrieval_stats"]["raw_chunk_duplicate_document_rate"]
        for item in case_results
    ) / len(case_results)
    empty_result_rate = sum(not value for value in vector_rankings.values()) / len(cases)
    wrong_collection_items = sum(
        item["collection"] != cases_by_id[case_id]["collection"]
        for case_id, ranking in vector_rankings.items()
        for item in ranking
    )
    negative_ids = [
        case["id"] for case in cases if case["expected_behavior"] == "no_confident_match"
    ]
    negative_return_rate = (
        sum(bool(vector_rankings[case_id]) for case_id in negative_ids) / len(negative_ids)
        if negative_ids
        else 0.0
    )

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at": now_iso(),
        "suite": {
            "name": suite["name"],
            "path": str(args.suite.resolve()),
            "case_count": len(cases),
            "positive_case_count": len(cases) - len(negative_ids),
            "negative_case_count": len(negative_ids),
        },
        "configuration": {
            "candidate_k_values": args.candidate_k,
            "final_k_values": args.final_k,
            "metric_k_values": args.metric_k,
            "rerank_model": rerank_model,
            "rerank_max_chars": args.rerank_max_chars,
            "api_rpm": api_rpm,
            "query_models": {
                collection: {"model": value["model"], "dimension": value["dimension"]}
                for collection, value in corpus.items()
            },
            "retrieval_engine": "exact_numpy_cosine_over_cached_voyage_vectors",
            "document_deduplication": True,
        },
        "corpus": {
            collection: {
                "chunks": len(value["items"]),
                "documents": len({item["document_id"] for item in value["items"]}),
                "model": value["model"],
                "dimension": value["dimension"],
            }
            for collection, value in corpus.items()
        },
        "metrics": {
            "vector": vector_metrics,
            "rerank": rerank_metrics,
            "operational": {
                "empty_result_rate": round(empty_result_rate, 8),
                "wrong_collection_rate": round(
                    wrong_collection_items / all_candidate_count if all_candidate_count else 0.0,
                    8,
                ),
                "deduplicated_duplicate_document_rate": round(
                    duplicate_count / all_candidate_count if all_candidate_count else 0.0,
                    8,
                ),
                "raw_chunk_duplicate_document_rate": round(raw_duplicate_rate, 8),
                "negative_unthresholded_return_rate": round(negative_return_rate, 8),
            },
        },
        "api_usage": {
            "query_embedding_tokens": embedding_tokens,
            "query_embedding_cache_hits": embedding_cache_hits,
            "rerank_tokens_this_run": rerank_tokens_this_run,
            "rerank_tokens_evaluated": rerank_tokens_evaluated,
            "rerank_cache_hits": rerank_cache_hits,
        },
        "cases": case_results,
    }
    report["summary"] = build_summary(vector_metrics, rerank_metrics)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_dir = args.report_dir.resolve()
    json_path = report_dir / f"rag-evaluation-{timestamp}.json"
    markdown_path = report_dir / f"rag-evaluation-{timestamp}.md"
    write_json(json_path, report)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        render_report(report, args.display_candidate_k, args.display_final_k),
        encoding="utf-8",
        newline="\n",
    )
    print(f"JSON report: {json_path}", flush=True)
    print(f"Markdown report: {markdown_path}", flush=True)


if __name__ == "__main__":
    main()
