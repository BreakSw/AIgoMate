import json
from collections import Counter
from pathlib import Path
from typing import Any


class RagOverviewService:
    """Build a read-only, auditable snapshot for the RAG dashboard."""

    _LIBRARIES = {
        "algorithm_concepts": {
            "label": "算法概念库",
            "source": "代码随想录",
            "quality_key": "algorithm-concepts",
            "distribution": "category",
        },
        "problem_bank": {
            "label": "题库",
            "source": "LeetCode 中国站",
            "quality_key": "problem-bank",
            "distribution": "difficulty",
        },
        "code_cases": {
            "label": "代码案例库",
            "source": "LeetCode 高赞题解",
            "quality_key": "code-cases",
            "distribution": "author",
        },
    }

    def __init__(self, project_root: Path, memory_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.vector_root = self.project_root / "rag-data" / "vector"
        self.quality_report_path = (
            self.project_root / "rag-data" / "processed" / "quality-report.json"
        )
        self.memory_root = memory_root.resolve()

    def overview(self, user_id: int) -> dict[str, Any]:
        plan = self._read_json(self.vector_root / "embedding-plan.json")
        import_report = self._read_json(self.vector_root / "milvus-import-result.json")
        quality = self._read_json(self.quality_report_path)
        plan_libraries = plan.get("libraries") or {}
        import_libraries = import_report.get("libraries") or {}
        quality_libraries = quality.get("libraries") or {}

        libraries: list[dict[str, Any]] = []
        for key, config in self._LIBRARIES.items():
            planned = plan_libraries.get(key) or {}
            imported = import_libraries.get(key) or {}
            quality_row = quality_libraries.get(config["quality_key"]) or {}
            chunk_summary = self._summarize_chunks(key, planned.get("chunk_file"))
            chunks = int(planned.get("chunks") or 0)
            imported_rows = int(imported.get("rows") or 0)
            libraries.append({
                "key": key,
                "label": config["label"],
                "source": config["source"],
                "storage": "Milvus 向量库",
                "retrieval_mode": "向量语义检索",
                "documents": int(planned.get("documents") or 0),
                "available_documents": int(
                    quality_row.get("included_for_embedding") or 0
                ),
                "chunks": chunks,
                "tokens": int(planned.get("tokens") or 0),
                "model": planned.get("model") or "未配置",
                "dimension": int(planned.get("dimension") or 0),
                "collection": imported.get("collection")
                or planned.get("collection")
                or "未创建",
                "imported_rows": imported_rows,
                "coverage": round(imported_rows / chunks * 100, 1) if chunks else 0.0,
                "status": "ready" if chunks > 0 and imported_rows >= chunks else "partial",
                "distribution": chunk_summary["distribution"],
                "samples": chunk_summary["samples"],
            })

        libraries.append(self._memory_summary(user_id))
        vector_libraries = libraries[:3]
        all_ready = all(item["status"] == "ready" for item in vector_libraries)
        return {
            "status": "ready" if all_ready else "partial",
            "storage": "Milvus Lite",
            "retrieval_mode": "Milvus 向量检索" if all_ready else "向量检索 + 文字回退",
            "embedding_provider": self._provider_for(vector_libraries),
            "generated_at": import_report.get("completed_at") or plan.get("generated_at"),
            "quality_status": quality.get("status") or "unknown",
            "total_documents": sum(item["documents"] for item in vector_libraries),
            "total_chunks": sum(item["chunks"] for item in vector_libraries),
            "total_tokens": sum(item["tokens"] for item in vector_libraries),
            "paired_problem_cases": int(
                (plan.get("selection") or {}).get("selected_paired_documents") or 0
            ),
            "libraries": libraries,
        }

    def _summarize_chunks(self, key: str, configured_path: str | None) -> dict[str, Any]:
        if not configured_path:
            return {"distribution": [], "samples": []}
        path = (self.project_root / configured_path).resolve()
        try:
            path.relative_to(self.project_root)
        except ValueError:
            return {"distribution": [], "samples": []}
        if not path.is_file():
            return {"distribution": [], "samples": []}

        documents: dict[str, dict[str, Any]] = {}
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    item = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                document_id = str(item.get("document_id") or item.get("chunk_id") or "")
                if document_id and document_id not in documents:
                    documents[document_id] = item

        dimension = self._LIBRARIES[key]["distribution"]
        distribution = Counter()
        samples: list[dict[str, Any]] = []
        for item in documents.values():
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            value = metadata.get(dimension)
            if key == "algorithm_concepts":
                value = value or "general"
            elif key == "problem_bank":
                value = value or "Unknown"
            else:
                value = value or "其他作者"
            distribution[str(value)] += 1
            if len(samples) < 5:
                samples.append({
                    "document_id": item.get("document_id"),
                    "title": item.get("title") or item.get("document_id") or "未命名文档",
                    "source_url": item.get("source_url"),
                    "metadata": {
                        name: metadata.get(name)
                        for name in (
                            "category", "problem_id", "difficulty", "author", "likes", "views"
                        )
                        if metadata.get(name) is not None
                    },
                })
        return {
            "distribution": [
                {"label": label, "count": count}
                for label, count in distribution.most_common(8)
            ],
            "samples": samples,
        }

    def _memory_summary(self, user_id: int) -> dict[str, Any]:
        user_root = self.memory_root / f"user-{user_id}"
        distribution = Counter()
        samples: list[dict[str, Any]] = []
        memory_count = 0
        session_count = 0
        latest_update: str | None = None
        if user_root.is_dir():
            for path in sorted(user_root.glob("session-*.json")):
                payload = self._read_json(path)
                values = payload.get("memories") if isinstance(payload.get("memories"), list) else []
                if values:
                    session_count += 1
                updated_at = payload.get("updated_at")
                if isinstance(updated_at, str) and (latest_update is None or updated_at > latest_update):
                    latest_update = updated_at
                for item in values:
                    if not isinstance(item, dict):
                        continue
                    memory_count += 1
                    distribution[str(item.get("kind") or "other")] += 1
                    if len(samples) < 5:
                        samples.append({
                            "document_id": item.get("memory_id"),
                            "title": str(item.get("content") or "未命名记忆")[:80],
                            "source_url": None,
                            "metadata": {
                                "kind": item.get("kind"),
                                "importance": item.get("importance"),
                                "session_id": payload.get("session_id"),
                            },
                        })
        return {
            "key": "user_memory",
            "label": "用户私有记忆库",
            "source": "当前用户 · 会话隔离",
            "storage": "本地持久化记忆",
            "retrieval_mode": "会话级相关性召回",
            "documents": memory_count,
            "available_documents": memory_count,
            "chunks": memory_count,
            "tokens": 0,
            "model": "动态记忆（未向量化）",
            "dimension": 0,
            "collection": f"user-{user_id} / {session_count} 个会话",
            "imported_rows": memory_count,
            "coverage": 100.0,
            "status": "growing",
            "updated_at": latest_update,
            "distribution": [
                {"label": label, "count": count}
                for label, count in distribution.most_common(8)
            ],
            "samples": samples,
        }

    @staticmethod
    def _provider_for(libraries: list[dict[str, Any]]) -> str:
        models = " ".join(str(item.get("model") or "") for item in libraries).lower()
        if "voyage" in models:
            return "Voyage AI"
        if "qwen" in models:
            return "Qwen"
        return "自定义 Embedding"

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}
