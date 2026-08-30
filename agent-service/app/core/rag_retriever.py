import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import voyageai
from pymilvus import MilvusClient

from app.models import CoordinatorPlan, RagEvidence, RagQuery


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _IndexRecord:
    collection: str
    title: str
    search_text: str
    source_url: str | None
    content_file: str | None
    metadata: dict


class LocalRagRetriever:
    """Read-only lexical fallback over crawler output.

    The interface is intentionally independent of storage. A Qdrant-backed
    implementation can replace this class without changing either agent
    protocol.
    """

    _SOURCES = {
        "algorithm_concepts": Path(
            "rag-data/raw/algorithm-concepts/programmercarl/manifest.jsonl"
        ),
        "problem_bank": Path(
            "rag-data/raw/problem-bank/leetcode/problem-manifest.jsonl"
        ),
        "code_cases": Path(
            "rag-data/raw/code-cases/leetcode/posts/manifest.jsonl"
        ),
    }
    _ALIASES = {
        "数组": "array",
        "链表": "linked list",
        "哈希": "hash table",
        "字符串": "string",
        "双指针": "two pointer",
        "二叉树": "binary tree",
        "回溯": "backtracking",
        "贪心": "greedy",
        "动态规划": "dynamic programming",
        "单调栈": "monotonic stack",
        "图论": "graph",
        "并查集": "union find",
        "最短路": "shortest path",
        "二分查找": "binary search",
        "两数之和": "two sum",
    }

    def __init__(
        self,
        project_root: Path,
        excerpt_chars: int = 3_500,
        total_context_chars: int = 12_000,
    ) -> None:
        self.project_root = project_root.resolve()
        self.excerpt_chars = excerpt_chars
        self.total_context_chars = total_context_chars
        self._cache: dict[str, tuple[tuple[int, int], list[_IndexRecord]]] = {}

    def availability(self) -> dict[str, bool]:
        result: dict[str, bool] = {}
        for collection, relative_path in self._SOURCES.items():
            path = self.project_root / relative_path
            result[collection] = path.is_file() and path.stat().st_size > 0
        return result

    def retrieve_for_plan(self, plan: CoordinatorPlan) -> list[RagEvidence]:
        collected: list[RagEvidence] = []
        used_chars = 0
        for query in plan.rag_queries:
            for candidate in self.retrieve(query):
                if used_chars >= self.total_context_chars:
                    break
                remaining = self.total_context_chars - used_chars
                content = candidate.content[:remaining]
                if not content:
                    continue
                collected.append(candidate.model_copy(update={"content": content}))
                used_chars += len(content)
            if used_chars >= self.total_context_chars:
                break
        return [
            item.model_copy(update={"evidence_id": f"R{index}"})
            for index, item in enumerate(collected, start=1)
        ]

    def retrieve(self, request: RagQuery) -> list[RagEvidence]:
        query = self._expand_aliases(request.query)
        records = self._records(request.collection)
        ranked = sorted(
            (
                (self._score(query, record.search_text), record)
                for record in records
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        evidence: list[RagEvidence] = []
        for score, record in ranked:
            if score <= 0 or len(evidence) >= request.top_k:
                break
            content = self._content_for(record)
            if not content:
                continue
            evidence.append(
                RagEvidence(
                    evidence_id="pending",
                    collection=request.collection,
                    title=record.title,
                    content=content,
                    source_url=record.source_url,
                    score=round(score, 4),
                    metadata=record.metadata,
                )
            )
        return evidence

    def _records(self, collection: str) -> list[_IndexRecord]:
        path = self.project_root / self._SOURCES[collection]
        if not path.is_file():
            return []
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        cached = self._cache.get(collection)
        if cached and cached[0] == signature:
            return cached[1]

        records: list[_IndexRecord] = []
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    # The crawler may be appending the last line concurrently.
                    continue
                record = self._to_record(collection, item)
                if record is not None:
                    records.append(record)
        self._cache[collection] = (signature, records)
        return records

    def _to_record(self, collection: str, item: dict) -> _IndexRecord | None:
        if collection == "algorithm_concepts":
            title = item.get("title") or item.get("source_url") or "算法概念"
            fields = [title, item.get("category"), item.get("source_url")]
            content_file = item.get("markdown_file")
            source_url = item.get("source_url")
            metadata = {"category": item.get("category")}
        elif collection == "problem_bank":
            title = item.get("title") or item.get("title_slug") or "LeetCode 题目"
            tags = item.get("topic_tags") or []
            tag_text = " ".join(
                str(tag.get("name") or tag.get("slug") or "")
                for tag in tags
                if isinstance(tag, dict)
            )
            categories = item.get("curriculum_categories") or []
            fields = [
                title,
                item.get("title_slug"),
                item.get("problem_id"),
                tag_text,
                " ".join(categories),
            ]
            slug = item.get("title_slug")
            content_file = (
                f"rag-data/raw/problem-bank/leetcode/pages-from-solution-lists/markdown/{slug}.md"
                if slug
                else None
            )
            source_url = item.get("problem_url")
            metadata = {
                "problem_id": item.get("problem_id"),
                "difficulty": item.get("difficulty"),
                "topic_tags": tags,
            }
        else:
            title = item.get("post_title") or item.get("problem_slug") or "代码案例"
            fields = [
                title,
                item.get("problem_slug"),
                item.get("problem_id"),
                item.get("author"),
            ]
            content_file = item.get("markdown_file")
            source_url = item.get("post_url") or item.get("problem_url")
            metadata = {
                "problem_id": item.get("problem_id"),
                "problem_slug": item.get("problem_slug"),
                "author": item.get("author"),
                "likes": item.get("likes"),
                "views": item.get("views"),
            }
        search_text = " ".join(str(value) for value in fields if value)
        return _IndexRecord(
            collection=collection,
            title=str(title),
            search_text=self._expand_aliases(search_text),
            source_url=source_url,
            content_file=content_file,
            metadata={key: value for key, value in metadata.items() if value is not None},
        )

    def _content_for(self, record: _IndexRecord) -> str:
        if record.content_file:
            candidate = (self.project_root / record.content_file).resolve()
            try:
                candidate.relative_to(self.project_root)
                if candidate.is_file():
                    text = candidate.read_text(encoding="utf-8", errors="replace")
                    return self._clean_excerpt(text)
            except (OSError, ValueError):
                pass
        fallback = [record.title]
        fallback.extend(f"{key}: {value}" for key, value in record.metadata.items())
        if record.source_url:
            fallback.append(f"source: {record.source_url}")
        return self._clean_excerpt("\n".join(fallback))

    def _clean_excerpt(self, text: str) -> str:
        text = text.replace("\x00", "").strip()
        if len(text) <= self.excerpt_chars:
            return text
        return text[: self.excerpt_chars].rstrip() + "\n[片段已截断]"

    def _expand_aliases(self, text: str) -> str:
        normalized = text.lower()
        additions = [english for chinese, english in self._ALIASES.items() if chinese in normalized]
        return " ".join([normalized, *additions])

    def _score(self, query: str, candidate: str) -> float:
        query = query.lower()
        candidate = candidate.lower()
        score = 0.0
        compact_query = re.sub(r"\s+", " ", query).strip()
        if compact_query and compact_query in candidate:
            score += 12.0
        query_tokens = self._tokens(query)
        candidate_tokens = self._tokens(candidate)
        if not query_tokens or not candidate_tokens:
            return score
        overlap = query_tokens & candidate_tokens
        score += sum(2.0 if len(token) > 2 else 1.0 for token in overlap)
        score += 4.0 * len(overlap) / len(query_tokens)
        return score

    @staticmethod
    def _tokens(text: str) -> set[str]:
        tokens = set(re.findall(r"[a-z0-9][a-z0-9+#._-]*", text.lower()))
        for sequence in re.findall(r"[\u3400-\u9fff]+", text):
            if len(sequence) == 1:
                tokens.add(sequence)
            else:
                tokens.update(sequence[index : index + 2] for index in range(len(sequence) - 1))
        return tokens


class MilvusRagRetriever(LocalRagRetriever):
    """Milvus semantic retrieval with the local lexical index as fallback.

    Each knowledge type has its own collection and Voyage model. The Milvus
    path is activated only after the embedding pipeline writes a successful
    import report, so the app remains usable while an initial import runs.
    """

    def __init__(
        self,
        project_root: Path,
        *,
        excerpt_chars: int = 3_500,
        total_context_chars: int = 12_000,
        embedding_base_url: str = "https://api.voyageai.com/v1",
        embedding_api_key: str | None = None,
        embedding_general_model: str = "voyage-4",
        embedding_code_model: str = "voyage-code-4",
        embedding_dimension: int = 1_024,
        milvus_uri: str = "rag-data/vector/algomate-milvus.db",
        milvus_token: str | None = None,
        collections: dict[str, str] | None = None,
    ) -> None:
        super().__init__(project_root, excerpt_chars, total_context_chars)
        self.embedding_api_key = embedding_api_key
        self.embedding_general_model = embedding_general_model
        self.embedding_code_model = embedding_code_model
        self.embedding_dimension = embedding_dimension
        self.collections = collections or {
            "algorithm_concepts": "algomate_algorithm_concepts_v1",
            "problem_bank": "algomate_problem_bank_v1",
            "code_cases": "algomate_code_cases_v1",
        }
        self.import_report_path = self.project_root / "rag-data/vector/milvus-import-result.json"
        self.milvus_uri = (
            milvus_uri
            if milvus_uri.startswith(("http://", "https://"))
            else str((self.project_root / milvus_uri).resolve())
        )
        self.milvus_token = milvus_token
        self._voyage = (
            voyageai.Client(
                api_key=embedding_api_key,
                base_url=embedding_base_url,
                max_retries=5,
                timeout=60,
            )
            if embedding_api_key
            else None
        )

    def availability(self) -> dict[str, bool]:
        fallback = super().availability()
        ready = self._ready_collections()
        return {
            collection: collection in ready or fallback.get(collection, False)
            for collection in self.collections
        }

    def retrieve(self, request: RagQuery) -> list[RagEvidence]:
        if self._voyage is None or request.collection not in self._ready_collections():
            return super().retrieve(request)
        try:
            evidence = self._retrieve_vector(request)
            if evidence:
                return evidence
        except Exception as exc:  # A lexical fallback is better than a failed user turn.
            logger.warning("Milvus retrieval failed; using lexical fallback: %s", exc)
        return super().retrieve(request)

    def _ready_collections(self) -> set[str]:
        if not self.import_report_path.is_file():
            return set()
        try:
            report = json.loads(self.import_report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return set()
        libraries = report.get("libraries") or {}
        return {
            key
            for key, value in libraries.items()
            if isinstance(value, dict)
            and value.get("collection") == self.collections.get(key)
            and int(value.get("rows") or 0) > 0
        }

    @lru_cache(maxsize=256)
    def _embed_query(self, collection: str, query: str) -> tuple[float, ...]:
        if self._voyage is None:
            raise RuntimeError("Voyage embedding client is unavailable")
        model = (
            self.embedding_code_model
            if collection == "code_cases"
            else self.embedding_general_model
        )
        response = self._voyage.embed(
            [query],
            model=model,
            input_type="query",
            truncation=False,
            output_dtype="float",
            output_dimension=self.embedding_dimension,
        )
        return tuple(float(value) for value in response.embeddings[0])

    def _retrieve_vector(self, request: RagQuery) -> list[RagEvidence]:
        vector = list(self._embed_query(request.collection, request.query))
        client = MilvusClient(uri=self.milvus_uri, token=self.milvus_token)
        try:
            hits = client.search(
                collection_name=self.collections[request.collection],
                data=[vector],
                limit=max(request.top_k * 3, request.top_k),
                output_fields=[
                    "document_id",
                    "chunk_index",
                    "title",
                    "content",
                    "source_url",
                    "token_count",
                    "model",
                    "metadata_json",
                ],
            )[0]
        finally:
            client.close()

        evidence: list[RagEvidence] = []
        seen_documents: set[str] = set()
        for hit in hits:
            entity = hit.get("entity") or {}
            document_id = str(entity.get("document_id") or hit.get("id"))
            if document_id in seen_documents:
                continue
            seen_documents.add(document_id)
            metadata: dict = {}
            raw_metadata = entity.get("metadata_json")
            if isinstance(raw_metadata, str):
                try:
                    metadata = json.loads(raw_metadata)
                except json.JSONDecodeError:
                    metadata = {}
            metadata.update(
                {
                    "document_id": document_id,
                    "chunk_index": entity.get("chunk_index"),
                    "token_count": entity.get("token_count"),
                    "embedding_model": entity.get("model"),
                    "retrieval_provider": "milvus_voyage",
                }
            )
            evidence.append(
                RagEvidence(
                    evidence_id="pending",
                    collection=request.collection,
                    title=str(entity.get("title") or document_id),
                    content=str(entity.get("content") or "")[: self.excerpt_chars],
                    source_url=entity.get("source_url") or None,
                    score=max(0.0, round(float(hit.get("distance") or 0.0), 6)),
                    metadata=metadata,
                )
            )
            if len(evidence) >= request.top_k:
                break
        return evidence
