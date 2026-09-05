import json
import logging
import math
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

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


@dataclass(frozen=True)
class _ChunkRecord:
    chunk_id: str
    document_id: str
    title: str
    content: str
    source_url: str | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _ScoredChunk:
    chunk: _ChunkRecord
    score: float


@dataclass(frozen=True)
class _FusedChunk:
    chunk: _ChunkRecord
    rrf_score: float
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None


class _Bm25Index:
    """Small in-memory BM25 index built from the chunks stored in Milvus.

    AlgoMate currently has only a few thousand chunks, so an in-process
    inverted index is both faster to deploy and easier to audit than creating
    a second external search service. The index is built lazily per knowledge
    collection and reused until the Milvus import report changes.
    """

    def __init__(
        self,
        records: list[_ChunkRecord],
        tokenizer: Callable[[str], list[str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.records = records
        self.k1 = k1
        self.b = b
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._document_frequency: Counter[str] = Counter()
        self._lengths: list[int] = []

        for index, record in enumerate(records):
            # Repeating the title gives exact topic names, problem numbers and
            # function names a modest field boost without hiding body matches.
            metadata_text = json.dumps(record.metadata, ensure_ascii=False)
            tokens = tokenizer(
                f"{record.title} {record.title} {record.title} "
                f"{metadata_text} {record.content}"
            )
            problem_id = record.metadata.get("problem_id")
            if problem_id is not None:
                tokens.extend([f"lc-id-{problem_id}"] * 8)
            normalized_document_id = re.sub(
                r"[^a-z0-9_-]+",
                "-",
                record.document_id.lower(),
            ).strip("-")
            if normalized_document_id:
                tokens.extend([f"doc-id-{normalized_document_id}"] * 4)
            frequencies = Counter(tokens)
            self._lengths.append(len(tokens))
            self._document_frequency.update(frequencies.keys())
            for term, frequency in frequencies.items():
                self._postings[term].append((index, frequency))

        self._average_length = (
            sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        )

    def search(
        self,
        query: str,
        tokenizer: Callable[[str], list[str]],
        limit: int,
    ) -> list[_ScoredChunk]:
        if not self.records or self._average_length <= 0 or limit <= 0:
            return []

        query_terms = Counter(tokenizer(query))
        scores: dict[int, float] = defaultdict(float)
        document_count = len(self.records)
        for term, query_frequency in query_terms.items():
            postings = self._postings.get(term)
            if not postings:
                continue
            document_frequency = self._document_frequency[term]
            # Robertson/Sparck Jones IDF in a positive-only BM25 form.
            inverse_document_frequency = math.log(
                1.0
                + (document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            for document_index, term_frequency in postings:
                length_normalization = self.k1 * (
                    1.0
                    - self.b
                    + self.b
                    * self._lengths[document_index]
                    / self._average_length
                )
                scores[document_index] += (
                    query_frequency
                    * inverse_document_frequency
                    * term_frequency
                    * (self.k1 + 1.0)
                    / (term_frequency + length_normalization)
                )

        ranked = sorted(
            scores.items(),
            key=lambda item: (-item[1], self.records[item[0]].chunk_id),
        )
        return [
            _ScoredChunk(self.records[index], float(score))
            for index, score in ranked[:limit]
            if score > 0
        ]


def _reciprocal_rank_fusion(
    dense: list[_ScoredChunk],
    sparse: list[_ScoredChunk],
    *,
    rrf_k: int,
    limit: int,
) -> list[_FusedChunk]:
    """Fuse heterogeneous rankings without comparing their raw score scales."""

    entries: dict[str, dict[str, Any]] = {}
    for source, ranking in (("dense", dense), ("bm25", sparse)):
        seen: set[str] = set()
        for rank, scored in enumerate(ranking, start=1):
            document_id = scored.chunk.document_id
            if document_id in seen:
                continue
            seen.add(document_id)
            entry = entries.setdefault(
                document_id,
                {"chunk": scored.chunk, "rrf_score": 0.0},
            )
            previous_best_rank = min(
                entry.get("dense_rank", 10**9),
                entry.get("bm25_rank", 10**9),
            )
            if rank < previous_best_rank:
                entry["chunk"] = scored.chunk
            entry["rrf_score"] += 1.0 / (rrf_k + rank)
            entry[f"{source}_rank"] = rank
            entry[f"{source}_score"] = scored.score

    fused = [
        _FusedChunk(
            chunk=value["chunk"],
            rrf_score=float(value["rrf_score"]),
            dense_rank=value.get("dense_rank"),
            dense_score=value.get("dense_score"),
            bm25_rank=value.get("bm25_rank"),
            bm25_score=value.get("bm25_score"),
        )
        for value in entries.values()
    ]
    fused.sort(
        key=lambda item: (
            -item.rrf_score,
            min(item.dense_rank or 10**9, item.bm25_rank or 10**9),
            item.chunk.document_id,
        )
    )
    return fused[:limit]


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
    """Hybrid Dense + BM25 + RRF + Voyage rerank retrieval over Milvus.

    Milvus remains the source of truth for embedded chunks. A BM25 inverted
    index is built lazily from the same rows, then dense and lexical rankings
    are fused with RRF. Voyage rerank refines the candidate pool when
    configured. Every stage can degrade independently, while the crawler
    manifest retriever remains the final failure fallback.
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
        dense_candidate_k: int = 20,
        bm25_candidate_k: int = 20,
        fusion_candidate_k: int = 20,
        rrf_k: int = 60,
        rerank_enabled: bool = True,
        rerank_model: str = "rerank-2.5",
        rerank_max_chars: int = 1_600,
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
        self.dense_candidate_k = max(1, dense_candidate_k)
        self.bm25_candidate_k = max(1, bm25_candidate_k)
        self.fusion_candidate_k = max(1, fusion_candidate_k)
        self.rrf_k = max(1, rrf_k)
        self.rerank_enabled = rerank_enabled
        self.rerank_model = rerank_model
        self.rerank_max_chars = max(200, rerank_max_chars)
        self._bm25_cache: dict[str, tuple[tuple[Any, ...], _Bm25Index]] = {}
        self._bm25_lock = threading.RLock()
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
        if request.collection not in self._ready_collections():
            return super().retrieve(request)
        try:
            evidence = self._retrieve_hybrid(request)
            if evidence:
                return evidence
        except Exception as exc:  # A local fallback is better than a failed user turn.
            logger.warning("Hybrid retrieval failed; using manifest fallback: %s", exc)
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

    def _retrieve_hybrid(self, request: RagQuery) -> list[RagEvidence]:
        dense: list[_ScoredChunk] = []
        sparse: list[_ScoredChunk] = []
        if self._voyage is not None:
            try:
                dense = self._deduplicate_document_ranking(
                    self._retrieve_dense(
                        request.collection,
                        request.query,
                        self.dense_candidate_k * 3,
                    ),
                    self.dense_candidate_k,
                )
            except Exception as exc:
                logger.warning("Dense retrieval failed; continuing with BM25: %s", exc)
        try:
            sparse = self._deduplicate_document_ranking(
                self._retrieve_bm25(
                    request.collection,
                    request.query,
                    self.bm25_candidate_k * 3,
                ),
                self.bm25_candidate_k,
            )
        except Exception as exc:
            logger.warning("BM25 retrieval failed; continuing with dense results: %s", exc)

        if not dense and not sparse:
            return []
        fused = _reciprocal_rank_fusion(
            dense,
            sparse,
            rrf_k=self.rrf_k,
            limit=max(self.fusion_candidate_k, request.top_k),
        )
        if not fused:
            return []

        rerank_applied = False
        if self.rerank_enabled and self._voyage is not None and self.rerank_model:
            try:
                fused = self._rerank(request.query, fused)
                rerank_applied = True
            except Exception as exc:
                logger.warning("Voyage rerank failed; using RRF order: %s", exc)
        return self._to_evidence(
            request,
            fused,
            rerank_applied=rerank_applied,
            dense_available=bool(dense),
            bm25_available=bool(sparse),
        )

    def _retrieve_dense(
        self,
        collection: str,
        query: str,
        limit: int,
    ) -> list[_ScoredChunk]:
        vector = list(self._embed_query(collection, query))
        client = self._new_milvus_client()
        collection_name = self.collections[collection]
        try:
            client.load_collection(collection_name)
            hits = client.search(
                collection_name=collection_name,
                data=[vector],
                limit=limit,
                output_fields=self._milvus_output_fields(),
            )[0]
        finally:
            client.close()

        candidates: list[_ScoredChunk] = []
        for hit in hits:
            entity = hit.get("entity") or {}
            chunk = self._chunk_from_entity(entity, fallback_id=hit.get("id"))
            if chunk is None:
                continue
            candidates.append(
                _ScoredChunk(
                    chunk=chunk,
                    score=max(0.0, float(hit.get("distance") or 0.0)),
                )
            )
        return candidates

    def _retrieve_bm25(
        self,
        collection: str,
        query: str,
        limit: int,
    ) -> list[_ScoredChunk]:
        index = self._bm25_index(collection)
        return index.search(
            self._expand_bm25_query(query),
            self._bm25_tokens,
            limit,
        )

    def _bm25_index(self, collection: str) -> _Bm25Index:
        signature = self._collection_signature(collection)
        with self._bm25_lock:
            cached = self._bm25_cache.get(collection)
            if cached and cached[0] == signature:
                return cached[1]
            records = self._load_milvus_chunks(collection, int(signature[2]))
            index = _Bm25Index(records, self._bm25_tokens)
            self._bm25_cache[collection] = (signature, index)
            return index

    def _collection_signature(self, collection: str) -> tuple[Any, ...]:
        report = json.loads(self.import_report_path.read_text(encoding="utf-8"))
        library = (report.get("libraries") or {}).get(collection) or {}
        return (
            self.import_report_path.stat().st_mtime_ns,
            library.get("collection"),
            int(library.get("rows") or 0),
        )

    def _load_milvus_chunks(
        self,
        collection: str,
        expected_rows: int,
    ) -> list[_ChunkRecord]:
        client = self._new_milvus_client()
        collection_name = self.collections[collection]
        try:
            client.load_collection(collection_name)
            rows = client.query(
                collection_name=collection_name,
                filter="",
                output_fields=self._milvus_output_fields(),
                limit=max(1, expected_rows),
            )
        finally:
            client.close()
        records: list[_ChunkRecord] = []
        for row in rows:
            chunk = self._chunk_from_entity(row, fallback_id=row.get("id"))
            if chunk is not None:
                records.append(chunk)
        return records

    def _rerank(
        self,
        query: str,
        candidates: list[_FusedChunk],
    ) -> list[_FusedChunk]:
        if self._voyage is None:
            return candidates
        documents = [
            f"{item.chunk.title}\n{item.chunk.content}"[: self.rerank_max_chars]
            for item in candidates
        ]
        response = self._voyage.rerank(
            query=query,
            documents=documents,
            model=self.rerank_model,
            top_k=len(documents),
            truncation=True,
        )
        reranked: list[_FusedChunk] = []
        for result in response.results:
            index = int(result.index)
            if index < 0 or index >= len(candidates):
                continue
            original = candidates[index]
            reranked.append(
                _FusedChunk(
                    chunk=original.chunk,
                    rrf_score=original.rrf_score,
                    dense_rank=original.dense_rank,
                    dense_score=original.dense_score,
                    bm25_rank=original.bm25_rank,
                    bm25_score=original.bm25_score,
                    rerank_score=float(result.relevance_score),
                )
            )
        if len(reranked) != len(candidates):
            raise RuntimeError("Voyage rerank returned an incomplete candidate ranking")
        return reranked

    def _to_evidence(
        self,
        request: RagQuery,
        candidates: list[_FusedChunk],
        *,
        rerank_applied: bool,
        dense_available: bool,
        bm25_available: bool,
    ) -> list[RagEvidence]:
        evidence: list[RagEvidence] = []
        seen_documents: set[str] = set()
        for item in candidates:
            if item.chunk.document_id in seen_documents:
                continue
            seen_documents.add(item.chunk.document_id)
            metadata = dict(item.chunk.metadata)
            metadata.update(
                {
                    "document_id": item.chunk.document_id,
                    "chunk_id": item.chunk.chunk_id,
                    "retrieval_provider": self._provider_label(
                        dense_available,
                        bm25_available,
                        rerank_applied,
                    ),
                    "dense_rank": item.dense_rank,
                    "dense_score": self._rounded(item.dense_score),
                    "bm25_rank": item.bm25_rank,
                    "bm25_score": self._rounded(item.bm25_score),
                    "rrf_score": round(item.rrf_score, 8),
                    "rerank_model": self.rerank_model if rerank_applied else None,
                    "rerank_score": self._rounded(item.rerank_score),
                }
            )
            score = (
                item.rerank_score
                if item.rerank_score is not None
                else item.rrf_score
            )
            evidence.append(
                RagEvidence(
                    evidence_id="pending",
                    collection=request.collection,
                    title=item.chunk.title,
                    content=item.chunk.content[: self.excerpt_chars],
                    source_url=item.chunk.source_url,
                    score=max(0.0, round(float(score), 6)),
                    metadata={
                        key: value
                        for key, value in metadata.items()
                        if value is not None
                    },
                )
            )
            if len(evidence) >= request.top_k:
                break
        return evidence

    def _new_milvus_client(self) -> MilvusClient:
        return MilvusClient(uri=self.milvus_uri, token=self.milvus_token)

    @staticmethod
    def _milvus_output_fields() -> list[str]:
        return [
            "id",
            "document_id",
            "chunk_index",
            "title",
            "content",
            "source_url",
            "token_count",
            "model",
            "metadata_json",
        ]

    @staticmethod
    def _chunk_from_entity(
        entity: dict[str, Any],
        *,
        fallback_id: Any,
    ) -> _ChunkRecord | None:
        chunk_id = str(entity.get("id") or fallback_id or "").strip()
        if not chunk_id:
            return None
        document_id = str(entity.get("document_id") or chunk_id)
        metadata: dict[str, Any] = {}
        raw_metadata = entity.get("metadata_json")
        if isinstance(raw_metadata, str):
            try:
                value = json.loads(raw_metadata)
                if isinstance(value, dict):
                    metadata = value
            except json.JSONDecodeError:
                pass
        metadata.update(
            {
                "chunk_index": entity.get("chunk_index"),
                "token_count": entity.get("token_count"),
                "embedding_model": entity.get("model"),
            }
        )
        return _ChunkRecord(
            chunk_id=chunk_id,
            document_id=document_id,
            title=str(entity.get("title") or document_id),
            content=str(entity.get("content") or ""),
            source_url=entity.get("source_url") or None,
            metadata={key: value for key, value in metadata.items() if value is not None},
        )

    @staticmethod
    def _bm25_tokens(text: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9][a-z0-9+#._-]*", text.lower())
        for sequence in re.findall(r"[\u3400-\u9fff]+", text):
            if len(sequence) <= 2:
                tokens.append(sequence)
                continue
            tokens.append(sequence)
            tokens.extend(
                sequence[index : index + 2]
                for index in range(len(sequence) - 1)
            )
        return tokens

    def _expand_bm25_query(self, query: str) -> str:
        expanded = self._expand_aliases(query)
        identifiers: list[str] = []
        identifier_patterns = (
            r"(?:leetcode|力扣|lc|题号)\s*[#第]?\s*(\d{1,6})",
            r"^\s*#?(\d{1,6})(?:\s|[.、：:])",
        )
        for pattern in identifier_patterns:
            for match in re.finditer(pattern, query, flags=re.IGNORECASE):
                identifiers.extend([f"lc-id-{match.group(1)}"] * 8)
        return " ".join([expanded, *identifiers])

    @staticmethod
    def _provider_label(
        dense_available: bool,
        bm25_available: bool,
        rerank_applied: bool,
    ) -> str:
        stages = []
        if dense_available:
            stages.append("milvus_voyage_dense")
        if bm25_available:
            stages.append("bm25")
        if len(stages) > 1:
            stages.append("rrf")
        if rerank_applied:
            stages.append("voyage_rerank")
        return "+".join(stages) or "manifest_lexical_fallback"

    @staticmethod
    def _deduplicate_document_ranking(
        ranking: list[_ScoredChunk],
        limit: int,
    ) -> list[_ScoredChunk]:
        result: list[_ScoredChunk] = []
        seen: set[str] = set()
        for item in ranking:
            if item.chunk.document_id in seen:
                continue
            seen.add(item.chunk.document_id)
            result.append(item)
            if len(result) >= limit:
                break
        return result

    @staticmethod
    def _rounded(value: float | None) -> float | None:
        return round(float(value), 8) if value is not None else None

    def _retrieve_vector(self, request: RagQuery) -> list[RagEvidence]:
        vector = list(self._embed_query(request.collection, request.query))
        client = MilvusClient(uri=self.milvus_uri, token=self.milvus_token)
        try:
            client.load_collection(self.collections[request.collection])
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
