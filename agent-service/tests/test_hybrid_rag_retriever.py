from types import SimpleNamespace

import pytest

from app.core.rag_retriever import (
    MilvusRagRetriever,
    _Bm25Index,
    _ChunkRecord,
    _FusedChunk,
    _ScoredChunk,
    _reciprocal_rank_fusion,
)
from app.models import RagQuery


def chunk(
    chunk_id: str,
    document_id: str,
    title: str,
    content: str = "",
    metadata: dict | None = None,
) -> _ChunkRecord:
    return _ChunkRecord(
        chunk_id=chunk_id,
        document_id=document_id,
        title=title,
        content=content,
        source_url=f"https://example.test/{document_id}",
        metadata=metadata or {},
    )


def test_bm25_prefers_explicit_problem_identifier() -> None:
    records = [
        chunk("c167", "two-sum-ii", "Two Sum II", metadata={"problem_id": "167"}),
        chunk("c1", "two-sum", "Two Sum", metadata={"problem_id": "1"}),
        chunk("c15", "3sum", "3Sum", metadata={"problem_id": "15"}),
    ]
    index = _Bm25Index(records, MilvusRagRetriever._bm25_tokens)

    ranking = index.search(
        "leetcode two sum lc-id-1 lc-id-1 lc-id-1",
        MilvusRagRetriever._bm25_tokens,
        3,
    )

    assert ranking[0].chunk.document_id == "two-sum"
    assert ranking[0].score > ranking[1].score


def test_rrf_rewards_documents_supported_by_both_retrievers() -> None:
    a = chunk("a1", "a", "A")
    b = chunk("b1", "b", "B")
    c = chunk("c1", "c", "C")
    dense = [_ScoredChunk(a, 0.9), _ScoredChunk(b, 0.8)]
    sparse = [_ScoredChunk(b, 12.0), _ScoredChunk(c, 10.0)]

    fused = _reciprocal_rank_fusion(dense, sparse, rrf_k=60, limit=3)

    assert [item.chunk.document_id for item in fused] == ["b", "a", "c"]
    assert fused[0].dense_rank == 2
    assert fused[0].bm25_rank == 1


def test_rrf_deduplicates_different_chunks_from_same_document() -> None:
    dense_chunk = chunk("same-dense", "same", "Same dense chunk")
    sparse_chunk = chunk("same-sparse", "same", "Same sparse chunk")

    fused = _reciprocal_rank_fusion(
        [_ScoredChunk(dense_chunk, 0.8)],
        [_ScoredChunk(sparse_chunk, 8.0)],
        rrf_k=60,
        limit=5,
    )

    assert len(fused) == 1
    assert fused[0].dense_rank == 1
    assert fused[0].bm25_rank == 1


def test_voyage_rerank_maps_results_back_to_fused_candidates(tmp_path) -> None:
    class FakeVoyage:
        def rerank(self, **kwargs):
            assert kwargs["model"] == "rerank-2.5"
            assert kwargs["top_k"] == 2
            return SimpleNamespace(
                results=[
                    SimpleNamespace(index=1, relevance_score=0.91),
                    SimpleNamespace(index=0, relevance_score=0.42),
                ]
            )

    retriever = MilvusRagRetriever(tmp_path, embedding_api_key=None)
    retriever._voyage = FakeVoyage()
    first = _FusedChunk(chunk("a", "a", "A"), rrf_score=0.03)
    second = _FusedChunk(chunk("b", "b", "B"), rrf_score=0.02)

    result = retriever._rerank("query", [first, second])

    assert [item.chunk.document_id for item in result] == ["b", "a"]
    assert result[0].rerank_score == pytest.approx(0.91)


def test_hybrid_retrieval_returns_reranked_top_k_with_auditable_metadata(
    tmp_path,
    monkeypatch,
) -> None:
    retriever = MilvusRagRetriever(tmp_path, embedding_api_key=None)
    retriever._voyage = object()
    a = chunk("a", "a", "A")
    b = chunk("b", "b", "B")
    c = chunk("c", "c", "C")
    monkeypatch.setattr(retriever, "_ready_collections", lambda: {"problem_bank"})
    monkeypatch.setattr(
        retriever,
        "_retrieve_dense",
        lambda *_args: [_ScoredChunk(a, 0.9), _ScoredChunk(b, 0.8)],
    )
    monkeypatch.setattr(
        retriever,
        "_retrieve_bm25",
        lambda *_args: [_ScoredChunk(b, 8.0), _ScoredChunk(c, 7.0)],
    )

    def fake_rerank(_query, candidates):
        by_id = {item.chunk.document_id: item for item in candidates}
        return [
            _FusedChunk(**{**by_id["a"].__dict__, "rerank_score": 0.95}),
            _FusedChunk(**{**by_id["b"].__dict__, "rerank_score": 0.90}),
            _FusedChunk(**{**by_id["c"].__dict__, "rerank_score": 0.20}),
        ]

    monkeypatch.setattr(retriever, "_rerank", fake_rerank)

    result = retriever.retrieve(
        RagQuery(
            collection="problem_bank",
            query="two sum",
            reason="test hybrid retrieval",
            top_k=2,
        )
    )

    assert [item.title for item in result] == ["A", "B"]
    assert result[0].score == pytest.approx(0.95)
    assert result[1].metadata["dense_rank"] == 2
    assert result[1].metadata["bm25_rank"] == 1
    assert result[1].metadata["retrieval_provider"] == (
        "milvus_voyage_dense+bm25+rrf+voyage_rerank"
    )
    assert result[1].metadata["rerank_model"] == "rerank-2.5"

