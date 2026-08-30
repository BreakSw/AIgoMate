import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_rag_retrieval import filtered_rerank, metric_for_case


def test_metric_for_case_uses_document_level_relevance() -> None:
    case = {
        "relevant_documents": [
            {"document_id": "expected", "relevance_grade": 3}
        ]
    }
    ranking = [
        {"document_id": "wrong-a"},
        {"document_id": "expected"},
        {"document_id": "wrong-b"},
    ]

    result = metric_for_case(case, ranking, 3)

    assert result["hit"] == 1.0
    assert result["precision"] == 1 / 3
    assert result["recall"] == 1.0
    assert result["reciprocal_rank"] == 0.5
    assert result["average_precision"] == 0.5
    assert result["ndcg"] == 1 / math.log2(3)


def test_filtered_rerank_respects_original_candidate_pool() -> None:
    ranking = [
        {"document_id": "late", "vector_rank": 8},
        {"document_id": "early-b", "vector_rank": 3},
        {"document_id": "early-a", "vector_rank": 1},
    ]

    assert [item["document_id"] for item in filtered_rerank(ranking, 5)] == [
        "early-b",
        "early-a",
    ]
