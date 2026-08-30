from app.core.progress_status import ProgressStatusStore


def test_progress_status_sequence_and_public_activity() -> None:
    store = ProgressStatusStore(max_entries=2)

    store.start(7)
    initial = store.get(7)
    assert initial is not None
    assert initial["sequence"] == 1
    assert initial["state"] == "active"

    store.update(
        7,
        "rag_retrieval",
        "正在检索题库",
        "RAG 检索 Agent",
        "查找与题面相关的候选内容",
    )
    progress = store.get(7)
    assert progress is not None
    assert progress["sequence"] == 2
    assert progress["phase"] == "rag_retrieval"
    assert progress["agent"] == "RAG 检索 Agent"
    assert progress["message"] == "正在检索题库"

    store.complete(7)
    completed = store.get(7)
    assert completed is not None
    assert completed["sequence"] == 3
    assert completed["state"] == "completed"


def test_progress_status_store_evicts_old_sessions() -> None:
    store = ProgressStatusStore(max_entries=2)
    store.start(1)
    store.start(2)
    store.start(3)

    assert store.get(1) is None
    assert store.get(2) is not None
    assert store.get(3) is not None


def test_old_request_cannot_overwrite_new_generation() -> None:
    store = ProgressStatusStore()
    old_generation = store.start(7)
    new_generation = store.start(7)

    store.update(
        7,
        "old_work",
        "旧请求仍在收尾",
        generation=old_generation,
    )
    store.complete(7, old_generation)

    current = store.get(7)
    assert current is not None
    assert current["generation"] == new_generation
    assert current["phase"] == "requesting"
    assert current["state"] == "active"
