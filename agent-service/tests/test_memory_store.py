import asyncio

from app.core.memory_store import DynamicSystemPromptBuilder, UserMemoryRepository
from app.models import AgentRequest, ContextSnapshot, MemoryScope, MemorySnapshot, MemoryUpdate


def test_durable_memory_is_isolated_between_sessions(tmp_path) -> None:
    repository = UserMemoryRepository(tmp_path, "memory")
    first = MemoryUpdate(
        kind="long_term_goal",
        content="学习动态规划",
        importance=0.9,
        reason="当前会话学习目标",
    )
    second = MemoryUpdate(
        kind="long_term_goal",
        content="学习图论",
        importance=0.9,
        reason="另一个会话学习目标",
    )

    asyncio.run(repository.upsert(1, 101, [first]))
    asyncio.run(repository.upsert(1, 202, [second]))

    first_session = asyncio.run(repository.load(1, 101))
    second_session = asyncio.run(repository.load(1, 202))

    assert [item.content for item in first_session] == ["学习动态规划"]
    assert [item.content for item in second_session] == ["学习图论"]
    assert first_session[0].memory_id != second_session[0].memory_id
    assert (tmp_path / "memory/user-1/session-101.json").is_file()
    assert (tmp_path / "memory/user-1/session-202.json").is_file()


def test_legacy_user_level_memory_is_not_loaded_into_a_session(tmp_path) -> None:
    memory_root = tmp_path / "memory"
    memory_root.mkdir()
    (memory_root / "user-1.json").write_text(
        '{"schema_version":"1.0","user_id":1,"memories":[]}',
        encoding="utf-8",
    )
    repository = UserMemoryRepository(tmp_path, "memory")

    assert asyncio.run(repository.load(1, 303)) == []


def test_dynamic_prompt_identifies_session_scope() -> None:
    prompt = DynamicSystemPromptBuilder().build(
        user_id=1,
        session_id=99,
        snapshot=MemorySnapshot(),
        durable_memory=[],
    )

    assert "会话标识：99" in prompt
    assert "记忆作用域：仅限当前会话" in prompt


def test_snapshot_scope_must_match_current_conversation() -> None:
    from app.core.orchestrator import AgentOrchestrator

    snapshot = ContextSnapshot.model_construct(
        memory_scope=MemoryScope(user_id=1, session_id=10)
    )
    matching = AgentRequest(
        user_id=1,
        session_id=10,
        message="继续",
        previous_context_snapshot=snapshot,
    )
    different_session = AgentRequest(
        user_id=1,
        session_id=11,
        message="新的对话",
        previous_context_snapshot=snapshot,
    )
    legacy_snapshot = ContextSnapshot.model_construct(memory_scope=None)
    legacy = AgentRequest(
        user_id=1,
        session_id=10,
        message="继续",
        previous_context_snapshot=legacy_snapshot,
    )

    assert AgentOrchestrator._scoped_previous_snapshot(matching) is snapshot
    assert AgentOrchestrator._scoped_previous_snapshot(different_session) is None
    assert AgentOrchestrator._scoped_previous_snapshot(legacy) is None


def test_first_turn_reset_prevents_reused_session_id_memory_leak(tmp_path) -> None:
    import asyncio

    from app.core.memory_store import UserMemoryRepository
    from app.models import MemoryUpdate

    async def scenario() -> None:
        repository = UserMemoryRepository(tmp_path, "memory")
        await repository.upsert(
            1,
            7,
            [MemoryUpdate(
                kind="long_term_goal",
                content="旧数据库中的动态规划学习目标",
                importance=0.9,
                reason="模拟旧会话残留",
            )],
        )

        await repository.reset_session(1, 7)

        assert await repository.load(1, 7) == []

    asyncio.run(scenario())
