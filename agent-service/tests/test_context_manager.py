import asyncio

from app.config import Settings
from app.core.context_compressor import ContextCompressionAgent
from app.core.context_manager import ContextManager, TokenEstimator
from app.models import (
    CompressedContext,
    ConversationMessage,
    DeliverySpec,
    MemorySnapshot,
    RoutingPlan,
    TaskSpec,
)


def make_settings(**updates) -> Settings:
    values = {
        "url": "https://model.example.test",
        "model": "deepseek-v4-pro",
        "api-key": "test-key",
        "context_window_tokens": 32_768,
        "context_soft_limit_tokens": 24_576,
        "context_hard_limit_tokens": 28_672,
        "context_output_reserve_tokens": 8_192,
        "context_recent_messages": 8,
        "context_recent_token_budget": 6_144,
        "context_compress_every_request": False,
    }
    values.update(updates)
    return Settings(_env_file=None, **values)


def make_task_spec(normalized_request: str = "继续使用 Python 获取提示") -> TaskSpec:
    return TaskSpec(
        primary_intent="guided_hint",
        normalized_request=normalized_request,
        user_goal="在不获取完整答案的情况下继续解题",
        recognition_summary="用户希望继续获得提示",
        constraints=["使用 Python", "只给提示"],
        response_mode="progressive_hint",
        delivery=DeliverySpec(assistance_level="hint_only", include_code=False),
        routing=RoutingPlan(primary_capability="algorithm_tutoring"),
        success_criteria=["只提供下一步提示"],
        confidence=0.95,
    )


class FakeCompressor:
    def __init__(self) -> None:
        self.calls = 0
        self.inputs: list[list[ConversationMessage]] = []

    async def compress(self, history, on_retry=None, source_message_count=None):
        self.calls += 1
        self.inputs.append(list(history))
        return (
            MemorySnapshot(
                current_goal="完成两数之和",
                working_memory=["已经确定使用哈希表"],
                pinned_constraints=["只给提示"],
            ),
            CompressedContext(
                summary="用户正在用 Python 解决两数之和，要求只给提示。",
                topics=["哈希表"],
                source_message_count=source_message_count or len(history),
                compression_model="deepseek-v4-pro",
                compression_provider="fake-provider",
            ),
        )

    def empty_snapshot(self):
        return MemorySnapshot(), CompressedContext(
            summary="当前上下文仍可直接装入窗口，尚未触发统一压缩。",
            compression_provider="not-required",
        )

    def to_context_message(self, memory, compressed):
        return ConversationMessage(
            role="system",
            content=f"压缩目标：{memory.current_goal}；摘要：{compressed.summary}",
        )

    def to_turn_context_message(self, turn):
        return ConversationMessage(
            role="assistant",
            content=f"本轮意图：{turn.primary_intent}；{turn.normalized_request}",
        )


def test_token_estimator_counts_cjk_and_latin_conservatively() -> None:
    estimator = TokenEstimator()

    assert estimator.estimate_text("算法") == 2
    assert estimator.estimate_text("abcdefgh") == 2
    assert estimator.estimate_message(ConversationMessage(role="user", content="算法")) == 6


def test_small_turn_inserts_intent_without_calling_compression_model() -> None:
    compressor = FakeCompressor()
    manager = ContextManager(make_settings(), compressor)

    prepared, preliminary = asyncio.run(manager.prepare("解释二分查找", []))
    snapshot = asyncio.run(manager.finalize_turn(
        "解释二分查找",
        [],
        prepared,
        preliminary,
        make_task_spec("解释二分查找的基本思想"),
        "fake-deepseek",
    ))

    assert prepared == []
    assert compressor.calls == 0
    assert snapshot.turn_context is not None
    assert snapshot.turn_context.primary_intent == "guided_hint"
    assert snapshot.memory.current_goal == "在不获取完整答案的情况下继续解题"
    assert "本轮意图：guided_hint" in snapshot.memory.working_memory
    assert snapshot.memory.pinned_constraints == ["使用 Python", "只给提示"]
    assert snapshot.checkpoint_memory is None
    assert snapshot.window.turn_metadata_tokens > 0
    assert snapshot.window.compression_triggered is False
    assert snapshot.window.compression_trigger_reason == "not_required"


def test_small_history_stays_raw_until_budget_is_insufficient() -> None:
    compressor = FakeCompressor()
    manager = ContextManager(make_settings(), compressor)
    history = [
        ConversationMessage(role="user", content="我在做两数之和，只给提示"),
        ConversationMessage(role="assistant", content="可以先想想如何快速查找补数"),
    ]

    prepared, snapshot = asyncio.run(manager.prepare("继续，使用 Python", history))

    assert compressor.calls == 0
    assert prepared == history
    assert snapshot.compressed_context.compression_provider == "not-required"
    assert snapshot.window.candidate_input_tokens <= snapshot.window.safe_input_budget_tokens


def test_preflight_overflow_compacts_active_history_exactly_once() -> None:
    compressor = FakeCompressor()
    settings = make_settings(
        context_window_tokens=12_000,
        context_soft_limit_tokens=8_000,
        context_hard_limit_tokens=9_000,
        context_output_reserve_tokens=4_000,
    )
    manager = ContextManager(settings, compressor)
    history = [
        ConversationMessage(role="user" if index % 2 == 0 else "assistant", content="x" * 8_000)
        for index in range(4)
    ]

    prepared, preliminary = asyncio.run(manager.prepare("继续", history))
    snapshot = asyncio.run(manager.finalize_turn(
        "继续", history, prepared, preliminary, make_task_spec(), "fake-deepseek"
    ))

    assert compressor.calls == 1
    assert prepared[0].role == "system"
    assert snapshot.window.compression_triggered is True
    assert snapshot.window.compression_trigger_reason == "preflight_budget_exceeded"
    assert snapshot.window.checkpoint_message_count == len(history)
    assert snapshot.window.new_messages_since_checkpoint == 2
    assert snapshot.checkpoint_memory is not None
    assert snapshot.checkpoint_memory.current_goal == "完成两数之和"


def test_task_spec_is_tried_before_commit_compaction() -> None:
    compressor = FakeCompressor()
    settings = make_settings(
        context_window_tokens=12_000,
        context_soft_limit_tokens=8_000,
        context_hard_limit_tokens=9_000,
        context_output_reserve_tokens=4_000,
    )
    manager = ContextManager(settings, compressor)
    history = [
        ConversationMessage(role="user", content="x" * 6_000),
        ConversationMessage(role="assistant", content="y" * 6_000),
    ]

    prepared, preliminary = asyncio.run(manager.prepare("继续", history))
    assert compressor.calls == 0
    snapshot = asyncio.run(manager.finalize_turn(
        "继续",
        history,
        prepared,
        preliminary,
        make_task_spec("z" * 8_000),
        "fake-deepseek",
    ))

    assert compressor.calls == 1
    assert any("本轮意图" in item.content for item in compressor.inputs[0])
    assert snapshot.window.compression_trigger_reason == "turn_commit_budget_exceeded"
    assert snapshot.window.checkpoint_message_count == len(history) + 2
    assert snapshot.window.new_messages_since_checkpoint == 0


def test_next_turn_reuses_checkpoint_without_recompressing_old_messages() -> None:
    compressor = FakeCompressor()
    settings = make_settings(
        context_window_tokens=12_000,
        context_soft_limit_tokens=8_000,
        context_hard_limit_tokens=9_000,
        context_output_reserve_tokens=4_000,
    )
    manager = ContextManager(settings, compressor)
    old_history = [
        ConversationMessage(role="user" if index % 2 == 0 else "assistant", content="x" * 8_000)
        for index in range(4)
    ]
    prepared, preliminary = asyncio.run(manager.prepare("继续", old_history))
    checkpoint = asyncio.run(manager.finalize_turn(
        "继续", old_history, prepared, preliminary, make_task_spec(), "fake-deepseek"
    ))
    full_history = [
        *old_history,
        ConversationMessage(role="user", content="继续"),
        ConversationMessage(role="assistant", content="TaskSpec v1.0\n主要意图：guided_hint"),
    ]

    reused_context, reused_snapshot = asyncio.run(manager.prepare(
        "再给一个提示", full_history, checkpoint
    ))

    assert compressor.calls == 1
    assert reused_context[0].role == "system"
    assert reused_context[1:] == full_history[checkpoint.compressed_context.source_message_count:]
    assert reused_snapshot.window.compression_reused is True
    assert reused_snapshot.window.compression_trigger_reason == "reused_checkpoint"
    assert reused_snapshot.checkpoint_memory == checkpoint.checkpoint_memory


def test_window_remaining_tokens_matches_active_and_safe_budgets() -> None:
    compressor = FakeCompressor()
    settings = make_settings()
    manager = ContextManager(settings, compressor)

    _, snapshot = asyncio.run(manager.prepare("继续讲解", [
        ConversationMessage(role="user", content="解释快速排序"),
    ]))

    assert snapshot.window.remaining_tokens == settings.context_window_tokens - snapshot.window.estimated_input_tokens
    assert snapshot.window.safe_remaining_tokens == (
        snapshot.window.safe_input_budget_tokens - snapshot.window.estimated_input_tokens
    )


def test_generated_context_audit_is_not_reinserted() -> None:
    compressor = FakeCompressor()
    manager = ContextManager(make_settings(), compressor)
    history = [
        ConversationMessage(role="user", content="解释快速排序"),
        ConversationMessage(
            role="assistant",
            content=(
                "TaskSpec v1.0\n主要意图：concept_explanation\n\n"
                "========== 上下文审查 ==========\n"
                "压缩后内容：这段生成元数据不能递归进入下一轮"
            ),
        ),
    ]

    prepared, snapshot = asyncio.run(manager.prepare("继续", history))

    assert all("这段生成元数据" not in message.content for message in prepared)
    assert snapshot.window.raw_history_tokens < TokenEstimator().estimate_messages(history)


def test_compression_agent_falls_back_when_model_returns_invalid_json() -> None:
    class InvalidJsonModelClient:
        async def complete_json(self, system_prompt, user_prompt, on_retry=None, max_tokens=None):
            assert max_tokens == 1800
            return "这不是 JSON", "fake-provider"

    compressor = ContextCompressionAgent(InvalidJsonModelClient(), "deepseek-v4-pro")
    history = [ConversationMessage(role="user", content="请记住后续都用 Python")]

    memory, compressed = asyncio.run(compressor.compress(history))

    assert "确定性摘要" in memory.working_memory[0]
    assert "user: 请记住后续都用 Python" in compressed.summary
    assert compressed.compression_provider == "fake-provider+deterministic-fallback"
