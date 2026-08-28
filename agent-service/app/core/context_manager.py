import math
import re

from app.config import Settings
from app.core.context_compressor import ContextCompressionAgent
from app.core.model_client import RetryCallback
from app.models import (
    CompressedContext,
    ContextSnapshot,
    ContextWindowStatus,
    ConversationMessage,
    MemorySnapshot,
    TaskSpec,
    TurnContext,
)


class TokenEstimator:
    CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

    def estimate_text(self, text: str) -> int:
        cjk_count = len(self.CJK_PATTERN.findall(text))
        non_cjk_count = max(0, len(text) - cjk_count)
        return max(1, cjk_count + math.ceil(non_cjk_count / 4))

    def estimate_message(self, message: ConversationMessage) -> int:
        return self.estimate_text(message.content) + 4

    def estimate_messages(self, messages: list[ConversationMessage]) -> int:
        return sum(self.estimate_message(message) for message in messages)


class ContextManager:
    """Append turns until the active context no longer fits, then compact once."""

    SYSTEM_PROMPT_RESERVE_TOKENS = 3_500
    CONTEXT_AUDIT_MARKER = "========== 上下文审查 =========="

    def __init__(
        self,
        settings: Settings,
        compressor: ContextCompressionAgent,
        estimator: TokenEstimator | None = None,
    ) -> None:
        self.settings = settings
        self.compressor = compressor
        self.estimator = estimator or TokenEstimator()

    async def prepare(
        self,
        current_message: str,
        history: list[ConversationMessage],
        previous_snapshot: ContextSnapshot | None = None,
        on_retry: RetryCallback | None = None,
    ) -> tuple[list[ConversationMessage], ContextSnapshot]:
        """Prepare enough context to recognize the current turn's intent."""

        clean_history = self._without_audit_metadata(history)
        raw_history_tokens = self.estimator.estimate_messages(clean_history)
        current_input_tokens = self.estimator.estimate_text(current_message)
        active, memory, checkpoint_memory, compressed, checkpoint_count = self._active_from_checkpoint(
            clean_history,
            previous_snapshot,
        )
        candidate_input = self._estimate_input(active, current_input_tokens)
        safe_budget = self._safe_input_budget()
        triggered = bool(active) and candidate_input > safe_budget

        if triggered:
            checkpoint_memory, compressed = await self.compressor.compress(
                active,
                on_retry,
                source_message_count=len(clean_history),
            )
            memory = checkpoint_memory
            prepared = [self.compressor.to_context_message(checkpoint_memory, compressed)]
            checkpoint_count = len(clean_history)
            reason = "preflight_budget_exceeded"
            reused = False
        else:
            prepared = active
            reused = checkpoint_count > 0
            reason = "reused_checkpoint" if reused else "not_required"

        estimated_input = self._estimate_input(prepared, current_input_tokens)
        window = self._window_status(
            raw_history_tokens=raw_history_tokens,
            current_input_tokens=current_input_tokens,
            context=prepared,
            candidate_input_tokens=candidate_input,
            turn_metadata_tokens=0,
            compression_triggered=triggered,
            compression_reused=reused,
            compression_trigger_reason=reason,
            messages_before_compression=len(active),
            checkpoint_message_count=checkpoint_count,
            total_committed_messages=len(clean_history),
            estimated_input_tokens=estimated_input,
        )
        return prepared, ContextSnapshot(
            memory=memory,
            compressed_context=compressed,
            window=window,
            checkpoint_memory=checkpoint_memory,
        )

    async def finalize_turn(
        self,
        current_message: str,
        history: list[ConversationMessage],
        prepared_context: list[ConversationMessage],
        preliminary_snapshot: ContextSnapshot,
        task_spec: TaskSpec,
        intent_provider: str,
        on_retry: RetryCallback | None = None,
    ) -> ContextSnapshot:
        """Insert this turn's TaskSpec, then compact once only if it will not fit."""

        clean_history = self._without_audit_metadata(history)
        turn = TurnContext(
            primary_intent=task_spec.primary_intent,
            normalized_request=task_spec.normalized_request,
            user_goal=task_spec.user_goal,
            constraints=task_spec.constraints,
            response_mode=task_spec.response_mode,
            primary_capability=task_spec.routing.primary_capability,
            success_criteria=task_spec.success_criteria,
            intent_model=self.settings.model,
            intent_provider=intent_provider,
        )
        committed_turn = [
            ConversationMessage(role="user", content=current_message),
            self.compressor.to_turn_context_message(turn),
        ]
        turn_metadata_tokens = self.estimator.estimate_messages(committed_turn)
        task_metadata_tokens = self.estimator.estimate_message(committed_turn[1])
        candidate = [*prepared_context, *committed_turn]
        candidate_input = self._estimate_input(candidate, 0)
        safe_budget = self._safe_input_budget()
        already_compacted = preliminary_snapshot.window.compression_triggered
        commit_compaction = bool(candidate) and candidate_input > safe_budget and not already_compacted

        memory = self._merge_active_memory(preliminary_snapshot.memory, task_spec)
        checkpoint_memory = preliminary_snapshot.checkpoint_memory
        compressed = preliminary_snapshot.compressed_context
        triggered = already_compacted
        reused = preliminary_snapshot.window.compression_reused
        reason = preliminary_snapshot.window.compression_trigger_reason
        checkpoint_count = preliminary_snapshot.window.checkpoint_message_count
        messages_before = len(candidate)

        if commit_compaction:
            checkpoint_memory, compressed = await self.compressor.compress(
                candidate,
                on_retry,
                source_message_count=len(clean_history) + 2,
            )
            memory = self._merge_active_memory(checkpoint_memory, task_spec)
            final_context = [self.compressor.to_context_message(checkpoint_memory, compressed)]
            triggered = True
            reused = False
            reason = "turn_commit_budget_exceeded"
            checkpoint_count = len(clean_history) + 2
        else:
            final_context = candidate

        final_context = self._enforce_hard_limit(final_context)
        estimated_input = self._estimate_input(final_context, 0)
        raw_history_tokens = self.estimator.estimate_messages(clean_history) + turn_metadata_tokens
        pre_compaction_candidate = max(
            candidate_input,
            preliminary_snapshot.window.candidate_input_tokens + task_metadata_tokens,
        )
        window = self._window_status(
            raw_history_tokens=raw_history_tokens,
            current_input_tokens=self.estimator.estimate_text(current_message),
            context=final_context,
            candidate_input_tokens=pre_compaction_candidate,
            turn_metadata_tokens=turn_metadata_tokens,
            compression_triggered=triggered,
            compression_reused=(checkpoint_count > 0 and not triggered) or reused,
            compression_trigger_reason=reason,
            messages_before_compression=messages_before,
            checkpoint_message_count=checkpoint_count,
            total_committed_messages=len(clean_history) + 2,
            estimated_input_tokens=estimated_input,
        )
        return ContextSnapshot(
            memory=memory,
            compressed_context=compressed,
            window=window,
            turn_context=turn,
            checkpoint_memory=checkpoint_memory,
        )

    def _active_from_checkpoint(
        self,
        history: list[ConversationMessage],
        previous_snapshot: ContextSnapshot | None,
    ) -> tuple[
        list[ConversationMessage],
        MemorySnapshot,
        MemorySnapshot | None,
        CompressedContext,
        int,
    ]:
        if previous_snapshot is not None:
            checkpoint = previous_snapshot.compressed_context.source_message_count
            provider = previous_snapshot.compressed_context.compression_provider
            if checkpoint > 0 and checkpoint <= len(history) and provider not in {None, "not-required"}:
                checkpoint_memory = previous_snapshot.checkpoint_memory or previous_snapshot.memory
                prefix = self.compressor.to_context_message(
                    checkpoint_memory,
                    previous_snapshot.compressed_context,
                )
                return (
                    [prefix, *history[checkpoint:]],
                    previous_snapshot.memory,
                    checkpoint_memory,
                    previous_snapshot.compressed_context,
                    checkpoint,
                )
            _, compressed = self.compressor.empty_snapshot()
            return list(history), previous_snapshot.memory, None, compressed, 0
        memory, compressed = self.compressor.empty_snapshot()
        return list(history), memory, None, compressed, 0

    def _merge_active_memory(
        self,
        previous: MemorySnapshot,
        task_spec: TaskSpec,
    ) -> MemorySnapshot:
        current_work = [
            f"本轮意图：{task_spec.primary_intent}",
            task_spec.recognition_summary,
            f"当前请求：{task_spec.normalized_request}",
        ]
        open_questions = list(task_spec.ambiguities)
        if task_spec.clarifying_question:
            open_questions.append(task_spec.clarifying_question)
        return MemorySnapshot(
            current_goal=task_spec.user_goal,
            working_memory=self._merge_unique(previous.working_memory, current_work, limit=8),
            long_term_memory=previous.long_term_memory,
            user_preferences=previous.user_preferences,
            pinned_constraints=self._merge_unique(
                previous.pinned_constraints,
                task_spec.constraints,
                limit=12,
            ),
            open_questions=self._merge_unique(previous.open_questions, open_questions, limit=8),
            resolved_items=previous.resolved_items,
        )

    @staticmethod
    def _merge_unique(*groups: list[str], limit: int) -> list[str]:
        result: list[str] = []
        for group in groups:
            for item in group:
                if item and item not in result:
                    result.append(item)
        return result[-limit:]

    def _without_audit_metadata(
        self,
        history: list[ConversationMessage],
    ) -> list[ConversationMessage]:
        cleaned: list[ConversationMessage] = []
        for message in history:
            content = message.content
            if message.role == "assistant" and self.CONTEXT_AUDIT_MARKER in content:
                content = content.split(self.CONTEXT_AUDIT_MARKER, maxsplit=1)[0].rstrip()
            cleaned.append(message.model_copy(update={"content": content}))
        return cleaned

    def _safe_input_budget(self) -> int:
        return min(
            self.settings.context_soft_limit_tokens,
            self.settings.context_window_tokens - self.settings.context_output_reserve_tokens,
        )

    def _estimate_input(
        self,
        messages: list[ConversationMessage],
        current_input_tokens: int,
    ) -> int:
        return self.SYSTEM_PROMPT_RESERVE_TOKENS + current_input_tokens + self.estimator.estimate_messages(messages)

    def _enforce_hard_limit(self, messages: list[ConversationMessage]) -> list[ConversationMessage]:
        result = list(messages)
        while len(result) > 1 and self._estimate_input(result, 0) > self.settings.context_hard_limit_tokens:
            removable = next(
                (index for index, item in enumerate(result) if item.role != "system"),
                None,
            )
            if removable is None:
                break
            result.pop(removable)
        return result

    def _window_status(
        self,
        *,
        raw_history_tokens: int,
        current_input_tokens: int,
        context: list[ConversationMessage],
        candidate_input_tokens: int,
        turn_metadata_tokens: int,
        compression_triggered: bool,
        compression_reused: bool,
        compression_trigger_reason: str,
        messages_before_compression: int,
        checkpoint_message_count: int,
        total_committed_messages: int,
        estimated_input_tokens: int,
    ) -> ContextWindowStatus:
        compressed_tokens = (
            self.estimator.estimate_message(context[0])
            if checkpoint_message_count > 0 and context and context[0].role == "system"
            else 0
        )
        appended_tokens = self.estimator.estimate_messages(context[1:] if compressed_tokens else context)
        safe_budget = self._safe_input_budget()
        return ContextWindowStatus(
            window_size_tokens=self.settings.context_window_tokens,
            soft_limit_tokens=self.settings.context_soft_limit_tokens,
            hard_limit_tokens=self.settings.context_hard_limit_tokens,
            output_reserved_tokens=self.settings.context_output_reserve_tokens,
            raw_history_tokens=raw_history_tokens,
            current_input_tokens=current_input_tokens,
            compressed_context_tokens=compressed_tokens,
            recent_messages_tokens=appended_tokens,
            estimated_input_tokens=estimated_input_tokens,
            remaining_tokens=max(0, self.settings.context_window_tokens - estimated_input_tokens),
            usage_ratio=round(estimated_input_tokens / self.settings.context_window_tokens, 4),
            state=self._state_for(estimated_input_tokens),
            compression_triggered=compression_triggered,
            messages_before_compression=messages_before_compression,
            messages_after_compression=len(context),
            candidate_input_tokens=candidate_input_tokens,
            safe_input_budget_tokens=safe_budget,
            safe_remaining_tokens=max(0, safe_budget - estimated_input_tokens),
            turn_metadata_tokens=turn_metadata_tokens,
            compression_reused=compression_reused,
            compression_trigger_reason=compression_trigger_reason,
            checkpoint_message_count=checkpoint_message_count,
            new_messages_since_checkpoint=max(0, total_committed_messages - checkpoint_message_count),
        )

    def _state_for(self, used_tokens: int) -> str:
        if used_tokens > self.settings.context_window_tokens:
            return "overflow"
        if used_tokens >= self.settings.context_hard_limit_tokens:
            return "hard_limit"
        if used_tokens >= self._safe_input_budget():
            return "soft_limit"
        return "normal"
