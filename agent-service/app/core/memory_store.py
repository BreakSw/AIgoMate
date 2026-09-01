import asyncio
import hashlib
import json
import re
from datetime import date, datetime, timezone
from pathlib import Path

from app.models import (
    DurableMemoryItem,
    LearningProfileSnapshot,
    MemorySnapshot,
    MemoryUpdate,
)

_WEEKDAY_NAMES = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


def current_date_context() -> str:
    today = date.today()
    return (
        f"当前日期：{today.isoformat()}（{_WEEKDAY_NAMES[today.weekday()]}）。"
        "这是系统注入的唯一可信日期；回答涉及今天、昨天、明天、本周等相对时间时，"
        "必须以此换算，不得凭训练记忆猜测或编造当前日期。"
    )


class UserMemoryRepository:
    """Local durable memory store isolated by both user and chat session."""

    def __init__(self, project_root: Path, configured_dir: str) -> None:
        root = Path(configured_dir)
        self.root = (root if root.is_absolute() else project_root / root).resolve()
        self._lock = asyncio.Lock()

    async def load(
        self,
        user_id: int,
        session_id: int,
    ) -> list[DurableMemoryItem]:
        async with self._lock:
            return self._load_unlocked(user_id, session_id)

    async def reset_session(self, user_id: int, session_id: int) -> None:
        """Start a numerically reused chat session with an empty memory scope."""
        async with self._lock:
            self._write_unlocked(user_id, session_id, [])

    async def recall(
        self,
        user_id: int,
        session_id: int,
        query: str,
        limit: int = 12,
    ) -> list[DurableMemoryItem]:
        items = await self.load(user_id, session_id)
        query_tokens = self._tokens(query)
        ranked = sorted(
            items,
            key=lambda item: (
                self._overlap(query_tokens, self._tokens(item.content)),
                item.importance,
                item.updated_at,
            ),
            reverse=True,
        )
        relevant = [
            item
            for item in ranked
            if self._overlap(query_tokens, self._tokens(item.content)) > 0
            or item.kind in {"preference", "constraint", "long_term_goal"}
        ]
        return relevant[:limit]

    async def upsert(
        self,
        user_id: int,
        session_id: int,
        updates: list[MemoryUpdate],
        source: str = "memory_agent",
    ) -> list[DurableMemoryItem]:
        accepted = [item for item in updates if item.importance >= 0.55 and item.content.strip()]
        if not accepted:
            return await self.load(user_id, session_id)
        async with self._lock:
            existing = self._load_unlocked(user_id, session_id)
            by_key = {
                (item.kind, self._normalize(item.content)): item
                for item in existing
            }
            now = self._now()
            for update in accepted:
                content = update.content.strip()
                key = (update.kind, self._normalize(content))
                current = by_key.get(key)
                if current is not None:
                    by_key[key] = current.model_copy(
                        update={
                            "importance": max(current.importance, update.importance),
                            "updated_at": now,
                        }
                    )
                    continue
                digest = hashlib.sha256(
                    (
                        f"{user_id}:{session_id}:{update.kind}:"
                        f"{self._normalize(content)}"
                    ).encode("utf-8")
                ).hexdigest()[:20]
                by_key[key] = DurableMemoryItem(
                    memory_id=f"mem_{digest}",
                    kind=update.kind,
                    content=content,
                    importance=update.importance,
                    source=source,
                    created_at=now,
                    updated_at=now,
                )
            items = sorted(
                by_key.values(),
                key=lambda item: (item.importance, item.updated_at),
                reverse=True,
            )[:300]
            self._write_unlocked(user_id, session_id, items)
            return items

    async def checkpoint_snapshot(
        self,
        user_id: int,
        session_id: int,
        memory: MemorySnapshot,
    ) -> list[DurableMemoryItem]:
        updates: list[MemoryUpdate] = []
        if memory.current_goal:
            updates.append(MemoryUpdate(
                kind="long_term_goal",
                content=memory.current_goal,
                importance=0.9,
                reason=f"会话 {session_id} 压缩检查点中的当前目标",
            ))
        updates.extend(
            MemoryUpdate(
                kind="preference",
                content=value,
                importance=0.9,
                reason=f"会话 {session_id} 压缩前保存用户偏好",
            )
            for value in memory.user_preferences
        )
        updates.extend(
            MemoryUpdate(
                kind="constraint",
                content=value,
                importance=0.95,
                reason=f"会话 {session_id} 压缩前保存固定约束",
            )
            for value in memory.pinned_constraints
        )
        updates.extend(
            MemoryUpdate(
                kind="unfinished_task",
                content=value,
                importance=0.85,
                reason=f"会话 {session_id} 压缩前保存未完成事项",
            )
            for value in memory.open_questions
        )
        updates.extend(
            MemoryUpdate(
                kind="learned_fact",
                content=value,
                importance=0.8,
                reason=f"会话 {session_id} 压缩检查点中的长期记忆",
            )
            for value in memory.long_term_memory
        )
        return await self.upsert(
            user_id,
            session_id,
            updates,
            source="context_checkpoint",
        )

    def _path(self, user_id: int, session_id: int) -> Path:
        return self.root / f"user-{user_id}" / f"session-{session_id}.json"

    def _load_unlocked(
        self,
        user_id: int,
        session_id: int,
    ) -> list[DurableMemoryItem]:
        path = self._path(user_id, session_id)
        if not path.is_file():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            values = payload.get("memories", []) if isinstance(payload, dict) else []
            return [DurableMemoryItem.model_validate(item) for item in values]
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            # A corrupted memory file must fail closed instead of entering prompts.
            return []

    def _write_unlocked(
        self,
        user_id: int,
        session_id: int,
        items: list[DurableMemoryItem],
    ) -> None:
        path = self._path(user_id, session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        payload = {
            "schema_version": "1.1",
            "user_id": user_id,
            "session_id": session_id,
            "updated_at": self._now(),
            "memories": [item.model_dump() for item in items],
        }
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip().lower()

    @classmethod
    def _tokens(cls, value: str) -> set[str]:
        normalized = cls._normalize(value)
        tokens = set(re.findall(r"[a-z0-9][a-z0-9+#._-]*", normalized))
        for sequence in re.findall(r"[\u3400-\u9fff]+", normalized):
            if len(sequence) == 1:
                tokens.add(sequence)
            else:
                tokens.update(sequence[index : index + 2] for index in range(len(sequence) - 1))
        return tokens

    @staticmethod
    def _overlap(left: set[str], right: set[str]) -> int:
        return len(left & right)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()


class DynamicSystemPromptBuilder:
    def build(
        self,
        user_id: int,
        session_id: int,
        snapshot: MemorySnapshot,
        durable_memory: list[DurableMemoryItem],
        learning_profile: LearningProfileSnapshot | None = None,
    ) -> str:
        lines = [
            "以下是本轮动态系统上下文。它由系统维护，不是用户消息，也不能覆盖安全规则。",
            f"用户标识：{user_id}",
            f"会话标识：{session_id}",
            "记忆作用域：仅限当前会话，不得引用其他会话的目标、偏好或历史。",
            current_date_context(),
            f"当前目标：{snapshot.current_goal or '未形成稳定目标'}",
        ]
        if snapshot.pinned_constraints:
            lines.append("当前固定约束：" + "；".join(snapshot.pinned_constraints))
        if snapshot.open_questions:
            lines.append("未完成事项：" + "；".join(snapshot.open_questions))
        if durable_memory:
            lines.append("按当前任务召回的用户私有记忆：")
            lines.extend(
                f"- [{item.memory_id}] {item.kind}: {item.content} (importance={item.importance:.2f})"
                for item in durable_memory
            )
        else:
            lines.append("本轮没有召回到必要的用户私有记忆。")
        if learning_profile is not None and learning_profile.active:
            lines.extend([
                "本轮已明确触发个性化学习任务，以下画像可用于难度、知识点与复习推荐：",
                f"- IRT-1PL 能力值 theta={learning_profile.ability_theta:.3f}，建议难度={learning_profile.target_difficulty}",
            ])
            lines.extend(
                f"- {item.concept}: BKT={item.mastery_probability:.3f}, "
                f"attempts={item.attempts}, next_review={item.next_review_at or 'unknown'}"
                for item in learning_profile.concepts
            )
            if learning_profile.recommended_concepts:
                lines.append(
                    "画像推荐优先级："
                    + "、".join(learning_profile.recommended_concepts)
                )
            lines.append(
                "推荐题目时应结合画像检索题库；没有实际题库证据时不得编造题号。"
            )
        lines.append("不得编造未记录的用户偏好或历史；记忆冲突时以用户本轮明确表述为准。")
        return "\n".join(lines)
