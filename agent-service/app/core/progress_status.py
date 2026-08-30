from collections import OrderedDict
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from threading import Lock
from typing import Literal, TypedDict


ProgressState = Literal["active", "completed", "failed"]
ProgressCallback = Callable[
    [str, str, str | None, str | None],
    Awaitable[None],
]


class ProgressStatus(TypedDict):
    generation: int
    sequence: int
    phase: str
    agent: str | None
    message: str
    detail: str | None
    state: ProgressState
    updated_at: str


class ProgressStatusStore:
    """Keep the latest public, user-facing execution activity per session."""

    def __init__(self, max_entries: int = 500) -> None:
        self.max_entries = max_entries
        self._states: OrderedDict[int, ProgressStatus] = OrderedDict()
        self._generations: dict[int, int] = {}
        self._lock = Lock()

    def start(self, session_id: int) -> int:
        with self._lock:
            generation = self._generations.get(session_id, 0) + 1
            self._generations[session_id] = generation
            self._set(
                session_id,
                generation=generation,
                sequence=1,
                phase="requesting",
                agent="系统",
                message="正在接收本轮任务",
                detail="准备输入、上下文与会话记忆",
                state="active",
            )
            return generation

    def update(
        self,
        session_id: int,
        phase: str,
        message: str,
        agent: str | None = None,
        detail: str | None = None,
        generation: int | None = None,
    ) -> None:
        with self._lock:
            if generation is not None and generation != self._generations.get(session_id):
                return
            current = self._states.get(session_id)
            sequence = int(current["sequence"]) + 1 if current else 1
            self._set(
                session_id,
                generation=self._generations.get(session_id, generation or 1),
                sequence=sequence,
                phase=phase,
                agent=agent,
                message=message,
                detail=detail,
                state="active",
            )

    def complete(self, session_id: int, generation: int | None = None) -> None:
        self._finish(
            session_id,
            "completed",
            "回答已经生成",
            "正在写入本轮会话",
            generation,
        )

    def fail(self, session_id: int, generation: int | None = None) -> None:
        self._finish(
            session_id,
            "failed",
            "本轮执行未完成",
            "请查看页面上的错误提示",
            generation,
        )

    def get(self, session_id: int) -> ProgressStatus | None:
        with self._lock:
            state = self._states.get(session_id)
            return dict(state) if state else None

    def _finish(
        self,
        session_id: int,
        state: ProgressState,
        message: str,
        detail: str,
        generation: int | None,
    ) -> None:
        with self._lock:
            if generation is not None and generation != self._generations.get(session_id):
                return
            current = self._states.get(session_id)
            sequence = int(current["sequence"]) + 1 if current else 1
            self._set(
                session_id,
                generation=self._generations.get(session_id, generation or 1),
                sequence=sequence,
                phase=state,
                agent="系统",
                message=message,
                detail=detail,
                state=state,
            )

    def _set(
        self,
        session_id: int,
        *,
        generation: int,
        sequence: int,
        phase: str,
        agent: str | None,
        message: str,
        detail: str | None,
        state: ProgressState,
    ) -> None:
        self._states[session_id] = ProgressStatus(
            generation=generation,
            sequence=sequence,
            phase=phase,
            agent=agent,
            message=message,
            detail=detail,
            state=state,
            updated_at=datetime.now(UTC).isoformat(),
        )
        self._states.move_to_end(session_id)
        while len(self._states) > self.max_entries:
            evicted_session, _ = self._states.popitem(last=False)
            self._generations.pop(evicted_session, None)
