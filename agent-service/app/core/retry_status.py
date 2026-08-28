from collections import OrderedDict
from datetime import UTC, datetime
from typing import Literal, TypedDict


RetryPhase = Literal["requesting", "retrying", "completed", "failed"]


class RetryStatus(TypedDict):
    phase: RetryPhase
    retry_count: int
    max_retries: int
    retry_delay_seconds: float | None
    updated_at: str


class RetryStatusStore:
    def __init__(self, max_retries: int, max_entries: int = 500) -> None:
        self.max_retries = max_retries
        self.max_entries = max_entries
        self._states: OrderedDict[int, RetryStatus] = OrderedDict()

    def start(self, session_id: int) -> None:
        self._set(session_id, "requesting", 0, None)

    def retry(self, session_id: int, retry_count: int, max_retries: int, delay: float) -> None:
        self._set(session_id, "retrying", retry_count, delay, max_retries)

    def complete(self, session_id: int) -> None:
        current = self._states.get(session_id)
        retry_count = current["retry_count"] if current else 0
        self._set(session_id, "completed", retry_count, None)

    def fail(self, session_id: int) -> None:
        current = self._states.get(session_id)
        retry_count = current["retry_count"] if current else 0
        self._set(session_id, "failed", retry_count, None)

    def get(self, session_id: int) -> RetryStatus | None:
        state = self._states.get(session_id)
        return dict(state) if state else None

    def _set(
        self,
        session_id: int,
        phase: RetryPhase,
        retry_count: int,
        retry_delay_seconds: float | None,
        max_retries: int | None = None,
    ) -> None:
        self._states[session_id] = RetryStatus(
            phase=phase,
            retry_count=retry_count,
            max_retries=max_retries if max_retries is not None else self.max_retries,
            retry_delay_seconds=retry_delay_seconds,
            updated_at=datetime.now(UTC).isoformat(),
        )
        self._states.move_to_end(session_id)
        while len(self._states) > self.max_entries:
            self._states.popitem(last=False)
