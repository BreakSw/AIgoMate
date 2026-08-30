from collections.abc import Callable
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo


class CurrentTimeTool:
    """Return the current wall-clock time in the application's configured timezone."""

    name = "get_current_time"
    description = "读取当前日期、时间、星期和时区，用于解析今天、当前、最新等相对时间。"

    def __init__(
        self,
        timezone_name: str = "Asia/Shanghai",
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.timezone_name = timezone_name
        self._timezone = ZoneInfo(timezone_name)
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def read(self) -> dict[str, str]:
        current = self._local_now()
        return {
            "tool_name": self.name,
            "local_datetime": current.isoformat(timespec="seconds"),
            "local_date": current.date().isoformat(),
            "chinese_date": f"{current.year}年{current.month}月{current.day}日",
            "weekday": current.strftime("%A"),
            "timezone": self.timezone_name,
            "utc_offset": current.strftime("%z"),
        }

    def current_date(self) -> date:
        return self._local_now().date()

    def _local_now(self) -> datetime:
        current = self._clock()
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        return current.astimezone(self._timezone)
