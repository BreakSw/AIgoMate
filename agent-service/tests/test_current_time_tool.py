from datetime import datetime, timezone

from app.core.current_time_tool import CurrentTimeTool


def test_current_time_tool_returns_configured_local_time() -> None:
    tool = CurrentTimeTool(
        "Asia/Shanghai",
        clock=lambda: datetime(2026, 8, 30, 2, 15, 16, tzinfo=timezone.utc),
    )

    result = tool.read()

    assert result == {
        "tool_name": "get_current_time",
        "local_datetime": "2026-08-30T10:15:16+08:00",
        "local_date": "2026-08-30",
        "chinese_date": "2026年8月30日",
        "weekday": "Sunday",
        "timezone": "Asia/Shanghai",
        "utc_offset": "+0800",
    }
    assert tool.current_date().isoformat() == "2026-08-30"
