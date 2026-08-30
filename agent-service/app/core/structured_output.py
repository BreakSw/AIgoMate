import json
import re
from typing import TypeVar

from pydantic import BaseModel


ModelT = TypeVar("ModelT", bound=BaseModel)


def parse_model_json(raw_response: str, model_type: type[ModelT]) -> ModelT:
    """Extract one JSON object and validate it against an agent protocol model."""

    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        raw_response.strip(),
        flags=re.IGNORECASE,
    )
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("模型没有返回有效的 JSON 对象")
    parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("智能体协议输出必须是 JSON 对象")
    return model_type.model_validate(parsed)
