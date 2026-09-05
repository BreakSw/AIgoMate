import json
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.core.model_client import IntentModelClient, RetryCallback
from app.core.structured_output import parse_model_json


ModelT = TypeVar("ModelT", bound=BaseModel)
Validator = Callable[[ModelT], None]


class AgentProtocolExhaustedError(ValueError):
    def __init__(
        self,
        agent_name: str,
        reflection_rounds: int,
        provider: str = "unknown-provider",
        validation_feedback: str | None = None,
    ) -> None:
        super().__init__(
            f"{agent_name} 连续 {reflection_rounds} 轮自我修正后仍未通过输出协议"
        )
        self.agent_name = agent_name
        self.reflection_rounds = reflection_rounds
        self.provider = provider
        self.validation_feedback = validation_feedback


async def complete_with_reflection(
    *,
    model_client: IntentModelClient,
    agent_name: str,
    system_prompt: str,
    request_payload: dict,
    model_type: type[ModelT],
    on_retry: RetryCallback | None,
    max_tokens: int,
    max_reflection_rounds: int = 10,
    validator: Validator[ModelT] | None = None,
) -> tuple[ModelT, str, int]:
    """Run an agent, validate its protocol, then reflect and revise if needed."""

    raw, provider = await model_client.complete_json(
        system_prompt,
        json.dumps(request_payload, ensure_ascii=False),
        on_retry,
        max_tokens=max_tokens,
    )
    for reflection_round in range(max_reflection_rounds + 1):
        try:
            result = parse_model_json(raw, model_type)
            if validator is not None:
                validator(result)
            reflected_provider = (
                provider
                if reflection_round == 0
                else f"{provider}+reflection:{reflection_round}"
            )
            return result, reflected_provider, reflection_round
        except (json.JSONDecodeError, ValidationError, ValueError, KeyError, TypeError) as error:
            if reflection_round >= max_reflection_rounds:
                raise AgentProtocolExhaustedError(
                    agent_name,
                    max_reflection_rounds,
                    provider,
                    _safe_validation_feedback(error),
                ) from error
            reflection_payload = {
                "original_request": request_payload,
                "previous_invalid_output": raw[:20_000],
                "validation_feedback": _safe_validation_feedback(error),
                "reflection_round": reflection_round + 1,
                "reflection_instruction": (
                    "先检查上一版是否履行本 Agent 的职责以及每个字段的类型、枚举和必填约束；"
                    "再针对反馈修正。只输出修正后的完整 JSON，不解释反思过程。"
                ),
            }
            raw, provider = await model_client.complete_json(
                system_prompt
                + "\n\n你上一版输出未通过协议。执行 Reflection：自检角色、事实边界和字段约束后，重写完整 JSON。",
                json.dumps(reflection_payload, ensure_ascii=False),
                on_retry,
                max_tokens=max_tokens,
            )

    raise AgentProtocolExhaustedError(agent_name, max_reflection_rounds, provider)


def _safe_validation_feedback(error: Exception) -> str:
    if isinstance(error, ValidationError):
        compact = []
        for item in error.errors(include_url=False)[:12]:
            location = ".".join(str(value) for value in item.get("loc", []))
            compact.append(f"{location or 'root'}: {item.get('msg', 'invalid value')}")
        return "；".join(compact)[:2_000]
    return str(error).replace("\n", " ")[:2_000]
