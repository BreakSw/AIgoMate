import asyncio
import json
from datetime import datetime

from app.core.learning_profile import LearningProfileService
from app.core.memory_store import DynamicSystemPromptBuilder
from app.models import (
    DeliverySpec,
    MemorySnapshot,
    RoutingPlan,
    TaskSpec,
)


def task_spec(request: str, intent: str = "review_planning") -> TaskSpec:
    return TaskSpec(
        primary_intent=intent,
        normalized_request=request,
        user_goal=request,
        recognition_summary=request,
        response_mode="direct_answer",
        delivery=DeliverySpec(assistance_level="explanation_only"),
        routing=RoutingPlan(primary_capability="review_planning"),
        confidence=0.95,
    )


def test_correct_answer_updates_bkt_irt_and_fsrs(tmp_path) -> None:
    service = LearningProfileService(tmp_path, "profiles")
    message = "我刚刚独立做对了一道中等难度的二分查找题，没有看提示。"

    snapshot = asyncio.run(service.process_turn(1, 10, message, task_spec(message)))

    assert snapshot.active is True
    assert snapshot.updated is True
    assert snapshot.updates[0].outcome == "correct"
    assert snapshot.updates[0].mastery_after > snapshot.updates[0].mastery_before
    assert snapshot.updates[0].ability_after > snapshot.updates[0].ability_before
    assert snapshot.updates[0].fsrs_rating == "Good"
    assert datetime.fromisoformat(snapshot.updates[0].next_review_at) > datetime.now().astimezone()
    assert snapshot.concepts[0].concept == "二分查找"


def test_incorrect_answer_lowers_mastery_and_ability(tmp_path) -> None:
    service = LearningProfileService(tmp_path, "profiles")
    correct = "我独立做对了一道中等动态规划题。"
    wrong = "我做动态规划的打家劫舍没做出来。"
    asyncio.run(service.process_turn(2, 10, correct, task_spec(correct)))

    snapshot = asyncio.run(service.process_turn(2, 11, wrong, task_spec(wrong)))

    update = snapshot.updates[0]
    assert update.outcome == "incorrect"
    assert update.mastery_after < update.mastery_before
    assert update.ability_after < update.ability_before
    assert update.fsrs_rating == "Again"


def test_hint_is_recorded_without_being_treated_as_success(tmp_path) -> None:
    service = LearningProfileService(tmp_path, "profiles")
    message = "这道滑动窗口题我不会，请先只给我提示。"

    snapshot = asyncio.run(service.process_turn(3, 20, message, task_spec(message)))

    assert snapshot.observations[0].outcome == "hinted"
    assert snapshot.concepts[0].hint_count == 1
    assert snapshot.concepts[0].correct_attempts == 0
    assert snapshot.updates[0].fsrs_rating == "Hard"


def test_ordinary_chat_does_not_read_or_create_profile(tmp_path) -> None:
    service = LearningProfileService(tmp_path, "profiles")
    message = "解释一下二分查找为什么是 O(log n)。"

    snapshot = asyncio.run(service.process_turn(4, 30, message, task_spec(message)))

    assert snapshot.active is False
    assert not (tmp_path / "profiles" / "user-4.json").exists()
    prompt = DynamicSystemPromptBuilder().build(
        4,
        30,
        MemorySnapshot(),
        [],
        snapshot,
    )
    assert "BKT=" not in prompt
    assert "IRT-1PL" not in prompt


def test_recommendation_reads_cross_session_profile_without_mutating_it(tmp_path) -> None:
    service = LearningProfileService(tmp_path, "profiles")
    learning = "我做动态规划题没做出来。"
    asyncio.run(service.process_turn(5, 40, learning, task_spec(learning)))
    profile_path = tmp_path / "profiles" / "user-5.json"
    before = profile_path.read_text(encoding="utf-8")
    recommendation = "根据我的学习画像推荐下一道题，并说明理由。"

    snapshot = asyncio.run(
        service.process_turn(5, 41, recommendation, task_spec(recommendation))
    )

    assert snapshot.active is True
    assert snapshot.updated is False
    assert snapshot.recommended_concepts[0] == "动态规划"
    assert profile_path.read_text(encoding="utf-8") == before
    prompt = DynamicSystemPromptBuilder().build(
        5,
        41,
        MemorySnapshot(),
        [],
        snapshot,
    )
    assert "BKT=" in prompt
    assert "不得编造题号" in prompt


def test_profile_file_contains_structured_evidence_not_raw_messages(tmp_path) -> None:
    service = LearningProfileService(tmp_path, "profiles")
    message = "我独立做对了一道简单哈希表题，备注 SECRET_SENTENCE。"
    asyncio.run(service.process_turn(6, 50, message, task_spec(message)))

    payload = json.loads(
        (tmp_path / "profiles" / "user-6.json").read_text(encoding="utf-8")
    )

    assert "SECRET_SENTENCE" not in json.dumps(payload, ensure_ascii=False)
    assert set(payload) == {"schema_version", "user_id", "ability_theta", "concepts", "updated_at"}


def test_repeated_success_increases_fsrs_interval(tmp_path) -> None:
    service = LearningProfileService(tmp_path, "profiles")
    message = "我又独立做对了一道中等难度的二分查找题。"

    first = asyncio.run(service.process_turn(7, 60, message, task_spec(message)))
    second = asyncio.run(service.process_turn(7, 61, message, task_spec(message)))

    first_stability = first.concepts[0].fsrs_stability_days
    second_stability = second.concepts[0].fsrs_stability_days
    assert second_stability > first_stability
    assert second.updates[0].mastery_after > first.updates[0].mastery_after


def test_requesting_multiple_problems_activates_recommendation_profile(tmp_path) -> None:
    service = LearningProfileService(tmp_path, "profiles")
    message = "帮我出两道难题，有关动态规划的。"

    snapshot = asyncio.run(service.process_turn(8, 70, message, task_spec(message)))

    assert snapshot.active is True
    assert snapshot.updated is False
    assert snapshot.concepts[0].concept == "动态规划"
