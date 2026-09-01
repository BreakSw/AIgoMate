import asyncio
import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from app.models import (
    LearningConceptState,
    LearningDifficulty,
    LearningObservation,
    LearningOutcome,
    LearningProfileSnapshot,
    LearningUpdateTrace,
    TaskSpec,
)


class StoredConceptState(BaseModel):
    concept: str
    mastery_probability: float = Field(default=0.25, ge=0, le=1)
    attempts: int = 0
    correct_attempts: int = 0
    hint_count: int = 0
    fsrs_difficulty: float = Field(default=5.0, ge=1, le=10)
    fsrs_stability_days: float = Field(default=0.0, ge=0)
    last_review_at: str | None = None
    next_review_at: str | None = None
    last_outcome: LearningOutcome | None = None


class StoredLearningProfile(BaseModel):
    schema_version: str = "1.0"
    user_id: int
    ability_theta: float = 0.0
    concepts: dict[str, StoredConceptState] = Field(default_factory=dict)
    updated_at: str


class LearningSignalDetector:
    """Extract only explicit learning evidence; never infer performance from an answer."""

    CONCEPT_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("动态规划", ("动态规划", "dp", "背包", "打家劫舍", "最长公共子序列", "最长递增子序列")),
        ("二分查找", ("二分查找", "二分答案", "二分")),
        ("哈希表", ("哈希表", "哈希", "hash", "两数之和")),
        ("滑动窗口", ("滑动窗口",)),
        ("双指针", ("双指针", "快慢指针")),
        ("单调栈", ("单调栈", "下一个更大元素")),
        ("拓扑排序", ("拓扑排序",)),
        ("并查集", ("并查集", "union find", "union-find")),
        ("最短路", ("最短路", "dijkstra", "floyd", "bellman-ford")),
        ("二叉树", ("二叉树", "树的遍历", "层序遍历")),
        ("图搜索", ("深度优先", "广度优先", "dfs", "bfs", "岛屿数量")),
        ("图论", ("图论", "邻接表", "邻接矩阵")),
        ("回溯", ("回溯", "排列组合", "组合总和")),
        ("贪心", ("贪心",)),
        ("链表", ("链表", "反转链表")),
        ("堆与优先队列", ("优先队列", "最小堆", "最大堆", "heap")),
        ("栈与队列", ("栈", "队列", "括号匹配")),
        ("字符串", ("字符串", "kmp")),
        ("数组", ("数组", "前缀和", "差分数组")),
    )

    RECOMMENDATION_PATTERN = re.compile(
        r"推荐|下一道|下一题|练习题|学习计划|复习计划|掌握度|学习情况|学习画像|"
        r"(?:出|来|给我|找).{0,6}(?:道|个).{0,6}(?:题|练习)|"
        r"应该学|薄弱|根据.{0,8}(情况|水平)",
        re.IGNORECASE,
    )
    CORRECT_PATTERN = re.compile(
        r"独立.{0,10}(做对|通过|ac)|(?<!没)(?<!未)(做对了|通过了|ac了|解决了)|"
        r"没有.{0,6}(提示|题解).{0,8}(做对|通过)",
        re.IGNORECASE,
    )
    INCORRECT_PATTERN = re.compile(
        r"没(有)?做对|没(有)?通过|没做出来|做错|不会|不理解|卡住|wa了|答案错误",
        re.IGNORECASE,
    )
    HINT_PATTERN = re.compile(
        r"给我.{0,6}提示|只要提示|需要提示|用了.{0,6}提示|看了提示",
        re.IGNORECASE,
    )
    SOLUTION_PATTERN = re.compile(
        r"看了.{0,6}(答案|题解|完整解法)|直接.{0,6}(答案|完整解法)|告诉我完整答案",
        re.IGNORECASE,
    )
    REVIEW_PATTERN = re.compile(r"复习了|刚复习|复习一下|重新复习", re.IGNORECASE)

    def detect(
        self,
        user_message: str,
        task_spec: TaskSpec,
    ) -> tuple[list[LearningObservation], bool, list[str]]:
        entity_text = " ".join(item.value for item in task_spec.entities)
        text = " ".join(
            (
                user_message,
                task_spec.normalized_request,
                task_spec.recognition_summary,
                entity_text,
            )
        )
        concepts = self._concepts(text, task_spec)
        outcome, confidence = self._outcome(user_message)
        difficulty = self._difficulty(text)
        observations = [
            LearningObservation(
                concept=concept,
                outcome=outcome,
                difficulty=difficulty,
                confidence=confidence,
                evidence=self._evidence_label(outcome),
            )
            for concept in concepts
            if outcome is not None
        ]
        recommendation_requested = bool(self.RECOMMENDATION_PATTERN.search(user_message))
        return observations, recommendation_requested, concepts

    def _concepts(self, text: str, task_spec: TaskSpec) -> list[str]:
        normalized = text.casefold()
        found: list[str] = []
        for concept, aliases in self.CONCEPT_ALIASES:
            if any(self._contains(normalized, alias.casefold()) for alias in aliases):
                found.append(concept)
        if not found:
            for entity in task_spec.entities:
                if entity.type.casefold() in {"algorithm", "concept", "topic", "data_structure"}:
                    value = re.sub(r"\s+", " ", entity.value).strip()
                    if 1 < len(value) <= 40 and value not in found:
                        found.append(value)
        return found[:3]

    @staticmethod
    def _contains(text: str, alias: str) -> bool:
        if re.fullmatch(r"[a-z0-9+#._-]+", alias):
            return re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", text) is not None
        return alias in text

    def _outcome(self, message: str) -> tuple[LearningOutcome | None, float]:
        if self.CORRECT_PATTERN.search(message) and not self.INCORRECT_PATTERN.search(message):
            return "correct", 0.95
        if self.SOLUTION_PATTERN.search(message):
            return "solution_viewed", 0.9
        if self.HINT_PATTERN.search(message):
            return "hinted", 0.88
        if self.INCORRECT_PATTERN.search(message):
            return "incorrect", 0.95
        if self.REVIEW_PATTERN.search(message):
            return "reviewed", 0.75
        return None, 0.0

    @staticmethod
    def _difficulty(text: str) -> LearningDifficulty:
        lowered = text.casefold()
        if any(value in lowered for value in ("困难", "hard", "难题")):
            return "hard"
        if any(value in lowered for value in ("简单", "easy", "入门")):
            return "easy"
        if any(value in lowered for value in ("中等", "medium")):
            return "medium"
        return "unknown"

    @staticmethod
    def _evidence_label(outcome: LearningOutcome) -> str:
        return {
            "correct": "用户明确表示独立完成或通过",
            "incorrect": "用户明确表示未通过或尚未理解",
            "hinted": "用户明确请求或使用了提示",
            "solution_viewed": "用户明确请求或查看了完整解法",
            "reviewed": "用户明确表示进行了复习",
        }[outcome]


class LearningProfileService:
    BKT_PRIOR = 0.25
    BKT_LEARN = 0.13
    BKT_GUESS = 0.20
    BKT_SLIP = 0.10
    IRT_DISCRIMINATION = 1.20
    IRT_LEARNING_RATE = 0.35

    OUTCOME_SCORE: dict[LearningOutcome, float] = {
        "correct": 1.0,
        "incorrect": 0.0,
        "hinted": 0.35,
        "solution_viewed": 0.20,
        "reviewed": 0.60,
    }
    FSRS_RATING: dict[LearningOutcome, tuple[str, int]] = {
        "correct": ("Good", 3),
        "incorrect": ("Again", 1),
        "hinted": ("Hard", 2),
        "solution_viewed": ("Again", 1),
        "reviewed": ("Hard", 2),
    }
    IRT_DIFFICULTY: dict[LearningDifficulty, float] = {
        "easy": -1.0,
        "medium": 0.0,
        "hard": 1.1,
        "unknown": 0.0,
    }

    def __init__(
        self,
        project_root: Path,
        configured_dir: str,
        detector: LearningSignalDetector | None = None,
    ) -> None:
        root = Path(configured_dir)
        self.root = (root if root.is_absolute() else project_root / root).resolve()
        self.detector = detector or LearningSignalDetector()
        self._lock = asyncio.Lock()

    async def process_turn(
        self,
        user_id: int,
        session_id: int,
        user_message: str,
        task_spec: TaskSpec,
    ) -> LearningProfileSnapshot:
        observations, recommendation_requested, mentioned = self.detector.detect(
            user_message,
            task_spec,
        )
        active = bool(observations or recommendation_requested)
        if not active:
            return LearningProfileSnapshot(
                active=False,
                user_id=user_id,
                session_id=session_id,
                summary="本轮没有明确学习结果或个性化推荐请求，学习画像未读取。",
            )

        async with self._lock:
            profile = self._load_unlocked(user_id)
            updates: list[LearningUpdateTrace] = []
            for observation in observations:
                updates.append(self._apply_observation(profile, observation))
            if updates:
                profile.updated_at = self._now().isoformat()
                self._write_unlocked(profile)

        recommended = self._recommend_concepts(profile)
        display_names = list(dict.fromkeys([
            *(item.concept for item in updates),
            *recommended,
            *mentioned,
        ]))[:5]
        concepts = [
            self._view_state(
                profile.concepts.get(name)
                or StoredConceptState(concept=name),
                self._priority(
                    profile.concepts.get(name)
                    or StoredConceptState(concept=name)
                ),
            )
            for name in display_names
        ]
        if updates:
            summary = "已根据用户明确反馈更新 BKT 掌握度、IRT 能力值与 FSRS 复习计划。"
        elif profile.concepts:
            summary = "已读取独立学习画像用于本轮推荐，未修改任何掌握度。"
        else:
            summary = "尚无明确做题结果，本轮只建立推荐入口，不猜测用户掌握度。"
        return LearningProfileSnapshot(
            active=True,
            updated=bool(updates),
            user_id=user_id,
            session_id=session_id,
            ability_theta=round(profile.ability_theta, 4),
            target_difficulty=self._target_difficulty(profile.ability_theta),
            summary=summary,
            observations=observations,
            updates=updates,
            concepts=concepts,
            recommended_concepts=recommended[:3],
        )

    def render_markdown(self, snapshot: LearningProfileSnapshot) -> str:
        if not snapshot.active:
            return ""
        difficulty_label = {
            "easy": "简单",
            "medium": "中等",
            "hard": "困难",
            "unknown": "未知",
        }[snapshot.target_difficulty]
        lines = [
            "---",
            "### 个性化学习画像",
            "",
            f"> {snapshot.summary}",
            "",
            f"- **IRT 能力值**：θ = {snapshot.ability_theta:.2f}，当前建议题目难度：**{difficulty_label}**",
        ]
        if snapshot.concepts:
            lines.extend([
                "",
                "| 知识点 | BKT 掌握度 | 练习情况 | FSRS 下次复习 |",
                "| --- | ---: | --- | --- |",
            ])
            for item in snapshot.concepts:
                attempt_text = (
                    f"独立成功 {item.correct_attempts} · 学习事件 {item.attempts}"
                )
                review = (item.next_review_at or "待产生有效记录")[:10]
                lines.append(
                    f"| {item.concept} | {item.mastery_probability:.0%} | {attempt_text} | {review} |"
                )
        if snapshot.updates:
            lines.append("")
            lines.append("本轮更新：" + "；".join(
                f"{item.concept} {item.mastery_before:.0%} → {item.mastery_after:.0%}，"
                f"FSRS={item.fsrs_rating}"
                for item in snapshot.updates
            ))
        if snapshot.recommended_concepts:
            lines.append("")
            lines.append("建议优先关注：" + "、".join(snapshot.recommended_concepts) + "。")
        lines.extend([
            "",
            "_学习画像与聊天记忆独立；只依据用户明确反馈更新，不会把模型猜测写入画像。_",
        ])
        return "\n".join(lines)

    def prompt_fragment(self, snapshot: LearningProfileSnapshot | None) -> str:
        if snapshot is None or not snapshot.active:
            return ""
        lines = [
            "以下学习画像只用于本轮明确的学习反馈、复习或推荐任务；不得在普通任务中主动提及。",
            f"IRT 能力值 theta={snapshot.ability_theta:.3f}，建议难度={snapshot.target_difficulty}。",
        ]
        if snapshot.concepts:
            lines.append("知识点状态：")
            lines.extend(
                f"- {item.concept}: BKT={item.mastery_probability:.3f}, "
                f"attempts={item.attempts}, next_review={item.next_review_at or 'unknown'}"
                for item in snapshot.concepts
            )
        if snapshot.recommended_concepts:
            lines.append("推荐优先级：" + "、".join(snapshot.recommended_concepts))
        lines.append("需要推荐题目时，应结合该画像检索题库；没有实际题库证据时不得编造题号。")
        return "\n".join(lines)

    def _apply_observation(
        self,
        profile: StoredLearningProfile,
        observation: LearningObservation,
    ) -> LearningUpdateTrace:
        state = profile.concepts.get(observation.concept) or StoredConceptState(
            concept=observation.concept,
            mastery_probability=self.BKT_PRIOR,
        )
        mastery_before = state.mastery_probability
        ability_before = profile.ability_theta
        score = self.OUTCOME_SCORE[observation.outcome]

        if observation.outcome == "reviewed":
            mastery_after = mastery_before
            predicted = self._irt_probability(
                ability_before,
                self.IRT_DIFFICULTY[observation.difficulty],
            )
            ability_after = ability_before
        else:
            mastery_after = self._bkt_update(mastery_before, score)
            item_difficulty = self.IRT_DIFFICULTY[observation.difficulty]
            predicted = self._irt_probability(ability_before, item_difficulty)
            ability_after = self._clamp(
                ability_before
                + self.IRT_LEARNING_RATE
                * self.IRT_DISCRIMINATION
                * (score - predicted)
                * observation.confidence,
                -3.0,
                3.0,
            )

        rating_name, rating_value = self.FSRS_RATING[observation.outcome]
        now = self._now()
        stability, fsrs_difficulty, next_review = self._fsrs_schedule(
            state,
            rating_value,
            now,
        )
        state.mastery_probability = mastery_after
        state.attempts += 1
        if observation.outcome == "correct":
            state.correct_attempts += 1
        if observation.outcome in {"hinted", "solution_viewed"}:
            state.hint_count += 1
        state.fsrs_stability_days = stability
        state.fsrs_difficulty = fsrs_difficulty
        state.last_review_at = now.isoformat()
        state.next_review_at = next_review.isoformat()
        state.last_outcome = observation.outcome
        profile.concepts[observation.concept] = state
        profile.ability_theta = ability_after
        return LearningUpdateTrace(
            concept=observation.concept,
            outcome=observation.outcome,
            mastery_before=round(mastery_before, 6),
            mastery_after=round(mastery_after, 6),
            ability_before=round(ability_before, 6),
            ability_after=round(ability_after, 6),
            predicted_success=round(predicted, 6),
            fsrs_rating=rating_name,
            next_review_at=next_review.isoformat(),
        )

    def _bkt_update(self, prior: float, score: float) -> float:
        correct_denominator = prior * (1 - self.BKT_SLIP) + (1 - prior) * self.BKT_GUESS
        incorrect_denominator = prior * self.BKT_SLIP + (1 - prior) * (1 - self.BKT_GUESS)
        posterior_correct = prior * (1 - self.BKT_SLIP) / max(correct_denominator, 1e-9)
        posterior_incorrect = prior * self.BKT_SLIP / max(incorrect_denominator, 1e-9)
        posterior = score * posterior_correct + (1 - score) * posterior_incorrect
        learned = posterior + (1 - posterior) * self.BKT_LEARN
        return self._clamp(learned, 0.01, 0.995)

    def _irt_probability(self, theta: float, difficulty: float) -> float:
        exponent = -self.IRT_DISCRIMINATION * (theta - difficulty)
        return 1.0 / (1.0 + math.exp(self._clamp(exponent, -30, 30)))

    def _fsrs_schedule(
        self,
        state: StoredConceptState,
        rating: int,
        now: datetime,
    ) -> tuple[float, float, datetime]:
        difficulty = self._clamp(
            state.fsrs_difficulty - 0.30 * (rating - 3),
            1.0,
            10.0,
        )
        if state.fsrs_stability_days <= 0:
            stability = {1: 0.4, 2: 1.0, 3: 2.5, 4: 5.0}[rating]
        else:
            elapsed = self._elapsed_days(state.last_review_at, now)
            retrievability = 0.9 ** (elapsed / max(state.fsrs_stability_days, 0.1))
            if rating == 1:
                stability = max(0.4, state.fsrs_stability_days * (0.25 + 0.15 * retrievability))
            else:
                growth = 1 + (11 - difficulty) * 0.08 * (rating - 1) * (
                    1.3 - retrievability
                )
                stability = state.fsrs_stability_days * max(1.05, growth)
        stability = self._clamp(stability, 0.4, 365.0)
        interval_days = max(1, min(365, round(stability)))
        return stability, difficulty, now + timedelta(days=interval_days)

    def _recommend_concepts(self, profile: StoredLearningProfile) -> list[str]:
        return [
            item.concept
            for item in sorted(
                profile.concepts.values(),
                key=lambda value: (self._priority(value), value.concept),
                reverse=True,
            )
        ]

    def _priority(self, state: StoredConceptState) -> float:
        due = 0.0
        if state.next_review_at:
            try:
                due_date = datetime.fromisoformat(state.next_review_at)
                days = (self._now() - due_date).total_seconds() / 86_400
                due = 1.0 if days >= 0 else max(0.0, 1.0 + days / 30)
            except ValueError:
                due = 0.0
        uncertainty = 1.0 / math.sqrt(state.attempts + 1)
        return round((1 - state.mastery_probability) * 0.65 + due * 0.25 + uncertainty * 0.10, 6)

    def _view_state(
        self,
        state: StoredConceptState,
        priority: float,
    ) -> LearningConceptState:
        return LearningConceptState(
            **state.model_dump(),
            priority_score=priority,
        )

    @staticmethod
    def _target_difficulty(theta: float) -> LearningDifficulty:
        if theta < -0.55:
            return "easy"
        if theta < 0.85:
            return "medium"
        return "hard"

    def _path(self, user_id: int) -> Path:
        return self.root / f"user-{user_id}.json"

    def _load_unlocked(self, user_id: int) -> StoredLearningProfile:
        path = self._path(user_id)
        if not path.is_file():
            return StoredLearningProfile(
                user_id=user_id,
                updated_at=self._now().isoformat(),
            )
        try:
            return StoredLearningProfile.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError, TypeError):
            return StoredLearningProfile(
                user_id=user_id,
                updated_at=self._now().isoformat(),
            )

    def _write_unlocked(self, profile: StoredLearningProfile) -> None:
        path = self._path(profile.user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(profile.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _elapsed_days(value: str | None, now: datetime) -> float:
        if not value:
            return 0.0
        try:
            previous = datetime.fromisoformat(value)
            return max(0.0, (now - previous).total_seconds() / 86_400)
        except ValueError:
            return 0.0

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)
