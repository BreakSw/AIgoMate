from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class InputRewriteResult(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    input_type: Literal["text", "code", "mixed"]
    formatted_input: str
    explicit_request: str
    requested_operations: list[Literal[
        "general_code_review",
        "explain_logic",
        "find_bugs",
        "analyze_complexity",
        "optimize_code",
        "solve_problem",
        "explain_concept",
        "provide_hint",
        "compare_solutions",
        "plan_learning",
        "general_request",
    ]]
    request_is_actionable: bool
    instruction_verbatim: str | None = None
    contextual_references: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    contains_code: bool = False
    programming_language: str | None = None
    rewrite_summary: str
    rewrite_model: str
    rewrite_provider: str


class AgentRequest(BaseModel):
    model_config = ConfigDict(
        alias_generator=lambda value: {
            "session_id": "sessionId",
            "previous_context_snapshot": "previousContextSnapshot",
        }.get(value, value),
        populate_by_name=True,
    )

    session_id: int
    message: str = Field(min_length=1, max_length=12_000)
    history: list[ConversationMessage] = Field(default_factory=list)
    previous_context_snapshot: "ContextSnapshot | None" = None


class AgentResponse(BaseModel):
    content: str
    intent: str
    context_messages_used: int
    task_spec: "TaskSpec"
    context_snapshot: "ContextSnapshot"
    model: str
    provider: str


PrimaryIntent = Literal[
    "concept_explanation",
    "guided_hint",
    "problem_solving",
    "code_generation",
    "code_diagnosis",
    "complexity_analysis",
    "solution_comparison",
    "mock_interview",
    "review_planning",
    "visual_explanation",
    "learning_consultation",
    "general_conversation",
]

ResponseMode = Literal[
    "direct_answer",
    "progressive_hint",
    "socratic_questioning",
    "code_review",
    "step_by_step_explanation",
    "study_plan",
    "clarification_first",
]


class IntentEntity(BaseModel):
    type: Literal[
        "algorithm",
        "data_structure",
        "programming_language",
        "problem",
        "code",
        "error",
        "test_case",
        "complexity_target",
        "learning_topic",
        "other",
    ]
    value: str
    role: str | None = None


class InputArtifacts(BaseModel):
    problem_statement: str | None = None
    code: str | None = None
    error_message: str | None = None
    test_cases: list[str] = Field(default_factory=list)
    programming_language: str | None = None


class DeliverySpec(BaseModel):
    assistance_level: Literal[
        "direct_solution",
        "hint_only",
        "explanation_only",
        "review_only",
        "interactive_guidance",
        "plan_only",
    ]
    explanation_depth: Literal["brief", "standard", "detailed"] = "standard"
    response_language: str = "zh-CN"
    expected_outputs: list[str] = Field(default_factory=list)
    include_code: bool | None = None


Capability = Literal[
    "algorithm_tutoring",
    "knowledge_retrieval",
    "code_sandbox",
    "code_diagnosis",
    "solution_comparison",
    "visualization",
    "review_planning",
    "interview_simulation",
]


class RoutingPlan(BaseModel):
    primary_capability: Capability
    supporting_capabilities: list[Capability] = Field(default_factory=list)
    execution_mode: Literal["single", "sequential", "parallel"] = "single"
    recommended_sequence: list[Capability] = Field(default_factory=list)
    tool_requirements: list[
        Literal["knowledge_base", "code_sandbox", "visualization_renderer", "none"]
    ] = Field(default_factory=list)


class ContextPlan(BaseModel):
    recent_messages: bool = True
    task_state: bool = False
    long_term_memory: bool = False
    user_learning_profile: bool = False
    algorithm_knowledge: bool = False


class MemorySnapshot(BaseModel):
    current_goal: str | None = None
    working_memory: list[str] = Field(default_factory=list)
    long_term_memory: list[str] = Field(default_factory=list)
    user_preferences: list[str] = Field(default_factory=list)
    pinned_constraints: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    resolved_items: list[str] = Field(default_factory=list)


class CompressedContext(BaseModel):
    summary: str
    topics: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    artifact_references: list[str] = Field(default_factory=list)
    source_message_count: int = 0
    compression_model: str | None = None
    compression_provider: str | None = None


class ContextWindowStatus(BaseModel):
    window_size_tokens: int
    soft_limit_tokens: int
    hard_limit_tokens: int
    output_reserved_tokens: int
    raw_history_tokens: int
    current_input_tokens: int
    compressed_context_tokens: int
    recent_messages_tokens: int
    estimated_input_tokens: int
    remaining_tokens: int
    usage_ratio: float = Field(ge=0)
    state: Literal["normal", "soft_limit", "hard_limit", "overflow"]
    compression_triggered: bool
    messages_before_compression: int
    messages_after_compression: int
    candidate_input_tokens: int = 0
    safe_input_budget_tokens: int = 0
    safe_remaining_tokens: int = 0
    turn_metadata_tokens: int = 0
    compression_reused: bool = False
    compression_trigger_reason: Literal[
        "not_required",
        "reused_checkpoint",
        "preflight_budget_exceeded",
        "turn_commit_budget_exceeded",
    ] = "not_required"
    checkpoint_message_count: int = 0
    new_messages_since_checkpoint: int = 0


class TurnContext(BaseModel):
    primary_intent: PrimaryIntent
    normalized_request: str
    user_goal: str
    constraints: list[str] = Field(default_factory=list)
    response_mode: ResponseMode
    primary_capability: Capability
    success_criteria: list[str] = Field(default_factory=list)
    intent_model: str
    intent_provider: str


class ContextSnapshot(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    memory: MemorySnapshot
    compressed_context: CompressedContext
    window: ContextWindowStatus
    turn_context: TurnContext | None = None
    input_rewrite: InputRewriteResult | None = None
    # Memory frozen at the last compaction boundary. Active `memory` can keep
    # advancing every turn without changing what the checkpoint represents.
    checkpoint_memory: MemorySnapshot | None = None


class TaskSpec(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    primary_intent: PrimaryIntent
    secondary_intents: list[PrimaryIntent] = Field(default_factory=list)
    normalized_request: str
    user_goal: str
    recognition_summary: str
    entities: list[IntentEntity] = Field(default_factory=list)
    input_artifacts: InputArtifacts = Field(default_factory=InputArtifacts)
    constraints: list[str] = Field(default_factory=list)
    response_mode: ResponseMode
    delivery: DeliverySpec
    routing: RoutingPlan
    context_plan: ContextPlan = Field(default_factory=ContextPlan)
    success_criteria: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    clarifying_question: str | None = None
    raw_metadata: dict[str, Any] = Field(default_factory=dict, exclude=True)
