from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class InputOrganizationResult(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    organized_input: str
    input_shape: Literal["text", "code", "mixed", "unclassified"]
    organization_summary: str
    organizer_model: str
    organizer_provider: str


class InputRewriteResult(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    input_type: Literal["text", "code", "mixed"]
    formatted_input: str
    explicit_request: str | None = None
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
            "user_id": "userId",
            "session_id": "sessionId",
            "previous_context_snapshot": "previousContextSnapshot",
        }.get(value, value),
        populate_by_name=True,
    )

    user_id: int = 1
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


class MemoryScope(BaseModel):
    user_id: int
    session_id: int


LearningOutcome = Literal[
    "correct",
    "incorrect",
    "hinted",
    "solution_viewed",
    "reviewed",
]
LearningDifficulty = Literal["easy", "medium", "hard", "unknown"]


class LearningObservation(BaseModel):
    concept: str
    outcome: LearningOutcome
    difficulty: LearningDifficulty = "unknown"
    confidence: float = Field(default=0.8, ge=0, le=1)
    evidence: str


class LearningUpdateTrace(BaseModel):
    concept: str
    outcome: LearningOutcome
    mastery_before: float = Field(ge=0, le=1)
    mastery_after: float = Field(ge=0, le=1)
    ability_before: float
    ability_after: float
    predicted_success: float = Field(ge=0, le=1)
    fsrs_rating: Literal["Again", "Hard", "Good", "Easy"]
    next_review_at: str


class LearningConceptState(BaseModel):
    concept: str
    mastery_probability: float = Field(ge=0, le=1)
    attempts: int = Field(ge=0)
    correct_attempts: int = Field(ge=0)
    hint_count: int = Field(ge=0)
    fsrs_difficulty: float = Field(ge=1, le=10)
    fsrs_stability_days: float = Field(ge=0)
    last_review_at: str | None = None
    next_review_at: str | None = None
    last_outcome: LearningOutcome | None = None
    priority_score: float = Field(default=0, ge=0)


class LearningProfileSnapshot(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    active: bool = False
    updated: bool = False
    scope: Literal["user_learning_profile"] = "user_learning_profile"
    user_id: int
    session_id: int
    ability_theta: float = 0
    target_difficulty: LearningDifficulty = "medium"
    summary: str
    observations: list[LearningObservation] = Field(default_factory=list)
    updates: list[LearningUpdateTrace] = Field(default_factory=list)
    concepts: list[LearningConceptState] = Field(default_factory=list)
    recommended_concepts: list[str] = Field(default_factory=list)
    algorithms: list[str] = Field(
        default_factory=lambda: ["BKT", "IRT-1PL", "FSRS-style"]
    )


class ContextSnapshot(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    memory: MemorySnapshot
    compressed_context: CompressedContext
    window: ContextWindowStatus
    turn_context: TurnContext | None = None
    input_organization: InputOrganizationResult | None = None
    input_rewrite: InputRewriteResult | None = None
    agent_execution: "AgentExecutionTrace | None" = None
    memory_scope: MemoryScope | None = None
    learning_profile: LearningProfileSnapshot | None = None
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

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_percentage_confidence(cls, value):
        if isinstance(value, (int, float)) and 1 < value <= 100:
            return value / 100
        return value


KnowledgeCollection = Literal[
    "algorithm_concepts",
    "problem_bank",
    "code_cases",
]

EvidenceCollection = Literal[
    "algorithm_concepts",
    "problem_bank",
    "code_cases",
    "web_search",
]

ExecutionAgent = Literal[
    "tutoring_agent",
    "problem_solving_agent",
    "code_analysis_agent",
    "problem_structuring_agent",
    "strategy_agent",
    "solution_review_agent",
    "implementation_agent",
    "verification_agent",
    "code_test_generation_agent",
    "learning_planning_agent",
    "conversation_agent",
    "clarification_agent",
]


class RagQuery(BaseModel):
    collection: KnowledgeCollection
    query: str = Field(min_length=1, max_length=500)
    reason: str = Field(min_length=1, max_length=300)
    top_k: int = Field(default=3, ge=1, le=5)
    required: bool = False


class MemorySelection(BaseModel):
    working_memory: bool = True
    long_term_memory: bool = False
    user_preferences: bool = False
    pinned_constraints: bool = True
    reason: str = Field(default="仅选择完成当前任务所需的记忆")


MemoryKind = Literal[
    "user_profile",
    "preference",
    "long_term_goal",
    "constraint",
    "decision",
    "unfinished_task",
    "learned_fact",
]


class MemoryUpdate(BaseModel):
    kind: MemoryKind
    content: str = Field(min_length=1, max_length=500)
    importance: float = Field(default=0.7, ge=0, le=1)
    reason: str = Field(min_length=1, max_length=300)


class MemoryUpdateBatch(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    updates: list[MemoryUpdate] = Field(default_factory=list)
    ignored_transient_details: list[str] = Field(default_factory=list)


class DurableMemoryItem(BaseModel):
    memory_id: str
    kind: MemoryKind
    content: str
    importance: float = Field(ge=0, le=1)
    source: Literal["memory_agent", "context_checkpoint", "user_explicit"]
    created_at: str
    updated_at: str


class HeadDecision(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    iteration: int = Field(ge=1)
    rationale: str = Field(min_length=1, max_length=500)
    action: Literal[
        "get_current_time",
        "retrieve_rag",
        "switch_to_native_reasoning",
        "search_web",
        "execute_code_tests",
        "delegate",
        "persist_memory",
        "ask_clarification",
        "finish",
    ]
    selected_agent: ExecutionAgent | None = None
    task_instruction: str | None = None
    rag_query: RagQuery | None = None
    web_query: str | None = Field(default=None, max_length=500)
    web_search_reason: str | None = Field(default=None, max_length=300)
    memory_updates: list[MemoryUpdate] = Field(default_factory=list)
    clarification_question: str | None = None
    finish_reason: str | None = None


class CoordinatorPlan(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    objective: str
    selected_agent: ExecutionAgent
    task_instruction: str
    planned_steps: list[str] = Field(default_factory=list)
    rag_queries: list[RagQuery] = Field(default_factory=list)
    memory_selection: MemorySelection = Field(default_factory=MemorySelection)
    execution_mode: Literal["rag_assisted", "native_reasoning"] = "native_reasoning"
    rag_status: Literal[
        "not_checked",
        "candidate_found",
        "hit",
        "miss",
        "insufficient",
        "unavailable",
        "error",
    ] = "not_checked"
    grounding_policy: Literal["no_rag", "prefer_rag", "require_rag"] = "no_rag"
    requires_clarification: bool = False
    clarification_question: str | None = None
    known_limits: list[str] = Field(default_factory=list)
    decision_trace: list[HeadDecision] = Field(default_factory=list)
    web_search_queries: list[str] = Field(default_factory=list)


class RagEvidence(BaseModel):
    evidence_id: str
    collection: EvidenceCollection
    title: str
    content: str
    source_url: str | None = None
    score: float = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentWorkRequest(BaseModel):
    protocol_version: Literal["1.0"] = "1.0"
    task_spec: TaskSpec
    coordinator_plan: CoordinatorPlan
    conversation_context: list[ConversationMessage] = Field(default_factory=list)
    memory: MemorySnapshot
    durable_memory: list[DurableMemoryItem] = Field(default_factory=list)
    dynamic_system_prompt: str = ""
    rag_evidence: list[RagEvidence] = Field(default_factory=list)
    prior_work_results: list["AgentWorkResult"] = Field(default_factory=list)
    code_execution_reports: list["CodeExecutionReport"] = Field(default_factory=list)


class AgentWorkResult(BaseModel):
    protocol_version: Literal["1.0"] = "1.0"
    agent: ExecutionAgent
    draft_answer: str
    used_evidence_ids: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    needs_follow_up: bool = False


CodeExecutionStatus = Literal[
    "passed",
    "failed",
    "unavailable",
    "unsupported",
    "error",
]


class CodeTestPlan(BaseModel):
    """Executable harness produced for one exact candidate-code revision."""

    protocol_version: Literal["1.0"] = "1.0"
    language: Literal["Python", "Java", "C++"]
    source_code_hash: str = Field(min_length=64, max_length=64)
    executable_source: str = Field(min_length=20)
    expected_output: str = Field(min_length=1)
    test_count: int = Field(ge=1, le=100)
    test_categories: list[str] = Field(default_factory=list)
    oracle_strategy: str = Field(min_length=1, max_length=500)
    semantic_reflection_rounds: int = Field(default=0, ge=0, le=5)
    review_summary: str | None = Field(default=None, max_length=1_000)
    review_confidence: float | None = Field(default=None, ge=0, le=1)


class CodeTestPlanReview(BaseModel):
    """Independent semantic critique of a generated algorithm test Harness."""

    protocol_version: Literal["1.0"] = "1.0"
    verdict: Literal["approved", "revise"]
    summary: str = Field(min_length=1, max_length=1_000)
    checked_dimensions: list[Literal[
        "candidate_integrity",
        "candidate_invocation",
        "oracle_independence",
        "oracle_correctness",
        "constraint_compliance",
        "edge_coverage",
        "language_compilability",
        "output_protocol",
    ]] = Field(default_factory=list)
    issues: list[str | dict[str, Any]] = Field(default_factory=list)
    revision_instructions: list[str | dict[str, Any]] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class CodeExecutionReport(BaseModel):
    """Normalized, user-visible report returned by the Judge0 tool boundary."""

    protocol_version: Literal["1.0"] = "1.0"
    provider: Literal["judge0-sdk"] = "judge0-sdk"
    source_code_hash: str = Field(min_length=64, max_length=64)
    language: str
    overall_status: CodeExecutionStatus
    verdict: str
    passed_tests: int = Field(default=0, ge=0)
    total_tests: int = Field(default=0, ge=0)
    test_categories: list[str] = Field(default_factory=list)
    oracle_strategy: str = ""
    semantic_reflection_rounds: int = Field(default=0, ge=0, le=5)
    test_plan_review: str | None = None
    test_plan_review_confidence: float | None = Field(default=None, ge=0, le=1)
    stdout: str | None = None
    stderr: str | None = None
    compile_output: str | None = None
    exit_code: int | None = None
    time_seconds: float | None = None
    memory_kb: int | None = None
    failure_reason: str | None = None


class PolishResult(BaseModel):
    protocol_version: Literal["1.0"] = "1.0"
    final_answer: str
    preserved_uncertainties: list[str] = Field(default_factory=list)
    style_changes: list[str] = Field(default_factory=list)
    added_factual_claims: bool = False


class OutputFormatResult(BaseModel):
    protocol_version: Literal["1.0"] = "1.0"
    formatted_answer: str
    formatting_changes: list[str] = Field(default_factory=list)
    added_factual_claims: bool = False


class AgentExecutionTrace(BaseModel):
    protocol_version: Literal["1.0"] = "1.0"
    task_spec: TaskSpec
    coordinator_plan: CoordinatorPlan
    rag_evidence: list[RagEvidence] = Field(default_factory=list)
    work_result: AgentWorkResult
    polish_result: PolishResult
    format_result: OutputFormatResult | None = None
    durable_memory: list[DurableMemoryItem] = Field(default_factory=list)
    memory_updates: list[MemoryUpdate] = Field(default_factory=list)
    model_call_trace: list[str] = Field(default_factory=list)
    code_execution_reports: list[CodeExecutionReport] = Field(default_factory=list)
