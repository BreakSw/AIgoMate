export type Role = 'USER' | 'ASSISTANT' | 'SYSTEM'

export interface ChatSession {
  id: number
  title: string
  summary?: string
  messageCount: number
  createdAt: string
  updatedAt: string
}

export interface ChatMessage {
  id: number
  role: Role
  content: string
  createdAt: string
  contextSnapshot?: ContextSnapshot | null
}

export interface Conversation {
  session: ChatSession
  messages: ChatMessage[]
}

export interface ModelConfigStatus {
  configured: boolean
  model?: string | null
  baseUrl?: string | null
  maskedApiKey?: string | null
  searchConfigured?: boolean
  maskedSerpapiApiKey?: string | null
  ttlSeconds?: number | null
  expiresAt?: string | null
}

export interface ModelConfigInput {
  apiKey?: string
  serpapiApiKey?: string | null
  model?: string
  baseUrl?: string
  ttlSeconds: number
  updateModel?: boolean
  updateSearch?: boolean
}

export interface IntentEntity {
  type: string
  value: string
  role?: string
}

export interface InputArtifacts {
  problem_statement?: string | null
  code?: string | null
  error_message?: string | null
  test_cases: string[]
  programming_language?: string | null
}

export interface DeliverySpec {
  assistance_level: string
  explanation_depth: string
  response_language: string
  expected_outputs: string[]
  include_code?: boolean | null
}

export interface RoutingPlan {
  primary_capability: string
  supporting_capabilities: string[]
  execution_mode: 'single' | 'sequential' | 'parallel'
  recommended_sequence: string[]
  tool_requirements: string[]
}

export interface ContextPlan {
  recent_messages: boolean
  task_state: boolean
  long_term_memory: boolean
  user_learning_profile: boolean
  algorithm_knowledge: boolean
}

export interface TaskSpec {
  schema_version: '1.0'
  primary_intent: string
  secondary_intents: string[]
  normalized_request: string
  user_goal: string
  recognition_summary: string
  entities: IntentEntity[]
  input_artifacts: InputArtifacts
  constraints: string[]
  response_mode: string
  delivery: DeliverySpec
  routing: RoutingPlan
  context_plan: ContextPlan
  success_criteria: string[]
  ambiguities: string[]
  risk_flags: string[]
  confidence: number
  clarifying_question?: string | null
}

export interface MemorySnapshot {
  current_goal?: string | null
  working_memory: string[]
  long_term_memory: string[]
  user_preferences: string[]
  pinned_constraints: string[]
  open_questions: string[]
  resolved_items: string[]
}

export interface InputOrganizationResult {
  schema_version: '1.0'
  organized_input: string
  input_shape: 'text' | 'code' | 'mixed' | 'unclassified'
  organization_summary: string
  organizer_model: string
  organizer_provider: string
}

export interface InputRewriteResult {
  schema_version: '1.0'
  input_type: 'text' | 'code' | 'mixed'
  formatted_input: string
  explicit_request?: string | null
  requested_operations: string[]
  request_is_actionable: boolean
  instruction_verbatim?: string | null
  contextual_references: string[]
  constraints: string[]
  ambiguities: string[]
  contains_code: boolean
  programming_language?: string | null
  rewrite_summary: string
  rewrite_model: string
  rewrite_provider: string
}

export interface CompressedContext {
  summary: string
  topics: string[]
  decisions: string[]
  artifact_references: string[]
  source_message_count: number
  compression_model?: string | null
  compression_provider?: string | null
}

export interface ContextWindowStatus {
  window_size_tokens: number
  soft_limit_tokens: number
  hard_limit_tokens: number
  output_reserved_tokens: number
  raw_history_tokens: number
  current_input_tokens: number
  compressed_context_tokens: number
  recent_messages_tokens: number
  estimated_input_tokens: number
  remaining_tokens: number
  usage_ratio: number
  state: 'normal' | 'soft_limit' | 'hard_limit' | 'overflow'
  compression_triggered: boolean
  messages_before_compression: number
  messages_after_compression: number
  candidate_input_tokens?: number
  safe_input_budget_tokens?: number
  safe_remaining_tokens?: number
  turn_metadata_tokens?: number
  compression_reused?: boolean
  compression_trigger_reason?: 'not_required' | 'reused_checkpoint' | 'preflight_budget_exceeded' | 'turn_commit_budget_exceeded'
  checkpoint_message_count?: number
  new_messages_since_checkpoint?: number
}

export interface TurnContext {
  primary_intent: string
  normalized_request: string
  user_goal: string
  constraints: string[]
  response_mode: string
  primary_capability: string
  success_criteria: string[]
  intent_model: string
  intent_provider: string
}

export type LearningOutcome = 'correct' | 'incorrect' | 'hinted' | 'solution_viewed' | 'reviewed'
export type LearningDifficulty = 'easy' | 'medium' | 'hard' | 'unknown'

export interface LearningObservation {
  concept: string
  outcome: LearningOutcome
  difficulty: LearningDifficulty
  confidence: number
  evidence: string
}

export interface LearningUpdateTrace {
  concept: string
  outcome: LearningOutcome
  mastery_before: number
  mastery_after: number
  ability_before: number
  ability_after: number
  predicted_success: number
  fsrs_rating: 'Again' | 'Hard' | 'Good' | 'Easy'
  next_review_at: string
}

export interface LearningConceptState {
  concept: string
  mastery_probability: number
  attempts: number
  correct_attempts: number
  hint_count: number
  fsrs_difficulty: number
  fsrs_stability_days: number
  last_review_at?: string | null
  next_review_at?: string | null
  last_outcome?: LearningOutcome | null
  priority_score: number
}

export interface LearningProfileSnapshot {
  schema_version: '1.0'
  active: boolean
  updated: boolean
  scope: 'user_learning_profile'
  user_id: number
  session_id: number
  ability_theta: number
  target_difficulty: LearningDifficulty
  summary: string
  observations: LearningObservation[]
  updates: LearningUpdateTrace[]
  concepts: LearningConceptState[]
  recommended_concepts: string[]
  algorithms: string[]
}

export interface ContextSnapshot {
  schema_version: '1.0'
  memory: MemorySnapshot
  compressed_context: CompressedContext
  window: ContextWindowStatus
  turn_context?: TurnContext | null
  input_organization?: InputOrganizationResult | null
  input_rewrite?: InputRewriteResult | null
  agent_execution?: AgentExecutionTrace | null
  learning_profile?: LearningProfileSnapshot | null
  checkpoint_memory?: MemorySnapshot | null
}

export type KnowledgeCollection = 'algorithm_concepts' | 'problem_bank' | 'code_cases'

export interface RagQuery {
  collection: KnowledgeCollection
  query: string
  reason: string
  top_k: number
  required: boolean
}

export interface MemorySelection {
  working_memory: boolean
  long_term_memory: boolean
  user_preferences: boolean
  pinned_constraints: boolean
  reason: string
}

export interface CoordinatorPlan {
  schema_version: '1.0'
  objective: string
  selected_agent: string
  task_instruction: string
  planned_steps?: string[]
  rag_queries?: RagQuery[]
  memory_selection: MemorySelection
  grounding_policy: 'no_rag' | 'prefer_rag' | 'require_rag'
  requires_clarification: boolean
  clarification_question?: string | null
  known_limits: string[]
  decision_trace: HeadDecision[]
  web_search_queries?: string[]
}

export interface MemoryUpdate {
  kind: string
  content: string
  importance: number
  reason: string
}

export interface DurableMemoryItem {
  memory_id: string
  kind: string
  content: string
  importance: number
  source: string
  created_at: string
  updated_at: string
}

export interface HeadDecision {
  schema_version: '1.0'
  iteration: number
  rationale: string
  action: 'get_current_time' | 'retrieve_rag' | 'switch_to_native_reasoning' | 'search_web' | 'execute_code_tests' | 'delegate' | 'persist_memory' | 'ask_clarification' | 'finish'
  selected_agent?: string | null
  task_instruction?: string | null
  rag_query?: RagQuery | null
  web_query?: string | null
  web_search_reason?: string | null
  memory_updates: MemoryUpdate[]
  clarification_question?: string | null
  finish_reason?: string | null
}

export interface RagEvidence {
  evidence_id: string
  collection: KnowledgeCollection | 'web_search'
  title: string
  content: string
  source_url?: string | null
  score: number
  metadata: Record<string, unknown>
}

export interface AgentWorkResult {
  protocol_version: '1.0'
  agent: string
  draft_answer: string
  used_evidence_ids: string[]
  uncertainties: string[]
  needs_follow_up: boolean
}

export interface PolishResult {
  protocol_version: '1.0'
  final_answer: string
  preserved_uncertainties: string[]
  style_changes: string[]
  added_factual_claims: boolean
}

export interface CodeExecutionReport {
  protocol_version: '1.0'
  provider: 'judge0-sdk'
  source_code_hash: string
  language: string
  overall_status: 'passed' | 'failed' | 'unavailable' | 'unsupported' | 'error'
  verdict: string
  passed_tests: number
  total_tests: number
  test_categories: string[]
  oracle_strategy: string
  semantic_reflection_rounds: number
  test_plan_review?: string | null
  test_plan_review_confidence?: number | null
  stdout?: string | null
  stderr?: string | null
  compile_output?: string | null
  exit_code?: number | null
  time_seconds?: number | null
  memory_kb?: number | null
  failure_reason?: string | null
}

export interface AgentExecutionTrace {
  protocol_version: '1.0'
  task_spec: TaskSpec
  coordinator_plan: CoordinatorPlan
  rag_evidence: RagEvidence[]
  work_result: AgentWorkResult
  polish_result: PolishResult
  durable_memory?: DurableMemoryItem[]
  memory_updates?: MemoryUpdate[]
  model_call_trace?: string[]
  code_execution_reports?: CodeExecutionReport[]
}

export interface IntentResult {
  content: string
  intent: string
  contextMessagesUsed: number
  task_spec: TaskSpec
  context_snapshot: ContextSnapshot
  model: string
  provider: string
}

export interface StreamStatus {
  phase: string
  message: string
  sequence?: number
  agent?: string
  detail?: string
  state?: 'active' | 'completed' | 'failed'
  retryCount?: number
  maxRetries?: number
}

export type RagLibraryKey = KnowledgeCollection | 'user_memory'

export interface RagDistributionItem {
  label: string
  count: number
}

export interface RagDocumentSample {
  document_id?: string | null
  title: string
  source_url?: string | null
  metadata: Record<string, unknown>
}

export interface RagLibraryOverview {
  key: RagLibraryKey
  label: string
  source: string
  storage: string
  retrieval_mode: string
  documents: number
  available_documents: number
  chunks: number
  tokens: number
  model: string
  dimension: number
  collection: string
  imported_rows: number
  coverage: number
  status: 'ready' | 'partial' | 'growing'
  updated_at?: string | null
  distribution: RagDistributionItem[]
  samples: RagDocumentSample[]
}

export interface RagOverview {
  status: 'ready' | 'partial'
  storage: string
  retrieval_mode: string
  embedding_provider: string
  generated_at?: string | null
  quality_status: string
  total_documents: number
  total_chunks: number
  total_tokens: number
  paired_problem_cases: number
  libraries: RagLibraryOverview[]
}
