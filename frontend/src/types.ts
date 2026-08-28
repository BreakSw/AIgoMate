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

export interface InputRewriteResult {
  schema_version: '1.0'
  input_type: 'text' | 'code' | 'mixed'
  formatted_input: string
  explicit_request: string
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

export interface ContextSnapshot {
  schema_version: '1.0'
  memory: MemorySnapshot
  compressed_context: CompressedContext
  window: ContextWindowStatus
  turn_context?: TurnContext | null
  input_rewrite?: InputRewriteResult | null
  checkpoint_memory?: MemorySnapshot | null
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
  retryCount?: number
  maxRetries?: number
}
