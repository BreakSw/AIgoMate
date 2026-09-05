<script setup lang="ts">
import { computed } from 'vue'
import type { ContextSnapshot } from '../types'

const props = withDefaults(defineProps<{ snapshot: ContextSnapshot; open?: boolean }>(), { open: false })

const usedPercent = computed(() => Math.min(100, Math.round(props.snapshot.window.usage_ratio * 100)))
const stateLabel = computed(() => ({
  normal: '充足',
  soft_limit: '准备压缩',
  hard_limit: '强制压缩',
  overflow: '已超限',
}[props.snapshot.window.state]))
const compressionLabel = computed(() => ({
  not_required: '试装成功 · 本轮 0 次压缩',
  reused_checkpoint: '复用检查点 · 本轮 0 次压缩',
  preflight_budget_exceeded: '历史试装超限 · 已统一压缩 1 次',
  turn_commit_budget_exceeded: '写入本轮意图后超限 · 已统一压缩 1 次',
}[props.snapshot.window.compression_trigger_reason || (
  props.snapshot.window.compression_triggered ? 'preflight_budget_exceeded' : 'not_required'
)]))
const displayGoal = computed(() => (
  props.snapshot.memory.current_goal
  || props.snapshot.turn_context?.user_goal
  || '本轮尚未识别出明确目标'
))
const displayWorkingMemory = computed(() => (
  props.snapshot.memory.working_memory.length
    ? props.snapshot.memory.working_memory
    : props.snapshot.turn_context
      ? [`本轮意图：${props.snapshot.turn_context.primary_intent}`, `当前请求：${props.snapshot.turn_context.normalized_request}`]
      : []
))
const displayConstraints = computed(() => (
  props.snapshot.memory.pinned_constraints.length
    ? props.snapshot.memory.pinned_constraints
    : props.snapshot.turn_context?.constraints || []
))

function formatTokens(value: number) {
  return new Intl.NumberFormat('zh-CN').format(value)
}

const collectionLabels: Record<string, string> = {
  algorithm_concepts: '算法概念库',
  problem_bank: '题库',
  code_cases: '代码案例库',
  web_search: '网页搜索',
}

const agentLabels: Record<string, string> = {
  tutoring_agent: '算法教学 Agent',
  problem_solving_agent: '问题求解 Agent',
  code_analysis_agent: '代码分析 Agent',
  problem_structuring_agent: '题面结构化 Agent',
  strategy_agent: '算法策略 Agent',
  solution_review_agent: '方案评审 Agent',
  implementation_agent: '代码实现 Agent',
  verification_agent: '验证 Agent',
  code_test_generation_agent: '算法测试生成 Agent',
  learning_planning_agent: '学习规划 Agent',
  conversation_agent: '通用对话 Agent',
  clarification_agent: '澄清 Agent',
}

function collectionLabel(value: string) {
  return collectionLabels[value] || value
}

function agentLabel(value: string) {
  return agentLabels[value] || value
}

function reflectionRounds(value: string) {
  const match = value.match(/\+(?:reflection|test-reflection):(\d+)/)
  return match ? Number(match[1]) : 0
}

function callLabel(value: string) {
  return value.replace(/\+(?:reflection|test-reflection):\d+/, '')
}

function difficultyLabel(value: string) {
  return ({ easy: '简单', medium: '中等', hard: '困难', unknown: '待估计' } as Record<string, string>)[value] || value
}

function reviewDate(value?: string | null) {
  return value ? new Date(value).toLocaleDateString('zh-CN') : '待记录'
}
</script>

<template>
  <details class="context-inspector" :open="open">
    <summary>
      <span class="context-symbol">◎</span>
      <span><strong>上下文审查</strong><small>记忆 · 压缩 · 窗口预算</small></span>
      <span class="window-remaining">剩余 {{ formatTokens(snapshot.window.remaining_tokens) }} tokens</span>
    </summary>

    <div class="context-body">
      <section class="window-overview">
        <div class="window-heading">
          <span>活跃窗口 {{ formatTokens(snapshot.window.estimated_input_tokens) }} / {{ formatTokens(snapshot.window.window_size_tokens) }}</span>
          <strong>{{ usedPercent }}% · {{ stateLabel }}</strong>
        </div>
        <div class="window-meter"><i :style="{ width: `${usedPercent}%` }"></i></div>
        <div class="window-stats">
          <span>完整试装 <b>{{ formatTokens(snapshot.window.candidate_input_tokens ?? snapshot.window.estimated_input_tokens) }}</b></span>
          <span>原始历史 <b>{{ formatTokens(snapshot.window.raw_history_tokens) }}</b></span>
          <span>压缩上下文 <b>{{ formatTokens(snapshot.window.compressed_context_tokens) }}</b></span>
          <span>检查点后新增 <b>{{ formatTokens(snapshot.window.recent_messages_tokens) }}</b></span>
          <span>本轮意图 <b>{{ formatTokens(snapshot.window.turn_metadata_tokens ?? 0) }}</b></span>
          <span>输出预留 <b>{{ formatTokens(snapshot.window.output_reserved_tokens) }}</b></span>
        </div>
      </section>

      <section v-if="snapshot.input_organization" class="organizer-card">
        <header>
          <span><small>FIRST LAYER · INPUT ORGANIZER</small><strong>用户输入整理</strong></span>
          <b>{{ snapshot.input_organization.input_shape }}</b>
        </header>
        <div class="organizer-content">
          <small>整理后内容</small>
          <pre>{{ snapshot.input_organization.organized_input }}</pre>
        </div>
        <p>{{ snapshot.input_organization.organization_summary }}</p>
        <small class="rewrite-provider">{{ snapshot.input_organization.organizer_model }} · {{ snapshot.input_organization.organizer_provider }} · 未做意图拆解</small>
      </section>

      <section v-if="snapshot.input_rewrite" class="rewrite-card">
        <header>
          <span><small>INPUT REWRITE AGENT</small><strong>模型输入改写</strong></span>
          <b>{{ snapshot.input_rewrite.input_type }}<template v-if="snapshot.input_rewrite.programming_language"> · {{ snapshot.input_rewrite.programming_language }}</template></b>
        </header>
        <div class="rewrite-row"><small>原意格式化</small><p>{{ snapshot.input_rewrite.formatted_input }}</p></div>
        <div class="rewrite-row"><small>识别出的明确要求</small><p>{{ snapshot.input_rewrite.explicit_request || '未提供明确操作要求' }}</p></div>
        <div class="rewrite-meta">
          <span v-for="item in snapshot.input_rewrite.requested_operations" :key="item">{{ item }}</span>
          <span v-for="item in snapshot.input_rewrite.constraints" :key="item">{{ item }}</span>
          <span v-for="item in snapshot.input_rewrite.ambiguities" :key="item" class="ambiguity">待确认：{{ item }}</span>
        </div>
        <small class="rewrite-provider">{{ snapshot.input_rewrite.rewrite_model }} · {{ snapshot.input_rewrite.rewrite_provider }}</small>
      </section>

      <section v-if="snapshot.agent_execution" class="intent-card">
        <header>
          <span><small>INTENT RECOGNITION</small><strong>意图识别结果</strong></span>
          <b>{{ snapshot.agent_execution.task_spec.primary_intent }} · {{ Math.round(snapshot.agent_execution.task_spec.confidence * 100) }}%</b>
        </header>
        <p>{{ snapshot.agent_execution.task_spec.recognition_summary }}</p>
        <div class="intent-summary-grid">
          <span><small>用户目标</small>{{ snapshot.agent_execution.task_spec.user_goal }}</span>
          <span><small>响应方式</small>{{ snapshot.agent_execution.task_spec.response_mode }}</span>
          <span><small>协助级别</small>{{ snapshot.agent_execution.task_spec.delivery.assistance_level }}</span>
          <span><small>能力路由</small>{{ snapshot.agent_execution.task_spec.routing.primary_capability }}</span>
        </div>
      </section>

      <section v-if="snapshot.learning_profile?.active" class="learning-profile-card">
        <header>
          <span><small>ADAPTIVE LEARNING MODEL</small><strong>个性化学习画像</strong></span>
          <b>{{ snapshot.learning_profile.updated ? '本轮已更新' : '本轮已读取' }}</b>
        </header>
        <p>{{ snapshot.learning_profile.summary }}</p>
        <div class="learning-algorithms">
          <span v-for="item in snapshot.learning_profile.algorithms" :key="item">{{ item }}</span>
        </div>
        <div class="learning-summary">
          <span><small>IRT-1PL 能力值</small><strong>θ = {{ snapshot.learning_profile.ability_theta.toFixed(2) }}</strong></span>
          <span><small>建议题目难度</small><strong>{{ difficultyLabel(snapshot.learning_profile.target_difficulty) }}</strong></span>
          <span><small>优先知识点</small><strong>{{ snapshot.learning_profile.recommended_concepts.join('、') || '等待更多记录' }}</strong></span>
        </div>
        <div v-if="snapshot.learning_profile.concepts.length" class="learning-concepts">
          <div v-for="item in snapshot.learning_profile.concepts" :key="item.concept" class="learning-concept-row">
            <span><strong>{{ item.concept }}</strong><small>独立成功 {{ item.correct_attempts }} · 学习事件 {{ item.attempts }}</small></span>
            <span class="mastery-value">BKT {{ Math.round(item.mastery_probability * 100) }}%</span>
            <i><b :style="{ width: `${Math.round(item.mastery_probability * 100)}%` }"></b></i>
            <small>FSRS-style：{{ reviewDate(item.next_review_at) }} 复习</small>
          </div>
        </div>
        <div v-if="snapshot.learning_profile.updates.length" class="learning-updates">
          <strong>本轮模型更新轨迹</strong>
          <span v-for="item in snapshot.learning_profile.updates" :key="`${item.concept}:${item.outcome}`">
            {{ item.concept }} · BKT {{ Math.round(item.mastery_before * 100) }}% → {{ Math.round(item.mastery_after * 100) }}% · IRT θ {{ item.ability_before.toFixed(2) }} → {{ item.ability_after.toFixed(2) }} · FSRS {{ item.fsrs_rating }}
          </span>
        </div>
      </section>

      <section v-if="snapshot.agent_execution" class="orchestration-card">
        <header>
          <span><small>HEAD ORCHESTRATOR</small><strong>首脑智能体决策</strong></span>
          <b>{{ agentLabel(snapshot.agent_execution.coordinator_plan.selected_agent) }}</b>
        </header>
        <p>{{ snapshot.agent_execution.coordinator_plan.objective }}</p>
        <div class="orchestration-meta">
          <span>{{ snapshot.agent_execution.coordinator_plan.grounding_policy }}</span>
          <span>{{ snapshot.agent_execution.coordinator_plan.memory_selection.reason }}</span>
        </div>
        <ol v-if="snapshot.agent_execution.coordinator_plan.planned_steps?.length" class="planned-steps">
          <li v-for="step in (snapshot.agent_execution.coordinator_plan.planned_steps || [])" :key="step">{{ step }}</li>
        </ol>
        <div v-if="snapshot.agent_execution.model_call_trace?.length" class="reflection-trace">
          <strong>Agent 协议校验与 Reflection</strong>
          <span
            v-for="item in (snapshot.agent_execution.model_call_trace || [])"
            :key="item"
            :class="{ reflected: reflectionRounds(item) > 0 }"
          >
            {{ callLabel(item) }} · {{ reflectionRounds(item) > 0 ? `已修正 ${reflectionRounds(item)} 轮` : '首次通过' }}
          </span>
        </div>
        <div class="rag-plan">
          <div class="rag-plan-heading">
            <strong>知识与工具调用</strong>
            <small v-if="!snapshot.agent_execution.coordinator_plan.rag_queries?.length && !snapshot.agent_execution.coordinator_plan.web_search_queries?.length">本轮无需检索</small>
          </div>
          <div v-for="query in (snapshot.agent_execution.coordinator_plan.rag_queries || [])" :key="query.collection" class="rag-query-row">
            <span>{{ collectionLabel(query.collection) }}</span>
            <p>{{ query.query }}</p>
            <small>{{ query.reason }}</small>
          </div>
          <div v-for="query in (snapshot.agent_execution.coordinator_plan.web_search_queries || [])" :key="query" class="rag-query-row web-query-row">
            <span>网页搜索</span>
            <p>{{ query }}</p>
            <small>由首脑智能体按实施或时效信息需求动态调用</small>
          </div>
        </div>
        <div v-if="snapshot.agent_execution.memory_updates?.length" class="memory-write-list">
          <strong>本轮持久记忆更新</strong>
          <span v-for="item in (snapshot.agent_execution.memory_updates || [])" :key="`${item.kind}:${item.content}`">{{ item.kind }} · {{ item.content }}</span>
        </div>
      </section>

      <section v-if="snapshot.agent_execution?.code_execution_reports?.length" class="execution-card">
        <header>
          <span><small>CODE EXECUTION TOOL</small><strong>Judge0 真实编译运行</strong></span>
          <b>{{ snapshot.agent_execution.code_execution_reports.at(-1)?.overall_status }}</b>
        </header>
        <div v-for="report in snapshot.agent_execution.code_execution_reports" :key="report.source_code_hash" class="execution-report">
          <div class="execution-summary">
            <span><small>语言</small><strong>{{ report.language }}</strong></span>
            <span><small>Verdict</small><strong>{{ report.verdict }}</strong></span>
            <span><small>用例</small><strong>{{ report.passed_tests }}/{{ report.total_tests }}</strong></span>
            <span><small>耗时 / 内存</small><strong>{{ report.time_seconds ?? '—' }}s / {{ report.memory_kb ?? '—' }} KB</strong></span>
          </div>
          <p><strong>覆盖：</strong>{{ report.test_categories.join('、') || '未生成' }}</p>
          <p><strong>Oracle：</strong>{{ report.oracle_strategy || '未提供' }}</p>
          <p><strong>Reflection：</strong>语义修订 {{ report.semantic_reflection_rounds }} 轮<template v-if="report.test_plan_review"> · {{ report.test_plan_review }}</template><template v-if="report.test_plan_review_confidence != null">（置信度 {{ Math.round(report.test_plan_review_confidence * 100) }}%）</template></p>
          <small class="source-hash">源码 SHA-256 · {{ report.source_code_hash }}</small>
          <pre v-if="report.failure_reason || report.compile_output || report.stderr || report.stdout">{{ report.failure_reason || report.compile_output || report.stderr || report.stdout }}</pre>
        </div>
      </section>

      <section v-if="snapshot.agent_execution" class="evidence-card">
        <header>
          <span><small>GROUNDING EVIDENCE</small><strong>实际召回证据</strong></span>
          <b>{{ snapshot.agent_execution.rag_evidence.length }} 条</b>
        </header>
        <p v-if="!snapshot.agent_execution.rag_evidence.length" class="empty-evidence">首脑判断本轮不需要知识库，或没有检索到满足条件的证据。</p>
        <div v-for="item in snapshot.agent_execution.rag_evidence" :key="item.evidence_id" class="evidence-row">
          <span>{{ item.evidence_id }}</span>
          <div>
            <a v-if="item.source_url" :href="item.source_url" target="_blank" rel="noreferrer">{{ item.title }}</a>
            <strong v-else>{{ item.title }}</strong>
            <small>{{ collectionLabel(item.collection) }} · 匹配分 {{ item.score.toFixed(2) }}</small>
          </div>
        </div>
      </section>

      <section v-if="snapshot.agent_execution" class="polish-card">
        <header>
          <span><small>LANGUAGE POLISH AGENT</small><strong>最终表达校验</strong></span>
          <b>{{ snapshot.agent_execution.polish_result.added_factual_claims ? '事实边界异常' : '未添加事实' }}</b>
        </header>
        <div class="polish-meta">
          <span v-for="item in snapshot.agent_execution.polish_result.style_changes" :key="item">{{ item }}</span>
          <span v-if="!snapshot.agent_execution.polish_result.style_changes.length">仅做必要语言整理</span>
        </div>
        <details class="draft-compare">
          <summary>查看润色前草稿</summary>
          <p>{{ snapshot.agent_execution.work_result.draft_answer }}</p>
        </details>
      </section>

      <section v-if="snapshot.turn_context" class="turn-context-card">
        <header>
          <span><small>TURN CONTEXT</small><strong>本轮意图已写入上下文</strong></span>
          <b>{{ snapshot.turn_context.primary_intent }}</b>
        </header>
        <p>{{ snapshot.turn_context.normalized_request }}</p>
        <div class="turn-meta">
          <span>{{ snapshot.turn_context.response_mode }}</span>
          <span>{{ snapshot.turn_context.primary_capability }}</span>
          <span v-for="item in snapshot.turn_context.constraints" :key="item">{{ item }}</span>
        </div>
      </section>

      <div class="context-grid">
        <section class="context-card memory-card">
          <header><span>MEMORY</span><strong>当前记忆</strong></header>
          <div class="memory-goal"><small>当前目标</small>{{ displayGoal }}</div>
          <div class="memory-group">
            <small>工作记忆</small>
            <ul><li v-for="item in displayWorkingMemory" :key="item">{{ item }}</li><li v-if="!displayWorkingMemory.length">无</li></ul>
          </div>
          <div class="memory-group">
            <small>用户偏好与长期记忆</small>
            <ul>
              <li v-for="item in [...snapshot.memory.user_preferences, ...snapshot.memory.long_term_memory]" :key="item">{{ item }}</li>
              <li v-if="!snapshot.memory.user_preferences.length && !snapshot.memory.long_term_memory.length">无</li>
            </ul>
          </div>
          <div class="memory-group">
            <small>固定约束</small>
            <ul><li v-for="item in displayConstraints" :key="item">{{ item }}</li><li v-if="!displayConstraints.length">无</li></ul>
          </div>
        </section>

        <section class="context-card compression-card">
          <header><span>COMPRESSION</span><strong>压缩后内容</strong></header>
          <p>{{ snapshot.compressed_context.summary }}</p>
          <div class="compression-meta">
            <span>{{ compressionLabel }}</span>
            <span v-if="snapshot.window.checkpoint_message_count">检查点覆盖 {{ snapshot.window.checkpoint_message_count }} 条 · 新增 {{ snapshot.window.new_messages_since_checkpoint ?? 0 }} 条</span>
            <span v-else>原始上下文直接保留</span>
          </div>
          <div v-if="snapshot.compressed_context.topics.length" class="topic-list">
            <span v-for="topic in snapshot.compressed_context.topics" :key="topic">{{ topic }}</span>
          </div>
          <small class="compression-provider">{{ snapshot.compressed_context.compression_model || '本地策略' }} · {{ snapshot.compressed_context.compression_provider }}</small>
        </section>
      </div>
    </div>
  </details>
</template>

<style scoped>
.context-inspector { width: min(760px, 100%); margin-top: 10px; overflow: hidden; border: 1px solid #d7dbe2; border-radius: 14px; background: #f7f7f3; color: #31405b; }
.context-inspector summary { display: flex; align-items: center; gap: 10px; padding: 11px 13px; cursor: pointer; list-style: none; background: linear-gradient(100deg, #eef3fa, #f8f5ec); }
.context-inspector summary::-webkit-details-marker { display: none; }
.context-symbol { display: grid; place-items: center; width: 27px; height: 27px; border-radius: 9px; color: #fff; background: #315e9d; }
.context-inspector summary > span:nth-child(2) { display: grid; gap: 1px; }
.context-inspector summary strong { font-size: 12px; }
.context-inspector summary small { color: #8791a1; font-size: 9px; }
.window-remaining { margin-left: auto; padding: 4px 8px; border: 1px solid #d7d3c7; border-radius: 999px; color: #6d675a; background: #fffdf7; font-size: 10px; }
.context-body { display: grid; gap: 10px; padding: 12px; border-top: 1px solid #dde0e5; }
.window-overview { padding: 11px; border-radius: 10px; background: #fff; }
.window-heading { display: flex; justify-content: space-between; color: #5e697b; font-size: 10px; }
.window-heading strong { color: #315e9d; }
.window-meter { height: 5px; margin: 8px 0; overflow: hidden; border-radius: 99px; background: #e4e7eb; }
.window-meter i { display: block; height: 100%; border-radius: inherit; background: linear-gradient(90deg, #3c71b3, #d6a35a); }
.window-stats { display: grid; grid-template-columns: repeat(6, 1fr); gap: 5px; color: #8a909b; font-size: 9px; }
.window-stats span { display: grid; gap: 2px; }
.window-stats b { color: #4a5669; font-size: 11px; }
.organizer-card { padding: 12px; border: 1px solid #ddd9cf; border-left: 3px solid #d09b55; border-radius: 10px; background: linear-gradient(105deg, #fffdf8, #fff); }
.organizer-card header { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
.organizer-card header > span { display: grid; gap: 2px; }
.organizer-card header small { color: #ad7b3f; font-size: 8px; letter-spacing: .12em; }
.organizer-card header strong { color: #34445c; font-size: 11px; }
.organizer-card header b { padding: 4px 7px; border-radius: 99px; color: #76592f; background: #f6ead7; font-size: 8px; }
.organizer-content { margin-top: 9px; }
.organizer-content > small { color: #8c744f; font-size: 8px; letter-spacing: .06em; }
.organizer-content pre { max-height: 220px; margin: 4px 0 0; padding: 9px; overflow: auto; border-radius: 8px; color: #42516a; background: #f7f7f3; font: 9px/1.6 ui-monospace, SFMono-Regular, Consolas, monospace; white-space: pre-wrap; word-break: break-word; }
.organizer-card > p { margin: 8px 0 0; color: #787063; font-size: 9px; }
.rewrite-card { padding: 12px; border: 1px solid #d6dee9; border-left: 3px solid #315e9d; border-radius: 10px; background: linear-gradient(105deg, #f7faff, #fff); }
.rewrite-card header { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
.rewrite-card header > span { display: grid; gap: 2px; }
.rewrite-card header small { color: #527fb8; font-size: 8px; letter-spacing: .12em; }
.rewrite-card header strong { color: #34445c; font-size: 11px; }
.rewrite-card header b { padding: 4px 7px; border-radius: 99px; color: #315e9d; background: #e7eff9; font-size: 8px; }
.rewrite-row { margin-top: 9px; }
.rewrite-row small { color: #8c744f; font-size: 8px; letter-spacing: .06em; }
.rewrite-row p { margin: 3px 0 0; color: #4b586b; font-size: 10px; line-height: 1.6; }
.rewrite-meta { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 8px; }
.rewrite-meta span { padding: 3px 6px; border-radius: 99px; color: #526c92; background: #edf2f8; font-size: 8px; }
.rewrite-meta span.ambiguity { color: #8a6539; background: #f7eedf; }
.rewrite-provider { display: block; margin-top: 8px; color: #a1a5ad; font-size: 8px; }
.intent-card, .learning-profile-card, .orchestration-card, .execution-card, .evidence-card, .polish-card { padding: 12px; border: 1px solid #dce1e8; border-radius: 10px; background: #fff; }
.intent-card header, .orchestration-card header, .evidence-card header, .polish-card header { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
.intent-card header > span, .orchestration-card header > span, .evidence-card header > span, .polish-card header > span { display: grid; gap: 2px; }
.intent-card header small, .orchestration-card header small, .evidence-card header small, .polish-card header small { color: #6b89b4; font-size: 8px; letter-spacing: .12em; }
.intent-card header strong, .orchestration-card header strong, .evidence-card header strong, .polish-card header strong { color: #34445c; font-size: 11px; }
.intent-card header b, .orchestration-card header b, .evidence-card header b, .polish-card header b { padding: 4px 7px; border-radius: 99px; color: #315e9d; background: #e9f0f9; font-size: 8px; }
.intent-card { border-left: 3px solid #466f9f; background: linear-gradient(105deg, #f7faff, #fff); }
.intent-card > p, .orchestration-card > p { margin: 8px 0 0; color: #4b586b; font-size: 10px; line-height: 1.6; }
.intent-summary-grid { display: grid; grid-template-columns: 1.4fr 1fr 1fr 1fr; gap: 6px; margin-top: 10px; }
.intent-summary-grid span { color: #4b586b; font-size: 9px; line-height: 1.45; }
.intent-summary-grid small { display: block; margin-bottom: 2px; color: #9b8460; font-size: 8px; }
.learning-profile-card { border-left: 3px solid #5e8c79; background: linear-gradient(110deg, #f2f8f6, #fffaf1); }
.learning-profile-card header { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
.learning-profile-card header > span { display: grid; gap: 2px; }
.learning-profile-card header small { color: #4d806d; font-size: 8px; letter-spacing: .12em; }
.learning-profile-card header strong { color: #34445c; font-size: 11px; }
.learning-profile-card header b { padding: 4px 7px; border-radius: 99px; color: #76592f; background: #f5ead7; font-size: 8px; }
.learning-profile-card > p { margin: 8px 0 0; color: #4b586b; font-size: 10px; line-height: 1.6; }
.learning-algorithms { display: flex; gap: 5px; margin-top: 9px; }
.learning-algorithms span { padding: 3px 7px; border: 1px solid #c8ded5; border-radius: 99px; color: #477463; background: #edf6f2; font-size: 8px; }
.learning-summary { display: grid; grid-template-columns: .8fr .8fr 1.6fr; gap: 7px; margin-top: 10px; }
.learning-summary span { padding: 8px; border-radius: 8px; background: rgba(255, 255, 255, .8); }
.learning-summary small { display: block; margin-bottom: 3px; color: #9b8460; font-size: 8px; }
.learning-summary strong { color: #405169; font-size: 10px; }
.learning-concepts { display: grid; gap: 7px; margin-top: 10px; }
.learning-concept-row { display: grid; grid-template-columns: 1fr auto; gap: 5px 8px; padding: 8px; border-radius: 8px; background: #fff; }
.learning-concept-row > span:first-child { display: grid; gap: 2px; color: #405169; font-size: 9px; }
.learning-concept-row small { color: #8d959f; font-size: 8px; }
.mastery-value { color: #477463; font-size: 9px; font-weight: 700; }
.learning-concept-row i { height: 4px; overflow: hidden; border-radius: 99px; background: #e4ebe8; }
.learning-concept-row i b { display: block; height: 100%; border-radius: inherit; background: linear-gradient(90deg, #4775aa, #63a185, #d0a158); }
.learning-updates { display: grid; gap: 5px; margin-top: 10px; padding-top: 9px; border-top: 1px solid #dfe9e4; }
.learning-updates strong { color: #526556; font-size: 9px; }
.learning-updates span { color: #647084; font-size: 8px; line-height: 1.5; }
.orchestration-card { border-left: 3px solid #d09b55; background: linear-gradient(105deg, #fffdf8, #fff); }
.orchestration-meta, .polish-meta { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 9px; }
.orchestration-meta span, .polish-meta span { padding: 3px 6px; border-radius: 99px; color: #6a5d49; background: #f4eddf; font-size: 8px; }
.planned-steps { margin: 9px 0 0; padding-left: 19px; color: #536073; font-size: 9px; line-height: 1.6; }
.reflection-trace { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 10px; padding-top: 9px; border-top: 1px solid #ebe8e0; }
.reflection-trace strong { width: 100%; color: #59667a; font-size: 9px; }
.reflection-trace span { padding: 4px 7px; border-radius: 7px; color: #607065; background: #edf3ee; font-size: 8px; }
.reflection-trace span.reflected { color: #8b5c2f; background: #fff0dc; box-shadow: inset 0 0 0 1px #edc38f; }
.rag-plan { margin-top: 10px; padding-top: 9px; border-top: 1px solid #ebe8e0; }
.rag-plan-heading { display: flex; align-items: center; justify-content: space-between; color: #59667a; font-size: 9px; }
.rag-plan-heading small { color: #9299a5; }
.rag-query-row { display: grid; grid-template-columns: 80px 1fr; gap: 3px 8px; margin-top: 7px; align-items: baseline; }
.rag-query-row > span { grid-row: span 2; padding: 3px 6px; border-radius: 6px; color: #315e9d; background: #eaf1f9; font-size: 8px; text-align: center; }
.rag-query-row p { margin: 0; color: #48566b; font-size: 9px; }
.rag-query-row small { color: #9299a5; font-size: 8px; }
.web-query-row > span { color: #6d5a8b; background: #eee9f5; }
.memory-write-list { display: grid; gap: 5px; margin-top: 10px; padding-top: 9px; border-top: 1px solid #ebe8e0; }
.memory-write-list strong { color: #59667a; font-size: 9px; }
.memory-write-list span { padding: 5px 7px; border-radius: 7px; color: #526556; background: #edf4ef; font-size: 8px; }
.execution-card { border-left: 3px solid #5e8c79; background: linear-gradient(105deg, #f4faf7, #fffaf2); }
.execution-card header { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
.execution-card header > span { display: grid; gap: 2px; }
.execution-card header small { color: #4f806f; font-size: 8px; letter-spacing: .12em; }
.execution-card header strong { color: #34445c; font-size: 11px; }
.execution-card header b { padding: 4px 7px; border-radius: 99px; color: #315e9d; background: #e9f0f9; font-size: 8px; }
.execution-report { margin-top: 9px; padding-top: 9px; border-top: 1px solid #dfe9e4; }
.execution-summary { display: grid; grid-template-columns: .7fr 1fr .7fr 1.4fr; gap: 6px; }
.execution-summary span { display: grid; gap: 2px; padding: 7px; border-radius: 7px; background: rgba(255,255,255,.84); }
.execution-summary small, .source-hash { color: #87939a; font-size: 8px; }
.execution-summary strong { color: #405169; font-size: 9px; }
.execution-report p { margin: 7px 0 0; color: #536073; font-size: 9px; line-height: 1.5; }
.source-hash { display: block; margin-top: 7px; overflow-wrap: anywhere; }
.execution-report pre { max-height: 160px; margin: 7px 0 0; padding: 8px; overflow: auto; border-radius: 7px; color: #7b4935; background: #fff3e8; font: 8px/1.5 ui-monospace, SFMono-Regular, Consolas, monospace; white-space: pre-wrap; }
.evidence-card { background: #f8fafc; }
.empty-evidence { margin: 9px 0 0; color: #8c939e; font-size: 9px; }
.evidence-row { display: flex; gap: 8px; margin-top: 8px; padding-top: 8px; border-top: 1px solid #e5e9ef; }
.evidence-row > span { display: grid; place-items: center; width: 25px; height: 21px; border-radius: 6px; color: #fff; background: #476f9f; font-size: 8px; }
.evidence-row > div { display: grid; gap: 2px; min-width: 0; }
.evidence-row a, .evidence-row strong { overflow: hidden; color: #3d5f8c; font-size: 9px; text-overflow: ellipsis; white-space: nowrap; }
.evidence-row small { color: #9299a5; font-size: 8px; }
.polish-card { border-left: 3px solid #5e8c79; background: linear-gradient(105deg, #f7fbf9, #fff); }
.draft-compare { margin-top: 9px; color: #647084; font-size: 9px; }
.draft-compare summary { display: block; padding: 0; color: #547060; background: transparent; font-size: 8px; }
.draft-compare p { margin: 7px 0 0; padding: 8px; border-radius: 7px; background: #f3f6f4; white-space: pre-wrap; line-height: 1.6; }
.turn-context-card { padding: 12px; border: 1px solid #d9dfdf; border-left: 3px solid #d09b55; border-radius: 10px; background: linear-gradient(105deg, #fff, #fbf7ee); }
.turn-context-card header { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
.turn-context-card header > span { display: grid; gap: 2px; }
.turn-context-card header small { color: #b07a39; font-size: 8px; letter-spacing: .12em; }
.turn-context-card header strong { color: #34445c; font-size: 11px; }
.turn-context-card header b { padding: 4px 7px; border-radius: 99px; color: #315e9d; background: #eaf1f9; font-size: 8px; }
.turn-context-card p { margin: 8px 0 0; color: #526071; font-size: 10px; line-height: 1.6; }
.turn-meta { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 8px; }
.turn-meta span { padding: 3px 6px; border-radius: 99px; color: #756243; background: #f3ebdc; font-size: 8px; }
.context-grid { display: grid; grid-template-columns: 1fr 1.25fr; gap: 10px; }
.context-card { padding: 12px; border: 1px solid #e0e2e5; border-radius: 10px; background: #fff; }
.context-card header { display: flex; align-items: baseline; gap: 7px; margin-bottom: 10px; }
.context-card header span { color: #6f91c4; font-size: 8px; letter-spacing: .12em; }
.context-card header strong { font-size: 11px; }
.memory-goal, .memory-group { margin-top: 8px; color: #445168; font-size: 10px; line-height: 1.55; }
.memory-goal small, .memory-group small { display: block; margin-bottom: 3px; color: #9b8460; font-size: 8px; letter-spacing: .08em; }
.memory-group ul { margin: 0; padding-left: 15px; }
.compression-card p { margin: 0; color: #4b5668; font-size: 10px; line-height: 1.7; white-space: pre-wrap; }
.compression-meta { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 10px; }
.compression-meta span, .topic-list span { padding: 3px 6px; border-radius: 99px; color: #526c92; background: #edf2f8; font-size: 8px; }
.topic-list { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 7px; }
.compression-provider { display: block; margin-top: 10px; color: #a1a5ad; font-size: 8px; }
@media (max-width: 760px) { .context-grid { grid-template-columns: 1fr; } .window-stats { grid-template-columns: 1fr 1fr; } .window-remaining { display: none; } .intent-summary-grid, .learning-summary { grid-template-columns: 1fr 1fr; } }
</style>
