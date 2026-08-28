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

      <section v-if="snapshot.input_rewrite" class="rewrite-card">
        <header>
          <span><small>INPUT REWRITE AGENT</small><strong>模型输入改写</strong></span>
          <b>{{ snapshot.input_rewrite.input_type }}<template v-if="snapshot.input_rewrite.programming_language"> · {{ snapshot.input_rewrite.programming_language }}</template></b>
        </header>
        <div class="rewrite-row"><small>原意格式化</small><p>{{ snapshot.input_rewrite.formatted_input }}</p></div>
        <div class="rewrite-row"><small>识别出的明确要求</small><p>{{ snapshot.input_rewrite.explicit_request }}</p></div>
        <div class="rewrite-meta">
          <span v-for="item in snapshot.input_rewrite.requested_operations" :key="item">{{ item }}</span>
          <span v-for="item in snapshot.input_rewrite.constraints" :key="item">{{ item }}</span>
          <span v-for="item in snapshot.input_rewrite.ambiguities" :key="item" class="ambiguity">待确认：{{ item }}</span>
        </div>
        <small class="rewrite-provider">{{ snapshot.input_rewrite.rewrite_model }} · {{ snapshot.input_rewrite.rewrite_provider }}</small>
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
@media (max-width: 760px) { .context-grid { grid-template-columns: 1fr; } .window-stats { grid-template-columns: 1fr 1fr; } .window-remaining { display: none; } }
</style>
