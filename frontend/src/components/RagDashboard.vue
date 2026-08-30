<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { chatApi } from '../api'
import type { RagDocumentSample, RagLibraryKey, RagLibraryOverview, RagOverview } from '../types'

const overview = ref<RagOverview | null>(null)
const loading = ref(true)
const error = ref('')
const selectedKey = ref<RagLibraryKey>('algorithm_concepts')

const libraryVisuals: Record<RagLibraryKey, { icon: string; className: string; summary: string }> = {
  algorithm_concepts: { icon: '∑', className: 'concepts', summary: '概念、方法与学习路线' },
  problem_bank: { icon: '⌁', className: 'problems', summary: '题目描述与约束条件' },
  code_cases: { icon: '</>', className: 'cases', summary: '高赞解析与代码模板' },
  user_memory: { icon: '◇', className: 'memory', summary: '目标、偏好与学习轨迹' },
}

const selectedLibrary = computed(() =>
  overview.value?.libraries.find((library) => library.key === selectedKey.value) ?? null,
)
const maxDistribution = computed(() =>
  Math.max(1, ...(selectedLibrary.value?.distribution.map((item) => item.count) ?? [])),
)

onMounted(loadOverview)

async function loadOverview() {
  loading.value = true
  error.value = ''
  try {
    overview.value = await chatApi.getRagOverview()
    if (!overview.value.libraries.some((item) => item.key === selectedKey.value)) {
      selectedKey.value = overview.value.libraries[0]?.key ?? 'algorithm_concepts'
    }
  } catch {
    error.value = '暂时无法读取 RAG 库状态，请确认 Agent Service 已启动。'
  } finally {
    loading.value = false
  }
}

function selectLibrary(key: RagLibraryKey) {
  selectedKey.value = key
}

function formatNumber(value: number) {
  return new Intl.NumberFormat('zh-CN').format(value)
}

function formatDate(value?: string | null) {
  if (!value) return '尚无更新时间'
  return new Intl.DateTimeFormat('zh-CN', {
    month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit',
  }).format(new Date(value))
}

function normalizedLabel(value: string) {
  const labels: Record<string, string> = {
    general: '综合路线', array: '数组', string: '字符串', 'linked-list': '链表',
    'hash-table': '哈希表', 'binary-tree': '二叉树', backtracking: '回溯',
    greedy: '贪心', 'dynamic-programming': '动态规划', graph: '图论',
    Easy: '简单', Medium: '中等', Hard: '困难', Unknown: '未标注',
    learned_fact: '学习事实', long_term_goal: '长期目标', preference: '用户偏好',
    constraint: '固定约束', unfinished_task: '未完成任务', other: '其他专题',
  }
  return labels[value] ?? value
}

function sampleMeta(sample: RagDocumentSample, library: RagLibraryOverview) {
  const metadata = sample.metadata
  if (library.key === 'algorithm_concepts') {
    return normalizedLabel(String(metadata.category ?? 'general'))
  }
  if (library.key === 'problem_bank') {
    return `LeetCode ${metadata.problem_id ?? '—'} · ${normalizedLabel(String(metadata.difficulty ?? 'Unknown'))}`
  }
  if (library.key === 'code_cases') {
    return `${metadata.author ?? '优质作者'} · ${formatNumber(Number(metadata.likes ?? 0))} 赞 · ${formatNumber(Number(metadata.views ?? 0))} 浏览`
  }
  return `${normalizedLabel(String(metadata.kind ?? 'other'))} · 重要度 ${Number(metadata.importance ?? 0).toFixed(2)}`
}
</script>

<template>
  <main class="rag-workspace">
    <header class="rag-topbar">
      <div>
        <span class="rag-eyebrow">KNOWLEDGE INFRASTRUCTURE</span>
        <h1>RAG 知识库可视化</h1>
        <p>查看智能体可检索的知识来源、向量覆盖率与私有记忆增长情况。</p>
      </div>
      <div class="rag-top-actions">
        <span v-if="overview" class="sync-state"><i></i>{{ overview.status === 'ready' ? '向量库可用' : '部分可用' }}</span>
        <button type="button" :disabled="loading" @click="loadOverview">↻ 刷新数据</button>
      </div>
    </header>

    <section v-if="loading" class="rag-state">正在读取 Milvus 与本地记忆状态…</section>
    <section v-else-if="error" class="rag-state rag-error">
      <strong>RAG 概览加载失败</strong>
      <p>{{ error }}</p>
      <button type="button" @click="loadOverview">重新加载</button>
    </section>

    <section v-else-if="overview" class="rag-dashboard">
      <div class="metric-grid">
        <article>
          <span>入选文档</span><strong>{{ formatNumber(overview.total_documents) }}</strong><small>清洗后进入 Embedding 计划</small>
        </article>
        <article>
          <span>向量分块</span><strong>{{ formatNumber(overview.total_chunks) }}</strong><small>Milvus 中可检索的语义单元</small>
        </article>
        <article>
          <span>语料 Tokens</span><strong>{{ formatNumber(overview.total_tokens) }}</strong><small>{{ overview.embedding_provider }} · 1024 维</small>
        </article>
        <article>
          <span>题目—案例配对</span><strong>{{ formatNumber(overview.paired_problem_cases) }}</strong><small>一题一解，可联合召回</small>
        </article>
      </div>

      <div class="rag-main-grid">
        <article class="knowledge-map-card">
          <div class="card-heading">
            <div><span>RETRIEVAL MAP</span><h2>知识检索拓扑</h2></div>
            <small>{{ overview.retrieval_mode }}</small>
          </div>
          <div class="knowledge-map">
            <svg viewBox="0 0 560 370" preserveAspectRatio="none" aria-hidden="true">
              <path d="M280 185 L115 82"/><path d="M280 185 L445 82"/>
              <path d="M280 185 L115 288"/><path d="M280 185 L445 288"/>
              <circle cx="280" cy="185" r="116"/><circle cx="280" cy="185" r="151"/>
            </svg>
            <div class="rag-core">
              <i></i><strong>Agent RAG</strong><small>{{ formatNumber(overview.total_chunks) }} vectors</small>
            </div>
            <button
              v-for="(library, index) in overview.libraries"
              :key="library.key"
              type="button"
              class="map-node"
              :class="[libraryVisuals[library.key].className, `node-${index + 1}`, { active: selectedKey === library.key }]"
              @click="selectLibrary(library.key)"
            >
              <span>{{ libraryVisuals[library.key].icon }}</span>
              <strong>{{ library.label }}</strong>
              <small>{{ formatNumber(library.documents) }} {{ library.key === 'user_memory' ? '条记忆' : '篇文档' }}</small>
            </button>
          </div>
          <div class="map-legend">
            <span><i class="ready"></i> 已完成向量导入</span>
            <span><i class="growing"></i> 动态增长</span>
            <span>更新于 {{ formatDate(overview.generated_at) }}</span>
          </div>
        </article>

        <article v-if="selectedLibrary" class="library-detail-card" :class="libraryVisuals[selectedLibrary.key].className">
          <div class="detail-title">
            <span class="detail-icon">{{ libraryVisuals[selectedLibrary.key].icon }}</span>
            <div><small>{{ selectedLibrary.source }}</small><h2>{{ selectedLibrary.label }}</h2></div>
            <span class="library-status" :class="selectedLibrary.status">
              {{ selectedLibrary.status === 'ready' ? 'READY' : selectedLibrary.status === 'growing' ? 'GROWING' : 'PARTIAL' }}
            </span>
          </div>
          <p class="detail-summary">{{ libraryVisuals[selectedLibrary.key].summary }}</p>
          <div class="detail-stats">
            <div><span>文档</span><strong>{{ formatNumber(selectedLibrary.documents) }}</strong></div>
            <div><span>分块</span><strong>{{ formatNumber(selectedLibrary.chunks) }}</strong></div>
            <div><span>可用原始数据</span><strong>{{ formatNumber(selectedLibrary.available_documents) }}</strong></div>
          </div>
          <div class="coverage-head"><span>存储覆盖率</span><strong>{{ selectedLibrary.coverage.toFixed(1) }}%</strong></div>
          <div class="coverage-track"><i :style="{ width: `${Math.min(100, selectedLibrary.coverage)}%` }"></i></div>
          <dl>
            <div><dt>检索方式</dt><dd>{{ selectedLibrary.retrieval_mode }}</dd></div>
            <div><dt>存储位置</dt><dd>{{ selectedLibrary.storage }}</dd></div>
            <div><dt>Embedding</dt><dd>{{ selectedLibrary.model }}</dd></div>
            <div><dt>Collection</dt><dd :title="selectedLibrary.collection">{{ selectedLibrary.collection }}</dd></div>
          </dl>
        </article>
      </div>

      <div v-if="selectedLibrary" class="rag-bottom-grid">
        <article class="distribution-card">
          <div class="card-heading">
            <div><span>DISTRIBUTION</span><h2>内容分布</h2></div>
            <small>Top {{ selectedLibrary.distribution.length }}</small>
          </div>
          <div v-if="selectedLibrary.distribution.length" class="bar-list">
            <div v-for="item in selectedLibrary.distribution" :key="item.label" class="bar-row">
              <span>{{ normalizedLabel(item.label) }}</span>
              <div><i :style="{ width: `${item.count / maxDistribution * 100}%` }"></i></div>
              <strong>{{ item.count }}</strong>
            </div>
          </div>
          <p v-else class="empty-data">该库正在积累数据，暂无可展示的分布。</p>
        </article>

        <article class="sample-card">
          <div class="card-heading">
            <div><span>DOCUMENT SAMPLES</span><h2>数据样例</h2></div>
            <small>只展示元数据，不加载完整正文</small>
          </div>
          <div v-if="selectedLibrary.samples.length" class="sample-list">
            <a
              v-for="sample in selectedLibrary.samples"
              :key="sample.document_id || sample.title"
              :href="sample.source_url || undefined"
              :target="sample.source_url ? '_blank' : undefined"
              :class="{ disabled: !sample.source_url }"
              rel="noreferrer"
            >
              <span>{{ libraryVisuals[selectedLibrary.key].icon }}</span>
              <div><strong>{{ sample.title }}</strong><small>{{ sampleMeta(sample, selectedLibrary) }}</small></div>
              <b>{{ sample.source_url ? '↗' : '·' }}</b>
            </a>
          </div>
          <p v-else class="empty-data">暂无样例数据。</p>
        </article>
      </div>
    </section>
  </main>
</template>

<style scoped>
.rag-workspace { height: 100vh; overflow-y: auto; color: #29313d; background: radial-gradient(circle at 15% 0, #f9f8f2 0, #f1f3f4 44%, #e9edf1 100%); }
.rag-topbar { position: sticky; z-index: 4; top: 0; display: flex; align-items: center; justify-content: space-between; min-height: 112px; padding: 22px 38px; border-bottom: 1px solid #d9dde1; background: rgba(247,247,243,.88); backdrop-filter: blur(16px); }
.rag-eyebrow, .card-heading span { color: #6680a8; font-size: 9px; font-weight: 800; letter-spacing: 1.7px; }
.rag-topbar h1 { margin: 5px 0 3px; font-family: 'Noto Serif SC', serif; font-size: 24px; font-weight: 650; }
.rag-topbar p { margin: 0; color: #858d98; font-size: 11px; }
.rag-top-actions { display: flex; align-items: center; gap: 10px; }
.rag-top-actions button, .rag-error button { padding: 9px 13px; border: 1px solid #d6d9dd; border-radius: 9px; color: #526073; background: #fff; font-size: 10px; }
.rag-top-actions button:hover, .rag-error button:hover { border-color: #8fa5c4; color: #284e82; }
.sync-state { display: flex; align-items: center; gap: 7px; padding: 8px 11px; border: 1px solid #d7ded8; border-radius: 99px; color: #627167; background: #f7faf7; font-size: 9px; }
.sync-state i { width: 6px; height: 6px; border-radius: 50%; background: #5fa379; box-shadow: 0 0 0 4px rgba(95,163,121,.1); }
.rag-state { display: grid; min-height: calc(100vh - 112px); place-content: center; color: #7f8997; font-size: 12px; text-align: center; }
.rag-error strong { color: #714139; font-size: 16px; }.rag-error p { margin: 8px 0 16px; }
.rag-dashboard { width: min(1280px, calc(100% - 52px)); margin: 0 auto; padding: 26px 0 42px; }
.metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.metric-grid article { position: relative; overflow: hidden; min-height: 116px; padding: 17px 19px; border: 1px solid #dce0e2; border-radius: 14px; background: rgba(255,255,252,.78); box-shadow: 0 8px 26px rgba(47,59,73,.04); }
.metric-grid article::after { content: ''; position: absolute; width: 70px; height: 70px; right: -18px; bottom: -28px; border-radius: 50%; background: rgba(76,113,170,.07); }
.metric-grid span, .metric-grid small { display: block; color: #8a929d; font-size: 9px; }.metric-grid strong { display: block; margin: 8px 0 5px; color: #273d60; font-family: 'Noto Serif SC', serif; font-size: 25px; }.metric-grid small { line-height: 1.5; }
.rag-main-grid { display: grid; grid-template-columns: minmax(580px, 1.5fr) minmax(300px, .78fr); gap: 14px; margin-top: 14px; }
.knowledge-map-card, .library-detail-card, .distribution-card, .sample-card { border: 1px solid #d9dee2; border-radius: 16px; background: rgba(253,253,250,.84); box-shadow: 0 12px 32px rgba(42,53,67,.05); }
.knowledge-map-card { padding: 19px 21px 14px; }.card-heading { display: flex; align-items: flex-start; justify-content: space-between; }.card-heading h2 { margin: 4px 0 0; font-family: 'Noto Serif SC', serif; font-size: 15px; }.card-heading small { color: #9199a3; font-size: 9px; }
.knowledge-map { position: relative; height: 376px; margin-top: 4px; overflow: hidden; }
.knowledge-map svg { position: absolute; inset: 0; width: 100%; height: 100%; }.knowledge-map path { fill: none; stroke: #b9c8da; stroke-width: 1.2; stroke-dasharray: 5 6; }.knowledge-map circle { fill: none; stroke: rgba(113,140,178,.13); stroke-width: 1; }
.rag-core { position: absolute; z-index: 2; left: 50%; top: 50%; display: grid; width: 112px; height: 112px; place-content: center; border: 1px solid #b8c9df; border-radius: 50%; color: #f4f7fb; background: radial-gradient(circle at 35% 25%, #416aa2, #1c355c 70%); text-align: center; transform: translate(-50%,-50%); box-shadow: 0 18px 40px rgba(38,69,111,.25), 0 0 0 12px rgba(68,103,153,.06); }
.rag-core i { position: absolute; width: 7px; height: 7px; right: 12px; top: 16px; border-radius: 50%; background: #f0b86d; box-shadow: 0 0 0 4px rgba(240,184,109,.14); }.rag-core strong { font-family: 'Noto Serif SC', serif; font-size: 13px; }.rag-core small { margin-top: 5px; color: #aebdd2; font-size: 8px; }
.map-node { position: absolute; z-index: 2; display: grid; grid-template-columns: 38px 1fr; width: 174px; padding: 10px 12px; border: 1px solid #d8dde3; border-radius: 12px; color: #4e5968; background: rgba(255,255,253,.94); text-align: left; box-shadow: 0 8px 22px rgba(48,60,74,.06); transition: .2s ease; }
.map-node:hover, .map-node.active { border-color: #8ba4c7; transform: translateY(-2px); box-shadow: 0 12px 26px rgba(42,63,91,.12); }.map-node > span { grid-row: span 2; display: grid; width: 31px; height: 31px; place-items: center; border-radius: 9px; color: #345f99; background: #e8eef7; font-size: 13px; font-weight: 700; }.map-node strong { align-self: end; font-size: 10px; }.map-node small { color: #939ba5; font-size: 8px; }.node-1 { left: 3%; top: 10%; }.node-2 { right: 3%; top: 10%; }.node-3 { left: 3%; bottom: 9%; }.node-4 { right: 3%; bottom: 9%; }
.map-node.problems > span { color: #8a6431; background: #f4ead9; }.map-node.cases > span { color: #4c7661; background: #e3eee8; }.map-node.memory > span { color: #7a5b83; background: #eee7f0; }
.map-legend { display: flex; align-items: center; gap: 16px; padding-top: 12px; border-top: 1px solid #e8e9e5; color: #8e969f; font-size: 8px; }.map-legend span:last-child { margin-left: auto; }.map-legend i { display: inline-block; width: 6px; height: 6px; margin-right: 5px; border-radius: 50%; }.map-legend .ready { background: #5fa379; }.map-legend .growing { background: #9b70aa; }
.library-detail-card { padding: 20px; border-top: 3px solid #4a76b3; }.library-detail-card.problems { border-top-color: #c18a45; }.library-detail-card.cases { border-top-color: #5f9678; }.library-detail-card.memory { border-top-color: #956ca1; }.detail-title { display: flex; align-items: center; gap: 11px; }.detail-icon { display: grid; width: 40px; height: 40px; place-items: center; border-radius: 11px; color: #315d96; background: #e7eef8; font-weight: 750; }.detail-title small { color: #9299a2; font-size: 8px; }.detail-title h2 { margin: 3px 0 0; font-family: 'Noto Serif SC', serif; font-size: 16px; }.library-status { margin-left: auto; padding: 5px 7px; border-radius: 99px; color: #4f8063; background: #e7f2eb; font-size: 7px; font-weight: 800; letter-spacing: .8px; }.library-status.growing { color: #7a5385; background: #efe6f2; }.library-status.partial { color: #9b6e30; background: #f7ead8; }
.detail-summary { margin: 14px 0; color: #7e8792; font-size: 10px; }.detail-stats { display: grid; grid-template-columns: repeat(3, 1fr); overflow: hidden; border: 1px solid #e2e4e4; border-radius: 10px; }.detail-stats div { padding: 11px; text-align: center; }.detail-stats div + div { border-left: 1px solid #e2e4e4; }.detail-stats span, .detail-stats strong { display: block; }.detail-stats span { color: #929aa4; font-size: 7px; }.detail-stats strong { margin-top: 5px; color: #34465e; font-size: 14px; }
.coverage-head { display: flex; justify-content: space-between; margin-top: 17px; color: #77818e; font-size: 9px; }.coverage-head strong { color: #426b9f; }.coverage-track { height: 5px; margin-top: 7px; overflow: hidden; border-radius: 99px; background: #e5e8e9; }.coverage-track i { display: block; height: 100%; border-radius: inherit; background: linear-gradient(90deg, #406da8, #7fa2cf); }
dl { margin: 17px 0 0; }dl div { display: grid; grid-template-columns: 84px minmax(0,1fr); gap: 8px; padding: 9px 0; border-top: 1px solid #e8e9e7; font-size: 8px; }dt { color: #969da5; }dd { margin: 0; color: #576373; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.rag-bottom-grid { display: grid; grid-template-columns: .8fr 1.2fr; gap: 14px; margin-top: 14px; }.distribution-card, .sample-card { min-height: 292px; padding: 19px 21px; }.bar-list { display: grid; gap: 12px; margin-top: 19px; }.bar-row { display: grid; grid-template-columns: 90px 1fr 28px; gap: 9px; align-items: center; color: #697482; font-size: 8px; }.bar-row > div { height: 6px; overflow: hidden; border-radius: 99px; background: #e7e9e8; }.bar-row i { display: block; height: 100%; border-radius: inherit; background: linear-gradient(90deg, #3e6ba6, #87a6cd); }.bar-row strong { color: #596777; text-align: right; }
.sample-list { display: grid; gap: 7px; margin-top: 14px; }.sample-list a { display: grid; grid-template-columns: 34px minmax(0,1fr) 20px; gap: 10px; align-items: center; padding: 9px 10px; border: 1px solid #e3e5e4; border-radius: 9px; color: inherit; background: #fbfbf8; text-decoration: none; transition: .18s ease; }.sample-list a:hover:not(.disabled) { border-color: #a7b8cf; background: #fff; transform: translateX(2px); }.sample-list a > span { display: grid; width: 30px; height: 30px; place-items: center; border-radius: 8px; color: #456f9f; background: #e9eff6; font-size: 9px; font-weight: 750; }.sample-list strong, .sample-list small { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }.sample-list strong { color: #485462; font-size: 9px; }.sample-list small { margin-top: 4px; color: #949ba3; font-size: 7px; }.sample-list b { color: #8397b1; font-size: 11px; text-align: center; }.empty-data { display: grid; min-height: 190px; margin: 0; place-items: center; color: #9da3aa; font-size: 9px; }
@media (max-width: 1080px) { .rag-main-grid { grid-template-columns: 1fr; }.rag-bottom-grid { grid-template-columns: 1fr 1fr; } }
@media (max-width: 820px) { .rag-dashboard { width: calc(100% - 28px); }.rag-topbar { padding: 18px 22px; }.rag-topbar p, .sync-state { display: none; }.metric-grid { grid-template-columns: repeat(2,1fr); }.rag-bottom-grid { grid-template-columns: 1fr; }.knowledge-map-card { padding-left: 10px; padding-right: 10px; } }
</style>
