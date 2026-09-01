<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { chatApi } from '../api'
import type { ModelConfigStatus } from '../types'

type DurationUnit = 'minutes' | 'hours' | 'days'

const status = ref<ModelConfigStatus>({ configured: false })
const loading = ref(true)
const savingModel = ref(false)
const savingSearch = ref(false)
const deletingModel = ref(false)
const deletingSearch = ref(false)
const error = ref('')
const success = ref('')
const showApiKey = ref(false)
const showSerpapiKey = ref(false)

const apiKey = ref('')
const serpapiApiKey = ref('')
const model = ref('deepseek-v4-pro')
const baseUrl = ref('https://api.deepseek.com')
const durationValue = ref(24)
const durationUnit = ref<DurationUnit>('hours')

const ttlSeconds = computed(() => {
  const multiplier = {
    minutes: 60,
    hours: 3_600,
    days: 86_400,
  }[durationUnit.value]
  return Math.round(Number(durationValue.value) * multiplier)
})

const expiryText = computed(() => {
  if (!status.value.expiresAt) return '—'
  return new Intl.DateTimeFormat('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(status.value.expiresAt))
})

onMounted(loadStatus)

async function loadStatus() {
  loading.value = true
  error.value = ''
  try {
    status.value = await chatApi.getModelConfig()
    if (status.value.configured) {
      model.value = status.value.model || model.value
      baseUrl.value = status.value.baseUrl || baseUrl.value
    }
  } catch (cause) {
    error.value = messageOf(cause, '无法读取 Redis 模型配置。')
  } finally {
    loading.value = false
  }
}

function validateTtl() {
  if (ttlSeconds.value < 300 || ttlSeconds.value > 2_592_000) {
    error.value = '保存时间必须在 5 分钟到 30 天之间。'
    return false
  }
  return true
}

async function saveModelConfig() {
  error.value = ''
  success.value = ''
  if (!apiKey.value.trim()) {
    error.value = '请输入新的模型 API Key；已保存的 Key 不会回填到表单。'
    return
  }
  if (!model.value.trim() || !baseUrl.value.trim() || !validateTtl()) {
    if (!error.value) error.value = '请填写模型名称和 API URL。'
    return
  }
  savingModel.value = true
  try {
    status.value = await chatApi.saveModelConfig({
      updateModel: true,
      apiKey: apiKey.value.trim(),
      model: model.value.trim(),
      baseUrl: baseUrl.value.trim(),
      ttlSeconds: ttlSeconds.value,
    })
    apiKey.value = ''
    showApiKey.value = false
    success.value = '模型配置已保存；原有 SerpAPI Key 未被修改。'
  } catch (cause) {
    error.value = messageOf(cause, '模型配置保存失败。')
  } finally {
    savingModel.value = false
  }
}

async function saveSearchConfig() {
  error.value = ''
  success.value = ''
  if (!serpapiApiKey.value.trim()) {
    error.value = '请输入新的 SerpAPI Key。'
    return
  }
  if (!validateTtl()) return
  savingSearch.value = true
  try {
    status.value = await chatApi.saveModelConfig({
      updateSearch: true,
      serpapiApiKey: serpapiApiKey.value.trim(),
      ttlSeconds: ttlSeconds.value,
    })
    serpapiApiKey.value = ''
    showSerpapiKey.value = false
    success.value = 'SerpAPI 配置已保存；原有模型配置未被修改。'
  } catch (cause) {
    error.value = messageOf(cause, 'SerpAPI 配置保存失败。')
  } finally {
    savingSearch.value = false
  }
}

async function deleteModelConfig() {
  if (!window.confirm('确定只删除模型连接配置吗？SerpAPI Key 会保留。')) return
  deletingModel.value = true
  error.value = ''
  success.value = ''
  try {
    await chatApi.deleteModelConnection()
    apiKey.value = ''
    status.value = await chatApi.getModelConfig()
    success.value = '模型配置已删除，SerpAPI 配置未受影响。'
  } catch (cause) {
    error.value = messageOf(cause, '模型配置删除失败。')
  } finally {
    deletingModel.value = false
  }
}

async function deleteSearchConfig() {
  if (!window.confirm('确定只删除 SerpAPI 配置吗？模型连接会保留。')) return
  deletingSearch.value = true
  error.value = ''
  success.value = ''
  try {
    await chatApi.deleteSearchConnection()
    serpapiApiKey.value = ''
    status.value = await chatApi.getModelConfig()
    success.value = 'SerpAPI 配置已删除，模型配置未受影响。'
  } catch (cause) {
    error.value = messageOf(cause, 'SerpAPI 配置删除失败。')
  } finally {
    deletingSearch.value = false
  }
}

function messageOf(cause: unknown, fallback: string) {
  return cause instanceof Error && cause.message ? cause.message : fallback
}
</script>

<template>
  <main class="settings-page">
    <header class="settings-header">
      <div>
        <p>MODEL &amp; SEARCH CONNECTION</p>
        <h1>模型与搜索设置</h1>
        <span>为当前部署保存一套模型和 SerpAPI 凭据，所有会话直接读取 Redis 全局配置。</span>
      </div>
      <div class="config-state" :class="{ ready: status.configured || status.searchConfigured }">
        <i></i>
        <span>{{ loading ? '正在读取' : status.configured || status.searchConfigured ? 'Redis 配置有效' : '尚未配置' }}</span>
      </div>
    </header>

    <div class="settings-scroll">
      <section class="settings-grid">
        <article class="settings-card connection-card">
          <div class="card-heading">
            <div><span>01</span><h2>连接参数</h2></div>
            <p>两类凭据独立保存，互不覆盖</p>
          </div>

          <div class="config-form">
            <label>
              <span>API URL</span>
              <input v-model="baseUrl" type="url" required maxlength="500" placeholder="https://api.deepseek.com" />
              <small>填写服务商的基础地址或完整 `/chat/completions` 地址，只允许 HTTPS。</small>
            </label>

            <label>
              <span>模型名称</span>
              <input v-model="model" type="text" required maxlength="200" placeholder="deepseek-v4-pro" />
              <small>必须与服务商控制台或模型文档中的 Model ID 完全一致。</small>
            </label>

            <label>
              <span>API Key</span>
              <div class="secret-input">
                <input
                  v-model="apiKey"
                  :type="showApiKey ? 'text' : 'password'"
                  minlength="8"
                  maxlength="512"
                  autocomplete="new-password"
                  placeholder="输入新的 API Key"
                />
                <button type="button" @click="showApiKey = !showApiKey">{{ showApiKey ? '隐藏' : '显示' }}</button>
              </div>
              <small>Key 只在保存请求中传输，之后仅显示掩码，不会返回明文。</small>
            </label>

            <div class="section-actions">
              <button class="save-config" type="button" :disabled="savingModel" @click="saveModelConfig">
                {{ savingModel ? '正在保存模型…' : '保存模型配置' }}
              </button>
              <button
                v-if="status.configured"
                class="delete-config"
                type="button"
                :disabled="deletingModel"
                @click="deleteModelConfig"
              >{{ deletingModel ? '正在删除…' : '删除模型配置' }}</button>
            </div>

            <div class="config-divider"><span>网页搜索（独立配置）</span></div>

            <label>
              <span>SerpAPI Key（可选）</span>
              <div class="secret-input">
                <input
                  v-model="serpapiApiKey"
                  :type="showSerpapiKey ? 'text' : 'password'"
                  minlength="8"
                  maxlength="512"
                  autocomplete="new-password"
                  placeholder="用于网页搜索的 SerpAPI Key"
                />
                <button type="button" @click="showSerpapiKey = !showSerpapiKey">{{ showSerpapiKey ? '隐藏' : '显示' }}</button>
              </div>
              <small>可以单独保存，不需要重新输入上方已经保存的模型 API Key。</small>
            </label>

            <fieldset>
              <legend>保存时长</legend>
              <div class="duration-row">
                <input v-model.number="durationValue" type="number" required min="1" step="1" />
                <select v-model="durationUnit">
                  <option value="minutes">分钟</option>
                  <option value="hours">小时</option>
                  <option value="days">天</option>
                </select>
              </div>
              <small>可设置 5 分钟至 30 天；到期后 Redis 自动删除，继续对话前需要重新保存。</small>
            </fieldset>

            <div v-if="error" class="settings-message error">{{ error }}</div>
            <div v-if="success" class="settings-message success">{{ success }}</div>

            <div class="section-actions">
              <button class="save-config search-save" type="button" :disabled="savingSearch" @click="saveSearchConfig">
                {{ savingSearch ? '正在保存搜索…' : '保存 SerpAPI 配置' }}
              </button>
              <button
                v-if="status.searchConfigured"
                class="delete-config"
                type="button"
                :disabled="deletingSearch"
                @click="deleteSearchConfig"
              >{{ deletingSearch ? '正在删除…' : '删除 SerpAPI 配置' }}</button>
            </div>
          </div>
        </article>

        <div class="settings-side">
          <article class="settings-card status-card">
            <div class="card-heading"><div><span>02</span><h2>当前状态</h2></div></div>
            <div v-if="status.configured || status.searchConfigured" class="status-list">
              <div><span>模型</span><strong>{{ status.configured ? status.model : '未配置' }}</strong></div>
              <div><span>API URL</span><strong>{{ status.configured ? status.baseUrl : '未配置' }}</strong></div>
              <div><span>API Key</span><strong>{{ status.configured ? status.maskedApiKey : '未配置' }}</strong></div>
              <div><span>SerpAPI</span><strong>{{ status.searchConfigured ? status.maskedSerpapiApiKey : '未配置' }}</strong></div>
              <div><span>自动过期</span><strong>{{ expiryText }}</strong></div>
            </div>
            <div v-else class="empty-config">
              <span>∅</span>
              <strong>没有可用配置</strong>
              <p>完成左侧表单后，聊天 Agent 才会调用模型。</p>
            </div>
          </article>

          <article class="settings-card security-card">
            <div class="card-heading"><div><span>03</span><h2>安全边界</h2></div></div>
            <ul>
              <li>无登录模式使用固定 Redis 全局配置键，不依赖浏览器令牌。</li>
              <li>模型与 SerpAPI 配置整体加密后写入 Redis，API Key 不进入 SQLite 和对话记录。</li>
              <li>Redis 不暴露公网端口；公网部署必须使用 HTTPS。</li>
              <li>重启服务或切换 localhost / 127.0.0.1 后仍可读取同一配置，直到主动删除或 TTL 到期。</li>
            </ul>
          </article>
        </div>
      </section>

      <section class="usage-guide">
        <div class="guide-title"><span>QUICK GUIDE</span><h2>前端使用文档</h2></div>
        <div class="guide-steps">
          <article><b>1</b><h3>获取服务商信息</h3><p>准备模型 API Key、OpenAI 兼容 URL、Model ID；需要网页搜索时再准备 SerpAPI Key。</p></article>
          <article><b>2</b><h3>分别保存</h3><p>模型与 SerpAPI 各有独立按钮；后补或更新一个 Key 时无需重输另一个。</p></article>
          <article><b>3</b><h3>确认状态</h3><p>当前状态会分别显示模型与网页搜索是否配置成功。</p></article>
          <article><b>4</b><h3>独立删除</h3><p>两个配置也可以分别撤销，不会误删另一类凭据。</p></article>
        </div>
        <div class="provider-examples">
          <div><span>服务商</span><span>API URL 示例</span><span>模型名称示例</span></div>
          <div><strong>DeepSeek</strong><code>https://api.deepseek.com</code><code>deepseek-chat</code></div>
          <div><strong>OpenAI</strong><code>https://api.openai.com/v1</code><code>gpt-4.1-mini</code></div>
          <div><strong>硅基流动</strong><code>https://api.siliconflow.cn/v1</code><code>服务商页面提供的 Model ID</code></div>
        </div>
        <p class="guide-note">不同平台的模型名称和兼容能力可能变化，请以对应服务商当前文档为准。本项目调用 `/chat/completions` 并要求模型能够输出 JSON。</p>
      </section>
    </div>
  </main>
</template>

<style scoped>
.settings-page { min-width: 0; height: 100vh; overflow: hidden; color: #273143; background: radial-gradient(circle at 84% 12%, rgba(86,125,188,.13), transparent 31%), linear-gradient(145deg, #f5f3ed, #eef2f6); }
.settings-header { display: flex; align-items: center; justify-content: space-between; min-height: 118px; padding: 24px 44px; border-bottom: 1px solid rgba(168,174,180,.42); background: rgba(249,248,243,.74); backdrop-filter: blur(12px); }
.settings-header p, .settings-header h1, .settings-header span { margin: 0; }.settings-header p { color: #4f76b0; font-size: 9px; font-weight: 800; letter-spacing: .18em; }.settings-header h1 { margin-top: 7px; font-family: 'Noto Serif SC', serif; font-size: 23px; }.settings-header > div:first-child > span { display: block; margin-top: 7px; color: #828781; font-size: 11px; }
.config-state { display: flex; align-items: center; gap: 8px; padding: 9px 13px; border: 1px solid #d6d5ce; border-radius: 999px; color: #8a6a42; background: #faf8f1; font-size: 10px; }.config-state i { width: 7px; height: 7px; border-radius: 50%; background: #d19a54; box-shadow: 0 0 0 4px rgba(209,154,84,.12); }.config-state.ready { color: #446d65; border-color: #b9d4cb; background: #f1f8f5; }.config-state.ready i { background: #4b9681; box-shadow: 0 0 0 4px rgba(75,150,129,.12); }
.settings-scroll { height: calc(100vh - 118px); overflow-y: auto; padding: 30px 38px 60px; }.settings-grid { display: grid; grid-template-columns: minmax(420px, 1.35fr) minmax(300px, .85fr); gap: 18px; max-width: 1160px; margin: auto; }.settings-side { display: grid; align-content: start; gap: 18px; }
.settings-card, .usage-guide { border: 1px solid rgba(177,181,181,.5); border-radius: 18px; background: rgba(255,254,250,.85); box-shadow: 0 16px 40px rgba(45,55,70,.055); }.settings-card { padding: 24px; }.card-heading { display: flex; align-items: center; justify-content: space-between; padding-bottom: 18px; border-bottom: 1px solid #e4e3dc; }.card-heading > div { display: flex; align-items: center; gap: 9px; }.card-heading span { display: grid; place-items: center; width: 25px; height: 25px; border-radius: 8px; color: #466da6; background: #e6edf8; font-size: 9px; font-weight: 800; }.card-heading h2, .card-heading p { margin: 0; }.card-heading h2 { font-family: 'Noto Serif SC', serif; font-size: 14px; }.card-heading p { color: #a09e96; font-size: 9px; }
form, .config-form { display: grid; gap: 17px; padding-top: 20px; }label, fieldset { display: grid; gap: 7px; margin: 0; padding: 0; border: 0; }label > span, legend { color: #565f6d; font-size: 10px; font-weight: 700; letter-spacing: .04em; }input, select { width: 100%; height: 43px; padding: 0 13px; border: 1px solid #d8d8d2; border-radius: 10px; outline: 0; color: #303a49; background: #fbfbf8; font-size: 12px; transition: .2s; }input:focus, select:focus { border-color: #7898ca; box-shadow: 0 0 0 3px rgba(88,126,187,.1); }label small, fieldset small { color: #979992; font-size: 9px; line-height: 1.5; }.secret-input { display: grid; grid-template-columns: 1fr auto; }.secret-input input { border-radius: 10px 0 0 10px; }.secret-input button { padding: 0 13px; border: 1px solid #d8d8d2; border-left: 0; border-radius: 0 10px 10px 0; color: #5273a5; background: #edf2f8; font-size: 10px; }.duration-row { display: grid; grid-template-columns: 1fr 110px; gap: 8px; }.config-divider { display: flex; align-items: center; gap: 10px; margin-top: 4px; color: #65758d; font-size: 9px; font-weight: 800; letter-spacing: .07em; }.config-divider::before, .config-divider::after { content: ''; height: 1px; flex: 1; background: #e1e3e5; }
.settings-message { padding: 10px 12px; border-radius: 9px; font-size: 10px; line-height: 1.5; }.settings-message.error { color: #8b4138; background: #f5e4df; }.settings-message.success { color: #356d5f; background: #e7f3ee; }.section-actions { display: flex; gap: 9px; padding-top: 3px; }.section-actions button { min-height: 42px; padding: 0 18px; border-radius: 10px; font-size: 11px; font-weight: 700; }.save-config { flex: 1; border: 0; color: #f7f4eb; background: linear-gradient(110deg, #203c69, #3f659f); }.save-config.search-save { background: linear-gradient(110deg, #285c62, #4d7d77); }.delete-config { border: 1px solid #e1bdb2; color: #8a493e; background: #fff7f4; }.section-actions button:disabled { cursor: wait; opacity: .55; }
.status-list { display: grid; gap: 0; padding-top: 12px; }.status-list > div { display: grid; grid-template-columns: 88px 1fr; gap: 10px; padding: 12px 0; border-bottom: 1px solid #ebe9e2; }.status-list > div:last-child { border: 0; }.status-list span { color: #96978f; font-size: 9px; }.status-list strong { min-width: 0; overflow-wrap: anywhere; color: #3e4856; font-size: 10px; font-weight: 600; }.empty-config { display: grid; place-items: center; padding: 32px 12px 16px; text-align: center; }.empty-config > span { display: grid; place-items: center; width: 42px; height: 42px; border: 1px solid #dedcd3; border-radius: 50%; color: #b1a58f; background: #f7f4ec; font-size: 20px; }.empty-config strong { margin-top: 12px; font-size: 12px; }.empty-config p { max-width: 210px; margin: 7px 0 0; color: #9b9b93; font-size: 9px; line-height: 1.6; }
.security-card ul { display: grid; gap: 10px; margin: 17px 0 0; padding: 0; list-style: none; }.security-card li { position: relative; padding-left: 18px; color: #717773; font-size: 10px; line-height: 1.6; }.security-card li::before { content: '✓'; position: absolute; left: 0; top: 1px; color: #4e86a9; font-weight: 800; }
.usage-guide { max-width: 1160px; margin: 20px auto 0; padding: 26px; }.guide-title span { color: #a37338; font-size: 8px; font-weight: 800; letter-spacing: .17em; }.guide-title h2 { margin: 5px 0 0; font-family: 'Noto Serif SC', serif; font-size: 16px; }.guide-steps { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 18px; }.guide-steps article { padding: 15px; border: 1px solid #e2e1da; border-radius: 12px; background: #faf9f5; }.guide-steps b { color: #ba8543; font-size: 10px; }.guide-steps h3 { margin: 7px 0 5px; font-size: 11px; }.guide-steps p { margin: 0; color: #898b85; font-size: 9px; line-height: 1.6; }.provider-examples { margin-top: 18px; overflow: hidden; border: 1px solid #dfded7; border-radius: 11px; }.provider-examples > div { display: grid; grid-template-columns: 120px 1.3fr 1fr; gap: 12px; padding: 10px 13px; border-bottom: 1px solid #e7e5de; align-items: center; }.provider-examples > div:first-child { color: #898b85; background: #f0f1ef; font-size: 8px; font-weight: 700; }.provider-examples > div:last-child { border: 0; }.provider-examples strong, .provider-examples code { overflow-wrap: anywhere; font-size: 9px; }.provider-examples code { color: #45648e; }.guide-note { margin: 12px 0 0; color: #969891; font-size: 9px; line-height: 1.6; }
@media (max-width: 980px) { .settings-grid { grid-template-columns: 1fr; }.guide-steps { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 620px) { .settings-header { padding: 20px; }.settings-header > div:first-child > span { display: none; }.config-state { align-self: flex-start; }.settings-scroll { padding: 18px 14px 40px; }.settings-grid { grid-template-columns: minmax(0, 1fr); }.settings-card, .usage-guide { padding: 18px; }.card-heading p { display: none; }.guide-steps { grid-template-columns: 1fr; }.provider-examples > div { grid-template-columns: 1fr; gap: 4px; }.provider-examples > div:first-child { display: none; } }
</style>
