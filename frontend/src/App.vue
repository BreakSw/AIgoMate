<script setup lang="ts">
import { computed, nextTick, onMounted, ref } from 'vue'
import { chatApi } from './api'
import LogoMark from './components/LogoMark.vue'
import ContextInspector from './components/ContextInspector.vue'
import type { ChatMessage, ChatSession, ContextSnapshot } from './types'

const sessions = ref<ChatSession[]>([])
const activeSessionId = ref<number | null>(null)
const messages = ref<ChatMessage[]>([])
const draft = ref('')
const loading = ref(true)
const sending = ref(false)
const error = ref('')
const streamStatus = ref('')
const retryStatus = ref<{ current: number; max: number } | null>(null)
const streamingText = ref('')
const streamingContext = ref<ContextSnapshot | null>(null)
const messageList = ref<HTMLElement | null>(null)

const activeSession = computed(() => sessions.value.find((item) => item.id === activeSessionId.value))
const hasMessages = computed(() => messages.value.length > 0)

const suggestions = [
  { icon: '⌁', title: '拆解一道题', text: '从思路开始理解二分查找' },
  { icon: '◫', title: '分析复杂度', text: '比较快速排序与归并排序' },
  { icon: '↗', title: '制定学习路径', text: '为我规划 30 天算法入门路线' },
]

onMounted(loadInitialData)

async function loadInitialData() {
  try {
    sessions.value = await chatApi.listSessions()
    if (sessions.value.length > 0) await selectSession(sessions.value[0].id)
  } catch {
    error.value = '暂时无法连接后端，请确认 Spring Boot 服务已启动。'
  } finally {
    loading.value = false
  }
}

async function createSession() {
  error.value = ''
  try {
    const session = await chatApi.createSession()
    sessions.value.unshift(session)
    activeSessionId.value = session.id
    messages.value = []
  } catch {
    error.value = '新建会话失败，请稍后重试。'
  }
}

async function selectSession(id: number) {
  activeSessionId.value = id
  error.value = ''
  try {
    const conversation = await chatApi.getConversation(id)
    messages.value = conversation.messages
    await scrollToBottom()
  } catch {
    error.value = '读取会话失败。'
  }
}

async function deleteSession(id: number) {
  try {
    await chatApi.deleteSession(id)
    sessions.value = sessions.value.filter((item) => item.id !== id)
    if (activeSessionId.value === id) {
      activeSessionId.value = null
      messages.value = []
      if (sessions.value.length > 0) await selectSession(sessions.value[0].id)
    }
  } catch {
    error.value = '删除会话失败。'
  }
}

function chooseSuggestion(text: string) {
  draft.value = text
}

async function sendMessage() {
  const content = draft.value.trim()
  if (!content || sending.value) return

  if (!activeSessionId.value) {
    await createSession()
    if (!activeSessionId.value) return
  }

  const sessionId = activeSessionId.value
  draft.value = ''
  sending.value = true
  error.value = ''
  streamStatus.value = '正在建立 SSE 连接'
  retryStatus.value = null
  streamingText.value = ''
  streamingContext.value = null
  messages.value.push({ id: -Date.now(), role: 'USER', content, createdAt: new Date().toISOString() })
  await scrollToBottom()

  try {
    await chatApi.streamIntent(sessionId, content, {
      onStatus(status) {
        streamStatus.value = status.message
        if (status.phase === 'reconnecting' && status.retryCount && status.maxRetries) {
          retryStatus.value = { current: status.retryCount, max: status.maxRetries }
        } else {
          retryStatus.value = null
        }
      },
      onIntent(result) {
        retryStatus.value = null
        streamStatus.value = `已由 ${result.model} 完成识别`
        streamingText.value = result.content
        streamingContext.value = result.context_snapshot
      },
      onComplete(conversation) {
        messages.value = conversation.messages
        const index = sessions.value.findIndex((item) => item.id === sessionId)
        if (index >= 0) sessions.value.splice(index, 1, conversation.session)
      },
    })
  } catch (cause) {
    try {
      const conversation = await chatApi.getConversation(sessionId)
      messages.value = conversation.messages
    } catch {
      messages.value = messages.value.filter((message) => message.id > 0)
      draft.value = content
    }
    const detail = cause instanceof Error ? cause.message : '未知错误'
    error.value = `意图识别失败：${detail}`
  } finally {
    sending.value = false
    streamStatus.value = ''
    retryStatus.value = null
    streamingText.value = ''
    streamingContext.value = null
    await scrollToBottom()
  }
}

function handleComposerKeydown(event: KeyboardEvent) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    void sendMessage()
  }
}

async function scrollToBottom() {
  await nextTick()
  messageList.value?.scrollTo({ top: messageList.value.scrollHeight, behavior: 'smooth' })
}

function formatTime(value: string) {
  return new Intl.DateTimeFormat('zh-CN', { hour: '2-digit', minute: '2-digit' }).format(new Date(value))
}

function visibleMessageContent(content: string) {
  return content.split('========== 上下文审查 ==========')[0].trim()
}
</script>

<template>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-mark"><LogoMark /></div>
        <div><strong>AlgoMate</strong><span>算法学习智能体</span></div>
      </div>

      <button class="new-chat" type="button" @click="createSession">
        <span>＋</span> 新建探索
      </button>

      <div class="session-label">最近对话</div>
      <nav class="session-list" aria-label="会话列表">
        <button
          v-for="session in sessions"
          :key="session.id"
          class="session-item"
          :class="{ active: session.id === activeSessionId }"
          type="button"
          @click="selectSession(session.id)"
        >
          <span class="session-icon">⌘</span>
          <span class="session-copy">
            <strong>{{ session.title }}</strong>
            <small>{{ session.messageCount }} 条消息</small>
          </span>
          <span class="delete-session" role="button" aria-label="删除会话" @click.stop="deleteSession(session.id)">×</span>
        </button>
        <div v-if="!loading && sessions.length === 0" class="empty-sessions">还没有对话，开始你的第一次探索吧。</div>
      </nav>

      <div class="sidebar-footer">
        <div class="avatar">学</div>
        <div><strong>算法学习者</strong><span>持续学习中</span></div>
        <span class="status-dot" title="服务状态"></span>
      </div>
    </aside>

    <main class="workspace">
      <header class="topbar">
        <div>
          <span class="eyebrow">当前探索</span>
          <h1>{{ activeSession?.title ?? '新的算法探索' }}</h1>
        </div>
        <div class="memory-pill"><span></span> 上下文已保存</div>
      </header>

      <section ref="messageList" class="conversation" aria-live="polite">
        <div v-if="loading" class="loading-state">正在载入学习空间…</div>

        <div v-else-if="!hasMessages" class="welcome">
          <div class="orb"><LogoMark size="large" /></div>
          <p class="kicker">THINK · PRACTICE · MASTER</p>
          <h2>今天想弄懂什么算法？</h2>
          <p class="welcome-copy">我会记住你的学习轨迹，把复杂问题拆成刚刚好的下一步。</p>
          <div class="suggestions">
            <button v-for="item in suggestions" :key="item.title" type="button" @click="chooseSuggestion(item.text)">
              <span class="suggestion-icon">{{ item.icon }}</span>
              <span><strong>{{ item.title }}</strong><small>{{ item.text }}</small></span>
              <span class="arrow">→</span>
            </button>
          </div>
        </div>

        <div v-else class="messages">
          <article v-for="message in messages" :key="message.id" class="message" :class="message.role.toLowerCase()">
            <div class="message-avatar">
              <span v-if="message.role === 'USER'">你</span>
              <LogoMark v-else size="small" />
            </div>
            <div class="bubble">
              <div class="message-meta">{{ message.role === 'USER' ? '你' : 'AlgoMate' }} · {{ formatTime(message.createdAt) }}</div>
              <p>{{ visibleMessageContent(message.content) }}</p>
              <ContextInspector v-if="message.contextSnapshot" :snapshot="message.contextSnapshot" />
            </div>
          </article>
          <article v-if="sending" class="message assistant">
            <div class="message-avatar"><LogoMark size="small" /></div>
            <div class="bubble live-intent">
              <div class="message-meta">AlgoMate · {{ streamStatus }}</div>
              <div v-if="retryStatus" class="retry-indicator" role="status">
                <span class="retry-icon">↻</span>
                <div class="retry-copy">
                  <strong>正在重新连接 {{ retryStatus.current }}/{{ retryStatus.max }}</strong>
                  <span class="retry-track"><i :style="{ width: `${retryStatus.current / retryStatus.max * 100}%` }"></i></span>
                </div>
              </div>
              <p v-if="streamingText">{{ streamingText }}</p>
              <ContextInspector v-if="streamingContext" :snapshot="streamingContext" open />
              <div v-else-if="!retryStatus" class="thinking"><span></span><span></span><span></span></div>
            </div>
          </article>
        </div>
      </section>

      <div class="composer-wrap">
        <div v-if="error" class="error-banner">{{ error }}</div>
        <form class="composer" @submit.prevent="sendMessage">
          <textarea
            v-model="draft"
            rows="1"
            placeholder="描述一道题、一个概念，或贴上你的代码…"
            aria-label="发送给 AlgoMate 的消息"
            @keydown="handleComposerKeydown"
          ></textarea>
          <button type="submit" :disabled="!draft.trim() || sending" aria-label="发送消息">↑</button>
        </form>
        <p>Enter 发送 · Shift + Enter 换行 · 学习记录将保存到本地数据库</p>
      </div>
    </main>
  </div>
</template>
