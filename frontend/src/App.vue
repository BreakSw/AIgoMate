<script setup lang="ts">
import { computed, nextTick, onMounted, ref } from 'vue'
import { chatApi } from './api'
import LogoMark from './components/LogoMark.vue'
import ContextInspector from './components/ContextInspector.vue'
import MessageContent from './components/MessageContent.vue'
import RagDashboard from './components/RagDashboard.vue'
import ModelSettings from './components/ModelSettings.vue'
import type { ChatMessage, ChatSession, ContextSnapshot } from './types'

interface StreamState {
  status: string
  retry: { current: number; max: number } | null
  activities: StreamActivity[]
  text: string
  context: ContextSnapshot | null
}

interface StreamActivity {
  sequence: number
  phase: string
  agent: string
  message: string
  detail: string | null
}

const sessions = ref<ChatSession[]>([])
const activeSessionId = ref<number | null>(null)
const messages = ref<ChatMessage[]>([])
const draft = ref('')
const loading = ref(true)
const error = ref('')
const streams = ref(new Map<number, StreamState>())
const streamControllers = new Map<number, AbortController>()
const messageList = ref<HTMLElement | null>(null)
const workspaceView = ref<'chat' | 'rag' | 'settings'>('chat')
const clearingConversation = ref(false)

const activeSession = computed(() => sessions.value.find((item) => item.id === activeSessionId.value))
const hasMessages = computed(() => messages.value.length > 0)
const activeStream = computed(() =>
  activeSessionId.value === null ? undefined : streams.value.get(activeSessionId.value),
)

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
  workspaceView.value = 'chat'
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
  workspaceView.value = 'chat'
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
    const controller = streamControllers.get(id)
    if (controller) {
      controller.abort()
      await chatApi.cancelIntent(id)
      streamControllers.delete(id)
    }
    await chatApi.deleteSession(id)
    streams.value.delete(id)
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

async function clearConversation() {
  const sessionId = activeSessionId.value
  if (sessionId === null || clearingConversation.value) return
  const confirmed = window.confirm(
    '确定清空当前会话吗？\n\n将删除本会话的消息、上下文快照和会话记忆；跨会话学习画像会保留。此操作不可撤销。',
  )
  if (!confirmed) return

  clearingConversation.value = true
  error.value = ''
  try {
    const controller = streamControllers.get(sessionId)
    if (controller) {
      controller.abort()
      await chatApi.cancelIntent(sessionId)
      streamControllers.delete(sessionId)
    }
    streams.value.delete(sessionId)
    const conversation = await chatApi.clearConversation(sessionId)
    const index = sessions.value.findIndex((item) => item.id === sessionId)
    if (index >= 0) sessions.value.splice(index, 1, conversation.session)
    if (activeSessionId.value === sessionId) {
      messages.value = []
      draft.value = ''
    }
  } catch (cause) {
    const detail = cause instanceof Error ? cause.message : '未知错误'
    error.value = `清空会话失败：${detail}`
  } finally {
    clearingConversation.value = false
  }
}

function chooseSuggestion(text: string) {
  draft.value = text
}

function openRagDashboard() {
  workspaceView.value = 'rag'
  error.value = ''
}

function openModelSettings() {
  workspaceView.value = 'settings'
  error.value = ''
}

async function sendMessage() {
  const content = draft.value.trim()
  if (!content) return

  if (!activeSessionId.value) {
    await createSession()
    if (!activeSessionId.value) return
  }

  const sessionId = activeSessionId.value
  if (streams.value.has(sessionId)) return

  draft.value = ''
  error.value = ''
  streams.value.set(sessionId, {
    status: '正在建立 SSE 连接',
    retry: null,
    activities: [{
      sequence: 0,
      phase: 'connecting',
      agent: '系统',
      message: '正在连接实时执行通道',
      detail: '连接建立后会持续显示每个 Agent 的当前任务',
    }],
    text: '',
    context: null,
  })
  const controller = new AbortController()
  streamControllers.set(sessionId, controller)
  messages.value.push({ id: -Date.now(), role: 'USER', content, createdAt: new Date().toISOString() })
  await scrollToBottom()

  const updateStream = (mutate: (state: StreamState) => void) => {
    const state = streams.value.get(sessionId)
    if (state) mutate(state)
  }

  try {
    await chatApi.streamIntent(sessionId, content, {
      onStatus(status) {
        updateStream((state) => {
          state.status = status.message
          state.retry = status.phase === 'reconnecting' && status.retryCount && status.maxRetries
            ? { current: status.retryCount, max: status.maxRetries }
            : null
          const activity: StreamActivity = {
            sequence: status.sequence ?? Date.now(),
            phase: status.phase,
            agent: status.agent ?? (status.phase === 'reconnecting' ? '模型连接' : '系统'),
            message: status.message,
            detail: status.detail ?? null,
          }
          const previous = state.activities[state.activities.length - 1]
          if (!previous || previous.sequence !== activity.sequence || previous.message !== activity.message) {
            state.activities = [...state.activities, activity].slice(-5)
          }
        })
      },
      onIntent(result) {
        updateStream((state) => {
          state.retry = null
          state.status = `已由 ${result.model} 完成回答与润色`
          state.text = result.content
          state.context = result.context_snapshot
        })
      },
      onComplete(conversation) {
        streams.value.delete(sessionId)
        const index = sessions.value.findIndex((item) => item.id === sessionId)
        if (index >= 0) sessions.value.splice(index, 1, conversation.session)
        if (activeSessionId.value === sessionId) {
          messages.value = conversation.messages
          void scrollToBottom()
        }
      },
    }, controller.signal)
  } catch (cause) {
    streams.value.delete(sessionId)
    if (controller.signal.aborted) return
    if (activeSessionId.value === sessionId) {
      try {
        const conversation = await chatApi.getConversation(sessionId)
        messages.value = conversation.messages
      } catch {
        messages.value = messages.value.filter((message) => message.id > 0)
        draft.value = content
      }
      const detail = cause instanceof Error ? cause.message : '未知错误'
      error.value = `智能体回答失败：${detail}`
    }
  } finally {
    if (streamControllers.get(sessionId) === controller) {
      streamControllers.delete(sessionId)
    }
  }
}

async function pauseGeneration() {
  const sessionId = activeSessionId.value
  if (sessionId === null || !streams.value.has(sessionId)) return

  error.value = ''
  streams.value.delete(sessionId)
  streamControllers.get(sessionId)?.abort()
  const optimisticUserMessages = messages.value.filter((message) => message.id < 0 && message.role === 'USER')

  try {
    await chatApi.cancelIntent(sessionId)
    const conversation = await chatApi.getConversation(sessionId)
    const index = sessions.value.findIndex((item) => item.id === sessionId)
    if (index >= 0) sessions.value.splice(index, 1, conversation.session)
    if (activeSessionId.value === sessionId) {
      const missingOptimisticMessages = optimisticUserMessages.filter((pending) =>
        !conversation.messages.some((saved) =>
          saved.role === 'USER'
          && saved.content === pending.content
          && Math.abs(new Date(saved.createdAt).getTime() - new Date(pending.createdAt).getTime()) < 60_000,
        ),
      )
      messages.value = [...conversation.messages, ...missingOptimisticMessages]
      await scrollToBottom()
    }
  } catch {
    if (activeSessionId.value === sessionId) {
      error.value = '已停止接收本次回答，但后端取消状态暂时无法确认。'
    }
  }
}

function handleComposerKeydown(event: KeyboardEvent) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    if (activeStream.value) return
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

function currentActivity(stream: StreamState) {
  return stream.activities[stream.activities.length - 1]
}

function previousActivities(stream: StreamState) {
  return stream.activities.slice(-4, -1)
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
          :class="{ active: workspaceView === 'chat' && session.id === activeSessionId }"
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

      <button
        class="sidebar-tool"
        :class="{ active: workspaceView === 'rag' }"
        type="button"
        @click="openRagDashboard"
      >
        <span class="tool-icon"><i></i><i></i><i></i></span>
        <span><strong>RAG 可视化</strong><small>知识库与向量状态</small></span>
        <b>→</b>
      </button>

      <button
        class="sidebar-tool model-settings-tool"
        :class="{ active: workspaceView === 'settings' }"
        type="button"
        @click="openModelSettings"
      >
        <span class="settings-tool-icon" aria-hidden="true">⌘</span>
        <span><strong>服务设置</strong><small>模型 · SerpAPI · Redis</small></span>
        <b>→</b>
      </button>

      <div class="sidebar-footer">
        <div class="avatar">学</div>
        <div><strong>算法学习者</strong><span>持续学习中</span></div>
        <span class="status-dot" title="服务状态"></span>
      </div>
    </aside>

    <main v-if="workspaceView === 'chat'" class="workspace">
      <header class="topbar">
        <div>
          <span class="eyebrow">当前探索</span>
          <h1>{{ activeSession?.title ?? '新的算法探索' }}</h1>
        </div>
        <div class="topbar-actions">
          <div class="memory-pill"><span></span> 上下文已保存</div>
          <button
            class="clear-conversation"
            type="button"
            :disabled="!activeSessionId || clearingConversation"
            @click="clearConversation"
          >{{ clearingConversation ? '正在清空…' : '清空对话' }}</button>
        </div>
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
              <MessageContent :content="visibleMessageContent(message.content)" :tone="message.role === 'USER' ? 'user' : 'assistant'" />
              <ContextInspector v-if="message.contextSnapshot" :snapshot="message.contextSnapshot" />
            </div>
          </article>
          <article v-if="activeStream" class="message assistant">
            <div class="message-avatar"><LogoMark size="small" /></div>
            <div class="bubble live-intent">
              <div class="message-meta">AlgoMate · 实时执行轨迹</div>
              <div v-if="activeStream.retry" class="retry-indicator" role="status">
                <span class="retry-icon">↻</span>
                <div class="retry-copy">
                  <strong>正在重新连接 {{ activeStream.retry.current }}/{{ activeStream.retry.max }}</strong>
                  <span class="retry-track"><i :style="{ width: `${activeStream.retry.current / activeStream.retry.max * 100}%` }"></i></span>
                </div>
              </div>
              <MessageContent v-if="activeStream.text" :content="activeStream.text" tone="live" />
              <ContextInspector v-if="activeStream.context" :snapshot="activeStream.context" open />
              <div v-else-if="!activeStream.retry" class="thinking-progress" role="status" aria-live="polite">
                <div class="activity-current">
                  <span class="activity-orbit" aria-hidden="true"><i></i></span>
                  <div class="activity-copy">
                    <small>{{ currentActivity(activeStream).agent }}</small>
                    <strong>{{ currentActivity(activeStream).message }}</strong>
                    <p v-if="currentActivity(activeStream).detail">{{ currentActivity(activeStream).detail }}</p>
                  </div>
                  <span class="activity-live">LIVE</span>
                </div>
                <div v-if="previousActivities(activeStream).length" class="activity-trail">
                  <div v-for="activity in previousActivities(activeStream)" :key="`${activity.sequence}-${activity.phase}`">
                    <span>✓</span>
                    <p><small>{{ activity.agent }}</small>{{ activity.message }}</p>
                  </div>
                </div>
              </div>
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
          <button
            v-if="activeStream"
            class="pause-generation"
            type="button"
            aria-label="暂停生成"
            title="暂停生成"
            @click="pauseGeneration"
          ><span aria-hidden="true"></span></button>
          <button v-else type="submit" :disabled="!draft.trim()" aria-label="发送消息">↑</button>
        </form>
        <p>{{ activeStream ? '点击停止按钮可暂停本次生成 · 已有对话不会丢失' : 'Enter 发送 · Shift + Enter 换行 · 学习记录将保存到本地数据库' }}</p>
      </div>
    </main>
    <ModelSettings v-else-if="workspaceView === 'settings'" />
    <RagDashboard v-else />
  </div>
</template>
