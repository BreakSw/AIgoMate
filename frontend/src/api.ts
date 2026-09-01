import type {
  ChatSession,
  Conversation,
  IntentResult,
  ModelConfigInput,
  ModelConfigStatus,
  RagOverview,
  StreamStatus,
} from './types'

const USER_ID = 1

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!response.ok) {
    const body = await response.json().catch(() => null) as { detail?: string } | null
    throw new Error(body?.detail || `请求失败（${response.status}）`)
  }
  return response.json() as Promise<T>
}

export const chatApi = {
  getModelConfig: () => request<ModelConfigStatus>('/api/model-config'),
  saveModelConfig: (payload: ModelConfigInput) =>
    request<ModelConfigStatus>('/api/model-config', {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),
  deleteModelConfig: async () => {
    const response = await fetch('/api/model-config', {
      method: 'DELETE',
    })
    if (!response.ok) throw new Error(`删除模型配置失败（${response.status}）`)
  },
  deleteModelConnection: async () => {
    const response = await fetch('/api/model-config/model', { method: 'DELETE' })
    if (!response.ok) throw new Error(`删除模型连接失败（${response.status}）`)
  },
  deleteSearchConnection: async () => {
    const response = await fetch('/api/model-config/search', { method: 'DELETE' })
    if (!response.ok) throw new Error(`删除搜索配置失败（${response.status}）`)
  },
  getRagOverview: () => request<RagOverview>(`/api/rag/overview?userId=${USER_ID}`),
  listSessions: () => request<ChatSession[]>(`/api/sessions?userId=${USER_ID}`),
  createSession: (title?: string) =>
    request<ChatSession>('/api/sessions', {
      method: 'POST',
      body: JSON.stringify({ userId: USER_ID, title }),
    }),
  getConversation: (sessionId: number) =>
    request<Conversation>(`/api/sessions/${sessionId}/messages?userId=${USER_ID}`),
  sendMessage: (sessionId: number, content: string) =>
    request<Conversation>(`/api/sessions/${sessionId}/messages`, {
      method: 'POST',
      body: JSON.stringify({ userId: USER_ID, content }),
    }),
  streamIntent: async (
    sessionId: number,
    content: string,
    handlers: {
      onStatus: (status: StreamStatus) => void
      onIntent: (result: IntentResult) => void
      onComplete: (conversation: Conversation) => void
    },
    signal?: AbortSignal,
  ) => {
    const response = await fetch(`/api/sessions/${sessionId}/messages/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      },
      body: JSON.stringify({ userId: USER_ID, content }),
      signal,
    })
    if (!response.ok || !response.body) {
      throw new Error(`SSE 连接失败（${response.status}）`)
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    const dispatch = (block: string) => {
      let eventName = 'message'
      const dataLines: string[] = []
      for (const line of block.split('\n')) {
        if (line.startsWith('event:')) eventName = line.slice(6).trim()
        if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart())
      }
      if (dataLines.length === 0) return
      const payload = JSON.parse(dataLines.join('\n'))
      if (eventName === 'status') handlers.onStatus(payload as StreamStatus)
      if (eventName === 'intent') handlers.onIntent(payload as IntentResult)
      if (eventName === 'complete') handlers.onComplete(payload as Conversation)
      if (eventName === 'failed') {
        throw new Error(payload.detail || payload.message || '意图识别失败')
      }
    }

    while (true) {
      const { value, done } = await reader.read()
      buffer += decoder.decode(value, { stream: !done }).replace(/\r\n/g, '\n')
      const blocks = buffer.split('\n\n')
      buffer = blocks.pop() ?? ''
      for (const block of blocks) dispatch(block)
      if (done) break
    }
    if (buffer.trim()) dispatch(buffer)
  },
  cancelIntent: async (sessionId: number) => {
    const response = await fetch(
      `/api/sessions/${sessionId}/messages/stream/cancel?userId=${USER_ID}`,
      { method: 'POST' },
    )
    if (!response.ok) throw new Error(`暂停失败（${response.status}）`)
  },
  deleteSession: async (sessionId: number) => {
    const response = await fetch(`/api/sessions/${sessionId}?userId=${USER_ID}`, {
      method: 'DELETE',
    })
    if (!response.ok) throw new Error(`删除失败（${response.status}）`)
  },
  clearConversation: (sessionId: number) =>
    request<Conversation>(
      `/api/sessions/${sessionId}/content?userId=${USER_ID}`,
      { method: 'DELETE' },
    ),
}
