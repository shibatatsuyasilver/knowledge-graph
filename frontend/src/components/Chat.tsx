import { useMemo, useState } from 'react'
import { chatGeneral, getQueryKgJob, startQueryKgAsync, toApiError } from '../api'
import type { ChatHistoryMessage, ExtractionProvider, QueryProgressSnapshot } from '../types'

type ChatMode = 'kg' | 'general'

interface ChatMessage extends ChatHistoryMessage {
  cypher?: string
  rows?: Array<Record<string, unknown>>
  agentic_trace?: {
    stage?: string
    round_count?: number
    replan_count?: number
    final_strategy?: string
    failure_chain?: string[]
    llm_provider?: string
    llm_model?: string
  }
}

const STORAGE_KEYS: Record<ChatMode, string> = {
  kg: 'genai_chat_kg_history_v1',
  general: 'genai_chat_general_history_v1',
}

function loadMessages(mode: ChatMode): ChatMessage[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS[mode])
    if (!raw) {
      return []
    }
    const parsed = JSON.parse(raw) as ChatMessage[]
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

function saveMessages(mode: ChatMode, messages: ChatMessage[]) {
  localStorage.setItem(STORAGE_KEYS[mode], JSON.stringify(messages))
}

function summarizeKgRows(rows: Array<Record<string, unknown>>): string {
  if (rows.length === 0) {
    return '找不到相關資訊。'
  }
  return `我查到 ${rows.length} 筆資料。可展開查看 Cypher 與原始資料。`
}

const STAGE_LABELS: Record<string, string> = {
  planner: 'Planner',
  react: 'Reactor',
  critic: 'Critic',
  replan: 'Replanner',
  done: 'Done',
  fail_fast: 'Fail Fast',
  exhausted: 'Exhausted',
}

const DEFAULT_OLLAMA_QUERY_MODEL = 'ministral-3:14b'
const DEFAULT_GEMINI_QUERY_MODEL = 'gemini-3-pro-preview'
const QUERY_JOB_POLL_INTERVAL_MS = 350
const QUERY_JOB_TIMEOUT_MS = 20 * 60 * 1000
const QUERY_JOB_MAX_POLLS = Math.ceil(QUERY_JOB_TIMEOUT_MS / QUERY_JOB_POLL_INTERVAL_MS)

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms)
  })
}

function resolveStageLabel(stage?: string): string {
  return STAGE_LABELS[stage ?? ''] ?? stage ?? 'Running'
}

function resolveStrategyLabel(strategy?: string): string {
  if (!strategy) {
    return 'single_query'
  }
  return strategy.replace(/_/g, ' ')
}

const Chat = () => {
  const [chatMode, setChatMode] = useState<ChatMode>('kg')
  const [kgMessages, setKgMessages] = useState<ChatMessage[]>(() => loadMessages('kg'))
  const [generalMessages, setGeneralMessages] = useState<ChatMessage[]>(() => loadMessages('general'))
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [loadingProgress, setLoadingProgress] = useState<QueryProgressSnapshot | null>(null)
  const [queryProvider, setQueryProvider] = useState<ExtractionProvider>('ollama')
  const [queryModel, setQueryModel] = useState(DEFAULT_OLLAMA_QUERY_MODEL)

  const activeMessages = useMemo(
    () => (chatMode === 'kg' ? kgMessages : generalMessages),
    [chatMode, kgMessages, generalMessages],
  )

  const setMessagesByMode = (mode: ChatMode, messages: ChatMessage[]) => {
    if (mode === 'kg') {
      setKgMessages(messages)
    } else {
      setGeneralMessages(messages)
    }
    saveMessages(mode, messages)
  }

  const clearCurrentMode = () => {
    setMessagesByMode(chatMode, [])
  }

  const handleSend = async () => {
    const text = input.trim()
    if (!text || loading) {
      return
    }

    const userMessage: ChatMessage = { role: 'user', content: text }
    const baseMessages = [...activeMessages, userMessage]

    setMessagesByMode(chatMode, baseMessages)
    setInput('')
    setLoading(true)
    setLoadingProgress(
      chatMode === 'kg'
        ? {
            stage: 'planner',
            round_count: 0,
            replan_count: 0,
            final_strategy: 'single_query',
            detail: 'Planning query strategy',
            llm_provider: queryProvider,
            llm_model: queryModel.trim() || undefined,
          }
        : null,
    )

    try {
      if (chatMode === 'kg') {
        const started = await startQueryKgAsync({
          question: text,
          nl2cypher_provider: queryProvider,
          nl2cypher_model: queryModel.trim() || undefined,
        })
        let response: {
          answer?: string
          cypher: string
          rows: Array<Record<string, unknown>>
          agentic_trace?: ChatMessage['agentic_trace']
        } | null = null

        for (let attempt = 0; attempt < QUERY_JOB_MAX_POLLS; attempt += 1) {
          const job = await getQueryKgJob(started.job_id)
          if (job.progress) {
            setLoadingProgress(job.progress)
          }
          if (job.status === 'completed') {
            if (!job.result) {
              throw new Error('Query job completed without result')
            }
            response = job.result
            break
          }
          if (job.status === 'failed') {
            const reason = job.error || job.progress?.detail || 'Query job failed'
            throw new Error(reason)
          }
          await sleep(QUERY_JOB_POLL_INTERVAL_MS)
        }

        if (!response) {
          throw new Error('Query job timed out after 20 minutes')
        }

        const assistantMessage: ChatMessage = {
          role: 'assistant',
          content: response.answer?.trim() || summarizeKgRows(response.rows),
          cypher: response.cypher,
          rows: response.rows,
          agentic_trace: response.agentic_trace,
        }
        setMessagesByMode('kg', [...baseMessages, assistantMessage])
      } else {
        const history = baseMessages.map((item) => ({ role: item.role, content: item.content }))
        const response = await chatGeneral(text, history)
        const assistantMessage: ChatMessage = {
          role: 'assistant',
          content: response.answer,
        }
        setMessagesByMode('general', [...baseMessages, assistantMessage])
      }
    } catch (err) {
      const assistantError: ChatMessage = {
        role: 'assistant',
        content: `❌ ${toApiError(err)}`,
      }
      setMessagesByMode(chatMode, [...baseMessages, assistantError])
    } finally {
      setLoading(false)
      setLoadingProgress(null)
    }
  }

  return (
    <section className="panel">
      <div className="panel-header with-actions">
        <div>
          <h2>Chat</h2>
          <p>切換 KG 問答與一般陪聊模式，歷史紀錄分開保存</p>
        </div>
        <button type="button" className="ghost-btn" onClick={clearCurrentMode}>
          Clear Current Mode
        </button>
      </div>

      <div className="mode-row" role="tablist" aria-label="Chat modes">
        <button
          type="button"
          className={`mode-btn ${chatMode === 'kg' ? 'active' : ''}`}
          onClick={() => setChatMode('kg')}
        >
          KG QA
        </button>
        <button
          type="button"
          className={`mode-btn ${chatMode === 'general' ? 'active' : ''}`}
          onClick={() => setChatMode('general')}
        >
          Companion
        </button>
      </div>

      {chatMode === 'kg' && (
        <div className="input-block">
          <div className="keyword-grid">
            <label className="field-label">
              <span>Query Provider</span>
              <select
                value={queryProvider}
                onChange={(event) => {
                  const next = event.target.value as ExtractionProvider
                  setQueryProvider(next)
                  setQueryModel(next === 'gemini' ? DEFAULT_GEMINI_QUERY_MODEL : DEFAULT_OLLAMA_QUERY_MODEL)
                }}
                className="text-input"
                disabled={loading}
              >
                <option value="ollama">Ollama</option>
                <option value="gemini">Gemini</option>
              </select>
            </label>

            <label className="field-label">
              <span>Query Model</span>
              <input
                type="text"
                value={queryModel}
                onChange={(event) => setQueryModel(event.target.value)}
                className="text-input"
                placeholder={queryProvider === 'gemini' ? DEFAULT_GEMINI_QUERY_MODEL : DEFAULT_OLLAMA_QUERY_MODEL}
                disabled={loading}
              />
            </label>
          </div>
        </div>
      )}

      <div className="chat-window" aria-live="polite">
        {activeMessages.length === 0 && (
          <p className="hint-text">
            {chatMode === 'kg'
              ? '在 KG QA 模式輸入問題，系統會先回覆文字答案，並提供可展開的 Cypher 與 Rows。'
              : '在 Companion 模式可以一般聊天，適合失眠陪聊情境。'}
          </p>
        )}

        {activeMessages.map((message, index) => (
          <article key={`${message.role}-${index}`} className={`bubble ${message.role}`}>
            <p className="bubble-role">{message.role === 'user' ? 'You' : 'Assistant'}</p>
            <p className="bubble-content">{message.content}</p>

            {chatMode === 'kg' && message.cypher && (
              <details>
                <summary>Cypher</summary>
                <pre className="json-preview">{message.cypher}</pre>
              </details>
            )}

            {chatMode === 'kg' && message.rows && message.rows.length > 0 && (
              <details>
                <summary>Rows</summary>
                <pre className="json-preview">{JSON.stringify(message.rows, null, 2)}</pre>
              </details>
            )}

            {chatMode === 'kg' && message.agentic_trace && (
              <details>
                <summary>Agentic Trace</summary>
                <pre className="json-preview">{JSON.stringify(message.agentic_trace, null, 2)}</pre>
              </details>
            )}
          </article>
        ))}

        {loading && chatMode === 'kg' && (
          <article className="agentic-progress">
            <p className="bubble-role">Agentic Progress</p>
            <p className="hint-text">
              Plan: <strong>{resolveStrategyLabel(loadingProgress?.final_strategy)}</strong>
            </p>
            <p className="hint-text">
              LLM:{' '}
              <strong>{loadingProgress?.llm_provider || queryProvider}</strong>
              {' · '}
              {(loadingProgress?.llm_model || queryModel).trim() || '-'}
            </p>
            <p className="hint-text">
              Current stage:{' '}
              <strong>{resolveStageLabel(loadingProgress?.stage)}</strong>
              {' · '}
              round {loadingProgress?.round_count ?? 0}
              {' · '}
              replan {loadingProgress?.replan_count ?? 0}
            </p>
            <p className="hint-text">{loadingProgress?.detail || 'Running agentic loop...'}</p>
            {loadingProgress?.failure_chain && loadingProgress.failure_chain.length > 0 && (
              <details>
                <summary>Failure Chain</summary>
                <pre className="json-preview">{JSON.stringify(loadingProgress.failure_chain, null, 2)}</pre>
              </details>
            )}
          </article>
        )}

        {loading && chatMode !== 'kg' && <p className="hint-text">Assistant is thinking...</p>}
      </div>

      <div className="chat-composer">
        <input
          type="text"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') {
              void handleSend()
            }
          }}
          placeholder={chatMode === 'kg' ? '輸入圖譜問題...' : '輸入聊天訊息...'}
          className="text-input"
          disabled={loading}
        />
        <button type="button" className="primary-btn" onClick={() => void handleSend()} disabled={loading}>
          Send
        </button>
      </div>
    </section>
  )
}

export default Chat
