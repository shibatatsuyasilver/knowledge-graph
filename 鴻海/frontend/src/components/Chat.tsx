import { useMemo, useState } from 'react'
import { chatGeneral, queryKg, toApiError } from '../api'
import type { ChatHistoryMessage } from '../types'

type ChatMode = 'kg' | 'general'

interface ChatMessage extends ChatHistoryMessage {
  cypher?: string
  rows?: Array<Record<string, unknown>>
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

const Chat = () => {
  const [chatMode, setChatMode] = useState<ChatMode>('kg')
  const [kgMessages, setKgMessages] = useState<ChatMessage[]>(() => loadMessages('kg'))
  const [generalMessages, setGeneralMessages] = useState<ChatMessage[]>(() => loadMessages('general'))
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)

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

    try {
      if (chatMode === 'kg') {
        const response = await queryKg(text)
        const assistantMessage: ChatMessage = {
          role: 'assistant',
          content: response.answer?.trim() || summarizeKgRows(response.rows),
          cypher: response.cypher,
          rows: response.rows,
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
          </article>
        ))}

        {loading && <p className="hint-text">Assistant is thinking...</p>}
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
