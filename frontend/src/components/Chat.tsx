import { useCallback, useMemo, useRef, useState } from 'react'
import type { SyntheticEvent, WheelEvent } from 'react'
import { chatGeneral, getQueryKgJob, queryKg, startQueryKgAsync, toApiError } from '../api'
import type { AgenticTrace, ChatHistoryMessage, ExtractionProvider, QueryEngine, QueryProgressSnapshot } from '../types'

type ChatMode = 'kg' | 'general'
type KgExecutionMode = 'async_manual' | 'sync_graph_chain'

interface ChatMessage extends ChatHistoryMessage {
  query_engine?: QueryEngine
  cypher?: string
  rows?: Array<Record<string, unknown>>
  graph_chain_raw?: Record<string, unknown>
  agentic_trace?: AgenticTrace
}

const STORAGE_KEYS: Record<ChatMode, string> = {
  kg: 'genai_chat_kg_history_v1',
  general: 'genai_chat_general_history_v1',
}

function loadMessages(mode: ChatMode): ChatMessage[] {
  // ─── 階段 1：從 localStorage 讀取該模式的歷史訊息 ─────────
  try {
    const raw = localStorage.getItem(STORAGE_KEYS[mode])
    if (!raw) {
      return []
    }
    // ─── 階段 2：解析與型別保底 ───────────────────────────
    const parsed = JSON.parse(raw) as ChatMessage[]
    return Array.isArray(parsed) ? parsed : []
  } catch {
    // ─── 階段 3：資料損壞時回退空陣列 ──────────────────────
    return []
  }
}

function saveMessages(mode: ChatMode, messages: ChatMessage[]) {
  // 將兩種模式 (`kg` / `general`) 的歷史分開儲存，避免上下文互相汙染。
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
  const [kgExecutionMode, setKgExecutionMode] = useState<KgExecutionMode>('async_manual')
  const chatWindowRef = useRef<HTMLDivElement | null>(null)

  const alignDebugDetailsIntoView = useCallback((detailsEl: HTMLDetailsElement) => {
    // ─── 階段 1：僅在 details 展開時調整捲動 ────────────────
    const chatWindow = chatWindowRef.current
    if (!chatWindow || !detailsEl.open) {
      return
    }

    const edgePadding = 12
    const containerRect = chatWindow.getBoundingClientRect()
    const detailsRect = detailsEl.getBoundingClientRect()
    const bottomOverflow = detailsRect.bottom - (containerRect.bottom - edgePadding)

    if (bottomOverflow > 0) {
      chatWindow.scrollTop += bottomOverflow
    }

    // ─── 階段 2：二次校正頂部溢出 ─────────────────────────
    const adjustedRect = detailsEl.getBoundingClientRect()
    const topOverflow = containerRect.top + edgePadding - adjustedRect.top
    if (topOverflow > 0) {
      chatWindow.scrollTop -= topOverflow
    }
  }, [])

  const handleDebugDetailsToggle = useCallback(
    (event: SyntheticEvent<HTMLDetailsElement>) => {
      const detailsEl = event.currentTarget
      if (!detailsEl.open) {
        return
      }
      window.requestAnimationFrame(() => {
        alignDebugDetailsIntoView(detailsEl)
      })
    },
    [alignDebugDetailsIntoView],
  )

  const handleDebugPreviewWheel = useCallback((event: WheelEvent<HTMLPreElement>) => {
    // ─── 階段 1：判斷內層 `pre` 是否可滾動 ──────────────────
    const preview = event.currentTarget
    const { scrollTop, scrollHeight, clientHeight } = preview
    if (scrollHeight <= clientHeight) {
      return
    }

    const atTop = scrollTop <= 0
    const atBottom = scrollTop + clientHeight >= scrollHeight - 1
    const scrollingDown = event.deltaY > 0
    const scrollingUp = event.deltaY < 0
    const canInnerScroll = (scrollingDown && !atBottom) || (scrollingUp && !atTop)

    // ─── 階段 2：若內層仍可滾動，阻止事件冒泡到外層聊天窗 ─────
    if (canInnerScroll) {
      event.stopPropagation()
    }
  }, [])

  const activeMessages = useMemo(
    () => (chatMode === 'kg' ? kgMessages : generalMessages),
    [chatMode, kgMessages, generalMessages],
  )

  const setMessagesByMode = (mode: ChatMode, messages: ChatMessage[]) => {
    // ─── 階段 1：同步 React state ─────────────────────────
    if (mode === 'kg') {
      setKgMessages(messages)
    } else {
      setGeneralMessages(messages)
    }
    // ─── 階段 2：同步 localStorage ─────────────────────────
    saveMessages(mode, messages)
  }

  const clearCurrentMode = () => {
    setMessagesByMode(chatMode, [])
  }

  const handleSend = async () => {
    // ─── 階段 1：await 之前（同步）────────────────────────
    const text = input.trim()
    if (!text || loading) {
      return
    }

    const userMessage: ChatMessage = { role: 'user', content: text }
    const baseMessages = [...activeMessages, userMessage]

    setMessagesByMode(chatMode, baseMessages)
    //  ↑ Optimistic Update，立刻渲染使用者的訊息
    setInput('')
    //  ↑ 清空輸入框
    setLoading(true)
    //  ↑ 標示載入中狀態，禁用輸入與送出按鈕
    setLoadingProgress(
      chatMode === 'kg' && kgExecutionMode === 'async_manual'
        ? {
            stage: 'planner',
            round_count: 0,
            replan_count: 0,
            final_strategy: 'single_query',
            detail: 'Planning query strategy',
            llm_provider: queryProvider,
            llm_model: queryModel.trim() || undefined,
            agentic_trace: {
              stage: 'planner',
              round_count: 0,
              replan_count: 0,
              final_strategy: 'single_query',
              failure_chain: [],
              llm_provider: queryProvider,
              llm_model: queryModel.trim() || undefined,
              rounds: [],
            },
          }
        : null,
    )
    //  ↑ 針對 async 模式，立刻初始化進度狀態，讓 UI 顯示準備階段

    // ─── 階段 2：等待 IO ──────────────────────────────────
    try {
      if (chatMode === 'kg') {
        if (kgExecutionMode === 'sync_graph_chain') {
          const response = await queryKg({
            question: text,
            nl2cypher_provider: queryProvider,
            nl2cypher_model: queryModel.trim() || undefined,
            query_engine: 'graph_chain',
          })
          //  ↑ 同步模式，發送 API 請求並等待回覆
          
          // ─── 階段 3：await 之後（React 19 自動標記 Transition）─
          const assistantMessage: ChatMessage = {
            role: 'assistant',
            content: response.answer?.trim() || summarizeKgRows(response.rows),
            query_engine: response.query_engine ?? 'graph_chain',
            cypher: response.cypher,
            rows: response.rows,
            graph_chain_raw: response.graph_chain_raw,
          }
          setMessagesByMode('kg', [...baseMessages, assistantMessage])
          //  ↑ 將伺服器回傳的 Assistant 訊息加入聊天紀錄中
        } else {
          const started = await startQueryKgAsync({
            question: text,
            nl2cypher_provider: queryProvider,
            nl2cypher_model: queryModel.trim() || undefined,
            query_engine: 'manual',
          })
          //  ↑ 發送非同步任務建立請求
          
          let response: {
            answer?: string
            cypher: string
            rows: Array<Record<string, unknown>>
            query_engine?: QueryEngine
            agentic_trace?: ChatMessage['agentic_trace']
          } | null = null
          let failedMessage: ChatMessage | null = null

          //  ↓ 輪詢機制，每隔 QUERY_JOB_POLL_INTERVAL_MS 查詢任務進度
          for (let attempt = 0; attempt < QUERY_JOB_MAX_POLLS; attempt += 1) {
            const job = await getQueryKgJob(started.job_id)
            if (job.progress) {
              setLoadingProgress(job.progress)
              //  ↑ 更新畫面的進度狀態
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
              failedMessage = {
                role: 'assistant',
                content: `❌ ${reason}`,
                query_engine: 'manual',
                agentic_trace: job.progress?.agentic_trace,
              }
              break
            }
            await sleep(QUERY_JOB_POLL_INTERVAL_MS)
          }

          // ─── 階段 3：await 之後（React 19 自動標記 Transition）─
          if (failedMessage) {
            setMessagesByMode('kg', [...baseMessages, failedMessage])
            //  ↑ 如果任務失敗，顯示錯誤訊息
            return
          }

          if (!response) {
            throw new Error('Query job timed out after 20 minutes')
          }

          const assistantMessage: ChatMessage = {
            role: 'assistant',
            content: response.answer?.trim() || summarizeKgRows(response.rows),
            query_engine: response.query_engine ?? 'manual',
            cypher: response.cypher,
            rows: response.rows,
            agentic_trace: response.agentic_trace,
          }
          setMessagesByMode('kg', [...baseMessages, assistantMessage])
          //  ↑ 將成功取得的結果轉換為訊息加入聊天紀錄
        }
      } else {
        const history = baseMessages.map((item) => ({ role: item.role, content: item.content }))
        const response = await chatGeneral(text, history)
        //  ↑ 一般模式的 API 請求
        
        // ─── 階段 3：await 之後（React 19 自動標記 Transition）─
        const assistantMessage: ChatMessage = {
          role: 'assistant',
          content: response.answer,
        }
        setMessagesByMode('general', [...baseMessages, assistantMessage])
        //  ↑ 加入伺服器回覆
      }
    } catch (err) {
      const assistantError: ChatMessage = {
        role: 'assistant',
        content: `❌ ${toApiError(err)}`,
      }
      // ─── 階段 3：await 之後（React 19 自動標記 Transition）─
      setMessagesByMode(chatMode, [...baseMessages, assistantError])
      //  ↑ 捕捉錯誤並顯示給使用者
    } finally {
      // ─── 階段 3：await 之後（React 19 自動標記 Transition）─
      setLoading(false)
      //  ↑ 結束 loading 狀態，還原 UI
      setLoadingProgress(null)
      //  ↑ 清空任務進度資料
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
              <span>Execution Mode</span>
              <select
                value={kgExecutionMode}
                onChange={(event) => setKgExecutionMode(event.target.value as KgExecutionMode)}
                className="text-input"
                disabled={loading}
              >
                <option value="async_manual">Agentic Async (Manual)</option>
                <option value="sync_graph_chain">GraphCypherQAChain (Sync)</option>
              </select>
            </label>

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

      <div className="chat-window" aria-live="polite" ref={chatWindowRef}>
        {activeMessages.length === 0 && (
          <p className="hint-text">
            {chatMode === 'kg'
              ? kgExecutionMode === 'sync_graph_chain'
                ? '在 GraphCypherQAChain (Sync) 模式輸入問題，系統會回覆文字答案，並提供可展開的 Cypher 與 Raw。'
                : '在 Agentic Async (Manual) 模式輸入問題，系統會先回覆文字答案，並提供可展開的 Cypher 與 Rows。'
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

            {chatMode === 'kg' && message.graph_chain_raw && (
              <details>
                <summary>Raw</summary>
                <pre className="json-preview">{JSON.stringify(message.graph_chain_raw, null, 2)}</pre>
              </details>
            )}

            {chatMode === 'kg' && message.agentic_trace && (
              <details onToggle={handleDebugDetailsToggle}>
                <summary>Plan</summary>
                <pre className="json-preview chat-debug-json-preview" onWheel={handleDebugPreviewWheel}>
                  {JSON.stringify(
                    {
                      plan_initial: message.agentic_trace.plan_initial,
                      planner_plan: message.agentic_trace.planner_plan,
                      plan_final: message.agentic_trace.plan_final,
                    },
                    null,
                    2,
                  )}
                </pre>
              </details>
            )}

            {chatMode === 'kg' && message.agentic_trace && (
              <details onToggle={handleDebugDetailsToggle}>
                <summary>ReAct Rounds</summary>
                <pre className="json-preview chat-debug-json-preview" onWheel={handleDebugPreviewWheel}>
                  {JSON.stringify(message.agentic_trace.rounds ?? [], null, 2)}
                </pre>
              </details>
            )}

            {chatMode === 'kg' && message.agentic_trace && (
              <details>
                <summary>Agentic Trace Meta</summary>
                <pre className="json-preview">
                  {JSON.stringify(
                    {
                      stage: message.agentic_trace.stage,
                      round_count: message.agentic_trace.round_count,
                      replan_count: message.agentic_trace.replan_count,
                      final_strategy: message.agentic_trace.final_strategy,
                      failure_chain: message.agentic_trace.failure_chain,
                      llm_provider: message.agentic_trace.llm_provider,
                      llm_model: message.agentic_trace.llm_model,
                    },
                    null,
                    2,
                  )}
                </pre>
              </details>
            )}
          </article>
        ))}

        {loading && chatMode === 'kg' && kgExecutionMode === 'async_manual' && (
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
            {loadingProgress?.agentic_trace && (
              <details>
                <summary>Debug Snapshot</summary>
                <pre className="json-preview">{JSON.stringify(loadingProgress.agentic_trace, null, 2)}</pre>
              </details>
            )}
          </article>
        )}

        {loading && chatMode === 'kg' && kgExecutionMode === 'sync_graph_chain' && (
          <p className="hint-text">GraphChain is thinking...</p>
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
