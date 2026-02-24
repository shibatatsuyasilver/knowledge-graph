// ─── 基礎列舉型別：前後端共用的模式與 provider ──────────────
export type Language = 'zh-tw' | 'en'
export type ExtractionProvider = 'ollama' | 'gemini'
export type QueryEngine = 'manual' | 'graph_chain'

// ─── Ingest / Build KG：統計與區塊明細 ─────────────────────
export interface KGStats {
  chunks_processed: number
  entities: number
  relations: number
  merged_entities: number
  dropped_relations: number
  json_retries: number
}

export interface ExtractionSummary {
  chunk_id: string
  entities: number
  relations: number
  source_url?: string
}

export interface ChunkProgress {
  order: number
  chunk_id: string
  source_url?: string
  title?: string
  chars: number
  status: 'queued' | 'processing' | 'processed' | 'skipped_by_limit' | 'failed'
  entities: number
  relations: number
  error?: string
  tokens?: number
}

export interface BuildKgResponse {
  stats: KGStats
  summary: ExtractionSummary[]
  chunk_limit?: number | null
  chunks_available?: number
  chunk_progress?: ChunkProgress[]
}

// ─── Keyword Crawl：搜尋與失敗網址紀錄 ────────────────────
export interface FailedUrl {
  url: string
  error: string
}

export interface KeywordProcessResponse extends BuildKgResponse {
  searched_keyword: string
  fetched_urls: string[]
  failed_urls: FailedUrl[]
}

// ─── 非同步 Job 狀態：Ingest / Keyword 共用契約 ────────────
export type KeywordJobStatus = 'running' | 'completed' | 'failed'

export interface IngestProgressSnapshot extends BuildKgResponse {
  status?: KeywordJobStatus
  current_url?: string
  error?: string
}

export interface KeywordProgressSnapshot extends KeywordProcessResponse {
  status?: KeywordJobStatus
  current_url?: string
  error?: string
}

export interface KeywordJobStartResponse {
  job_id: string
  status: 'running'
}

export interface KeywordJobStateResponse {
  job_id: string
  status: KeywordJobStatus
  progress?: KeywordProgressSnapshot
  result?: KeywordProcessResponse
  error?: string
}

export interface IngestJobStartResponse {
  job_id: string
  status: 'running'
}

export interface IngestJobStateResponse {
  job_id: string
  status: KeywordJobStatus
  progress?: IngestProgressSnapshot
  result?: BuildKgResponse
  error?: string
}

// ─── Request Payload：建立圖譜與查詢輸入格式 ───────────────
export interface KeywordRequest {
  keyword: string
  max_results?: number
  language?: Language
  site_allowlist?: string[]
  chunk_limit?: number
  extraction_provider?: ExtractionProvider
  extraction_model?: string
}

export interface IngestRequest {
  chunk_limit?: number
  extraction_provider?: ExtractionProvider
  extraction_model?: string
}

export interface TextIngestRequest extends IngestRequest {
  text: string
}

export interface UrlIngestRequest extends IngestRequest {
  url: string
}

// ─── Query KG：同步/非同步查詢結果與追蹤資訊 ───────────────
export interface QueryResponse {
  question: string
  cypher: string
  rows: Array<Record<string, unknown>>
  answer?: string
  attempt?: number
  answer_source?: 'qa_llm' | 'template_fallback'
  query_engine?: QueryEngine
  graph_chain_raw?: Record<string, unknown>
  engine_provider?: ExtractionProvider
  engine_model?: string
  agentic_trace?: AgenticTrace
}

export interface QueryRequest {
  question: string
  nl2cypher_provider?: ExtractionProvider
  nl2cypher_model?: string
  query_engine?: QueryEngine
}

export type QueryJobStatus = 'running' | 'completed' | 'failed'

// ─── Agentic Trace：Planner/ReAct/Critic/Replan 除錯欄位 ────
export interface AgenticIssue {
  code: string
  message: string
  severity?: string
}

export interface AgenticPlanSnapshot {
  intent?: string
  strategy?: string
  must_have_paths?: string[]
  forbidden_patterns?: string[]
  output_contract?: {
    columns?: string[]
  }
  risk_hypotheses?: string[]
}

export interface AgenticRoundTrace {
  round?: number
  strategy_before?: string
  reactor?: {
    cypher?: string
    assumptions?: string[]
    self_checks?: {
      schema_grounded?: boolean
      projection_consistent?: boolean
    }
  }
  candidate_cypher?: string
  linked_cypher?: string
  checked_cypher?: string
  deterministic_issues?: AgenticIssue[]
  runtime_error?: string
  rows_count?: number
  critic?: {
    verdict?: string
    issues?: AgenticIssue[]
    repair_actions?: string[]
    next_strategy?: string
  }
  verdict?: string
  replan?: {
    strategy?: string
    delta_actions?: string[]
    tightened_constraints?: string[]
    stop_if?: string[]
  }
  strategy_after?: string
}

export interface AgenticTrace {
  stage?: string
  round_count?: number
  replan_count?: number
  final_strategy?: string
  failure_chain?: string[]
  llm_provider?: string
  llm_model?: string
  plan_initial?: AgenticPlanSnapshot
  planner_plan?: AgenticPlanSnapshot
  plan_final?: AgenticPlanSnapshot
  rounds?: AgenticRoundTrace[]
}

export interface QueryProgressSnapshot {
  status?: QueryJobStatus
  question?: string
  stage?: string
  round_count?: number
  replan_count?: number
  final_strategy?: string
  failure_chain?: string[]
  detail?: string
  llm_provider?: string
  llm_model?: string
  agentic_trace?: AgenticTrace
}

// ─── Query Job / General Chat：最外層 API 回應 ─────────────
export interface QueryJobStartResponse {
  job_id: string
  status: 'running'
}

export interface QueryJobStateResponse {
  job_id: string
  status: QueryJobStatus
  progress?: QueryProgressSnapshot
  result?: QueryResponse
  error?: string
}

export interface ChatHistoryMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface GeneralChatResponse {
  answer: string
  model: string
}
