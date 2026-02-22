export type Language = 'zh-tw' | 'en'
export type ExtractionProvider = 'ollama' | 'gemini'

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

export interface FailedUrl {
  url: string
  error: string
}

export interface KeywordProcessResponse extends BuildKgResponse {
  searched_keyword: string
  fetched_urls: string[]
  failed_urls: FailedUrl[]
}

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

export interface QueryResponse {
  question: string
  cypher: string
  rows: Array<Record<string, unknown>>
  answer?: string
  attempt?: number
  answer_source?: 'qa_llm' | 'template_fallback'
  agentic_trace?: {
    stage: string
    round_count: number
    replan_count: number
    final_strategy: string
    failure_chain: string[]
    llm_provider?: string
    llm_model?: string
  }
}

export interface QueryRequest {
  question: string
  nl2cypher_provider?: ExtractionProvider
  nl2cypher_model?: string
}

export type QueryJobStatus = 'running' | 'completed' | 'failed'

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
}

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
