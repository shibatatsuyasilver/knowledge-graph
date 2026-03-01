import axios from 'axios'
import type {
  ChatHistoryMessage,
  GeneralChatResponse,
  IngestJobStartResponse,
  IngestJobStateResponse,
  KeywordJobStartResponse,
  KeywordJobStateResponse,
  KeywordRequest,
  QueryRequest,
  QueryJobStartResponse,
  QueryJobStateResponse,
  QueryResponse,
  TextIngestRequest,
  UrlIngestRequest,
} from './types'

const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? '',
  timeout: 900000,
  headers: {
    'Content-Type': 'application/json',
  },
})

export function toApiError(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data?.detail
    if (typeof detail === 'string' && detail.trim()) {
      return detail
    }
    return error.message
  }
  if (error instanceof Error) {
    return error.message
  }
  return 'Unknown error'
}

// ─── Internal helpers ─────────────────────────────────────────
async function _postJob<P, R>(path: string, payload: P): Promise<R> {
  return (await apiClient.post<R>(path, payload)).data
}

async function _getJob<R>(path: string): Promise<R> {
  return (await apiClient.get<R>(path)).data
}

// ─── Public API ────────────────────────────────────────────────
export async function startTextProcessAsync(payload: TextIngestRequest): Promise<IngestJobStartResponse> {
  return _postJob('/api/process_text_async/start', payload)
}

export async function getTextProcessJob(jobId: string): Promise<IngestJobStateResponse> {
  return _getJob(`/api/process_text_async/${jobId}`)
}

export async function startUrlProcessAsync(payload: UrlIngestRequest): Promise<IngestJobStartResponse> {
  return _postJob('/api/process_url_async/start', payload)
}

export async function getUrlProcessJob(jobId: string): Promise<IngestJobStateResponse> {
  return _getJob(`/api/process_url_async/${jobId}`)
}

export async function startKeywordProcessAsync(payload: KeywordRequest): Promise<KeywordJobStartResponse> {
  return _postJob('/api/process_keyword_async/start', payload)
}

export async function getKeywordProcessJob(jobId: string): Promise<KeywordJobStateResponse> {
  return _getJob(`/api/process_keyword_async/${jobId}`)
}

export async function queryKg(payload: QueryRequest): Promise<QueryResponse> {
  return _postJob('/api/query', payload)
}

export async function startQueryKgAsync(payload: QueryRequest): Promise<QueryJobStartResponse> {
  return _postJob('/api/query_async/start', payload)
}

export async function getQueryKgJob(jobId: string): Promise<QueryJobStateResponse> {
  return _getJob(`/api/query_async/${jobId}`)
}

export async function chatGeneral(message: string, history: ChatHistoryMessage[]): Promise<GeneralChatResponse> {
  return _postJob('/api/chat_general', { message, history })
}
