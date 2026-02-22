import axios from 'axios'
import type {
  BuildKgResponse,
  ChatHistoryMessage,
  GeneralChatResponse,
  IngestJobStartResponse,
  IngestJobStateResponse,
  KeywordJobStartResponse,
  KeywordJobStateResponse,
  KeywordRequest,
  KeywordProcessResponse,
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

export async function processText(payload: TextIngestRequest): Promise<BuildKgResponse> {
  const response = await apiClient.post<BuildKgResponse>('/api/process_text', payload)
  return response.data
}

export async function processUrl(payload: UrlIngestRequest): Promise<BuildKgResponse> {
  const response = await apiClient.post<BuildKgResponse>('/api/process_url', payload)
  return response.data
}

export async function startTextProcessAsync(payload: TextIngestRequest): Promise<IngestJobStartResponse> {
  const response = await apiClient.post<IngestJobStartResponse>('/api/process_text_async/start', payload)
  return response.data
}

export async function getTextProcessJob(jobId: string): Promise<IngestJobStateResponse> {
  const response = await apiClient.get<IngestJobStateResponse>(`/api/process_text_async/${jobId}`)
  return response.data
}

export async function startUrlProcessAsync(payload: UrlIngestRequest): Promise<IngestJobStartResponse> {
  const response = await apiClient.post<IngestJobStartResponse>('/api/process_url_async/start', payload)
  return response.data
}

export async function getUrlProcessJob(jobId: string): Promise<IngestJobStateResponse> {
  const response = await apiClient.get<IngestJobStateResponse>(`/api/process_url_async/${jobId}`)
  return response.data
}

export async function processKeyword(payload: KeywordRequest): Promise<KeywordProcessResponse> {
  const response = await apiClient.post<KeywordProcessResponse>('/api/process_keyword', payload)
  return response.data
}

export async function startKeywordProcessAsync(payload: KeywordRequest): Promise<KeywordJobStartResponse> {
  const response = await apiClient.post<KeywordJobStartResponse>('/api/process_keyword_async/start', payload)
  return response.data
}

export async function getKeywordProcessJob(jobId: string): Promise<KeywordJobStateResponse> {
  const response = await apiClient.get<KeywordJobStateResponse>(`/api/process_keyword_async/${jobId}`)
  return response.data
}

export async function queryKg(payload: QueryRequest): Promise<QueryResponse> {
  const response = await apiClient.post<QueryResponse>('/api/query', payload)
  return response.data
}

export async function startQueryKgAsync(payload: QueryRequest): Promise<QueryJobStartResponse> {
  const response = await apiClient.post<QueryJobStartResponse>('/api/query_async/start', payload)
  return response.data
}

export async function getQueryKgJob(jobId: string): Promise<QueryJobStateResponse> {
  const response = await apiClient.get<QueryJobStateResponse>(`/api/query_async/${jobId}`)
  return response.data
}

export async function chatGeneral(message: string, history: ChatHistoryMessage[]): Promise<GeneralChatResponse> {
  const response = await apiClient.post<GeneralChatResponse>('/api/chat_general', { message, history })
  return response.data
}
