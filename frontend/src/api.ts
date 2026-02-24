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

// ─── 階段 1：建立共用 HTTP Client ─────────────────────────
// 所有 API 呼叫都走同一個 `axios` instance，集中 timeout 與 headers 設定。
const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? '',
  timeout: 900000,
  headers: {
    'Content-Type': 'application/json',
  },
})

export function toApiError(error: unknown): string {
  // ─── 階段 1：優先解析 Axios 錯誤結構 ────────────────────
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data?.detail
    if (typeof detail === 'string' && detail.trim()) {
      return detail
    }
    return error.message
  }
  // ─── 階段 2：退回一般 Error 物件 ─────────────────────────
  if (error instanceof Error) {
    return error.message
  }
  // ─── 階段 3：最後保底錯誤字串 ───────────────────────────
  return 'Unknown error'
}

export async function processText(payload: TextIngestRequest): Promise<BuildKgResponse> {
  // ─── 階段 1：等待 IO ──────────────────────────────────
  const response = await apiClient.post<BuildKgResponse>('/api/process_text', payload)
  //  ↑ 呼叫後端 API，傳送文本以建立圖譜
  
  // ─── 階段 2：await 之後（解析結果）────────────────────
  return response.data
  //  ↑ 回傳解析後的圖譜建立結果
}

export async function processUrl(payload: UrlIngestRequest): Promise<BuildKgResponse> {
  // ─── 階段 1：等待 IO ──────────────────────────────────
  const response = await apiClient.post<BuildKgResponse>('/api/process_url', payload)
  //  ↑ 呼叫後端 API，傳送網址以建立圖譜

  // ─── 階段 2：await 之後（解析結果）────────────────────
  return response.data
  //  ↑ 回傳解析後的圖譜建立結果
}

export async function startTextProcessAsync(payload: TextIngestRequest): Promise<IngestJobStartResponse> {
  // ─── 階段 1：等待 IO ──────────────────────────────────
  const response = await apiClient.post<IngestJobStartResponse>('/api/process_text_async/start', payload)
  //  ↑ 發送非同步任務建立請求 (Text)

  // ─── 階段 2：await 之後（解析結果）────────────────────
  return response.data
  //  ↑ 回傳包含任務 ID 的狀態
}

export async function getTextProcessJob(jobId: string): Promise<IngestJobStateResponse> {
  // ─── 階段 1：等待 IO ──────────────────────────────────
  const response = await apiClient.get<IngestJobStateResponse>(`/api/process_text_async/${jobId}`)
  //  ↑ 查詢非同步任務進度 (Text)

  // ─── 階段 2：await 之後（解析結果）────────────────────
  return response.data
  //  ↑ 回傳最新的任務進度或完成結果
}

export async function startUrlProcessAsync(payload: UrlIngestRequest): Promise<IngestJobStartResponse> {
  // ─── 階段 1：等待 IO ──────────────────────────────────
  const response = await apiClient.post<IngestJobStartResponse>('/api/process_url_async/start', payload)
  //  ↑ 發送非同步任務建立請求 (Url)

  // ─── 階段 2：await 之後（解析結果）────────────────────
  return response.data
  //  ↑ 回傳包含任務 ID 的狀態
}

export async function getUrlProcessJob(jobId: string): Promise<IngestJobStateResponse> {
  // ─── 階段 1：等待 IO ──────────────────────────────────
  const response = await apiClient.get<IngestJobStateResponse>(`/api/process_url_async/${jobId}`)
  //  ↑ 查詢非同步任務進度 (Url)

  // ─── 階段 2：await 之後（解析結果）────────────────────
  return response.data
  //  ↑ 回傳最新的任務進度或完成結果
}

export async function processKeyword(payload: KeywordRequest): Promise<KeywordProcessResponse> {
  // ─── 階段 1：等待 IO ──────────────────────────────────
  const response = await apiClient.post<KeywordProcessResponse>('/api/process_keyword', payload)
  //  ↑ 發送實體/關鍵字至後端建立節點

  // ─── 階段 2：await 之後（解析結果）────────────────────
  return response.data
  //  ↑ 回傳建立結果
}

export async function startKeywordProcessAsync(payload: KeywordRequest): Promise<KeywordJobStartResponse> {
  // ─── 階段 1：等待 IO ──────────────────────────────────
  const response = await apiClient.post<KeywordJobStartResponse>('/api/process_keyword_async/start', payload)
  //  ↑ 發送非同步任務建立請求 (Keyword)

  // ─── 階段 2：await 之後（解析結果）────────────────────
  return response.data
  //  ↑ 回傳包含任務 ID 的狀態
}

export async function getKeywordProcessJob(jobId: string): Promise<KeywordJobStateResponse> {
  // ─── 階段 1：等待 IO ──────────────────────────────────
  const response = await apiClient.get<KeywordJobStateResponse>(`/api/process_keyword_async/${jobId}`)
  //  ↑ 查詢非同步任務進度 (Keyword)

  // ─── 階段 2：await 之後（解析結果）────────────────────
  return response.data
  //  ↑ 回傳最新的任務進度或完成結果
}

export async function queryKg(payload: QueryRequest): Promise<QueryResponse> {
  // ─── 階段 1：等待 IO ──────────────────────────────────
  const response = await apiClient.post<QueryResponse>('/api/query', payload)
  //  ↑ 同步模式，發送 Cypher / 自然語言查詢並等待回覆

  // ─── 階段 2：await 之後（解析結果）────────────────────
  return response.data
  //  ↑ 回傳查詢結果與對應的 Cypher
}

export async function startQueryKgAsync(payload: QueryRequest): Promise<QueryJobStartResponse> {
  // ─── 階段 1：等待 IO ──────────────────────────────────
  const response = await apiClient.post<QueryJobStartResponse>('/api/query_async/start', payload)
  //  ↑ 發送非同步任務建立請求 (Query KG)

  // ─── 階段 2：await 之後（解析結果）────────────────────
  return response.data
  //  ↑ 回傳包含任務 ID 的狀態
}

export async function getQueryKgJob(jobId: string): Promise<QueryJobStateResponse> {
  // ─── 階段 1：等待 IO ──────────────────────────────────
  const response = await apiClient.get<QueryJobStateResponse>(`/api/query_async/${jobId}`)
  //  ↑ 查詢非同步任務進度 (Query KG)

  // ─── 階段 2：await 之後（解析結果）────────────────────
  return response.data
  //  ↑ 回傳最新的任務進度或完成結果
}

export async function chatGeneral(message: string, history: ChatHistoryMessage[]): Promise<GeneralChatResponse> {
  // ─── 階段 1：等待 IO ──────────────────────────────────
  const response = await apiClient.post<GeneralChatResponse>('/api/chat_general', { message, history })
  //  ↑ 一般模式的 API 請求，傳送訊息與歷史紀錄

  // ─── 階段 2：await 之後（解析結果）────────────────────
  return response.data
  //  ↑ 回傳助理的回覆內容
}
