import { type ChangeEvent, useMemo, useState } from 'react'
import {
  getTextProcessJob,
  getUrlProcessJob,
  getKeywordProcessJob,
  startKeywordProcessAsync,
  startTextProcessAsync,
  startUrlProcessAsync,
  toApiError,
} from '../api'
import type {
  BuildKgResponse,
  ExtractionProvider,
  IngestProgressSnapshot,
  KeywordProcessResponse,
  KeywordProgressSnapshot,
  Language,
} from '../types'

type IngestMode = 'text' | 'file' | 'keyword' | 'url'
type IngestResult = BuildKgResponse | KeywordProcessResponse

const DEFAULT_URL = 'https://www.who.int/news-room/fact-sheets/detail/diabetes'
const DEFAULT_OLLAMA_EXTRACTION_MODEL = 'sam860/deepseek-r1-0528-qwen3:8b'
const DEFAULT_GEMINI_EXTRACTION_MODEL = 'gemini-3-pro-preview'

function isKeywordResult(result: IngestResult | null): result is KeywordProcessResponse {
  // 用 discriminated-like 欄位 `searched_keyword` 判斷是否為 keyword 回傳格式。
  return Boolean(result && 'searched_keyword' in result)
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms)
  })
}

const BuildKG = () => {
  const [mode, setMode] = useState<IngestMode>('text')
  const [textInput, setTextInput] = useState('')
  const [urlInput, setUrlInput] = useState(DEFAULT_URL)
  const [keywordInput, setKeywordInput] = useState('半導體')
  const [language, setLanguage] = useState<Language>('zh-tw')
  const [maxResults, setMaxResults] = useState(5)
  const [chunkLimit, setChunkLimit] = useState(5)
  const [extractionProvider, setExtractionProvider] = useState<ExtractionProvider>('ollama')
  const [extractionModel, setExtractionModel] = useState(DEFAULT_OLLAMA_EXTRACTION_MODEL)
  const [siteAllowlistText, setSiteAllowlistText] = useState('')
  const [uploadedFileName, setUploadedFileName] = useState('')
  const [uploadedText, setUploadedText] = useState('')

  const [result, setResult] = useState<IngestResult | null>(null)
  const [liveProgress, setLiveProgress] = useState<KeywordProgressSnapshot | IngestProgressSnapshot | null>(null)
  const [activeJobId, setActiveJobId] = useState('')
  const [showSkippedChunks, setShowSkippedChunks] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const siteAllowlist = useMemo(
    () =>
      siteAllowlistText
        .split(/[\n,]/)
        .map((entry) => entry.trim())
        .filter((entry) => entry.length > 0),
    [siteAllowlistText],
  )

  const waitForKeywordJob = async (jobId: string): Promise<KeywordProcessResponse> => {
    // ─── 階段 1：持續 polling 任務狀態 ─────────────────────
    for (;;) {
      const job = await getKeywordProcessJob(jobId)
      if (job.progress) {
        setLiveProgress(job.progress)
      }
      // ─── 階段 2：完成/失敗分流 ───────────────────────────
      if (job.status === 'completed') {
        if (job.result) {
          return job.result
        }
        if (job.progress) {
          return job.progress
        }
        throw new Error('Keyword job completed without result')
      }
      if (job.status === 'failed') {
        throw new Error(job.error || job.progress?.error || 'Keyword job failed')
      }
      // ─── 階段 3：尚未完成時，延遲後繼續輪詢 ─────────────────
      await sleep(1000)
    }
  }

  const waitForTextJob = async (jobId: string): Promise<BuildKgResponse> => {
    // ─── 階段 1：持續 polling 任務狀態 ─────────────────────
    for (;;) {
      const job = await getTextProcessJob(jobId)
      if (job.progress) {
        setLiveProgress(job.progress)
      }
      // ─── 階段 2：完成/失敗分流 ───────────────────────────
      if (job.status === 'completed') {
        if (job.result) {
          return job.result
        }
        if (job.progress) {
          return job.progress
        }
        throw new Error('Text job completed without result')
      }
      if (job.status === 'failed') {
        throw new Error(job.error || job.progress?.error || 'Text job failed')
      }
      // ─── 階段 3：尚未完成時，延遲後繼續輪詢 ─────────────────
      await sleep(1000)
    }
  }

  const waitForUrlJob = async (jobId: string): Promise<BuildKgResponse> => {
    // ─── 階段 1：持續 polling 任務狀態 ─────────────────────
    for (;;) {
      const job = await getUrlProcessJob(jobId)
      if (job.progress) {
        setLiveProgress(job.progress)
      }
      // ─── 階段 2：完成/失敗分流 ───────────────────────────
      if (job.status === 'completed') {
        if (job.result) {
          return job.result
        }
        if (job.progress) {
          return job.progress
        }
        throw new Error('URL job completed without result')
      }
      if (job.status === 'failed') {
        throw new Error(job.error || job.progress?.error || 'URL job failed')
      }
      // ─── 階段 3：尚未完成時，延遲後繼續輪詢 ─────────────────
      await sleep(1000)
    }
  }

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    // ─── 階段 1：檔案存在與副檔名驗證 ───────────────────────
    const file = event.target.files?.[0]
    if (!file) {
      return
    }

    if (!/\.(txt|md)$/i.test(file.name)) {
      setError('僅支援 .txt 或 .md 檔案')
      setUploadedFileName('')
      setUploadedText('')
      return
    }

    // ─── 階段 2：透過 FileReader 非同步讀取內容 ─────────────
    const reader = new FileReader()
    reader.onload = () => {
      const content = typeof reader.result === 'string' ? reader.result : ''
      setUploadedFileName(file.name)
      setUploadedText(content)
      setError('')
    }
    reader.onerror = () => {
      setError('檔案讀取失敗，請重試')
      setUploadedFileName('')
      setUploadedText('')
    }
    // ─── 階段 3：觸發讀取流程 ─────────────────────────────
    reader.readAsText(file)
  }

  const handleSubmit = async () => {
    // ─── 階段 1：await 之前（同步）────────────────────────
    setLoading(true)
    //  ↑ 開始載入狀態，避免重複提交
    setResult(null)
    setLiveProgress(null)
    setActiveJobId('')
    setError('')
    //  ↑ 重置之前的狀態與錯誤訊息

    // ─── 階段 2：等待 IO ──────────────────────────────────
    try {
      let response: IngestResult
      if (mode === 'text') {
        if (!textInput.trim()) {
          throw new Error('請輸入文本內容')
        }
        const payload = {
          text: textInput.trim(),
          chunk_limit: chunkLimit,
          extraction_provider: extractionProvider,
          extraction_model: extractionModel.trim() || undefined,
        }
        const job = await startTextProcessAsync(payload)
        //  ↑ 發送非同步任務建立請求
        setActiveJobId(job.job_id)
        //  ↑ 更新畫面上顯示的 Job ID
        response = await waitForTextJob(job.job_id)
        //  ↑ 進入輪詢，等待圖譜建立完成
      } else if (mode === 'file') {
        if (!uploadedText.trim()) {
          throw new Error('請先上傳 .txt 或 .md 文字檔')
        }
        const payload = {
          text: uploadedText.trim(),
          chunk_limit: chunkLimit,
          extraction_provider: extractionProvider,
          extraction_model: extractionModel.trim() || undefined,
        }
        const job = await startTextProcessAsync(payload)
        //  ↑ 發送非同步任務建立請求
        setActiveJobId(job.job_id)
        //  ↑ 更新畫面上顯示的 Job ID
        response = await waitForTextJob(job.job_id)
        //  ↑ 進入輪詢，等待圖譜建立完成
      } else if (mode === 'url') {
        if (!urlInput.trim()) {
          throw new Error('請輸入網址')
        }
        const payload = {
          url: urlInput.trim(),
          chunk_limit: chunkLimit,
          extraction_provider: extractionProvider,
          extraction_model: extractionModel.trim() || undefined,
        }
        const job = await startUrlProcessAsync(payload)
        //  ↑ 發送非同步任務建立請求
        setActiveJobId(job.job_id)
        //  ↑ 更新畫面上顯示的 Job ID
        response = await waitForUrlJob(job.job_id)
        //  ↑ 進入輪詢，等待圖譜建立完成
      } else {
        if (!keywordInput.trim()) {
          throw new Error('請輸入關鍵字')
        }
        const payload = {
          keyword: keywordInput.trim(),
          max_results: maxResults,
          language,
          site_allowlist: siteAllowlist.length > 0 ? siteAllowlist : undefined,
          chunk_limit: chunkLimit,
          extraction_provider: extractionProvider,
          extraction_model: extractionModel.trim() || undefined,
        }
        const job = await startKeywordProcessAsync(payload)
        //  ↑ 發送非同步任務建立請求
        setActiveJobId(job.job_id)
        //  ↑ 更新畫面上顯示的 Job ID
        response = await waitForKeywordJob(job.job_id)
        //  ↑ 進入輪詢，等待圖譜建立完成
      }

      // ─── 階段 3：await 之後（React 19 自動標記 Transition）─
      setResult(response)
      //  ↑ 任務完成，儲存並顯示結果
    } catch (err) {
      // ─── 階段 3：await 之後（React 19 自動標記 Transition）─
      setError(toApiError(err))
      //  ↑ 捕捉錯誤並顯示給使用者
    } finally {
      // ─── 階段 3：await 之後（React 19 自動標記 Transition）─
      setLoading(false)
      //  ↑ 結束 loading 狀態
    }
  }

  const displayResult: IngestResult | KeywordProgressSnapshot | IngestProgressSnapshot | null = result ?? liveProgress
  const allChunkRows = displayResult?.chunk_progress ?? []
  const visibleChunkRows = showSkippedChunks
    ? allChunkRows
    : allChunkRows.filter((row) => row.status !== 'skipped_by_limit')
  // `processableChunks` 代表真正會送入抽取流程的 chunk 數量，不含 limit skip。
  const processableChunks = allChunkRows.filter((row) => row.status !== 'skipped_by_limit').length
  // `completedOrFailedChunks` 用來估算整體任務進度（完成 + 失敗都算已結束）。
  const completedOrFailedChunks = allChunkRows.filter((row) => row.status === 'processed' || row.status === 'failed').length

  return (
    <section className="panel">
      <div className="panel-header">
        <h2>Build Knowledge Graph</h2>
        <p>支援貼文字、上傳文字檔、URL 與關鍵字爬取後入圖譜</p>
      </div>

      <div className="mode-row" role="tablist" aria-label="Ingestion modes">
        <button type="button" className={`mode-btn ${mode === 'text' ? 'active' : ''}`} onClick={() => setMode('text')}>
          Text
        </button>
        <button type="button" className={`mode-btn ${mode === 'file' ? 'active' : ''}`} onClick={() => setMode('file')}>
          File
        </button>
        <button type="button" className={`mode-btn ${mode === 'keyword' ? 'active' : ''}`} onClick={() => setMode('keyword')}>
          Keyword Crawl
        </button>
        <button type="button" className={`mode-btn ${mode === 'url' ? 'active' : ''}`} onClick={() => setMode('url')}>
          URL
        </button>
      </div>

      <div className="input-block">
        <div className="keyword-grid">
          <label className="field-label">
            <span>Extraction Provider</span>
            <select
              value={extractionProvider}
              onChange={(event) => {
                const next = event.target.value as ExtractionProvider
                setExtractionProvider(next)
                setExtractionModel(
                  next === 'gemini' ? DEFAULT_GEMINI_EXTRACTION_MODEL : DEFAULT_OLLAMA_EXTRACTION_MODEL,
                )
              }}
              className="text-input"
            >
              <option value="ollama">Ollama</option>
              <option value="gemini">Gemini</option>
            </select>
          </label>

          <label className="field-label">
            <span>Extraction Model</span>
            <input
              type="text"
              value={extractionModel}
              onChange={(event) => setExtractionModel(event.target.value)}
              className="text-input"
              placeholder={extractionProvider === 'gemini' ? DEFAULT_GEMINI_EXTRACTION_MODEL : DEFAULT_OLLAMA_EXTRACTION_MODEL}
            />
          </label>

          <label className="field-label">
            <span>Chunk Limit (per URL)</span>
            <input
              type="number"
              min={1}
              max={200}
              value={chunkLimit}
              onChange={(event) => {
                const parsed = Number(event.target.value) || 1
                setChunkLimit(Math.max(1, Math.min(200, parsed)))
              }}
              className="text-input"
            />
          </label>
        </div>
      </div>

      <div className="input-block">
        {mode === 'text' && (
          <label className="field-label">
            <span>Text Content</span>
            <textarea
              value={textInput}
              onChange={(event) => setTextInput(event.target.value)}
              placeholder="貼上要抽取的文本內容..."
              className="text-area"
            />
          </label>
        )}

        {mode === 'file' && (
          <div className="file-section">
            <label className="field-label" htmlFor="file-upload">
              <span>Upload .txt / .md</span>
            </label>
            <input id="file-upload" type="file" accept=".txt,.md,text/plain,text/markdown" onChange={handleFileChange} />
            <p className="hint-text">
              {uploadedFileName
                ? `已載入檔案：${uploadedFileName}（${uploadedText.length} chars）`
                : '尚未選擇檔案'}
            </p>
          </div>
        )}

        {mode === 'url' && (
          <label className="field-label">
            <span>Source URL</span>
            <input
              type="url"
              value={urlInput}
              onChange={(event) => setUrlInput(event.target.value)}
              placeholder="https://example.com/article"
              className="text-input"
            />
          </label>
        )}

        {mode === 'keyword' && (
          <div className="keyword-grid">
            <label className="field-label">
              <span>Keyword</span>
              <input
                type="text"
                value={keywordInput}
                onChange={(event) => setKeywordInput(event.target.value)}
                className="text-input"
                placeholder="例如：失眠、半導體、糖尿病"
              />
            </label>

            <label className="field-label">
              <span>Max Results (1-10)</span>
              <input
                type="number"
                min={1}
                max={10}
                value={maxResults}
                onChange={(event) => {
                  const parsed = Number(event.target.value) || 1
                  setMaxResults(Math.max(1, Math.min(10, parsed)))
                }}
                className="text-input"
              />
            </label>

            <label className="field-label">
              <span>Language</span>
              <select
                value={language}
                onChange={(event) => setLanguage(event.target.value as Language)}
                className="text-input"
              >
                <option value="zh-tw">繁中 (zh-tw)</option>
                <option value="en">English (en)</option>
              </select>
            </label>

            <label className="field-label keyword-allowlist">
              <span>Site Allowlist (optional, comma/newline separated)</span>
              <textarea
                value={siteAllowlistText}
                onChange={(event) => setSiteAllowlistText(event.target.value)}
                className="text-area compact"
                placeholder="wikipedia.org, who.int"
              />
            </label>
          </div>
        )}
      </div>

      <div className="action-row">
        <button type="button" className="primary-btn" onClick={handleSubmit} disabled={loading}>
          {loading ? 'Processing...' : 'Start Processing'}
        </button>
        {loading && mode === 'keyword' && liveProgress && 'searched_keyword' in liveProgress && (
          <p className="hint-text">
            Job {activeJobId.slice(0, 8)}:{' '}
            {liveProgress.current_url ? `處理中 ${liveProgress.current_url}` : '準備中'}
          </p>
        )}
        {loading && mode !== 'keyword' && (
          <p className="hint-text">
            Job {activeJobId.slice(0, 8)}:{' '}
            {liveProgress?.current_url ? `處理中 ${liveProgress.current_url}` : '處理中，請稍候...'}
          </p>
        )}
        {error && <p className="error-text">{error}</p>}
      </div>

      {displayResult && (
        <div className="result-block">
          <p className="hint-text">
            Chunk Processing: {displayResult.stats.chunks_processed}
            {processableChunks > 0 ? ` / ${processableChunks}` : ''}
            {typeof displayResult.chunk_limit === 'number' ? ` (limit ${displayResult.chunk_limit} per URL)` : ' (no limit)'}
            {typeof displayResult.chunks_available === 'number' ? `, raw chunks ${displayResult.chunks_available}` : ''}
          </p>
          {allChunkRows.length > 0 && (
            <p className="hint-text">
              完成/失敗：{completedOrFailedChunks}，處理中：{allChunkRows.filter((row) => row.status === 'processing').length}，
              佇列中：{allChunkRows.filter((row) => row.status === 'queued').length}，
              跳過：{allChunkRows.filter((row) => row.status === 'skipped_by_limit').length}
            </p>
          )}
          <div className="stats-grid">
            <article className="stat-card">
              <p className="stat-label">Chunks</p>
              <p className="stat-value">{displayResult.stats.chunks_processed}</p>
            </article>
            <article className="stat-card">
              <p className="stat-label">Entities</p>
              <p className="stat-value">{displayResult.stats.entities}</p>
            </article>
            <article className="stat-card">
              <p className="stat-label">Relations</p>
              <p className="stat-value">{displayResult.stats.relations}</p>
            </article>
          </div>

          <details>
            <summary>Detailed Stats</summary>
            <pre className="json-preview">{JSON.stringify(displayResult.stats, null, 2)}</pre>
          </details>

          {allChunkRows.length > 0 && (
            <div className="table-wrap">
              <div className="mode-row">
                <h3>Chunk Progress</h3>
                <button
                  type="button"
                  className="ghost-btn"
                  onClick={() => setShowSkippedChunks((prev) => !prev)}
                >
                  {showSkippedChunks ? 'Hide Skipped' : 'Show Skipped'}
                </button>
              </div>
              <table>
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Chunk ID</th>
                    <th>Status</th>
                    <th>Entities</th>
                    <th>Relations</th>
                    <th>Chars</th>
                    <th>Source URL</th>
                  </tr>
                </thead>
                <tbody>
                  {visibleChunkRows.map((item, index) => (
                    <tr key={`${item.chunk_id}-${item.source_url ?? 'n/a'}-${index}`}>
                      <td>{item.order}</td>
                      <td>{item.chunk_id}</td>
                      <td>{item.status}</td>
                      <td>{item.entities}</td>
                      <td>{item.relations}</td>
                      <td>{item.chars}</td>
                      <td>{item.source_url ?? '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {displayResult.summary.length > 0 && (
            <div className="table-wrap">
              <h3>Extraction Summary</h3>
              <table>
                <thead>
                  <tr>
                    <th>Chunk ID</th>
                    <th>Entities</th>
                    <th>Relations</th>
                    <th>Source URL</th>
                  </tr>
                </thead>
                <tbody>
                  {displayResult.summary.map((item, index) => (
                    <tr key={`${item.chunk_id}-${item.source_url ?? 'n/a'}-${index}`}>
                      <td>{item.chunk_id}</td>
                      <td>{item.entities}</td>
                      <td>{item.relations}</td>
                      <td>{item.source_url ?? '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {isKeywordResult(displayResult) && (
            <div className="keyword-result">
              <h3>Crawl Result: {displayResult.searched_keyword}</h3>
              <p className="hint-text">成功抓取 {displayResult.fetched_urls.length} 個網址</p>
              {displayResult.fetched_urls.length > 0 && (
                <ul>
                  {displayResult.fetched_urls.map((url) => (
                    <li key={url}>{url}</li>
                  ))}
                </ul>
              )}

              {displayResult.failed_urls.length > 0 && (
                <div className="table-wrap">
                  <h4>Failed URLs</h4>
                  <table>
                    <thead>
                      <tr>
                        <th>URL</th>
                        <th>Error</th>
                      </tr>
                    </thead>
                    <tbody>
                      {displayResult.failed_urls.map((item, index) => (
                        <tr key={`${item.url}-${index}`}>
                          <td>{item.url}</td>
                          <td>{item.error}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </section>
  )
}

export default BuildKG
