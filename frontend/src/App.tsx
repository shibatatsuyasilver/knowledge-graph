import { useState } from 'react'
import BuildKG from './components/BuildKG'
import Chat from './components/Chat'

type PageTab = 'ingest' | 'chat'

function App() {
  // ─── 階段 1：建立頁面主狀態 ─────────────────────────────
  // `activeTab` 決定主畫面顯示 `BuildKG` 或 `Chat`。
  const [activeTab, setActiveTab] = useState<PageTab>('ingest')

  // ─── 階段 2：依狀態渲染對應區塊 ─────────────────────────
  return (
    <div className="app-shell">
      <header className="app-header">
        <p className="app-eyebrow">Knowledge Graph + Companion Chat</p>
        <h1>GenAI Knowledge Studio</h1>
      </header>

      <div className="tab-row" role="tablist" aria-label="Main sections">
        <button
          className={`tab-btn ${activeTab === 'ingest' ? 'active' : ''}`}
          onClick={() => setActiveTab('ingest')}
          role="tab"
          aria-selected={activeTab === 'ingest'}
          type="button"
        >
          Build KG
        </button>
        <button
          className={`tab-btn ${activeTab === 'chat' ? 'active' : ''}`}
          onClick={() => setActiveTab('chat')}
          role="tab"
          aria-selected={activeTab === 'chat'}
          type="button"
        >
          Chat
        </button>
      </div>

      {/* ─── 階段 3：切換主內容面板 ───────────────────────── */}
      {/* 不保留兩個元件同時掛載，避免無效渲染與狀態干擾。 */}
      <main className="main-panel">{activeTab === 'ingest' ? <BuildKG /> : <Chat />}</main>
    </div>
  )
}

export default App
