import { useState } from 'react'
import BuildKG from './components/BuildKG'
import Chat from './components/Chat'

type PageTab = 'ingest' | 'chat'

function App() {
  const [activeTab, setActiveTab] = useState<PageTab>('ingest')

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

      <main className="main-panel">{activeTab === 'ingest' ? <BuildKG /> : <Chat />}</main>
    </div>
  )
}

export default App
