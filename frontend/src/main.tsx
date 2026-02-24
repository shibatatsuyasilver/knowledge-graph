import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css'

// ─── 階段 1：定位根節點並建立 React Root ───────────────────
// Vite 範本保留 `StrictMode`，用於開發期提前暴露副作用問題。
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    {/* ─── 階段 2：掛載應用主元件 ───────────────────────── */}
    <App />
  </React.StrictMode>,
)
