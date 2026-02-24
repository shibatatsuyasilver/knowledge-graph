import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// ─── 階段 1：解析本機開發時的 API 代理目標 ─────────────────
// 未指定時預設走本機後端 `:8000`。
const proxyTarget = process.env.VITE_API_PROXY_TARGET ?? 'http://localhost:8000'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    // ─── 階段 2：前端開發伺服器代理設定 ───────────────────
    // 將 `/api` 請求轉給後端，避免瀏覽器 CORS 問題。
    proxy: {
      '/api': {
        target: proxyTarget,
        changeOrigin: true,
        timeout: 900000,
        proxyTimeout: 900000,
      },
    },
  },
})
