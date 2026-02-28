"""FastAPI app assembly for backend services."""

from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load env first so imported modules can read runtime settings.
load_dotenv()

from backend.api.routers.ingest import router as ingest_router
from backend.api.routers.llm_compat import router as llm_compat_router
from backend.api.routers.qa import router as qa_router
from backend.api.routers.root import router as root_router


def create_app() -> FastAPI:
    """`create_app` 的主要流程入口。

主要用途：
- 串接此函式負責的核心步驟並回傳既有格式。
- 例外沿用現行錯誤處理策略，避免破壞呼叫端契約。

維護重點：
- 調整流程時需保持 API 欄位、狀態轉移與錯誤語意一致。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    app = FastAPI(title="GenAI KG Backend")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, replace with specific origins.
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(root_router)
    app.include_router(ingest_router)
    app.include_router(qa_router)
    app.include_router(llm_compat_router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
