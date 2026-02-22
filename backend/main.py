"""FastAPI app assembly for backend services."""

from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load env first so imported modules can read runtime settings.
load_dotenv()

# Backward compatibility for tests monkeypatching main_module.logic.
import backend.logic as logic  # noqa: F401
from backend.api.routers.ingest import router as ingest_router
from backend.api.routers.llm_compat import router as llm_compat_router
from backend.api.routers.qa import router as qa_router
from backend.api.routers.root import router as root_router


def create_app() -> FastAPI:
    """執行 `create_app` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
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
