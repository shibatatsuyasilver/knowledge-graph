"""FastAPI app assembly for backend services."""

from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load env first so imported modules can read runtime settings.
load_dotenv()

from backend.api.routers.health_check import router as health_check_router
from backend.api.routers.ingest import router as ingest_router
from backend.api.routers.qa import router as qa_router


def create_app() -> FastAPI:
    """初始化並組裝 FastAPI 應用，包含 CORS 中介軟體和三個路由器（health_check/ingest/qa）。"""
    app = FastAPI(title="GenAI KG Backend")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, replace with specific origins.
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_check_router)
    app.include_router(ingest_router)
    app.include_router(qa_router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
