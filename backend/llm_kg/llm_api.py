"""Local deployment wrapper with provider switch (OpenAI-compatible / Ollama)."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from . import llm_client
except ImportError:  # pragma: no cover - direct script execution
    import llm_client  # type: ignore[no-redef]

app = FastAPI(title="LLM Knowledge QA Service", version="1.0.0")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, description="User question")
    model: str | None = Field(default=None, description="Override model name")


class ChatResponse(BaseModel):
    answer: str
    model: str


@app.get("/health")
def health() -> Dict[str, Any]:
    """處理 `GET /health` 請求並回傳既有服務結果。
    函式會沿用目前驗證與例外處理策略，維持 API 契約與回應格式一致。
    """
    cfg = llm_client.get_runtime_config()
    upstream = llm_client.health_check(timeout_seconds=3.0)
    return {
        "status": "ok",
        "service": "llm_api",
        "provider": upstream["provider"],
        "model": upstream["model"],
        "upstream": {"type": upstream["upstream"], "status": upstream["status"], "reachable": upstream["reachable"]},
        # Legacy field for backward compatibility.
        "ollama": upstream["status"] if cfg.provider == "ollama" else "legacy-n/a",
        "defaultModel": cfg.model,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest) -> ChatResponse:
    """處理 `POST /api/chat` 請求並回傳既有服務結果。
    函式會沿用目前驗證與例外處理策略，維持 API 契約與回應格式一致。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    try:
        cfg = llm_client.get_runtime_config()
        model = req.model or cfg.model
        answer = llm_client.chat_text(
            messages=[
                {
                    "role": "system",
                    "content": "你是一個知識圖譜專家助手。請使用繁體中文回答，內容要精確且簡潔。",
                },
                {"role": "user", "content": req.question},
            ],
            model=model,
        )
        return ChatResponse(answer=answer, model=model)
    except llm_client.LLMTimeoutError as exc:  # pragma: no cover - network dependent
        raise HTTPException(status_code=504, detail="LLM response timed out") from exc
    except llm_client.LLMError as exc:  # pragma: no cover - network dependent
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("llm_api:app", host="0.0.0.0", port=8000, reload=False)
