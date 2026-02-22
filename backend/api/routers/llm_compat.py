"""Compatibility routes migrated from legacy llm_deploy API."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from backend.api.models import LlmCompatChatRequest, LlmCompatChatResponse
from backend.config.settings import get_general_chat_settings
from backend.llm_kg import llm_client

router = APIRouter()


@router.get("/health")
def health() -> Dict[str, Any]:
    """處理 `GET /health` 請求並回傳既有服務結果。
    函式會沿用目前驗證與例外處理策略，維持 API 契約與回應格式一致。
    """
    cfg = llm_client.get_runtime_config()
    upstream = llm_client.health_check(timeout_seconds=3.0)
    return {
        "status": "ok",
        "service": "backend-llm-compat",
        "provider": upstream["provider"],
        "model": upstream["model"],
        "upstream": {
            "type": upstream["upstream"],
            "status": upstream["status"],
            "reachable": upstream["reachable"],
        },
        # Legacy field for backward compatibility.
        "ollama": upstream["status"] if cfg.provider == "ollama" else "legacy-n/a",
        "defaultModel": cfg.model,
    }


@router.post("/api/chat", response_model=LlmCompatChatResponse)
def chat_endpoint(req: LlmCompatChatRequest) -> LlmCompatChatResponse:
    """處理 `POST /api/chat` 請求並回傳既有服務結果。
    函式會沿用目前驗證與例外處理策略，維持 API 契約與回應格式一致。
    """
    try:
        cfg = llm_client.get_runtime_config()
        chat_settings = get_general_chat_settings()
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
            timeout_seconds=chat_settings.timeout_seconds,
        )
        return LlmCompatChatResponse(answer=answer, model=model)
    except llm_client.LLMTimeoutError as exc:  # pragma: no cover - network dependent
        raise HTTPException(status_code=504, detail="LLM response timed out") from exc
    except llm_client.LLMError as exc:  # pragma: no cover - network dependent
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc
