"""QA-related API routes."""

from __future__ import annotations

from fastapi import APIRouter

import backend.logic as logic
from backend.api.errors import raise_http_error
from backend.api.models import GeneralChatRequest, QueryRequest

router = APIRouter()


@router.post("/api/query")
def query_sync(req: QueryRequest):
    """處理 `POST /api/query` 請求並回傳既有服務結果。
    函式會沿用目前驗證與例外處理策略，維持 API 契約與回應格式一致。
    """
    try:
        return logic.query_kg(req.question)
    except Exception as exc:
        raise_http_error(exc)


@router.post("/api/chat_general")
def chat_general_sync(req: GeneralChatRequest):
    """處理 `POST /api/chat_general` 請求並回傳既有服務結果。
    函式會沿用目前驗證與例外處理策略，維持 API 契約與回應格式一致。
    """
    try:
        history = [{"role": msg.role, "content": msg.content} for msg in (req.history or [])]
        return logic.chat_general(req.message, history=history)
    except Exception as exc:
        raise_http_error(exc)
