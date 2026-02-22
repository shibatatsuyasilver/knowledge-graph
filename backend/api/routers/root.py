"""Root and simple health routes."""

from __future__ import annotations

from typing import Dict

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def root() -> Dict[str, str]:
    """處理 `GET /` 請求並回傳既有服務結果。
    函式會沿用目前驗證與例外處理策略，維持 API 契約與回應格式一致。
    """
    return {"message": "GenAI KG Backend API is running"}
