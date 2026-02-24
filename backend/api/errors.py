"""HTTP error translation utilities."""

from __future__ import annotations

import requests
from fastapi import HTTPException


def raise_http_error(exc: Exception) -> None:
    """`raise_http_error` 的主要流程入口。

主要用途：
- 串接此函式負責的核心步驟並回傳既有格式。
- 例外沿用現行錯誤處理策略，避免破壞呼叫端契約。

維護重點：
- 調整流程時需保持 API 欄位、狀態轉移與錯誤語意一致。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    if isinstance(exc, HTTPException):
        raise exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, requests.Timeout):
        raise HTTPException(status_code=504, detail="Upstream service timeout")
    if isinstance(exc, requests.RequestException):
        raise HTTPException(status_code=502, detail=f"Upstream service error: {exc}")
    raise HTTPException(status_code=500, detail=f"Internal server error: {exc}")
