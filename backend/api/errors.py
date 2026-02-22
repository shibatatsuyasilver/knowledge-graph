"""HTTP error translation utilities."""

from __future__ import annotations

import requests
from fastapi import HTTPException


def raise_http_error(exc: Exception) -> None:
    """執行 `raise_http_error` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
    if isinstance(exc, HTTPException):
        raise exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, requests.Timeout):
        raise HTTPException(status_code=504, detail="Upstream service timeout")
    if isinstance(exc, requests.RequestException):
        raise HTTPException(status_code=502, detail=f"Upstream service error: {exc}")
    raise HTTPException(status_code=500, detail=f"Internal server error: {exc}")
