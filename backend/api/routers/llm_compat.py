"""Compatibility routes migrated from legacy llm_api (llm_kg standalone service)."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter

from backend.llm_kg import llm_client

router = APIRouter()


@router.get("/health")
def health() -> Dict[str, Any]:
    """處理 `GET /health` 請求並回傳既有服務結果。
    函式會沿用目前驗證與例外處理策略，維持 API 契約與回應格式一致。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
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
