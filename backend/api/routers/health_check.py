"""Root and health check routes."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter

from backend.llm_kg import llm_client

router = APIRouter()


@router.get("/")
def root() -> Dict[str, str]:
    return {"message": "GenAI KG Backend API is running"}


@router.get("/health")
def health() -> Dict[str, Any]:
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
        "ollama": upstream["status"] if cfg.provider == "ollama" else "legacy-n/a",
        "defaultModel": cfg.model,
    }
