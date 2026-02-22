"""QA-related API routes."""

from __future__ import annotations

import inspect
import threading
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

import backend.logic as logic
from backend.api.errors import raise_http_error
from backend.api.models import GeneralChatRequest, QueryRequest
from backend.jobs.runtime import query_job_store

router = APIRouter()


def _query_with_overrides(
    *,
    question: str,
    nl2cypher_provider: str | None = None,
    nl2cypher_model: str | None = None,
    progress_callback: Any = None,
):
    """以相容方式呼叫 logic.query_kg，支援舊版測試替身簽名。"""
    kwargs: Dict[str, Any] = {}
    if progress_callback is not None:
        kwargs["progress_callback"] = progress_callback
    if nl2cypher_provider:
        kwargs["nl2cypher_provider"] = nl2cypher_provider
    if nl2cypher_model:
        kwargs["nl2cypher_model"] = nl2cypher_model

    try:
        signature = inspect.signature(logic.query_kg)
    except (TypeError, ValueError):
        return logic.query_kg(question, **kwargs)

    params = signature.parameters
    supports_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
    if supports_kwargs:
        return logic.query_kg(question, **kwargs)

    filtered_kwargs = {key: value for key, value in kwargs.items() if key in params}
    return logic.query_kg(question, **filtered_kwargs)


@router.post("/api/query")
def query_sync(req: QueryRequest):
    """處理 `POST /api/query` 請求並回傳既有服務結果。
    函式會沿用目前驗證與例外處理策略，維持 API 契約與回應格式一致。
    """
    try:
        return _query_with_overrides(
            question=req.question,
            nl2cypher_provider=req.nl2cypher_provider,
            nl2cypher_model=req.nl2cypher_model,
        )
    except Exception as exc:
        raise_http_error(exc)


def _build_initial_query_progress(
    question: str,
    *,
    nl2cypher_provider: str | None = None,
    nl2cypher_model: str | None = None,
) -> Dict[str, Any]:
    """執行 `_build_initial_query_progress` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    progress = {
        "status": "running",
        "question": question,
        "stage": "planner",
        "round_count": 0,
        "replan_count": 0,
        "final_strategy": "single_query",
        "failure_chain": [],
        "detail": "Planning query strategy",
    }
    if nl2cypher_provider:
        progress["llm_provider"] = nl2cypher_provider
    if nl2cypher_model:
        progress["llm_model"] = nl2cypher_model
    return progress


def _query_job_status(job_id: str) -> Dict[str, Any]:
    """執行 `_query_job_status` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    job = query_job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or expired")
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "progress": job.get("progress"),
        "result": job.get("result"),
        "error": job.get("error"),
    }


@router.post("/api/query_async/start")
def query_async_start(req: QueryRequest):
    """處理 `POST /api/query_async/start` 請求並回傳既有服務結果。
    函式會沿用目前驗證與例外處理策略，維持 API 契約與回應格式一致。
    """
    try:
        question = req.question.strip()
        if not question:
            raise ValueError("Question cannot be empty")

        initial_progress = _build_initial_query_progress(
            question,
            nl2cypher_provider=req.nl2cypher_provider,
            nl2cypher_model=req.nl2cypher_model,
        )
        job_id = query_job_store.create(
            {
                "status": "running",
                "progress": initial_progress,
                "result": None,
                "error": None,
            }
        )

        def _run() -> None:
            """執行 `_run` 的內部輔助流程。
            此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
            """
            try:
                def _update_progress(event: Dict[str, Any]) -> None:
                    """執行 `_update_progress` 的內部輔助流程。
                    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
                    """
                    if event.get("type") != "agentic_progress":
                        return

                    def _mutate(job: Dict[str, Any]) -> None:
                        """執行 `_mutate` 的內部輔助流程。
                        此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
                        """
                        progress = dict(job.get("progress") or {})
                        progress["status"] = "running"
                        progress["question"] = question
                        progress["stage"] = str(event.get("stage") or progress.get("stage") or "planner")
                        progress["round_count"] = int(event.get("round_count") or progress.get("round_count") or 0)
                        progress["replan_count"] = int(event.get("replan_count") or progress.get("replan_count") or 0)
                        progress["final_strategy"] = str(
                            event.get("final_strategy") or progress.get("final_strategy") or "single_query"
                        )
                        failure_chain = event.get("failure_chain")
                        if isinstance(failure_chain, list):
                            progress["failure_chain"] = [str(item) for item in failure_chain]
                        detail = str(event.get("detail") or "").strip()
                        if detail:
                            progress["detail"] = detail
                        provider = str(event.get("llm_provider") or "").strip()
                        if provider:
                            progress["llm_provider"] = provider
                        model = str(event.get("llm_model") or "").strip()
                        if model:
                            progress["llm_model"] = model
                        job["progress"] = progress

                    query_job_store.update(job_id, _mutate)

                result = _query_with_overrides(
                    question=question,
                    nl2cypher_provider=req.nl2cypher_provider,
                    nl2cypher_model=req.nl2cypher_model,
                    progress_callback=_update_progress,
                )

                def _mark_completed(job: Dict[str, Any]) -> None:
                    """執行 `_mark_completed` 的內部輔助流程。
                    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
                    """
                    job["status"] = "completed"
                    job["result"] = result
                    progress = dict(job.get("progress") or {})
                    progress["status"] = "completed"
                    progress["question"] = question
                    trace = result.get("agentic_trace")
                    if isinstance(trace, dict):
                        progress["stage"] = str(trace.get("stage") or progress.get("stage") or "done")
                        progress["round_count"] = int(trace.get("round_count") or progress.get("round_count") or 0)
                        progress["replan_count"] = int(trace.get("replan_count") or progress.get("replan_count") or 0)
                        progress["final_strategy"] = str(
                            trace.get("final_strategy") or progress.get("final_strategy") or "single_query"
                        )
                        failure_chain = trace.get("failure_chain")
                        if isinstance(failure_chain, list):
                            progress["failure_chain"] = [str(item) for item in failure_chain]
                        provider = str(trace.get("llm_provider") or "").strip()
                        if provider:
                            progress["llm_provider"] = provider
                        model = str(trace.get("llm_model") or "").strip()
                        if model:
                            progress["llm_model"] = model
                    job["progress"] = progress

                query_job_store.update(job_id, _mark_completed)
            except Exception as exc:  # pragma: no cover - integration path
                def _mark_failed(job: Dict[str, Any]) -> None:
                    """執行 `_mark_failed` 的內部輔助流程。
                    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
                    """
                    job["status"] = "failed"
                    job["error"] = str(exc)
                    progress = dict(job.get("progress") or {})
                    progress["status"] = "failed"
                    progress["detail"] = str(exc)
                    job["progress"] = progress

                query_job_store.update(job_id, _mark_failed)

        worker = threading.Thread(target=_run, daemon=True, name=f"query-job-{job_id[:8]}")
        worker.start()
        return {"job_id": job_id, "status": "running"}
    except Exception as exc:
        raise_http_error(exc)


@router.get("/api/query_async/{job_id}")
def query_async_status(job_id: str):
    """處理 `GET /api/query_async/{job_id}` 請求並回傳既有服務結果。
    函式會沿用目前驗證與例外處理策略，維持 API 契約與回應格式一致。
    """
    return _query_job_status(job_id)


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
