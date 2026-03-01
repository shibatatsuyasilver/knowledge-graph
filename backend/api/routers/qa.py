"""QA-related API routes."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter

from backend.services.qa import qa_service
from backend.api.errors import raise_http_error
from backend.api.models import GeneralChatRequest, QueryRequest
from backend.api.utils.job_runner import create_async_job
from backend.api.utils.router_utils import get_job_status
from backend.jobs.runtime import query_job_store

router = APIRouter()



@router.post("/api/query")
def query_sync(req: QueryRequest):
    """知識圖譜問答 (同步 API)。"""
    try:
        return qa_service.query_kg(
            req.question,
            nl2cypher_provider=req.nl2cypher_provider or None,
            nl2cypher_model=req.nl2cypher_model or None,
            query_engine=req.query_engine or None,
        )
    except Exception as exc:
        raise_http_error(exc)


def _build_initial_query_progress(
    question: str,
    *,
    nl2cypher_provider: str | None = None,
    nl2cypher_model: str | None = None,
) -> Dict[str, Any]:
    """建構非同步問答 Job 的初始進度狀態結構。"""
    progress = {
        "status": "running",
        "question": question,
        "stage": "planner",
        "round_count": 0,
        "replan_count": 0,
        "final_strategy": "single_query",
        "failure_chain": [],
        "detail": "Planning query strategy",
        "agentic_trace": {
            "stage": "planner",
            "round_count": 0,
            "replan_count": 0,
            "final_strategy": "single_query",
            "failure_chain": [],
            "rounds": [],
        },
    }
    if nl2cypher_provider:
        progress["llm_provider"] = nl2cypher_provider
        progress["agentic_trace"]["llm_provider"] = nl2cypher_provider
    if nl2cypher_model:
        progress["llm_model"] = nl2cypher_model
        progress["agentic_trace"]["llm_model"] = nl2cypher_model
    return progress


@router.post("/api/query_async/start")
def query_async_start(req: QueryRequest):
    """發起知識圖譜問答 (非同步 API)。"""
    try:
        if req.query_engine == "graph_chain":
            raise ValueError("query_engine=graph_chain is only supported by /api/query (sync)")

        question = req.question.strip()
        if not question:
            raise ValueError("Question cannot be empty")

        initial_progress = _build_initial_query_progress(
            question,
            nl2cypher_provider=req.nl2cypher_provider,
            nl2cypher_model=req.nl2cypher_model,
        )

        def _run(job_id: str) -> Dict[str, Any]:
            def _update_progress(event: Dict[str, Any]) -> None:
                if event.get("type") != "agentic_progress":
                    return

                def _mutate(job: Dict[str, Any]) -> None:
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
                    event_trace = event.get("agentic_trace")
                    if isinstance(event_trace, dict):
                        progress["agentic_trace"] = event_trace
                    job["progress"] = progress

                query_job_store.update(job_id, _mutate)

            return qa_service.query_kg(
                question,
                progress_callback=_update_progress,
                nl2cypher_provider=req.nl2cypher_provider or None,
                nl2cypher_model=req.nl2cypher_model or None,
                query_engine=req.query_engine or None,
            )

        def _on_complete(result: Dict[str, Any], progress: Dict[str, Any]) -> Dict[str, Any]:
            progress["status"] = "completed"
            progress["question"] = question
            trace = result.get("agentic_trace")
            if isinstance(trace, dict):
                progress["agentic_trace"] = trace
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
            return progress

        return create_async_job(
            job_store=query_job_store,
            initial_progress=initial_progress,
            run_job=_run,
            name_prefix="query-job",
            on_complete=_on_complete,
            fail_progress_field="detail",
        )
    except Exception as exc:
        raise_http_error(exc)


@router.get("/api/query_async/{job_id}")
def query_async_status(job_id: str):
    """前端透過此端點輪詢非同步問答任務的當前進度。"""
    return get_job_status(query_job_store, job_id)


@router.post("/api/chat_general")
def chat_general_sync(req: GeneralChatRequest):
    """無關知識圖譜的純文字通用聊天。"""
    try:
        history = [{"role": msg.role, "content": msg.content} for msg in (req.history or [])]
        return qa_service.chat_general(req.message, history=history)
    except Exception as exc:
        raise_http_error(exc)
