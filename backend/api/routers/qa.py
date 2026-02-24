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
    query_engine: str | None = None,
    progress_callback: Any = None,
):
    """呼叫邏輯層 `logic.query_kg` 執行問答查詢，並自動處理參數相容性。
    
    由於 `logic.query_kg` 的簽名在不同版本或測試環境下（Monkeypatch）可能會有所不同，
    這個函式會先利用 `inspect.signature` 檢查目的端支援哪些參數。
    只有當目的端支援時，才會把 `nl2cypher_provider` 等參數傳進去，避免拋出 TypeError。
    
    參數:
        question: 使用者的自然語言問題。
        nl2cypher_provider: 可覆寫的 LLM Provider (例如 'openai')。
        nl2cypher_model: 可覆寫的 LLM 模型 (例如 'gpt-4o')。
        query_engine: 指定要用手動提示 ('manual') 還是 LangChain GraphQA ('graph_chain') 處理。
        progress_callback: 用來傳送非同步進度的回呼函式。
        
    回傳值:
        Dict[str, Any]: 圖譜查詢與 LLM 總結的結果。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    kwargs: Dict[str, Any] = {}
    if progress_callback is not None:
        kwargs["progress_callback"] = progress_callback
    if nl2cypher_provider:
        kwargs["nl2cypher_provider"] = nl2cypher_provider
    if nl2cypher_model:
        kwargs["nl2cypher_model"] = nl2cypher_model
    if query_engine:
        kwargs["query_engine"] = query_engine

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
    """知識圖譜問答 (同步 API)。
    
    接收前端送來的 QueryRequest (包含問題、指定的 LLM 模型等)，
    並以同步的方式阻塞執行直到生成最終回答，直接回傳給前端。
    
    如果中間發生任何錯誤 (例如連線 Neo4j 失敗，或 LLM Timeout)，
    會被 `raise_http_error` 捕捉並轉換為標準的 HTTP 500/400 Response。
    """
    try:
        return _query_with_overrides(
            question=req.question,
            nl2cypher_provider=req.nl2cypher_provider,
            nl2cypher_model=req.nl2cypher_model,
            query_engine=req.query_engine,
        )
    except Exception as exc:
        raise_http_error(exc)


def _build_initial_query_progress(
    question: str,
    *,
    nl2cypher_provider: str | None = None,
    nl2cypher_model: str | None = None,
) -> Dict[str, Any]:
    """建構非同步問答 Job 的初始進度 (Progress) 狀態結構。
    
    當發起一個非同步請求時，會先在 Job Store 裡寫入這份初始狀態，
    好讓前端輪詢 (Polling) 時能立刻知道「任務已經啟動且正在思考中」。
    
    回傳的格式包含了 `agentic_trace` 欄位，用來追蹤後續 AI Agent 在規劃或重試時的軌跡。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
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


def _query_job_status(job_id: str) -> Dict[str, Any]:
    """查詢特定非同步問答任務的當前狀態。
    
    從記憶體中的 `query_job_store` 提取 `job_id` 對應的資料。
    若任務不存在或已過期被清除，則拋出 404 錯誤。
    
    回傳值:
        Dict[str, Any]: 包含任務狀態、當前進度 (progress)、結果 (result) 或是錯誤訊息 (error)。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
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
    """發起知識圖譜問答 (非同步 API)。
    
    與 `/api/query` 不同，這個 Endpoint 會立即回傳一個 `job_id`，並在背景
    啟動一個獨立的執行緒來處理問答流程。
    適用於處理時間可能較長，前端不想保持連線阻塞的情境。
    
    目前限制：若是使用 `graph_chain` 引擎，因為其架構不支援非同步進度回報，
    會拒絕在此非同步端點中使用。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
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
        job_id = query_job_store.create(
            {
                "status": "running",
                "progress": initial_progress,
                "result": None,
                "error": None,
            }
        )

        def _run() -> None:
            """執行於背景執行緒的問答任務主要流程。"""
            # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
            # ─── 階段 2：核心處理流程 ─────────────────────────────────
            # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
            try:
                def _update_progress(event: Dict[str, Any]) -> None:
                    """當底層引擎回報進度 (Agentic Progress) 時，更新任務的狀態。"""
                    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
                    # ─── 階段 2：核心處理流程 ─────────────────────────────────
                    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
                    if event.get("type") != "agentic_progress":
                        return

                    def _mutate(job: Dict[str, Any]) -> None:
                        """更新 Job Store 中的 progress 資料。"""
                        # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
                        # ─── 階段 2：核心處理流程 ─────────────────────────────────
                        # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
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

                result = _query_with_overrides(
                    question=question,
                    nl2cypher_provider=req.nl2cypher_provider,
                    nl2cypher_model=req.nl2cypher_model,
                    query_engine=req.query_engine,
                    progress_callback=_update_progress,
                )

                def _mark_completed(job: Dict[str, Any]) -> None:
                    """背景任務成功執行完畢，將 result 寫入並標記為 completed。"""
                    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
                    # ─── 階段 2：核心處理流程 ─────────────────────────────────
                    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
                    job["status"] = "completed"
                    job["result"] = result
                    progress = dict(job.get("progress") or {})
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
                    job["progress"] = progress

                query_job_store.update(job_id, _mark_completed)
            except Exception as exc:  # pragma: no cover - integration path
                def _mark_failed(job: Dict[str, Any]) -> None:
                    """背景任務執行失敗，記錄錯誤訊息並標記為 failed。"""
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
    """前端透過此端點輪詢 (Polling) 非同步問答任務的當前進度。
    
    前端會定期以 `job_id` 呼叫此 API。當狀態為 "running" 時，會一併回傳最新的
    `agentic_trace` 等進度資料；當狀態變成 "completed" 時，就可以取得最終的 "result"。
    """
    return _query_job_status(job_id)


@router.post("/api/chat_general")
def chat_general_sync(req: GeneralChatRequest):
    """無關知識圖譜的純文字通用聊天。
    
    讓使用者可以直接把大語言模型當作一般的心理陪伴機器人對話。
    會將前端傳來的歷史紀錄 `history` 和當下訊息 `message` 餵給 `logic.chat_general` 處理。
    """
    try:
        history = [{"role": msg.role, "content": msg.content} for msg in (req.history or [])]
        return logic.chat_general(req.message, history=history)
    except Exception as exc:
        raise_http_error(exc)
