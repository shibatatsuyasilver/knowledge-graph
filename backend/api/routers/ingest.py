"""Ingestion API routes and async job orchestration."""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException

import backend.logic as logic
from backend.api.errors import raise_http_error
from backend.api.models import KeywordRequest, TextRequest, UrlRequest
from backend.config.settings import get_neo4j_settings
from backend.jobs.runtime import ingest_job_store, keyword_job_store

router = APIRouter()


def _neo4j_credentials() -> tuple[str, str, str]:
    """從設定檔提取 Neo4j 圖形資料庫的連線憑證。
    
    回傳值:
        tuple[str, str, str]: 包含 (URI, 帳號, 密碼) 的 Tuple，例如 ("bolt://localhost:7687", "neo4j", "password")
    """
    cfg = get_neo4j_settings()
    return cfg.uri, cfg.user, cfg.password


def _build_initial_keyword_progress(req: KeywordRequest) -> Dict[str, Any]:
    """建構關鍵字爬取任務 (Keyword Ingestion) 的初始狀態結構。
    
    當發起背景關鍵字任務時，系統需要一個空白的進度表以供後續填寫與前端輪詢。
    
    參數:
        req (KeywordRequest): 包含使用者搜尋參數的請求物件。
        
    回傳值:
        Dict[str, Any]: 初始的 progress 狀態字典，包含空陣列與歸零的計數器。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    return {
        "status": "running",
        "searched_keyword": req.keyword.strip(),
        "fetched_urls": [],
        "failed_urls": [],
        "stats": {
            "chunks_processed": 0,
            "entities": 0,
            "relations": 0,
            "merged_entities": 0,
            "dropped_relations": 0,
            "json_retries": 0,
        },
        "summary": [],
        "chunk_limit": req.chunk_limit,
        "chunks_available": 0,
        "chunk_progress": [],
    }


def _build_initial_ingest_progress(chunk_limit: Optional[int], *, current_url: Optional[str] = None) -> Dict[str, Any]:
    """建構純文本或單一 URL 爬取任務的初始狀態結構。
    
    參數:
        chunk_limit: 本次允許處理的最大 Chunk 數量。
        current_url: (可選) 正在處理的目標網址。
        
    回傳值:
        Dict[str, Any]: 初始的 progress 狀態字典。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    payload: Dict[str, Any] = {
        "status": "running",
        "stats": {
            "chunks_processed": 0,
            "entities": 0,
            "relations": 0,
            "merged_entities": 0,
            "dropped_relations": 0,
            "json_retries": 0,
        },
        "summary": [],
        "chunk_limit": chunk_limit,
        "chunks_available": 0,
        "chunk_progress": [],
    }
    if current_url:
        payload["current_url"] = current_url
    return payload


def _ingest_job_status(job_id: str) -> Dict[str, Any]:
    """查詢非同步 Ingest Job 的當前狀態。
    
    透過 `job_id` 從記憶體中的 `ingest_job_store` 讀取最新進度。
    
    回傳值:
        Dict[str, Any]: 包含 "running"、"completed" 或 "failed" 等狀態與資料。
    """
    job = ingest_job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or expired")
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "progress": job.get("progress"),
        "result": job.get("result"),
        "error": job.get("error"),
    }


def _create_ingest_job(
    *,
    initial_progress: Dict[str, Any],
    run_job: Any,
    name_prefix: str,
) -> Dict[str, str]:
    """共用的建立非同步任務工廠函式。
    
    為不同的 Ingest 操作 (如 URL 解析、純文字解析) 提供一個標準化的背景執行緒封裝，
    並處理任務建立、完成、與例外錯誤的狀態更新。
    
    參數:
        initial_progress: 任務的初始狀態字典。
        run_job: 實際要在背景執行的處理函式 (需接受 job_id)。
        name_prefix: 用來命名執行緒的前綴字串 (例如 "url-job" 或 "text-job")。
        
    回傳值:
        Dict[str, str]: 包含任務 ID 與初始狀態的字典。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    job_id = ingest_job_store.create(
        {
            "status": "running",
            "progress": initial_progress,
            "result": None,
            "error": None,
        }
    )

    def _wrapped() -> None:
        """包裝在背景執行緒中執行的函式，負責捕捉任務成功與失敗的狀態。"""
        # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
        # ─── 階段 2：核心處理流程 ─────────────────────────────────
        # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
        try:
            result = run_job(job_id)

            def _mark_completed(job: Dict[str, Any]) -> None:
                """任務成功完成，更新最終結果到 Store 中。"""
                job["status"] = "completed"
                job["result"] = result
                progress = dict(job.get("progress") or {})
                progress.update(result)
                progress["status"] = "completed"
                job["progress"] = progress

            ingest_job_store.update(job_id, _mark_completed)
        except Exception as exc:  # pragma: no cover - integration path
            def _mark_failed(job: Dict[str, Any]) -> None:
                """任務執行失敗，記錄錯誤到 Store 中供前端查詢。"""
                job["status"] = "failed"
                job["error"] = str(exc)
                progress = dict(job.get("progress") or {})
                progress["status"] = "failed"
                progress["error"] = str(exc)
                job["progress"] = progress

            ingest_job_store.update(job_id, _mark_failed)

    worker = threading.Thread(target=_wrapped, daemon=True, name=f"{name_prefix}-{job_id[:8]}")
    worker.start()
    return {"job_id": job_id, "status": "running"}


@router.post("/api/process_url")
def process_url_sync(req: UrlRequest):
    """處理指定的單一 URL，將內容轉換為知識圖譜 (同步 API)。
    
    此 API 會阻塞連線，直到目標網址的內容完全抓取、清洗並交由 LLM 萃取出圖譜為止。
    適用於小規模網頁測試或沒有非同步需求的情境。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    try:
        uri, user, pwd = _neo4j_credentials()
        return logic.process_url_to_kg(
            req.url,
            uri,
            user,
            pwd,
            chunk_limit=req.chunk_limit,
            extraction_provider=req.extraction_provider,
            extraction_model=req.extraction_model,
        )
    except Exception as exc:
        raise_http_error(exc)


@router.post("/api/process_text")
def process_text_sync(req: TextRequest):
    """將一段指定的純文字內容轉換為知識圖譜 (同步 API)。
    
    與 URL 不同，這是直接把文字餵入，不需要爬蟲處理。
    """
    try:
        uri, user, pwd = _neo4j_credentials()
        return logic.process_text_to_kg(
            req.text,
            uri,
            user,
            pwd,
            chunk_limit=req.chunk_limit,
            extraction_provider=req.extraction_provider,
            extraction_model=req.extraction_model,
        )
    except Exception as exc:
        raise_http_error(exc)


@router.post("/api/process_text_async/start")
def process_text_async_start(req: TextRequest):
    """發起處理指定純文字為知識圖譜的非同步任務。
    
    將長時間的 LLM 萃取與資料庫寫入放在背景執行緒，並立即回傳 `job_id` 供前端後續查詢。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    try:
        initial_progress = _build_initial_ingest_progress(req.chunk_limit)

        def _run(job_id: str) -> Dict[str, Any]:
            """背景任務的實際執行函式。"""
            # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
            # ─── 階段 2：核心處理流程 ─────────────────────────────────
            # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
            
            def _update_progress(event: Dict[str, Any]) -> None:
                """當處理每個 Chunk 完成時的回呼，用來更新記憶體中該 Job 的進度。"""
                # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
                # ─── 階段 2：核心處理流程 ─────────────────────────────────
                # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
                if event.get("type") != "chunk_update":
                    return

                def _mutate(job: Dict[str, Any]) -> None:
                    """安全的在鎖定範圍內更新 Job 狀態。"""
                    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
                    # ─── 階段 2：核心處理流程 ─────────────────────────────────
                    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
                    progress = dict(job.get("progress") or {})
                    chunk_row = event.get("chunk")
                    if isinstance(chunk_row, dict):
                        existing_rows = [dict(item) for item in progress.get("chunk_progress", [])]
                        chunk_id = str(chunk_row.get("chunk_id", ""))
                        found = False
                        # 若已有相同的 chunk_id，就更新它
                        for idx, row in enumerate(existing_rows):
                            if str(row.get("chunk_id", "")) == chunk_id:
                                existing_rows[idx] = {**row, **chunk_row}
                                found = True
                                break
                        # 沒有則新增一筆
                        if not found:
                            existing_rows.append(dict(chunk_row))
                        progress["chunk_progress"] = existing_rows
                    if isinstance(event.get("stats"), dict):
                        progress["stats"] = dict(event["stats"])
                    if event.get("chunk_limit") is not None:
                        progress["chunk_limit"] = event.get("chunk_limit")
                    if event.get("chunks_available") is not None:
                        progress["chunks_available"] = event.get("chunks_available")
                    progress["status"] = "running"
                    job["progress"] = progress

                ingest_job_store.update(job_id, _mutate)

            uri, user, pwd = _neo4j_credentials()
            return logic.process_text_to_kg(
                req.text,
                uri,
                user,
                pwd,
                chunk_limit=req.chunk_limit,
                extraction_provider=req.extraction_provider,
                extraction_model=req.extraction_model,
                progress_callback=_update_progress,
            )

        return _create_ingest_job(
            initial_progress=initial_progress,
            run_job=_run,
            name_prefix="text-job",
        )
    except Exception as exc:
        raise_http_error(exc)


@router.get("/api/process_text_async/{job_id}")
def process_text_async_status(job_id: str):
    """查詢由 `/api/process_text_async/start` 發起的任務狀態。"""
    return _ingest_job_status(job_id)


@router.post("/api/process_url_async/start")
def process_url_async_start(req: UrlRequest):
    """發起處理單一網址為知識圖譜的非同步任務。"""
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    try:
        initial_progress = _build_initial_ingest_progress(req.chunk_limit, current_url=req.url.strip())

        def _run(job_id: str) -> Dict[str, Any]:
            """背景任務的實際執行函式。"""
            # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
            # ─── 階段 2：核心處理流程 ─────────────────────────────────
            # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
            
            def _update_progress(event: Dict[str, Any]) -> None:
                """攔截進度更新並寫回 Store 中。"""
                # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
                # ─── 階段 2：核心處理流程 ─────────────────────────────────
                # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
                if event.get("type") != "chunk_update":
                    return

                def _mutate(job: Dict[str, Any]) -> None:
                    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
                    # ─── 階段 2：核心處理流程 ─────────────────────────────────
                    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
                    progress = dict(job.get("progress") or {})
                    chunk_row = event.get("chunk")
                    if isinstance(chunk_row, dict):
                        existing_rows = [dict(item) for item in progress.get("chunk_progress", [])]
                        chunk_id = str(chunk_row.get("chunk_id", ""))
                        found = False
                        for idx, row in enumerate(existing_rows):
                            if str(row.get("chunk_id", "")) == chunk_id:
                                existing_rows[idx] = {**row, **chunk_row}
                                found = True
                                break
                        if not found:
                            existing_rows.append(dict(chunk_row))
                        progress["chunk_progress"] = existing_rows
                    if isinstance(event.get("stats"), dict):
                        progress["stats"] = dict(event["stats"])
                    if event.get("chunk_limit") is not None:
                        progress["chunk_limit"] = event.get("chunk_limit")
                    if event.get("chunks_available") is not None:
                        progress["chunks_available"] = event.get("chunks_available")
                    progress["current_url"] = req.url.strip()
                    progress["status"] = "running"
                    job["progress"] = progress

                ingest_job_store.update(job_id, _mutate)

            uri, user, pwd = _neo4j_credentials()
            return logic.process_url_to_kg(
                req.url,
                uri,
                user,
                pwd,
                chunk_limit=req.chunk_limit,
                extraction_provider=req.extraction_provider,
                extraction_model=req.extraction_model,
                progress_callback=_update_progress,
            )

        return _create_ingest_job(
            initial_progress=initial_progress,
            run_job=_run,
            name_prefix="url-job",
        )
    except Exception as exc:
        raise_http_error(exc)


@router.get("/api/process_url_async/{job_id}")
def process_url_async_status(job_id: str):
    """查詢由 `/api/process_url_async/start` 發起的任務狀態。"""
    return _ingest_job_status(job_id)


@router.post("/api/process_keyword")
def process_keyword_sync(req: KeywordRequest):
    """將給定關鍵字自動搜尋、爬取多個網頁並建立知識圖譜 (同步 API)。
    
    這個流程可能會花費數分鐘，因為包含爬蟲、多次呼叫 LLM 以及寫入資料庫。
    對於正式環境，建議使用 `/api/process_keyword_async/start`。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    try:
        uri, user, pwd = _neo4j_credentials()
        return logic.process_keyword_to_kg(
            keyword=req.keyword,
            uri=uri,
            user=user,
            pwd=pwd,
            max_results=req.max_results,
            language=req.language,
            site_allowlist=req.site_allowlist,
            chunk_limit=req.chunk_limit,
            extraction_provider=req.extraction_provider,
            extraction_model=req.extraction_model,
        )
    except Exception as exc:
        raise_http_error(exc)


@router.post("/api/process_keyword_async/start")
def process_keyword_async_start(req: KeywordRequest):
    """發起將關鍵字轉知識圖譜的複雜非同步任務。
    
    由於這個任務會跨越多個不同的網址，並擁有自己的 Store (`keyword_job_store`)，
    它會在背景收集來自 `logic.process_keyword_to_kg` 中每個網址的執行進度並更新狀態。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    try:
        job_id = keyword_job_store.create(
            {
                "status": "running",
                "progress": _build_initial_keyword_progress(req),
                "result": None,
                "error": None,
            }
        )

        def _update_progress(payload: Dict[str, Any]) -> None:
            """將底層回傳的整體進度直接覆寫掉 Job 中的 progress。"""
            def _mutate(job: Dict[str, Any]) -> None:
                job["progress"] = payload

            keyword_job_store.update(job_id, _mutate)

        def _run_job() -> None:
            """在背景執行緒中執行關鍵字爬取與圖譜建立。"""
            # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
            # ─── 階段 2：核心處理流程 ─────────────────────────────────
            # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
            try:
                uri, user, pwd = _neo4j_credentials()
                result = logic.process_keyword_to_kg(
                    keyword=req.keyword,
                    uri=uri,
                    user=user,
                    pwd=pwd,
                    max_results=req.max_results,
                    language=req.language,
                    site_allowlist=req.site_allowlist,
                    chunk_limit=req.chunk_limit,
                    extraction_provider=req.extraction_provider,
                    extraction_model=req.extraction_model,
                    progress_callback=_update_progress,
                )

                def _mark_completed(job: Dict[str, Any]) -> None:
                    job["status"] = "completed"
                    job["result"] = result
                    job["progress"] = result

                keyword_job_store.update(job_id, _mark_completed)
            except Exception as exc:  # pragma: no cover - integration path
                def _mark_failed(job: Dict[str, Any]) -> None:
                    job["status"] = "failed"
                    job["error"] = str(exc)
                    progress = dict(job.get("progress") or {})
                    progress["status"] = "failed"
                    progress["error"] = str(exc)
                    job["progress"] = progress

                keyword_job_store.update(job_id, _mark_failed)

        worker = threading.Thread(target=_run_job, daemon=True, name=f"keyword-job-{job_id[:8]}")
        worker.start()
        return {"job_id": job_id, "status": "running"}
    except Exception as exc:
        raise_http_error(exc)


@router.get("/api/process_keyword_async/{job_id}")
def process_keyword_async_status(job_id: str):
    """查詢由 `/api/process_keyword_async/start` 發起的關鍵字任務狀態。"""
    job = keyword_job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or expired")
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "progress": job.get("progress"),
        "result": job.get("result"),
        "error": job.get("error"),
    }
