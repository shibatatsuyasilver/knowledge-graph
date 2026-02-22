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
    """執行 `_neo4j_credentials` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    cfg = get_neo4j_settings()
    return cfg.uri, cfg.user, cfg.password


def _build_initial_keyword_progress(req: KeywordRequest) -> Dict[str, Any]:
    """執行 `_build_initial_keyword_progress` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
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
    """執行 `_build_initial_ingest_progress` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
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
    """執行 `_ingest_job_status` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
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
    """執行 `_create_ingest_job` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    job_id = ingest_job_store.create(
        {
            "status": "running",
            "progress": initial_progress,
            "result": None,
            "error": None,
        }
    )

    def _wrapped() -> None:
        """執行 `_wrapped` 的內部輔助流程。
        此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
        """
        try:
            result = run_job(job_id)

            def _mark_completed(job: Dict[str, Any]) -> None:
                """執行 `_mark_completed` 的內部輔助流程。
                此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
                """
                job["status"] = "completed"
                job["result"] = result
                progress = dict(job.get("progress") or {})
                progress.update(result)
                progress["status"] = "completed"
                job["progress"] = progress

            ingest_job_store.update(job_id, _mark_completed)
        except Exception as exc:  # pragma: no cover - integration path
            def _mark_failed(job: Dict[str, Any]) -> None:
                """執行 `_mark_failed` 的內部輔助流程。
                此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
                """
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
    """處理 `POST /api/process_url` 請求並回傳既有服務結果。
    函式會沿用目前驗證與例外處理策略，維持 API 契約與回應格式一致。
    """
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
    """處理 `POST /api/process_text` 請求並回傳既有服務結果。
    函式會沿用目前驗證與例外處理策略，維持 API 契約與回應格式一致。
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
    """處理 `POST /api/process_text_async/start` 請求並回傳既有服務結果。
    函式會沿用目前驗證與例外處理策略，維持 API 契約與回應格式一致。
    """
    try:
        initial_progress = _build_initial_ingest_progress(req.chunk_limit)

        def _run(job_id: str) -> Dict[str, Any]:
            """執行 `_run` 的內部輔助流程。
            此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
            """
            def _update_progress(event: Dict[str, Any]) -> None:
                """執行 `_update_progress` 的內部輔助流程。
                此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
                """
                if event.get("type") != "chunk_update":
                    return

                def _mutate(job: Dict[str, Any]) -> None:
                    """執行 `_mutate` 的內部輔助流程。
                    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
                    """
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
    """處理 `GET /api/process_text_async/{job_id}` 請求並回傳既有服務結果。
    函式會沿用目前驗證與例外處理策略，維持 API 契約與回應格式一致。
    """
    return _ingest_job_status(job_id)


@router.post("/api/process_url_async/start")
def process_url_async_start(req: UrlRequest):
    """處理 `POST /api/process_url_async/start` 請求並回傳既有服務結果。
    函式會沿用目前驗證與例外處理策略，維持 API 契約與回應格式一致。
    """
    try:
        initial_progress = _build_initial_ingest_progress(req.chunk_limit, current_url=req.url.strip())

        def _run(job_id: str) -> Dict[str, Any]:
            """執行 `_run` 的內部輔助流程。
            此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
            """
            def _update_progress(event: Dict[str, Any]) -> None:
                """執行 `_update_progress` 的內部輔助流程。
                此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
                """
                if event.get("type") != "chunk_update":
                    return

                def _mutate(job: Dict[str, Any]) -> None:
                    """執行 `_mutate` 的內部輔助流程。
                    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
                    """
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
    """處理 `GET /api/process_url_async/{job_id}` 請求並回傳既有服務結果。
    函式會沿用目前驗證與例外處理策略，維持 API 契約與回應格式一致。
    """
    return _ingest_job_status(job_id)


@router.post("/api/process_keyword")
def process_keyword_sync(req: KeywordRequest):
    """處理 `POST /api/process_keyword` 請求並回傳既有服務結果。
    函式會沿用目前驗證與例外處理策略，維持 API 契約與回應格式一致。
    """
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
    """處理 `POST /api/process_keyword_async/start` 請求並回傳既有服務結果。
    函式會沿用目前驗證與例外處理策略，維持 API 契約與回應格式一致。
    """
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
            """執行 `_update_progress` 的內部輔助流程。
            此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
            """
            def _mutate(job: Dict[str, Any]) -> None:
                """執行 `_mutate` 的內部輔助流程。
                此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
                """
                job["progress"] = payload

            keyword_job_store.update(job_id, _mutate)

        def _run_job() -> None:
            """執行 `_run_job` 的內部輔助流程。
            此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
            """
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
                    """執行 `_mark_completed` 的內部輔助流程。
                    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
                    """
                    job["status"] = "completed"
                    job["result"] = result
                    job["progress"] = result

                keyword_job_store.update(job_id, _mark_completed)
            except Exception as exc:  # pragma: no cover - integration path
                def _mark_failed(job: Dict[str, Any]) -> None:
                    """執行 `_mark_failed` 的內部輔助流程。
                    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
                    """
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
    """處理 `GET /api/process_keyword_async/{job_id}` 請求並回傳既有服務結果。
    函式會沿用目前驗證與例外處理策略，維持 API 契約與回應格式一致。
    """
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
