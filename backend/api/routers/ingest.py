"""Ingestion API routes and async job orchestration."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException

from backend.services.ingest import ingest_service
from backend.api.errors import raise_http_error
from backend.api.models import KeywordRequest, TextRequest, UrlRequest
from backend.api.utils.job_runner import create_async_job
from backend.api.utils.router_utils import get_job_status
from backend.config.settings import get_neo4j_settings
from backend.jobs.runtime import ingest_job_store, keyword_job_store

router = APIRouter()


def _neo4j_credentials() -> tuple[str, str, str]:
    """從設定檔提取 Neo4j 圖形資料庫的連線憑證。

    回傳值:
        tuple[str, str, str]: 包含 (URI, 帳號, 密碼) 的 Tuple。
    """
    cfg = get_neo4j_settings()
    return cfg.uri, cfg.user, cfg.password


def _build_initial_keyword_progress(req: KeywordRequest) -> Dict[str, Any]:
    """建構關鍵字爬取任務的初始狀態結構。"""
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
    """建構純文本或單一 URL 爬取任務的初始狀態結構。"""
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


def _merge_chunk_progress(
    event: Dict[str, Any],
    job_id: str,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> None:
    """Handle a ``chunk_update`` event by merging it into the job's progress.

    If ``event["type"]`` is not ``"chunk_update"``, returns immediately without
    modifying the job.

    Args:
        event: The progress event dict (should have ``type == "chunk_update"``).
        job_id: The ingest job to update.
        extra_fields: Optional extra key/value pairs written into progress
            (e.g. ``{"current_url": ...}`` for URL jobs).
    """
    if event.get("type") != "chunk_update":
        return

    def _mutate(job: Dict[str, Any]) -> None:
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
        if extra_fields:
            progress.update(extra_fields)
        progress["status"] = "running"
        job["progress"] = progress

    ingest_job_store.update(job_id, _mutate)


@router.post("/api/process_text_async/start")
def process_text_async_start(req: TextRequest):
    """發起處理指定純文字為知識圖譜的非同步任務。"""
    try:
        initial_progress = _build_initial_ingest_progress(req.chunk_limit)

        def _run(job_id: str) -> Dict[str, Any]:
            def _update_progress(event: Dict[str, Any]) -> None:
                _merge_chunk_progress(event, job_id)

            uri, user, pwd = _neo4j_credentials()
            return ingest_service.process_text_to_kg(
                req.text,
                uri,
                user,
                pwd,
                chunk_limit=req.chunk_limit,
                extraction_provider=req.extraction_provider,
                extraction_model=req.extraction_model,
                progress_callback=_update_progress,
            )

        return create_async_job(
            job_store=ingest_job_store,
            initial_progress=initial_progress,
            run_job=_run,
            name_prefix="text-job",
        )
    except Exception as exc:
        raise_http_error(exc)


@router.get("/api/process_text_async/{job_id}")
def process_text_async_status(job_id: str):
    """查詢由 `/api/process_text_async/start` 發起的任務狀態。"""
    return get_job_status(ingest_job_store, job_id)


@router.post("/api/process_url_async/start")
def process_url_async_start(req: UrlRequest):
    """發起處理單一網址為知識圖譜的非同步任務。"""
    try:
        initial_progress = _build_initial_ingest_progress(req.chunk_limit, current_url=req.url.strip())

        def _run(job_id: str) -> Dict[str, Any]:
            def _update_progress(event: Dict[str, Any]) -> None:
                _merge_chunk_progress(event, job_id, extra_fields={"current_url": req.url.strip()})

            uri, user, pwd = _neo4j_credentials()
            return ingest_service.process_url_to_kg(
                req.url,
                uri,
                user,
                pwd,
                chunk_limit=req.chunk_limit,
                extraction_provider=req.extraction_provider,
                extraction_model=req.extraction_model,
                progress_callback=_update_progress,
            )

        return create_async_job(
            job_store=ingest_job_store,
            initial_progress=initial_progress,
            run_job=_run,
            name_prefix="url-job",
        )
    except Exception as exc:
        raise_http_error(exc)


@router.get("/api/process_url_async/{job_id}")
def process_url_async_status(job_id: str):
    """查詢由 `/api/process_url_async/start` 發起的任務狀態。"""
    return get_job_status(ingest_job_store, job_id)


@router.post("/api/process_keyword_async/start")
def process_keyword_async_start(req: KeywordRequest):
    """發起將關鍵字轉知識圖譜的複雜非同步任務。"""
    try:
        initial_progress = _build_initial_keyword_progress(req)

        def _run(job_id: str) -> Dict[str, Any]:
            def _update_progress(payload: Dict[str, Any]) -> None:
                def _mutate(job: Dict[str, Any]) -> None:
                    job["progress"] = payload

                keyword_job_store.update(job_id, _mutate)

            uri, user, pwd = _neo4j_credentials()
            return ingest_service.process_keyword_to_kg(
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

        def _keyword_on_complete(result: Dict[str, Any], progress: Dict[str, Any]) -> Dict[str, Any]:
            return result

        return create_async_job(
            job_store=keyword_job_store,
            initial_progress=initial_progress,
            run_job=_run,
            name_prefix="keyword-job",
            on_complete=_keyword_on_complete,
        )
    except Exception as exc:
        raise_http_error(exc)


@router.get("/api/process_keyword_async/{job_id}")
def process_keyword_async_status(job_id: str):
    """查詢由 `/api/process_keyword_async/start` 發起的關鍵字任務狀態。"""
    return get_job_status(keyword_job_store, job_id)
