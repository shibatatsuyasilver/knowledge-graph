import os
import sys
import threading
import time
import uuid
from typing import Any, Dict, List, Literal, Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load env
load_dotenv()

# Setup path to include project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import logic
# We use try-except to handle different execution contexts
try:
    import backend.logic as logic
except ImportError:
    try:
        import logic
    except ImportError as e:
        print(f"Error importing logic: {e}")
        raise

app = FastAPI(title="GenAI KG Backend")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Models
class UrlRequest(BaseModel):
    url: str
    chunk_limit: Optional[int] = None
    extraction_provider: Optional[Literal["ollama", "gemini"]] = None
    extraction_model: Optional[str] = None


class TextRequest(BaseModel):
    text: str
    chunk_limit: Optional[int] = None
    extraction_provider: Optional[Literal["ollama", "gemini"]] = None
    extraction_model: Optional[str] = None


class QueryRequest(BaseModel):
    question: str


class KeywordRequest(BaseModel):
    keyword: str
    max_results: int = 5
    language: Literal["zh-tw", "en"] = "zh-tw"
    site_allowlist: Optional[List[str]] = None
    chunk_limit: Optional[int] = None
    extraction_provider: Optional[Literal["ollama", "gemini"]] = None
    extraction_model: Optional[str] = None


class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class GeneralChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatHistoryMessage]] = None


# Config
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
KEYWORD_JOB_TTL_SECONDS = int(os.getenv("KEYWORD_JOB_TTL_SECONDS", "3600"))
INGEST_JOB_TTL_SECONDS = int(os.getenv("INGEST_JOB_TTL_SECONDS", "3600"))


keyword_jobs: Dict[str, Dict[str, Any]] = {}
keyword_jobs_lock = threading.Lock()
ingest_jobs: Dict[str, Dict[str, Any]] = {}
ingest_jobs_lock = threading.Lock()


def _cleanup_keyword_jobs() -> None:
    now = time.time()
    expired: List[str] = []
    with keyword_jobs_lock:
        for job_id, job in keyword_jobs.items():
            updated_at = float(job.get("updated_at", now))
            if now - updated_at > KEYWORD_JOB_TTL_SECONDS:
                expired.append(job_id)
        for job_id in expired:
            keyword_jobs.pop(job_id, None)


def _cleanup_ingest_jobs() -> None:
    now = time.time()
    expired: List[str] = []
    with ingest_jobs_lock:
        for job_id, job in ingest_jobs.items():
            updated_at = float(job.get("updated_at", now))
            if now - updated_at > INGEST_JOB_TTL_SECONDS:
                expired.append(job_id)
        for job_id in expired:
            ingest_jobs.pop(job_id, None)


def _build_initial_keyword_progress(req: KeywordRequest) -> Dict[str, Any]:
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


def _create_ingest_job(
    *,
    initial_progress: Dict[str, Any],
    run_job: Any,
    name_prefix: str,
) -> Dict[str, str]:
    _cleanup_ingest_jobs()
    job_id = uuid.uuid4().hex
    now = time.time()
    with ingest_jobs_lock:
        ingest_jobs[job_id] = {
            "job_id": job_id,
            "status": "running",
            "progress": initial_progress,
            "result": None,
            "error": None,
            "created_at": now,
            "updated_at": now,
        }

    def _wrapped() -> None:
        try:
            result = run_job(job_id)
            with ingest_jobs_lock:
                job = ingest_jobs.get(job_id)
                if not job:
                    return
                job["status"] = "completed"
                job["result"] = result
                progress = dict(job.get("progress") or {})
                progress.update(result)
                progress["status"] = "completed"
                job["progress"] = progress
                job["updated_at"] = time.time()
        except Exception as exc:  # pragma: no cover - integration path
            with ingest_jobs_lock:
                job = ingest_jobs.get(job_id)
                if not job:
                    return
                job["status"] = "failed"
                job["error"] = str(exc)
                progress = dict(job.get("progress") or {})
                progress["status"] = "failed"
                progress["error"] = str(exc)
                job["progress"] = progress
                job["updated_at"] = time.time()

    worker = threading.Thread(target=_wrapped, daemon=True, name=f"{name_prefix}-{job_id[:8]}")
    worker.start()
    return {"job_id": job_id, "status": "running"}


def _ingest_job_status(job_id: str) -> Dict[str, Any]:
    _cleanup_ingest_jobs()
    with ingest_jobs_lock:
        job = ingest_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found or expired")
        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "progress": job.get("progress"),
            "result": job.get("result"),
            "error": job.get("error"),
        }


def _raise_http_error(exc: Exception) -> None:
    if isinstance(exc, HTTPException):
        raise exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, requests.Timeout):
        raise HTTPException(status_code=504, detail="Upstream service timeout")
    if isinstance(exc, requests.RequestException):
        raise HTTPException(status_code=502, detail=f"Upstream service error: {exc}")
    raise HTTPException(status_code=500, detail=f"Internal server error: {exc}")


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "GenAI KG Backend API is running"}


@app.post("/api/process_url")
def process_url_sync(req: UrlRequest):
    try:
        return logic.process_url_to_kg(
            req.url,
            NEO4J_URI,
            NEO4J_USER,
            NEO4J_PASSWORD,
            chunk_limit=req.chunk_limit,
            extraction_provider=req.extraction_provider,
            extraction_model=req.extraction_model,
        )
    except Exception as exc:
        _raise_http_error(exc)


@app.post("/api/process_text")
def process_text_sync(req: TextRequest):
    try:
        return logic.process_text_to_kg(
            req.text,
            NEO4J_URI,
            NEO4J_USER,
            NEO4J_PASSWORD,
            chunk_limit=req.chunk_limit,
            extraction_provider=req.extraction_provider,
            extraction_model=req.extraction_model,
        )
    except Exception as exc:
        _raise_http_error(exc)


@app.post("/api/process_text_async/start")
def process_text_async_start(req: TextRequest):
    try:
        initial_progress = _build_initial_ingest_progress(req.chunk_limit)

        def _run(job_id: str) -> Dict[str, Any]:
            def _update_progress(event: Dict[str, Any]) -> None:
                if event.get("type") != "chunk_update":
                    return
                with ingest_jobs_lock:
                    job = ingest_jobs.get(job_id)
                    if not job:
                        return
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
                    job["updated_at"] = time.time()

            return logic.process_text_to_kg(
                req.text,
                NEO4J_URI,
                NEO4J_USER,
                NEO4J_PASSWORD,
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
        _raise_http_error(exc)


@app.get("/api/process_text_async/{job_id}")
def process_text_async_status(job_id: str):
    return _ingest_job_status(job_id)


@app.post("/api/process_url_async/start")
def process_url_async_start(req: UrlRequest):
    try:
        initial_progress = _build_initial_ingest_progress(req.chunk_limit, current_url=req.url.strip())

        def _run(job_id: str) -> Dict[str, Any]:
            def _update_progress(event: Dict[str, Any]) -> None:
                if event.get("type") != "chunk_update":
                    return
                with ingest_jobs_lock:
                    job = ingest_jobs.get(job_id)
                    if not job:
                        return
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
                    job["updated_at"] = time.time()

            return logic.process_url_to_kg(
                req.url,
                NEO4J_URI,
                NEO4J_USER,
                NEO4J_PASSWORD,
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
        _raise_http_error(exc)


@app.get("/api/process_url_async/{job_id}")
def process_url_async_status(job_id: str):
    return _ingest_job_status(job_id)


@app.post("/api/process_keyword")
def process_keyword_sync(req: KeywordRequest):
    try:
        return logic.process_keyword_to_kg(
            keyword=req.keyword,
            uri=NEO4J_URI,
            user=NEO4J_USER,
            pwd=NEO4J_PASSWORD,
            max_results=req.max_results,
            language=req.language,
            site_allowlist=req.site_allowlist,
            chunk_limit=req.chunk_limit,
            extraction_provider=req.extraction_provider,
            extraction_model=req.extraction_model,
        )
    except Exception as exc:
        _raise_http_error(exc)


@app.post("/api/process_keyword_async/start")
def process_keyword_async_start(req: KeywordRequest):
    try:
        _cleanup_keyword_jobs()
        job_id = uuid.uuid4().hex
        now = time.time()

        with keyword_jobs_lock:
            keyword_jobs[job_id] = {
                "job_id": job_id,
                "status": "running",
                "progress": _build_initial_keyword_progress(req),
                "result": None,
                "error": None,
                "created_at": now,
                "updated_at": now,
            }

        def _update_progress(payload: Dict[str, Any]) -> None:
            with keyword_jobs_lock:
                job = keyword_jobs.get(job_id)
                if not job:
                    return
                job["progress"] = payload
                job["updated_at"] = time.time()

        def _run_job() -> None:
            try:
                result = logic.process_keyword_to_kg(
                    keyword=req.keyword,
                    uri=NEO4J_URI,
                    user=NEO4J_USER,
                    pwd=NEO4J_PASSWORD,
                    max_results=req.max_results,
                    language=req.language,
                    site_allowlist=req.site_allowlist,
                    chunk_limit=req.chunk_limit,
                    extraction_provider=req.extraction_provider,
                    extraction_model=req.extraction_model,
                    progress_callback=_update_progress,
                )
                with keyword_jobs_lock:
                    job = keyword_jobs.get(job_id)
                    if not job:
                        return
                    job["status"] = "completed"
                    job["result"] = result
                    job["progress"] = result
                    job["updated_at"] = time.time()
            except Exception as exc:  # pragma: no cover - integration path
                with keyword_jobs_lock:
                    job = keyword_jobs.get(job_id)
                    if not job:
                        return
                    job["status"] = "failed"
                    job["error"] = str(exc)
                    progress = dict(job.get("progress") or {})
                    progress["status"] = "failed"
                    progress["error"] = str(exc)
                    job["progress"] = progress
                    job["updated_at"] = time.time()

        worker = threading.Thread(target=_run_job, daemon=True, name=f"keyword-job-{job_id[:8]}")
        worker.start()
        return {"job_id": job_id, "status": "running"}
    except Exception as exc:
        _raise_http_error(exc)


@app.get("/api/process_keyword_async/{job_id}")
def process_keyword_async_status(job_id: str):
    _cleanup_keyword_jobs()
    with keyword_jobs_lock:
        job = keyword_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found or expired")
        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "progress": job.get("progress"),
            "result": job.get("result"),
            "error": job.get("error"),
        }


@app.post("/api/query")
def query_sync(req: QueryRequest):
    try:
        return logic.query_kg(req.question)
    except Exception as exc:
        _raise_http_error(exc)


@app.post("/api/chat_general")
def chat_general_sync(req: GeneralChatRequest):
    try:
        history = [{"role": msg.role, "content": msg.content} for msg in (req.history or [])]
        return logic.chat_general(req.message, history=history)
    except Exception as exc:
        _raise_http_error(exc)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
