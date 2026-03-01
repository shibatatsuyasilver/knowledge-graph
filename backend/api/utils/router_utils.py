"""Generic router-level utilities shared across API routers."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import HTTPException

from backend.jobs.store import JobStore


def get_job_status(job_store: JobStore, job_id: str) -> Dict[str, Any]:
    """Return the standard job status response dict, or raise 404 if not found."""
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or expired")
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "progress": job.get("progress"),
        "result": job.get("result"),
        "error": job.get("error"),
    }
