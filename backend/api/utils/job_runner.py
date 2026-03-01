"""Shared async job orchestration for background thread jobs."""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, Optional

from backend.jobs.store import JobStore


def create_async_job(
    job_store: JobStore,
    initial_progress: Dict[str, Any],
    run_job: Callable[[str], Any],
    name_prefix: str,
    on_complete: Optional[Callable[[Any, Dict[str, Any]], Dict[str, Any]]] = None,
    fail_progress_field: str = "error",
) -> Dict[str, str]:
    """Create and start a background job thread, returning job_id immediately.

    Args:
        job_store: The JobStore instance to persist job state.
        initial_progress: The initial progress dict stored when the job is created.
        run_job: Callable receiving ``job_id`` and returning the final result dict.
        name_prefix: Thread name prefix (e.g. ``"text-job"``).
        on_complete: Optional ``(result, progress) -> new_progress`` callback.
            When omitted, the default behaviour merges result into progress via
            ``progress.update(result)`` and sets ``progress["status"] = "completed"``.
        fail_progress_field: Key used to store the error string in progress on
            failure.  Defaults to ``"error"``; pass ``"detail"`` for QA jobs.

    Returns:
        ``{"job_id": ..., "status": "running"}``
    """
    job_id = job_store.create(
        {
            "status": "running",
            "progress": initial_progress,
            "result": None,
            "error": None,
        }
    )

    def _wrapped() -> None:
        try:
            result = run_job(job_id)

            def _mark_completed(job: Dict[str, Any]) -> None:
                job["status"] = "completed"
                job["result"] = result
                progress = dict(job.get("progress") or {})
                if on_complete is not None:
                    progress = on_complete(result, progress)
                else:
                    progress.update(result)
                    progress["status"] = "completed"
                job["progress"] = progress

            job_store.update(job_id, _mark_completed)
        except Exception as exc:  # pragma: no cover - integration path
            def _mark_failed(job: Dict[str, Any]) -> None:
                job["status"] = "failed"
                job["error"] = str(exc)
                progress = dict(job.get("progress") or {})
                progress["status"] = "failed"
                progress[fail_progress_field] = str(exc)
                job["progress"] = progress

            job_store.update(job_id, _mark_failed)

    worker = threading.Thread(target=_wrapped, daemon=True, name=f"{name_prefix}-{job_id[:8]}")
    worker.start()
    return {"job_id": job_id, "status": "running"}
