"""Runtime singleton job stores."""

from __future__ import annotations

from backend.config.settings import get_job_ttl_settings
from backend.jobs.store import JobStore

_ttls = get_job_ttl_settings()

keyword_job_store = JobStore(_ttls.keyword_job_ttl_seconds)
ingest_job_store = JobStore(_ttls.ingest_job_ttl_seconds)
