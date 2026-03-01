"""Thread-safe in-memory job store with TTL cleanup."""

from __future__ import annotations

import threading
import time
import uuid
from copy import deepcopy
from typing import Any, Callable, Dict, Optional


class JobStore:
    def __init__(self, ttl_seconds: int) -> None:
        """初始化任務儲存：設定 TTL 秒數、建立執行緒鎖與空任務字典。"""
        self._ttl_seconds = max(1, int(ttl_seconds))
        self._lock = threading.Lock()
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def _cleanup_locked(self) -> None:
        """掃除過期任務：刪除超過 TTL 未更新的任務記錄（需在持有鎖時呼叫）。"""
        now = time.time()
        expired = []
        for job_id, job in self._jobs.items():
            updated_at = float(job.get("updated_at", now))
            if now - updated_at > self._ttl_seconds:
                expired.append(job_id)
        for job_id in expired:
            self._jobs.pop(job_id, None)

    def cleanup(self) -> None:
        """以執行緒安全方式執行過期任務清理。"""
        with self._lock:
            self._cleanup_locked()

    def create(self, payload: Dict[str, Any]) -> str:
        """建立新任務記錄，設定 ID 與時間戳，先清理過期項目後回傳任務 ID。"""
        job_id = uuid.uuid4().hex
        now = time.time()
        with self._lock:
            self._cleanup_locked()
            row = deepcopy(payload)
            row.setdefault("job_id", job_id)
            row.setdefault("created_at", now)
            row.setdefault("updated_at", now)
            self._jobs[job_id] = row
        return job_id

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        """以執行緒安全方式取得指定 ID 的任務，過期或不存在時回傳 None。"""
        with self._lock:
            self._cleanup_locked()
            job = self._jobs.get(job_id)
            if not job:
                return None
            return deepcopy(job)

    def update(self, job_id: str, mutator: Callable[[Dict[str, Any]], None]) -> bool:
        """以執行緒安全方式執行 mutator 回呼更新任務狀態，並更新時間戳。"""
        with self._lock:
            self._cleanup_locked()
            job = self._jobs.get(job_id)
            if not job:
                return False
            mutator(job)
            job["updated_at"] = time.time()
            return True
