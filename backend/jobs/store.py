"""Thread-safe in-memory job store with TTL cleanup."""

from __future__ import annotations

import threading
import time
import uuid
from copy import deepcopy
from typing import Any, Callable, Dict, Optional


class JobStore:
    def __init__(self, ttl_seconds: int) -> None:
        """初始化物件狀態並保存後續流程所需依賴。
        此方法會依目前參數設定實例欄位，供其他方法在生命週期內重複使用。
        """
        self._ttl_seconds = max(1, int(ttl_seconds))
        self._lock = threading.Lock()
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def _cleanup_locked(self) -> None:
        """執行 `_cleanup_locked` 的內部輔助流程。
        此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
        """
        now = time.time()
        expired = []
        for job_id, job in self._jobs.items():
            updated_at = float(job.get("updated_at", now))
            if now - updated_at > self._ttl_seconds:
                expired.append(job_id)
        for job_id in expired:
            self._jobs.pop(job_id, None)

    def cleanup(self) -> None:
        """執行 `cleanup` 的主要流程。
        函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
        """
        with self._lock:
            self._cleanup_locked()

    def create(self, payload: Dict[str, Any]) -> str:
        """執行 `create` 的主要流程。
        函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
        """
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
        """執行 `get` 的主要流程。
        函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
        """
        with self._lock:
            self._cleanup_locked()
            job = self._jobs.get(job_id)
            if not job:
                return None
            return deepcopy(job)

    def update(self, job_id: str, mutator: Callable[[Dict[str, Any]], None]) -> bool:
        """執行 `update` 的主要流程。
        函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
        """
        with self._lock:
            self._cleanup_locked()
            job = self._jobs.get(job_id)
            if not job:
                return False
            mutator(job)
            job["updated_at"] = time.time()
            return True
