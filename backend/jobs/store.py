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
        """`_cleanup_locked` 的內部輔助函式。

主要用途：
- 封裝局部步驟，讓主流程維持可讀性。
- 集中處理細節與邊界條件，避免重複邏輯分散。

回傳約定：
- 保持既有輸入/輸出契約，不改變對外行為。
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
        """`cleanup` 的主要流程入口。

主要用途：
- 串接此函式負責的核心步驟並回傳既有格式。
- 例外沿用現行錯誤處理策略，避免破壞呼叫端契約。

維護重點：
- 調整流程時需保持 API 欄位、狀態轉移與錯誤語意一致。
        """
        with self._lock:
            self._cleanup_locked()

    def create(self, payload: Dict[str, Any]) -> str:
        """`create` 的主要流程入口。

主要用途：
- 串接此函式負責的核心步驟並回傳既有格式。
- 例外沿用現行錯誤處理策略，避免破壞呼叫端契約。

維護重點：
- 調整流程時需保持 API 欄位、狀態轉移與錯誤語意一致。
        """
        # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
        # ─── 階段 2：核心處理流程 ─────────────────────────────────
        # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
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
        """`get` 的主要流程入口。

主要用途：
- 串接此函式負責的核心步驟並回傳既有格式。
- 例外沿用現行錯誤處理策略，避免破壞呼叫端契約。

維護重點：
- 調整流程時需保持 API 欄位、狀態轉移與錯誤語意一致。
        """
        with self._lock:
            self._cleanup_locked()
            job = self._jobs.get(job_id)
            if not job:
                return None
            return deepcopy(job)

    def update(self, job_id: str, mutator: Callable[[Dict[str, Any]], None]) -> bool:
        """`update` 的主要流程入口。

主要用途：
- 串接此函式負責的核心步驟並回傳既有格式。
- 例外沿用現行錯誤處理策略，避免破壞呼叫端契約。

維護重點：
- 調整流程時需保持 API 欄位、狀態轉移與錯誤語意一致。
        """
        with self._lock:
            self._cleanup_locked()
            job = self._jobs.get(job_id)
            if not job:
                return False
            mutator(job)
            job["updated_at"] = time.time()
            return True
