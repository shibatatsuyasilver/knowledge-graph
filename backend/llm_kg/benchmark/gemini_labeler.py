"""Gemini-based offline labeler for benchmark gold triples/answers."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from .schema import ANSWER_TYPES

DEFAULT_GEMINI_MODEL = "gemini-3-pro-preview"
DEFAULT_GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta"


class GeminiLabelError(RuntimeError):
    """Raised when Gemini label generation fails."""


@dataclass(frozen=True)
class GeminiLabelerConfig:
    model: str = DEFAULT_GEMINI_MODEL
    timeout_seconds: int = 60
    max_retries: int = 3
    retry_backoff_seconds: float = 1.5
    rate_limit_seconds: float = 0.4


class GeminiLabeler:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        config: Optional[GeminiLabelerConfig] = None,
    ) -> None:
        """初始化物件狀態並保存後續流程所需依賴。
        此方法會依目前參數設定實例欄位，供其他方法在生命週期內重複使用。
        """
        self.api_key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        if not self.api_key:
            raise GeminiLabelError("Missing GEMINI_API_KEY")
        self.config = config or GeminiLabelerConfig()
        self._last_request_ts = 0.0

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """`_strip_code_fence` 的內部輔助函式。

主要用途：
- 封裝局部步驟，讓主流程維持可讀性。
- 集中處理細節與邊界條件，避免重複邏輯分散。

回傳約定：
- 保持既有輸入/輸出契約，不改變對外行為。
        """
        stripped = text.strip()
        match = re.match(r"^```(?:json)?\\s*(.*?)\\s*```$", stripped, flags=re.DOTALL | re.IGNORECASE)
        return (match.group(1) if match else stripped).strip()

    @staticmethod
    def _validate_label_payload(payload: Dict[str, Any], answer_type: str) -> Dict[str, Any]:
        """`_validate_label_payload` 的內部輔助函式。

主要用途：
- 封裝局部步驟，讓主流程維持可讀性。
- 集中處理細節與邊界條件，避免重複邏輯分散。

回傳約定：
- 保持既有輸入/輸出契約，不改變對外行為。
        """
        # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
        # ─── 階段 2：核心處理流程 ─────────────────────────────────
        # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
        if answer_type not in ANSWER_TYPES:
            raise GeminiLabelError(f"Unsupported answer_type: {answer_type}")
        if not isinstance(payload, dict):
            raise GeminiLabelError("Gemini output must be a JSON object")

        triples = payload.get("gold_triples")
        if not isinstance(triples, list) or not triples:
            raise GeminiLabelError("gold_triples must be a non-empty array")

        normalized_triples: List[Dict[str, str]] = []
        for triple in triples:
            if not isinstance(triple, dict):
                continue
            subject = str(triple.get("subject", "")).strip()
            relation = str(triple.get("relation", "")).strip()
            object_ = str(triple.get("object", "")).strip()
            if not (subject and relation and object_):
                continue
            normalized_triples.append(
                {
                    "subject": subject,
                    "relation": relation,
                    "object": object_,
                }
            )
        if not normalized_triples:
            raise GeminiLabelError("gold_triples does not contain valid triples")

        canonical = str(payload.get("canonical", "")).strip()
        if not canonical:
            raise GeminiLabelError("canonical must be non-empty")

        aliases_raw = payload.get("accepted_aliases", [])
        if not isinstance(aliases_raw, list):
            raise GeminiLabelError("accepted_aliases must be an array")
        aliases = [str(alias).strip() for alias in aliases_raw if str(alias).strip()]

        required_raw = payload.get("required_entities", [])
        if not isinstance(required_raw, list):
            raise GeminiLabelError("required_entities must be an array")
        required = [str(entity).strip() for entity in required_raw if str(entity).strip()]

        return {
            "gold_triples": normalized_triples,
            "gold_answer": {
                "answer_type": answer_type,
                "canonical": canonical,
                "accepted_aliases": aliases,
                "required_entities": required,
            },
        }

    def _rate_limit(self) -> None:
        """`_rate_limit` 的內部輔助函式。

主要用途：
- 封裝局部步驟，讓主流程維持可讀性。
- 集中處理細節與邊界條件，避免重複邏輯分散。

回傳約定：
- 保持既有輸入/輸出契約，不改變對外行為。
        """
        now = time.time()
        elapsed = now - self._last_request_ts
        target = self.config.rate_limit_seconds
        if elapsed < target:
            time.sleep(target - elapsed)

    def _request_gemini(self, prompt: str) -> str:
        """`_request_gemini` 的內部輔助函式。

主要用途：
- 封裝局部步驟，讓主流程維持可讀性。
- 集中處理細節與邊界條件，避免重複邏輯分散。

回傳約定：
- 保持既有輸入/輸出契約，不改變對外行為。
        """
        # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
        # ─── 階段 2：核心處理流程 ─────────────────────────────────
        # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
        self._rate_limit()
        url = f"{DEFAULT_GEMINI_ENDPOINT}/models/{self.config.model}:generateContent"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
            },
        }
        response = requests.post(
            url,
            params={"key": self.api_key},
            json=payload,
            timeout=max(5, int(self.config.timeout_seconds)),
        )
        self._last_request_ts = time.time()
        if response.status_code >= 400:
            raise GeminiLabelError(f"Gemini HTTP {response.status_code}: {response.text[:240]}")

        try:
            body = response.json()
        except ValueError as exc:
            raise GeminiLabelError("Gemini returned non-JSON response body") from exc

        candidates = body.get("candidates", [])
        if not isinstance(candidates, list) or not candidates:
            raise GeminiLabelError("Gemini response missing candidates")

        content = candidates[0].get("content", {})
        parts = content.get("parts", []) if isinstance(content, dict) else []
        if not isinstance(parts, list) or not parts:
            raise GeminiLabelError("Gemini response missing content parts")

        text = str(parts[0].get("text", "")).strip() if isinstance(parts[0], dict) else str(parts[0]).strip()
        if not text:
            raise GeminiLabelError("Gemini response text is empty")
        return text

    def _build_prompt(
        self,
        *,
        question_zh_tw: str,
        context_text: str,
        answer_type: str,
    ) -> str:
        """`_build_prompt` 的內部輔助函式。

主要用途：
- 封裝局部步驟，讓主流程維持可讀性。
- 集中處理細節與邊界條件，避免重複邏輯分散。

回傳約定：
- 保持既有輸入/輸出契約，不改變對外行為。
        """
        # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
        # ─── 階段 2：核心處理流程 ─────────────────────────────────
        # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
        return f"""
你是知識圖譜標註器。請根據問題與上下文，輸出 JSON（不可有其他文字）。

問題（繁中）:
{question_zh_tw}

上下文（繁中）:
{context_text}

請輸出以下 JSON 結構：
{{
  "gold_triples": [
    {{"subject":"...","relation":"...","object":"..."}}
  ],
  "canonical": "標準答案字串",
  "accepted_aliases": ["可接受同義答案"],
  "required_entities": ["答案中必須出現的實體"]
}}

規則：
1. relation 使用英文字母與底線（例如 SUPPLIES_TO）。
2. 僅使用上下文能證實的資訊，不可猜測。
3. answer_type 固定為 {answer_type}，請保持 canonical 對應此型別語意。
4. 如果答案是集合，canonical 使用頓號分隔（例如 A、B、C）。
""".strip()

    def label_candidate(
        self,
        *,
        question_zh_tw: str,
        context_text: str,
        answer_type: str,
    ) -> Dict[str, Any]:
        """`label_candidate` 的主要流程入口。

主要用途：
- 串接此函式負責的核心步驟並回傳既有格式。
- 例外沿用現行錯誤處理策略，避免破壞呼叫端契約。

維護重點：
- 調整流程時需保持 API 欄位、狀態轉移與錯誤語意一致。
        """
        # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
        # ─── 階段 2：核心處理流程 ─────────────────────────────────
        # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
        prompt = self._build_prompt(
            question_zh_tw=question_zh_tw,
            context_text=context_text,
            answer_type=answer_type,
        )

        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                raw = self._request_gemini(prompt)
                payload = json.loads(self._strip_code_fence(raw))
                return self._validate_label_payload(payload, answer_type)
            except Exception as exc:  # noqa: BLE001 - return structured retry context
                last_error = exc
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_backoff_seconds * attempt)
                continue

        raise GeminiLabelError(f"Gemini labeling failed after {self.config.max_retries} attempts: {last_error}")
