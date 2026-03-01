"""Shared LLM client with provider switch (OpenAI-compatible / Ollama / Gemini)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, NamedTuple, Optional

import requests
from backend.config import settings as app_settings

DEFAULT_MODEL = app_settings.DEFAULT_MODEL
DEFAULT_PROVIDER = app_settings.DEFAULT_PROVIDER
DEFAULT_OPENAI_BASE_URL = app_settings.DEFAULT_OPENAI_BASE_URL
DEFAULT_OLLAMA_BASE_URL = app_settings.DEFAULT_OLLAMA_BASE_URL
DEFAULT_GEMINI_BASE_URL = app_settings.DEFAULT_GEMINI_BASE_URL
DEFAULT_GEMINI_MODEL = app_settings.DEFAULT_GEMINI_MODEL
DEFAULT_GEMINI_INPUT_TOKEN_LIMIT = app_settings.DEFAULT_GEMINI_INPUT_TOKEN_LIMIT
DEFAULT_GEMINI_OUTPUT_TOKEN_LIMIT = app_settings.DEFAULT_GEMINI_OUTPUT_TOKEN_LIMIT
DEFAULT_ERROR_DETAIL_MAX_CHARS = app_settings.DEFAULT_ERROR_DETAIL_MAX_CHARS
THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
THINK_CAPTURE_PATTERN = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL | re.IGNORECASE)


class LLMError(requests.RequestException):
    """Base exception for LLM provider errors."""


class LLMTimeoutError(requests.Timeout, LLMError):
    """Timeout while calling upstream LLM provider."""


class LLMHTTPError(LLMError):
    """HTTP error returned by upstream LLM provider."""


class LLMResponseError(LLMError):
    """Unexpected response payload from upstream LLM provider."""


class LLMParseError(LLMError):
    """Unable to parse JSON content returned by LLM."""


@dataclass(frozen=True)
class LLMRuntimeConfig:
    provider: str
    model: str
    timeout_seconds: float
    temperature: float
    max_tokens: int
    openai_base_url: str
    openai_api_key: str
    ollama_base_url: str
    gemini_base_url: str
    gemini_api_key: str
    gemini_input_token_limit: int
    gemini_output_token_limit: int
    ollama_think: bool
    ollama_think_json: bool



def _error_detail_max_chars() -> int:
    """從應用設定中取得錯誤訊息的最大字元限制。"""
    return app_settings.get_llm_misc_settings().error_detail_max_chars


def _trim_error_detail(text: str) -> str:
    """將錯誤訊息截斷至設定的字元限制，超出時附加截斷提示（... [truncated N chars]）。"""
    cleaned = str(text or "").strip()
    limit = max(0, _error_detail_max_chars())
    if limit == 0 or len(cleaned) <= limit:
        return cleaned
    omitted = len(cleaned) - limit
    return f"{cleaned[:limit]}... [truncated {omitted} chars]"



def get_runtime_config() -> LLMRuntimeConfig:
    """從環境變數讀取並構建 LLM 運行時組態物件（LLMRuntimeConfig）。"""
    cfg = app_settings.get_llm_runtime_settings()

    return LLMRuntimeConfig(
        provider=cfg.provider,
        model=cfg.model,
        timeout_seconds=cfg.timeout_seconds,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        openai_base_url=cfg.openai_base_url,
        openai_api_key=cfg.openai_api_key,
        ollama_base_url=cfg.ollama_base_url,
        gemini_base_url=cfg.gemini_base_url,
        gemini_api_key=cfg.gemini_api_key,
        gemini_input_token_limit=cfg.gemini_input_token_limit,
        gemini_output_token_limit=cfg.gemini_output_token_limit,
        ollama_think=cfg.ollama_think,
        ollama_think_json=cfg.ollama_think_json,
    )


def _resolve_effective_max_tokens(
    *,
    cfg: LLMRuntimeConfig,
    provider: str,
    requested_max_tokens: Optional[int],
) -> int:
    """根據 LLM 提供者解析有效的最大輸出 token 數，Gemini 套用專屬上限。"""
    if requested_max_tokens is None:
        if provider == "gemini":
            return cfg.gemini_output_token_limit
        return cfg.max_tokens

    requested = max(1, int(requested_max_tokens))
    if provider == "gemini":
        return min(requested, cfg.gemini_output_token_limit)
    return requested


def _strip_code_fence(raw: str) -> str:
    """移除 Markdown 代碼欄標記（```json ... ```），取出其中的純文字內容。"""
    text = raw.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else text).strip()


def _first_json_start(text: str) -> int:
    """找出文字中第一個 JSON 開始符號（'{' 或 '['）的索引位置，找不到則回傳 -1。"""
    starts = [idx for idx in (text.find("{"), text.find("[")) if idx >= 0]
    return min(starts) if starts else -1


def _extract_balanced_json_from(text: str, start_idx: int) -> Optional[str]:
    """從指定位置開始以堆疊追蹤平衡括號，提取完整的 JSON 物件或陣列，括號不平衡時回傳 None。"""
    if start_idx < 0 or start_idx >= len(text):
        return None
    if text[start_idx] not in "{[":
        return None

    stack: List[str] = [text[start_idx]]
    in_string = False
    escaped = False

    for idx in range(start_idx + 1, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == "\"":
                in_string = False
            continue

        if ch == "\"":
            in_string = True
            continue
        if ch in "{[":
            stack.append(ch)
            continue
        if ch in "}]":
            if not stack:
                return None
            opener = stack[-1]
            if (opener == "{" and ch == "}") or (opener == "[" and ch == "]"):
                stack.pop()
                if not stack:
                    return text[start_idx : idx + 1]
            else:
                return None
    return None


def _json_parse_candidates(raw: str) -> List[str]:
    """生成多組候選 JSON 字串（含去除代碼欄、移除 think 標籤、提取平衡 JSON 塊等版本），去重後回傳。"""
    text = _strip_code_fence(raw)
    candidates: List[str] = [text]

    without_think = THINK_TAG_PATTERN.sub(" ", text).strip()
    if without_think and without_think != text:
        candidates.append(without_think)

    # Some reasoning models may emit an open "<think>" prelude without closing tag.
    # Keep the most likely JSON segment from the first JSON token onward.
    if "<think" in text.lower():
        first_idx = _first_json_start(text)
        if first_idx >= 0:
            candidates.append(text[first_idx:].strip())

    # Also try extracting balanced JSON from candidate texts.
    expanded: List[str] = []
    for candidate in candidates:
        expanded.append(candidate)
        for idx, ch in enumerate(candidate):
            if ch not in "{[":
                continue
            block = _extract_balanced_json_from(candidate, idx)
            if block:
                expanded.append(block)

    deduped: List[str] = []
    seen = set()
    for candidate in expanded:
        norm = candidate.strip()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        deduped.append(norm)
    return deduped


def _parse_json_strict(raw: str) -> Dict[str, Any]:
    """逐一嘗試候選字串解析 JSON，全部失敗時以 LLMParseError 拋出最後一個錯誤。"""
    last_error: json.JSONDecodeError | None = None
    for candidate in _json_parse_candidates(raw):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue

    raise LLMParseError(f"Invalid JSON from LLM: {last_error}")


def _request_json(
    method: str,
    url: str,
    *,
    timeout: float,
    headers: Optional[Dict[str, str]] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """發送 GET 或 POST 請求並解析 JSON 回應，統一將網路與 HTTP 錯誤映射為 LLMError 子類。"""
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers, timeout=timeout)
        else:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    except requests.Timeout as exc:
        raise LLMTimeoutError(f"LLM request timed out: {exc}") from exc
    except requests.RequestException as exc:
        raise LLMHTTPError(f"LLM request failed: {exc}") from exc

    if response.status_code >= 400:
        text = response.text.strip()
        raise LLMHTTPError(
            f"LLM provider returned {response.status_code}: {_trim_error_detail(text)}"
        )

    try:
        return response.json()
    except ValueError as exc:
        raise LLMResponseError("LLM provider returned non-JSON payload") from exc


def _extract_openai_content(body: Dict[str, Any]) -> str:
    """從 OpenAI-compatible 回應中提取文字內容，支援 vision 多媒體格式（content 為陣列時拼接）。"""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMResponseError("OpenAI-compatible response missing choices")

    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        content = "".join(parts)

    content_text = str(content).strip()
    if not content_text:
        raise LLMResponseError("OpenAI-compatible response has empty content")
    return content_text


def _extract_ollama_content(body: Dict[str, Any]) -> str:
    """從 Ollama 回應中提取文字內容，並在推理模型因 token 預算耗盡時提供明確的錯誤指引。"""
    message = body.get("message", {}) if isinstance(body.get("message"), dict) else {}
    content = str(message.get("content", "")).strip()
    if not content:
        thinking = str(message.get("thinking", "")).strip()
        done_reason = str(body.get("done_reason", "")).strip().lower()
        if thinking and done_reason == "length":
            raise LLMResponseError(
                "Ollama response has empty content (reasoning consumed token budget). "
                "Increase LLM_MAX_TOKENS/EXTRACTION_NUM_PREDICT/NL2CYPHER_NUM_PREDICT "
                "or set OLLAMA_THINK=false."
            )
        if thinking:
            raise LLMResponseError("Ollama response has empty content while thinking is present")
        raise LLMResponseError("Ollama response has empty content")
    return content


def _is_ollama_thinking_only(body: Dict[str, Any]) -> bool:
    """檢測 Ollama 推理模型是否因 token 預算耗盡而只輸出了推理過程、未產生正式回應。"""
    message = body.get("message", {}) if isinstance(body.get("message"), dict) else {}
    content = str(message.get("content", "")).strip()
    thinking = str(message.get("thinking", "")).strip()
    done_reason = str(body.get("done_reason", "")).strip().lower()
    return (not content) and bool(thinking) and done_reason == "length"


def _next_retry_tokens(current_tokens: int) -> int:
    """計算推理模型重試時的 token 預算（當前值加倍），限制在 1024–8192 之間。

    深度推理模型在輸出正式回應前會消耗大量 token 於思考過程，加倍預算以確保有足夠空間輸出內容。
    """
    return min(max(current_tokens * 2, 1024), 8192)


def _think_log_enabled() -> bool:
    """從應用設定中讀取是否啟用推理過程日誌記錄。"""
    return app_settings.get_llm_misc_settings().think_log_enabled


def _think_log_path() -> str:
    """從應用設定中取得推理過程日誌的檔案路徑。"""
    return app_settings.get_llm_misc_settings().think_log_path


def _extract_think_from_content(content: str) -> str:
    """從模型輸出文字中提取推理思考過程，支援完整的 <think>...</think> 標籤及未結束的開放標籤。"""
    text = str(content or "")
    if not text:
        return ""

    captured = [chunk.strip() for chunk in THINK_CAPTURE_PATTERN.findall(text) if chunk.strip()]
    if captured:
        return "\n\n".join(captured)

    # Handle malformed output where "<think>" exists but is never closed.
    lower = text.lower()
    marker = lower.find("<think>")
    if marker < 0:
        return ""

    tail = text[marker + len("<think>") :].strip()
    if not tail:
        return ""
    json_start = _first_json_start(tail)
    if json_start > 0:
        tail = tail[:json_start].strip()
    return tail


def _extract_ollama_thinking(body: Dict[str, Any]) -> str:
    """提取 Ollama 回應中的推理思考內容，優先使用專用 thinking 欄位，否則從 content 中解析。"""
    message = body.get("message", {}) if isinstance(body.get("message"), dict) else {}
    thinking = str(message.get("thinking", "")).strip()
    if thinking:
        return thinking
    return _extract_think_from_content(str(message.get("content", "")))


def _append_ollama_think_log(*, body: Dict[str, Any], model: str, mode: str) -> None:
    """若推理日誌啟用，將本次推理過程以 JSON 行格式附加至日誌檔案，IO 失敗時靜默略過。"""
    if not _think_log_enabled():
        return

    thinking = _extract_ollama_thinking(body)
    if not thinking:
        return

    log_path = _think_log_path()
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "provider": "ollama",
        "model": model,
        "mode": mode,
        "done_reason": str(body.get("done_reason", "")).strip(),
        "thinking": thinking,
    }
    line = json.dumps(event, ensure_ascii=False)

    try:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        # Logging must not affect online request flow.
        return


def _resolve_provider_override(provider: Optional[str], default_provider: str) -> str:
    """正規化並驗證 LLM 提供者名稱，僅接受 openai/ollama/gemini，否則拋出 LLMResponseError。"""
    use_provider = (provider or default_provider).strip().lower()
    if use_provider not in {"openai", "ollama", "gemini"}:
        raise LLMResponseError(f"Unsupported provider: {use_provider}")
    return use_provider


def _resolve_model_for_provider(
    *,
    provider: str,
    model: Optional[str],
    default_cfg_model: str,
) -> str:
    """解析有效的模型名稱：優先使用傳入值，否則依提供者從設定中查詢預設模型。"""
    if model:
        return model
    if provider in {"ollama", "gemini", "openai"}:
        return app_settings.resolve_llm_model(provider)
    return default_cfg_model


def _openai_headers(api_key: str) -> Dict[str, str]:
    """構建 OpenAI-compatible 請求 header，包含 Content-Type 和可選的 Bearer 認證（本地部署可不帶）。"""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _gemini_headers(api_key: str) -> Dict[str, str]:
    """構建 Gemini API 請求 header，包含 Content-Type 和必需的 x-goog-api-key 認證。"""
    if not api_key:
        raise LLMResponseError("GEMINI_API_KEY is required when provider=gemini")
    headers = {"Content-Type": "application/json"}
    headers["x-goog-api-key"] = api_key
    return headers


def _messages_to_gemini_payload(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """將 OpenAI 訊息格式（system/user/assistant）轉換為 Gemini API 格式，分離系統指示與對話內容。"""
    system_chunks: List[str] = []
    contents: List[Dict[str, Any]] = []

    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if not content:
            continue

        if role == "system":
            system_chunks.append(content)
            continue

        gemini_role = "model" if role == "assistant" else "user"
        contents.append({"role": gemini_role, "parts": [{"text": content}]})

    if not contents:
        raise LLMResponseError("Gemini request has no user/model content")

    payload: Dict[str, Any] = {"contents": contents}
    if system_chunks:
        payload["system_instruction"] = {"parts": [{"text": "\n\n".join(system_chunks)}]}
    return payload


def _extract_gemini_content(body: Dict[str, Any]) -> str:
    """從 Gemini API 回應中逐層提取文字內容（candidates → content → parts），支援多 part 拼接。"""
    candidates = body.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise LLMResponseError("Gemini response missing candidates")

    content = candidates[0].get("content", {})
    parts = content.get("parts", []) if isinstance(content, dict) else []
    texts: List[str] = []
    for part in parts:
        if isinstance(part, dict):
            text = str(part.get("text", ""))
            if text:
                texts.append(text)

    content_text = "".join(texts).strip()
    if not content_text:
        raise LLMResponseError("Gemini response has empty content")
    return content_text


class _CallParams(NamedTuple):
    cfg: Any
    provider: str
    model: str
    temp: float
    max_tokens: int
    timeout: float


def _resolve_call_params(
    provider: Optional[str],
    model: Optional[str],
    temperature: Optional[float],
    max_tokens: Optional[int],
    timeout_seconds: Optional[float],
) -> _CallParams:
    """Resolve the five shared call parameters used by both chat_text and chat_json."""
    cfg = get_runtime_config()
    use_provider = _resolve_provider_override(provider, cfg.provider)
    use_model = _resolve_model_for_provider(provider=use_provider, model=model, default_cfg_model=cfg.model)
    use_temp = cfg.temperature if temperature is None else temperature
    use_max_tokens = _resolve_effective_max_tokens(cfg=cfg, provider=use_provider, requested_max_tokens=max_tokens)
    use_timeout = cfg.timeout_seconds if timeout_seconds is None else max(1.0, float(timeout_seconds))
    return _CallParams(cfg, use_provider, use_model, use_temp, use_max_tokens, use_timeout)


def chat_text(
    *,
    messages: List[Dict[str, str]],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout_seconds: Optional[float] = None,
) -> str:
    """調用 LLM 取得文字回應，支援 OpenAI/Ollama/Gemini 三種提供者；Ollama 推理模型在 token 耗盡時自動加倍預算重試。"""
    p = _resolve_call_params(provider, model, temperature, max_tokens, timeout_seconds)
    cfg, use_provider, use_model, use_temp, use_max_tokens, use_timeout = p

    if use_provider == "openai":
        body = _request_json(
            "POST",
            f"{cfg.openai_base_url}/chat/completions",
            headers=_openai_headers(cfg.openai_api_key),
            payload={
                "model": use_model,
                "messages": messages,
                "temperature": use_temp,
                "max_tokens": use_max_tokens,
            },
            timeout=use_timeout,
        )
        return _extract_openai_content(body)

    if use_provider == "ollama":
        payload = {
            "model": use_model,
            "messages": messages,
            "stream": False,
            "think": cfg.ollama_think,
            "options": {
                "temperature": use_temp,
                "num_predict": use_max_tokens,
            },
        }
        body = _request_json(
            "POST",
            f"{cfg.ollama_base_url}/api/chat",
            payload=payload,
            timeout=use_timeout,
        )
        _append_ollama_think_log(body=body, model=use_model, mode="chat_text")
        if cfg.ollama_think and _is_ollama_thinking_only(body):
            retry_payload = dict(payload)
            retry_options = dict(payload["options"])
            retry_options["num_predict"] = _next_retry_tokens(use_max_tokens)
            retry_payload["options"] = retry_options
            body = _request_json(
                "POST",
                f"{cfg.ollama_base_url}/api/chat",
                payload=retry_payload,
                timeout=use_timeout,
            )
            _append_ollama_think_log(body=body, model=use_model, mode="chat_text")
        return _extract_ollama_content(body)

    if use_provider == "gemini":
        payload = _messages_to_gemini_payload(messages)
        payload["generationConfig"] = {
            "temperature": use_temp,
            "maxOutputTokens": use_max_tokens,
        }
        body = _request_json(
            "POST",
            f"{cfg.gemini_base_url}/models/{use_model}:generateContent",
            headers=_gemini_headers(cfg.gemini_api_key),
            payload=payload,
            timeout=use_timeout,
        )
        return _extract_gemini_content(body)

    raise LLMResponseError(f"Unsupported provider: {use_provider}")


def chat_json(
    *,
    messages: List[Dict[str, str]],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """調用 LLM 取得 JSON 結構化回應，自適應各提供者的 JSON 模式（OpenAI 不支援時自動回退，Gemini 使用 MIME type）。"""
    p = _resolve_call_params(provider, model, temperature, max_tokens, timeout_seconds)
    cfg, use_provider, use_model, use_temp, use_max_tokens, use_timeout = p

    if use_provider == "ollama":
        payload = {
            "model": use_model,
            "messages": messages,
            "stream": False,
            "think": cfg.ollama_think_json,
            "format": "json",
            "options": {
                "temperature": use_temp,
                "num_predict": use_max_tokens,
            },
        }
        body = _request_json(
            "POST",
            f"{cfg.ollama_base_url}/api/chat",
            payload=payload,
            timeout=use_timeout,
        )
        _append_ollama_think_log(body=body, model=use_model, mode="chat_json")
        if cfg.ollama_think_json and _is_ollama_thinking_only(body):
            retry_payload = dict(payload)
            retry_options = dict(payload["options"])
            retry_options["num_predict"] = _next_retry_tokens(use_max_tokens)
            retry_payload["options"] = retry_options
            body = _request_json(
                "POST",
                f"{cfg.ollama_base_url}/api/chat",
                payload=retry_payload,
                timeout=use_timeout,
            )
            _append_ollama_think_log(body=body, model=use_model, mode="chat_json")
        return _parse_json_strict(_extract_ollama_content(body))

    if use_provider == "openai":
        payload = {
            "model": use_model,
            "messages": messages,
            "temperature": use_temp,
            "max_tokens": use_max_tokens,
            "response_format": {"type": "json_object"},
        }
        try:
            body = _request_json(
                "POST",
                f"{cfg.openai_base_url}/chat/completions",
                headers=_openai_headers(cfg.openai_api_key),
                payload=payload,
                timeout=use_timeout,
            )
        except LLMHTTPError:
            # Some OpenAI-compatible backends don't accept response_format.
            payload.pop("response_format", None)
            body = _request_json(
                "POST",
                f"{cfg.openai_base_url}/chat/completions",
                headers=_openai_headers(cfg.openai_api_key),
                payload=payload,
                timeout=use_timeout,
            )
        return _parse_json_strict(_extract_openai_content(body))

    if use_provider == "gemini":
        payload = _messages_to_gemini_payload(messages)
        payload["generationConfig"] = {
            "temperature": use_temp,
            "maxOutputTokens": use_max_tokens,
            "responseMimeType": "application/json",
        }
        body = _request_json(
            "POST",
            f"{cfg.gemini_base_url}/models/{use_model}:generateContent",
            headers=_gemini_headers(cfg.gemini_api_key),
            payload=payload,
            timeout=use_timeout,
        )
        return _parse_json_strict(_extract_gemini_content(body))

    raise LLMResponseError(f"Unsupported provider: {use_provider}")


def health_check(timeout_seconds: float = 3.0) -> Dict[str, Any]:
    """檢查 LLM 提供者的可用性：OpenAI 查詢 /models，Gemini 查詢 /models，Ollama 查詢 /api/tags；回傳結構化狀態報告。"""
    cfg = get_runtime_config()
    if cfg.provider == "openai":
        try:
            _request_json(
                "GET",
                f"{cfg.openai_base_url}/models",
                headers=_openai_headers(cfg.openai_api_key),
                timeout=max(1.0, timeout_seconds),
            )
            status = "ok"
            reachable = True
        except LLMError:
            status = "down"
            reachable = False
        return {
            "provider": cfg.provider,
            "model": cfg.model,
            "upstream": "openai-compatible",
            "status": status,
            "reachable": reachable,
        }

    if cfg.provider == "gemini":
        try:
            _request_json(
                "GET",
                f"{cfg.gemini_base_url}/models",
                headers=_gemini_headers(cfg.gemini_api_key),
                timeout=max(1.0, timeout_seconds),
            )
            status = "ok"
            reachable = True
        except LLMError:
            status = "down"
            reachable = False
        return {
            "provider": cfg.provider,
            "model": cfg.model,
            "upstream": "gemini",
            "status": status,
            "reachable": reachable,
        }

    try:
        _request_json(
            "GET",
            f"{cfg.ollama_base_url}/api/tags",
            timeout=max(1.0, timeout_seconds),
        )
        status = "ok"
        reachable = True
    except LLMError:
        status = "down"
        reachable = False

    return {
        "provider": cfg.provider,
        "model": cfg.model,
        "upstream": "ollama",
        "status": status,
        "reachable": reachable,
    }
