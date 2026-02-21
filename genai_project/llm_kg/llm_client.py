"""Shared LLM client with provider switch (OpenAI-compatible / Ollama / Gemini)."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

DEFAULT_MODEL = "mlx-community/Qwen3-8B-4bit-DWQ-053125"
DEFAULT_PROVIDER = "openai"
DEFAULT_OPENAI_BASE_URL = "http://localhost:8080/v1"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_GEMINI_MODEL = "gemini-3-pro-preview"
DEFAULT_GEMINI_INPUT_TOKEN_LIMIT = 1_048_576
DEFAULT_GEMINI_OUTPUT_TOKEN_LIMIT = 65_536
DEFAULT_ERROR_DETAIL_MAX_CHARS = 4000
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


def _safe_float(value: Optional[str], default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Optional[str], default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _error_detail_max_chars() -> int:
    return _safe_int(os.getenv("LLM_ERROR_DETAIL_MAX_CHARS"), DEFAULT_ERROR_DETAIL_MAX_CHARS)


def _trim_error_detail(text: str) -> str:
    cleaned = str(text or "").strip()
    limit = max(0, _error_detail_max_chars())
    if limit == 0 or len(cleaned) <= limit:
        return cleaned
    omitted = len(cleaned) - limit
    return f"{cleaned[:limit]}... [truncated {omitted} chars]"


def _resolve_provider() -> str:
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit in {"openai", "ollama", "gemini"}:
        return explicit
    if explicit:
        raise ValueError(f"Unsupported LLM_PROVIDER: {explicit}")

    # Backward compatibility: if legacy Ollama env exists, use Ollama.
    if os.getenv("OLLAMA_BASE_URL"):
        return "ollama"
    return DEFAULT_PROVIDER


def get_runtime_config() -> LLMRuntimeConfig:
    provider = _resolve_provider()
    if provider == "ollama":
        model = (
            os.getenv("OLLAMA_MODEL")
            or os.getenv("LLM_MODEL")
            or DEFAULT_MODEL
        )
    elif provider == "gemini":
        model = (
            os.getenv("GEMINI_MODEL")
            or os.getenv("LLM_MODEL")
            or DEFAULT_GEMINI_MODEL
        )
    else:
        model = (
            os.getenv("LLM_MODEL")
            or os.getenv("OLLAMA_MODEL")
            or DEFAULT_MODEL
        )
    timeout_seconds = _safe_float(
        os.getenv("LLM_TIMEOUT_SECONDS") or os.getenv("OLLAMA_TIMEOUT_SECONDS"),
        180.0,
    )
    temperature = _safe_float(
        os.getenv("LLM_TEMPERATURE") or os.getenv("OLLAMA_TEMPERATURE"),
        0.1,
    )
    gemini_input_token_limit = _safe_int(
        os.getenv("GEMINI_INPUT_TOKEN_LIMIT"),
        DEFAULT_GEMINI_INPUT_TOKEN_LIMIT,
    )
    gemini_output_token_limit = _safe_int(
        os.getenv("GEMINI_OUTPUT_TOKEN_LIMIT"),
        DEFAULT_GEMINI_OUTPUT_TOKEN_LIMIT,
    )
    max_tokens_env = os.getenv("LLM_MAX_TOKENS") or os.getenv("OLLAMA_NUM_PREDICT")
    if max_tokens_env is None and provider == "gemini":
        max_tokens = gemini_output_token_limit
    else:
        max_tokens = _safe_int(max_tokens_env, 512)
    openai_base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL).rstrip("/")
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).rstrip("/")
    gemini_base_url = os.getenv("GEMINI_BASE_URL", DEFAULT_GEMINI_BASE_URL).rstrip("/")
    gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
    ollama_think = _safe_bool(os.getenv("OLLAMA_THINK"), False)
    ollama_think_json = _safe_bool(os.getenv("OLLAMA_THINK_JSON"), ollama_think)

    return LLMRuntimeConfig(
        provider=provider,
        model=model,
        timeout_seconds=max(1.0, timeout_seconds),
        temperature=temperature,
        max_tokens=max(1, max_tokens),
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
        ollama_base_url=ollama_base_url,
        gemini_base_url=gemini_base_url,
        gemini_api_key=gemini_api_key,
        gemini_input_token_limit=max(1, gemini_input_token_limit),
        gemini_output_token_limit=max(1, gemini_output_token_limit),
        ollama_think=ollama_think,
        ollama_think_json=ollama_think_json,
    )


def _resolve_effective_max_tokens(
    *,
    cfg: LLMRuntimeConfig,
    provider: str,
    requested_max_tokens: Optional[int],
) -> int:
    if requested_max_tokens is None:
        if provider == "gemini":
            return cfg.gemini_output_token_limit
        return cfg.max_tokens

    requested = max(1, int(requested_max_tokens))
    if provider == "gemini":
        return min(requested, cfg.gemini_output_token_limit)
    return requested


def _strip_code_fence(raw: str) -> str:
    text = raw.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else text).strip()


def _first_json_start(text: str) -> int:
    starts = [idx for idx in (text.find("{"), text.find("[")) if idx >= 0]
    return min(starts) if starts else -1


def _extract_balanced_json_from(text: str, start_idx: int) -> Optional[str]:
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
    message = body.get("message", {}) if isinstance(body.get("message"), dict) else {}
    content = str(message.get("content", "")).strip()
    thinking = str(message.get("thinking", "")).strip()
    done_reason = str(body.get("done_reason", "")).strip().lower()
    return (not content) and bool(thinking) and done_reason == "length"


def _next_retry_tokens(current_tokens: int) -> int:
    # Deep reasoning models may consume many tokens in "thinking" before content output.
    return min(max(current_tokens * 2, 1024), 8192)


def _think_log_enabled() -> bool:
    value = os.getenv("LLM_THINK_LOG_ENABLED") or os.getenv("OLLAMA_THINK_LOG_ENABLED")
    return _safe_bool(value, False)


def _think_log_path() -> str:
    return (
        os.getenv("LLM_THINK_LOG_PATH")
        or os.getenv("OLLAMA_THINK_LOG_PATH")
        or "/tmp/ollama_think.log"
    )


def _extract_think_from_content(content: str) -> str:
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
    message = body.get("message", {}) if isinstance(body.get("message"), dict) else {}
    thinking = str(message.get("thinking", "")).strip()
    if thinking:
        return thinking
    return _extract_think_from_content(str(message.get("content", "")))


def _append_ollama_think_log(*, body: Dict[str, Any], model: str, mode: str) -> None:
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
    if model:
        return model
    if provider == "ollama":
        return os.getenv("OLLAMA_MODEL") or os.getenv("LLM_MODEL") or DEFAULT_MODEL
    if provider == "gemini":
        return os.getenv("GEMINI_MODEL") or os.getenv("LLM_MODEL") or DEFAULT_GEMINI_MODEL
    if provider == "openai":
        return os.getenv("LLM_MODEL") or os.getenv("OLLAMA_MODEL") or DEFAULT_MODEL
    return default_cfg_model


def _openai_headers(api_key: str) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _gemini_headers(api_key: str) -> Dict[str, str]:
    if not api_key:
        raise LLMResponseError("GEMINI_API_KEY is required when provider=gemini")
    headers = {"Content-Type": "application/json"}
    headers["x-goog-api-key"] = api_key
    return headers


def _messages_to_gemini_payload(messages: List[Dict[str, str]]) -> Dict[str, Any]:
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


def chat_text(
    *,
    messages: List[Dict[str, str]],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout_seconds: Optional[float] = None,
) -> str:
    cfg = get_runtime_config()
    use_provider = _resolve_provider_override(provider, cfg.provider)
    use_model = _resolve_model_for_provider(provider=use_provider, model=model, default_cfg_model=cfg.model)
    use_temp = cfg.temperature if temperature is None else temperature
    use_max_tokens = _resolve_effective_max_tokens(
        cfg=cfg,
        provider=use_provider,
        requested_max_tokens=max_tokens,
    )
    use_timeout = cfg.timeout_seconds if timeout_seconds is None else max(1.0, float(timeout_seconds))

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
    cfg = get_runtime_config()
    use_provider = _resolve_provider_override(provider, cfg.provider)
    use_model = _resolve_model_for_provider(provider=use_provider, model=model, default_cfg_model=cfg.model)
    use_temp = cfg.temperature if temperature is None else temperature
    use_max_tokens = _resolve_effective_max_tokens(
        cfg=cfg,
        provider=use_provider,
        requested_max_tokens=max_tokens,
    )
    use_timeout = cfg.timeout_seconds if timeout_seconds is None else max(1.0, float(timeout_seconds))

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
