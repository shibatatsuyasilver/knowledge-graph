from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
import requests

from backend.llm_kg import llm_client


class DummyResponse:
    def __init__(self, status_code: int, payload: Dict[str, Any] | None = None, text: str = ""):
        """初始化物件狀態並保存後續流程所需依賴。
        此方法會依目前參數設定實例欄位，供其他方法在生命週期內重複使用。
        """
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self) -> Dict[str, Any]:
        """執行 `json` 的主要流程。
        函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
        """
        return self._payload


def test_openai_chat_text_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 `test_openai_chat_text_success` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8080/v1")
    monkeypatch.setenv("LLM_MODEL", "mlx-community/Qwen3-8B-4bit-DWQ-053125")

    captured: Dict[str, Any] = {}

    def fake_post(url: str, headers=None, json=None, timeout=0):
        """提供 `fake_post` 測試替身以模擬外部依賴或固定回傳。
        此函式讓測試可注入可預測資料，將驗證焦點集中在流程與邏輯判斷。
        """
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = json
        captured["timeout"] = timeout
        return DummyResponse(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": "ok",
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(llm_client.requests, "post", fake_post)

    text = llm_client.chat_text(messages=[{"role": "user", "content": "hi"}], max_tokens=64, temperature=0.0)

    assert text == "ok"
    assert captured["url"] == "http://localhost:8080/v1/chat/completions"
    assert captured["payload"]["model"] == "mlx-community/Qwen3-8B-4bit-DWQ-053125"
    assert captured["payload"]["max_tokens"] == 64
    assert captured["payload"]["temperature"] == 0.0


def test_gemini_chat_json_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 `test_gemini_chat_json_success` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3-pro-preview")

    captured: Dict[str, Any] = {}

    def fake_post(url: str, headers=None, json=None, timeout=0):
        """提供 `fake_post` 測試替身以模擬外部依賴或固定回傳。
        此函式讓測試可注入可預測資料，將驗證焦點集中在流程與邏輯判斷。
        """
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = json
        return DummyResponse(
            200,
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": '{"ok": true, "provider": "gemini"}'}],
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(llm_client.requests, "post", fake_post)
    parsed = llm_client.chat_json(
        messages=[
            {"role": "system", "content": "Return strict JSON."},
            {"role": "user", "content": "hello"},
        ],
        temperature=0.0,
    )

    assert parsed["ok"] is True
    assert parsed["provider"] == "gemini"
    assert captured["url"].endswith("/models/gemini-3-pro-preview:generateContent")
    assert captured["headers"]["x-goog-api-key"] == "test-key"
    assert captured["payload"]["generationConfig"]["responseMimeType"] == "application/json"


def test_chat_text_provider_override_to_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 `test_chat_text_provider_override_to_gemini` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "deepseek-r1:8b")
    monkeypatch.setenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    captured: Dict[str, Any] = {}

    def fake_post(url: str, headers=None, json=None, timeout=0):
        """提供 `fake_post` 測試替身以模擬外部依賴或固定回傳。
        此函式讓測試可注入可預測資料，將驗證焦點集中在流程與邏輯判斷。
        """
        captured["url"] = url
        captured["payload"] = json
        return DummyResponse(
            200,
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "gemini-ok"}],
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(llm_client.requests, "post", fake_post)
    text = llm_client.chat_text(
        provider="gemini",
        model="gemini-3-pro-preview",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert text == "gemini-ok"
    assert captured["url"].endswith("/models/gemini-3-pro-preview:generateContent")


def test_gemini_chat_text_default_max_output_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 `test_gemini_chat_text_default_max_output_tokens` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("LLM_MAX_TOKENS", raising=False)
    monkeypatch.setenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3-pro-preview")
    monkeypatch.delenv("GEMINI_OUTPUT_TOKEN_LIMIT", raising=False)

    captured: Dict[str, Any] = {}

    def fake_post(url: str, headers=None, json=None, timeout=0):
        """提供 `fake_post` 測試替身以模擬外部依賴或固定回傳。
        此函式讓測試可注入可預測資料，將驗證焦點集中在流程與邏輯判斷。
        """
        captured["payload"] = json
        return DummyResponse(
            200,
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "ok"}],
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(llm_client.requests, "post", fake_post)
    text = llm_client.chat_text(messages=[{"role": "user", "content": "hello"}])

    assert text == "ok"
    assert captured["payload"]["generationConfig"]["maxOutputTokens"] == 65536


def test_ollama_chat_json_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 `test_ollama_chat_json_success` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("LLM_MODEL", "deepseek-r1:8b")

    def fake_post(url: str, headers=None, json=None, timeout=0):
        """提供 `fake_post` 測試替身以模擬外部依賴或固定回傳。
        此函式讓測試可注入可預測資料，將驗證焦點集中在流程與邏輯判斷。
        """
        assert url == "http://localhost:11434/api/chat"
        assert json["format"] == "json"
        assert json["think"] is False
        return DummyResponse(
            200,
            {
                "message": {
                    "content": '{"intent":"list_symptoms","entityType":"Disease","entityName":"Diabetes"}',
                }
            },
        )

    monkeypatch.setattr(llm_client.requests, "post", fake_post)
    parsed = llm_client.chat_json(messages=[{"role": "user", "content": "x"}], temperature=0.0)

    assert parsed["intent"] == "list_symptoms"
    assert parsed["entityType"] == "Disease"


def test_ollama_prefers_ollama_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 `test_ollama_prefers_ollama_model` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2:latest")
    monkeypatch.setenv("LLM_MODEL", "mlx-community/Qwen3-8B-4bit-DWQ-053125")

    captured: Dict[str, Any] = {}

    def fake_post(url: str, headers=None, json=None, timeout=0):
        """提供 `fake_post` 測試替身以模擬外部依賴或固定回傳。
        此函式讓測試可注入可預測資料，將驗證焦點集中在流程與邏輯判斷。
        """
        captured["url"] = url
        captured["payload"] = json
        return DummyResponse(
            200,
            {
                "message": {
                    "content": "ok",
                }
            },
        )

    monkeypatch.setattr(llm_client.requests, "post", fake_post)
    text = llm_client.chat_text(messages=[{"role": "user", "content": "hello"}])

    assert text == "ok"
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["payload"]["model"] == "llama3.2:latest"


def test_ollama_think_can_be_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 `test_ollama_think_can_be_enabled` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "deepseek-r1:8b")
    monkeypatch.setenv("OLLAMA_THINK", "true")

    captured: Dict[str, Any] = {}

    def fake_post(url: str, headers=None, json=None, timeout=0):
        """提供 `fake_post` 測試替身以模擬外部依賴或固定回傳。
        此函式讓測試可注入可預測資料，將驗證焦點集中在流程與邏輯判斷。
        """
        captured["payload"] = json
        return DummyResponse(
            200,
            {
                "message": {
                    "content": "ok",
                }
            },
        )

    monkeypatch.setattr(llm_client.requests, "post", fake_post)
    llm_client.chat_text(messages=[{"role": "user", "content": "hello"}])

    assert captured["payload"]["think"] is True


def test_ollama_think_json_can_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 `test_ollama_think_json_can_override` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "deepseek-r1:8b")
    monkeypatch.setenv("OLLAMA_THINK", "true")
    monkeypatch.setenv("OLLAMA_THINK_JSON", "false")

    captured: Dict[str, Any] = {}

    def fake_post(url: str, headers=None, json=None, timeout=0):
        """提供 `fake_post` 測試替身以模擬外部依賴或固定回傳。
        此函式讓測試可注入可預測資料，將驗證焦點集中在流程與邏輯判斷。
        """
        captured["payload"] = json
        return DummyResponse(
            200,
            {
                "message": {
                    "content": '{"ok": true}',
                }
            },
        )

    monkeypatch.setattr(llm_client.requests, "post", fake_post)
    parsed = llm_client.chat_json(messages=[{"role": "user", "content": "hello"}])

    assert parsed["ok"] is True
    assert captured["payload"]["think"] is False


def test_ollama_think_retry_when_content_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 `test_ollama_think_retry_when_content_empty` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "deepseek-r1:8b")
    monkeypatch.setenv("OLLAMA_THINK", "true")

    calls: List[Dict[str, Any]] = []

    def fake_post(url: str, headers=None, json=None, timeout=0):
        """提供 `fake_post` 測試替身以模擬外部依賴或固定回傳。
        此函式讓測試可注入可預測資料，將驗證焦點集中在流程與邏輯判斷。
        """
        calls.append({"url": url, "payload": json, "timeout": timeout})
        if len(calls) == 1:
            return DummyResponse(
                200,
                {
                    "message": {"content": "", "thinking": "reasoning..."},
                    "done_reason": "length",
                },
            )
        return DummyResponse(
            200,
            {
                "message": {"content": "ok"},
                "done_reason": "stop",
            },
        )

    monkeypatch.setattr(llm_client.requests, "post", fake_post)

    text = llm_client.chat_text(messages=[{"role": "user", "content": "hello"}], max_tokens=128)

    assert text == "ok"
    assert len(calls) == 2
    assert calls[0]["payload"]["options"]["num_predict"] == 128
    assert calls[1]["payload"]["options"]["num_predict"] == 1024


def test_ollama_think_retry_for_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 `test_ollama_think_retry_for_json` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "deepseek-r1:8b")
    monkeypatch.setenv("OLLAMA_THINK", "true")

    calls: List[Dict[str, Any]] = []

    def fake_post(url: str, headers=None, json=None, timeout=0):
        """提供 `fake_post` 測試替身以模擬外部依賴或固定回傳。
        此函式讓測試可注入可預測資料，將驗證焦點集中在流程與邏輯判斷。
        """
        calls.append({"url": url, "payload": json, "timeout": timeout})
        if len(calls) == 1:
            return DummyResponse(
                200,
                {
                    "message": {"content": "", "thinking": "reasoning..."},
                    "done_reason": "length",
                },
            )
        return DummyResponse(
            200,
            {
                "message": {"content": '{"ok": true}'},
                "done_reason": "stop",
            },
        )

    monkeypatch.setattr(llm_client.requests, "post", fake_post)

    parsed = llm_client.chat_json(messages=[{"role": "user", "content": "hello"}], max_tokens=256)

    assert parsed["ok"] is True
    assert len(calls) == 2
    assert calls[0]["payload"]["options"]["num_predict"] == 256
    assert calls[1]["payload"]["options"]["num_predict"] == 1024


def test_ollama_chat_json_parses_content_with_think_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 `test_ollama_chat_json_parses_content_with_think_tags` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "deepseek-r1:8b")
    monkeypatch.setenv("OLLAMA_THINK", "true")
    monkeypatch.setenv("OLLAMA_THINK_JSON", "true")

    def fake_post(url: str, headers=None, json=None, timeout=0):
        """提供 `fake_post` 測試替身以模擬外部依賴或固定回傳。
        此函式讓測試可注入可預測資料，將驗證焦點集中在流程與邏輯判斷。
        """
        return DummyResponse(
            200,
            {
                "message": {
                    "content": "<think>reasoning...</think>\n{\"ok\": true, \"source\": \"json\"}",
                }
            },
        )

    monkeypatch.setattr(llm_client.requests, "post", fake_post)
    parsed = llm_client.chat_json(messages=[{"role": "user", "content": "hello"}])

    assert parsed["ok"] is True
    assert parsed["source"] == "json"


def test_ollama_chat_json_parses_unclosed_think_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 `test_ollama_chat_json_parses_unclosed_think_prefix` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "deepseek-r1:8b")
    monkeypatch.setenv("OLLAMA_THINK", "true")
    monkeypatch.setenv("OLLAMA_THINK_JSON", "true")

    def fake_post(url: str, headers=None, json=None, timeout=0):
        """提供 `fake_post` 測試替身以模擬外部依賴或固定回傳。
        此函式讓測試可注入可預測資料，將驗證焦點集中在流程與邏輯判斷。
        """
        return DummyResponse(
            200,
            {
                "message": {
                    "content": "<think>\nreasoning...\n{\"ok\": true, \"value\": 3}",
                }
            },
        )

    monkeypatch.setattr(llm_client.requests, "post", fake_post)
    parsed = llm_client.chat_json(messages=[{"role": "user", "content": "hello"}])

    assert parsed["ok"] is True
    assert parsed["value"] == 3


def test_ollama_chat_json_writes_think_log_from_thinking_field(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """驗證 `test_ollama_chat_json_writes_think_log_from_thinking_field` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "deepseek-r1:8b")
    monkeypatch.setenv("LLM_THINK_LOG_ENABLED", "true")
    log_path = tmp_path / "think.log"
    monkeypatch.setenv("LLM_THINK_LOG_PATH", str(log_path))

    def fake_post(url: str, headers=None, json=None, timeout=0):
        """提供 `fake_post` 測試替身以模擬外部依賴或固定回傳。
        此函式讓測試可注入可預測資料，將驗證焦點集中在流程與邏輯判斷。
        """
        return DummyResponse(
            200,
            {
                "message": {
                    "thinking": "step-1\nstep-2",
                    "content": '{"ok": true}',
                },
                "done_reason": "stop",
            },
        )

    monkeypatch.setattr(llm_client.requests, "post", fake_post)

    parsed = llm_client.chat_json(messages=[{"role": "user", "content": "hello"}])

    assert parsed["ok"] is True
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["provider"] == "ollama"
    assert entry["mode"] == "chat_json"
    assert entry["model"] == "deepseek-r1:8b"
    assert entry["thinking"] == "step-1\nstep-2"


def test_ollama_chat_json_writes_think_log_from_content_tags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """驗證 `test_ollama_chat_json_writes_think_log_from_content_tags` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "deepseek-r1:8b")
    monkeypatch.setenv("LLM_THINK_LOG_ENABLED", "true")
    log_path = tmp_path / "think.log"
    monkeypatch.setenv("LLM_THINK_LOG_PATH", str(log_path))

    def fake_post(url: str, headers=None, json=None, timeout=0):
        """提供 `fake_post` 測試替身以模擬外部依賴或固定回傳。
        此函式讓測試可注入可預測資料，將驗證焦點集中在流程與邏輯判斷。
        """
        return DummyResponse(
            200,
            {
                "message": {
                    "content": "<think>trace A\ntrace B</think>\n{\"ok\": true}",
                },
                "done_reason": "stop",
            },
        )

    monkeypatch.setattr(llm_client.requests, "post", fake_post)

    parsed = llm_client.chat_json(messages=[{"role": "user", "content": "hello"}])

    assert parsed["ok"] is True
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["mode"] == "chat_json"
    assert entry["thinking"] == "trace A\ntrace B"


def test_chat_text_timeout_raises_llm_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 `test_chat_text_timeout_raises_llm_timeout` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8080/v1")

    def fake_post(url: str, headers=None, json=None, timeout=0):
        """提供 `fake_post` 測試替身以模擬外部依賴或固定回傳。
        此函式讓測試可注入可預測資料，將驗證焦點集中在流程與邏輯判斷。
        """
        raise requests.Timeout("timeout")

    monkeypatch.setattr(llm_client.requests, "post", fake_post)

    with pytest.raises(llm_client.LLMTimeoutError):
        llm_client.chat_text(messages=[{"role": "user", "content": "hello"}])


def test_chat_json_http_error_raises_llm_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 `test_chat_json_http_error_raises_llm_http_error` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8080/v1")

    def fake_post(url: str, headers=None, json=None, timeout=0):
        """提供 `fake_post` 測試替身以模擬外部依賴或固定回傳。
        此函式讓測試可注入可預測資料，將驗證焦點集中在流程與邏輯判斷。
        """
        return DummyResponse(502, text="bad gateway")

    monkeypatch.setattr(llm_client.requests, "post", fake_post)

    with pytest.raises(llm_client.LLMHTTPError):
        llm_client.chat_json(messages=[{"role": "user", "content": "hello"}])


def test_chat_json_invalid_payload_raises_llm_parse_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 `test_chat_json_invalid_payload_raises_llm_parse_error` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8080/v1")

    def fake_post(url: str, headers=None, json=None, timeout=0):
        """提供 `fake_post` 測試替身以模擬外部依賴或固定回傳。
        此函式讓測試可注入可預測資料，將驗證焦點集中在流程與邏輯判斷。
        """
        return DummyResponse(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": "not-a-json",
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(llm_client.requests, "post", fake_post)

    with pytest.raises(llm_client.LLMParseError):
        llm_client.chat_json(messages=[{"role": "user", "content": "hello"}])
