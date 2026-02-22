from __future__ import annotations

from backend.config import settings


def test_llm_runtime_settings_openai(monkeypatch) -> None:
    """驗證 `test_llm_runtime_settings_openai` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8080/v1")
    monkeypatch.setenv("LLM_MODEL", "mlx-community/Qwen3-8B-4bit-DWQ-053125")

    cfg = settings.get_llm_runtime_settings()

    assert cfg.provider == "openai"
    assert cfg.model == "mlx-community/Qwen3-8B-4bit-DWQ-053125"
    assert cfg.openai_base_url == "http://localhost:8080/v1"


def test_llm_runtime_settings_gemini_defaults_max_tokens(monkeypatch) -> None:
    """驗證 `test_llm_runtime_settings_gemini_defaults_max_tokens` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("LLM_MAX_TOKENS", raising=False)
    monkeypatch.delenv("OLLAMA_NUM_PREDICT", raising=False)
    monkeypatch.setenv("GEMINI_OUTPUT_TOKEN_LIMIT", "12345")

    cfg = settings.get_llm_runtime_settings()

    assert cfg.provider == "gemini"
    assert cfg.max_tokens == 12345


def test_kg_qa_settings_parsing(monkeypatch) -> None:
    """驗證 `test_kg_qa_settings_parsing` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    monkeypatch.setenv("KG_QA_USE_LLM", "0")
    monkeypatch.setenv("KG_QA_MODEL", "")
    monkeypatch.setenv("KG_QA_TEMPERATURE", "0.3")
    monkeypatch.setenv("KG_QA_MAX_TOKENS", "2048")

    qa = settings.get_kg_qa_settings()

    assert qa.use_llm is False
    assert qa.model is None
    assert qa.temperature == 0.3
    assert qa.max_tokens == 2048


def test_resolve_extraction_provider_validation(monkeypatch) -> None:
    """驗證 `test_resolve_extraction_provider_validation` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    monkeypatch.setenv("EXTRACTION_PROVIDER", "gemini")
    assert settings.resolve_extraction_provider(None) == "gemini"

    monkeypatch.setenv("EXTRACTION_PROVIDER", "invalid-provider")
    try:
        settings.resolve_extraction_provider(None)
        raised = False
    except ValueError:
        raised = True
    assert raised is True
