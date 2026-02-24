from __future__ import annotations

import pytest

from backend.llm_kg import kg_builder, nl2cypher


def _builder_without_driver() -> kg_builder.KnowledgeGraphBuilder:
    """`_builder_without_driver` 的內部輔助函式。

主要用途：
- 封裝局部步驟，讓主流程維持可讀性。
- 集中處理細節與邊界條件，避免重複邏輯分散。

回傳約定：
- 保持既有輸入/輸出契約，不改變對外行為。
    """
    return kg_builder.KnowledgeGraphBuilder.__new__(kg_builder.KnowledgeGraphBuilder)


def test_kg_builder_uses_extraction_model_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 `test_kg_builder_uses_extraction_model_override` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    builder = _builder_without_driver()
    monkeypatch.setenv("EXTRACTION_MODEL", "sam860/deepseek-r1-0528-qwen3:8b")
    monkeypatch.setattr(kg_builder, "EXTRACTION_JSON_MODE", "strict_json")

    captured: dict[str, object] = {}

    def fake_chat_json(**kwargs):
        """提供 `fake_chat_json` 測試替身以模擬外部依賴或固定回傳。
        此函式讓測試可注入可預測資料，將驗證焦點集中在流程與邏輯判斷。
        """
        captured.update(kwargs)
        return {"entities": [{"name": "台積電", "type": "Organization"}], "relations": []}

    monkeypatch.setattr(kg_builder.llm_client, "chat_json", fake_chat_json)

    parsed, retries = builder._extract_json_with_retry("test prompt")

    assert captured["model"] == "sam860/deepseek-r1-0528-qwen3:8b"
    assert retries == 0
    assert parsed["entities"][0]["name"] == "台積電"


def test_kg_builder_gemini_uses_gemini_model_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 `test_kg_builder_gemini_uses_gemini_model_by_default` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    builder = _builder_without_driver()
    monkeypatch.setenv("EXTRACTION_MODEL", "sam860/deepseek-r1-0528-qwen3:8b")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3-pro-preview")
    monkeypatch.setattr(kg_builder, "EXTRACTION_JSON_MODE", "strict_json")

    captured: dict[str, object] = {}

    def fake_chat_json(**kwargs):
        """提供 `fake_chat_json` 測試替身以模擬外部依賴或固定回傳。
        此函式讓測試可注入可預測資料，將驗證焦點集中在流程與邏輯判斷。
        """
        captured.update(kwargs)
        return {"entities": [], "relations": []}

    monkeypatch.setattr(kg_builder.llm_client, "chat_json", fake_chat_json)

    parsed, retries = builder._extract_json_with_retry("test prompt", provider="gemini")

    assert captured["provider"] == "gemini"
    assert captured["model"] == "gemini-3-pro-preview"
    assert retries == 0
    assert parsed["entities"] == []


def test_nl2cypher_uses_model_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 `test_nl2cypher_uses_model_override` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    monkeypatch.setenv("NL2CYPHER_MODEL", "ministral-3:14b")

    captured: dict[str, object] = {}

    def fake_chat_text(**kwargs):
        """提供 `fake_chat_text` 測試替身以模擬外部依賴或固定回傳。
        此函式讓測試可注入可預測資料，將驗證焦點集中在流程與邏輯判斷。
        """
        captured.update(kwargs)
        return "MATCH (n) RETURN n LIMIT 1"

    monkeypatch.setattr(nl2cypher.llm_client, "chat_text", fake_chat_text)

    cypher = nl2cypher.natural_language_to_cypher(
        question="台積電在哪裡？",
        schema="Node Labels:\nOrganization\n\nRelationships:\n- (Organization)-[:HEADQUARTERED_IN]->(Location)",
    )

    assert captured["model"] == "ministral-3:14b"
    assert cypher == "MATCH (n) RETURN n LIMIT 1"
