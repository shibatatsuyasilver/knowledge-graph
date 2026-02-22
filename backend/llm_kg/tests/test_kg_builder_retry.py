from __future__ import annotations

import pytest

from backend.llm_kg import kg_builder


def _builder_without_driver() -> kg_builder.KnowledgeGraphBuilder:
    """執行 `_builder_without_driver` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    return kg_builder.KnowledgeGraphBuilder.__new__(kg_builder.KnowledgeGraphBuilder)


def test_extract_json_retries_until_5th_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 `test_extract_json_retries_until_5th_attempt` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    builder = _builder_without_driver()
    monkeypatch.setattr(kg_builder, "EXTRACTION_JSON_MODE", "strict_json")
    monkeypatch.setattr(kg_builder, "EXTRACTION_MAX_JSON_RETRIES", 4)

    calls = {"count": 0}

    def fake_chat_json(**_kwargs):
        """提供 `fake_chat_json` 測試替身以模擬外部依賴或固定回傳。
        此函式讓測試可注入可預測資料，將驗證焦點集中在流程與邏輯判斷。
        """
        calls["count"] += 1
        if calls["count"] < 5:
            return {"oops": "invalid"}
        return {"entities": [{"name": "台積電", "type": "Organization"}], "relations": []}

    monkeypatch.setattr(kg_builder.llm_client, "chat_json", fake_chat_json)

    parsed, retries = builder._extract_json_with_retry("test prompt")

    assert calls["count"] == 5
    assert retries == 4
    assert parsed["entities"][0]["name"] == "台積電"


def test_extract_json_fails_after_5_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 `test_extract_json_fails_after_5_attempts` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    builder = _builder_without_driver()
    monkeypatch.setattr(kg_builder, "EXTRACTION_JSON_MODE", "strict_json")
    monkeypatch.setattr(kg_builder, "EXTRACTION_MAX_JSON_RETRIES", 4)
    monkeypatch.setattr(kg_builder.llm_client, "chat_json", lambda **_kwargs: {"invalid": True})

    with pytest.raises(ValueError, match="after 5 attempts"):
        builder._extract_json_with_retry("test prompt")
