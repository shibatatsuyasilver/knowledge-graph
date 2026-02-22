from __future__ import annotations

import pytest

from backend.llm_kg import kg_builder


def _builder_without_driver() -> kg_builder.KnowledgeGraphBuilder:
    """執行 `_builder_without_driver` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    return kg_builder.KnowledgeGraphBuilder.__new__(kg_builder.KnowledgeGraphBuilder)


def test_gemini_two_pass_prefills_missing_entities(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 `test_gemini_two_pass_prefills_missing_entities` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    builder = _builder_without_driver()
    monkeypatch.setattr(kg_builder, "GEMINI_TWO_PASS_EXTRACTION", True)

    prompts: list[str] = []

    def fake_extract_json_with_retry(prompt: str, **_kwargs):
        """提供 `fake_extract_json_with_retry` 測試替身以模擬外部依賴或固定回傳。
        此函式讓測試可注入可預測資料，將驗證焦點集中在流程與邏輯判斷。
        """
        prompts.append(prompt)
        if "Phase-1 Entity Inventory Rules" in prompt:
            return (
                {
                    "entities": [
                        {"name": "鴻海", "type": "Organization"},
                        {"name": "劉揚偉", "type": "Person"},
                    ],
                    "relations": [],
                },
                1,
            )
        return (
            {
                "entities": [
                    {"name": "鴻海", "type": "Organization"},
                    {"name": "劉揚偉", "type": "Person"},
                ],
                "relations": [
                    {"source": "鴻海", "relation": "CHAIRED_BY", "target": "劉揚偉"},
                ],
            },
            2,
        )

    monkeypatch.setattr(builder, "_extract_json_with_retry", fake_extract_json_with_retry)
    monkeypatch.setattr(builder, "_fetch_existing_entity_keys", lambda _entities: {("鴻海", "Organization")})

    created: list[tuple[str, str]] = []

    def fake_create_entity(name: str, entity_type: str, **_kwargs):
        """提供 `fake_create_entity` 測試替身以模擬外部依賴或固定回傳。
        此函式讓測試可注入可預測資料，將驗證焦點集中在流程與邏輯判斷。
        """
        created.append((name, entity_type))

    monkeypatch.setattr(builder, "_create_entity", fake_create_entity)

    extracted = builder.extract_entities_relations(
        "鴻海董事長是劉揚偉",
        provider="gemini",
        model="gemini-3-pro-preview",
    )

    assert len(prompts) == 2
    assert "Phase-1 Entity Inventory Rules" in prompts[0]
    assert "Phase-1 Seed Entities" in prompts[1]
    assert created == [("劉揚偉", "Person")]
    assert extracted["meta"]["two_pass"] is True
    assert extracted["meta"]["phase1_entities"] == 2
    assert extracted["meta"]["prefilled_missing_entities"] == 1
    assert extracted["meta"]["phase1_json_retries"] == 1
    assert extracted["meta"]["phase2_json_retries"] == 2
    assert extracted["meta"]["json_retries"] == 3


def test_non_gemini_provider_keeps_single_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 `test_non_gemini_provider_keeps_single_pass` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    builder = _builder_without_driver()
    monkeypatch.setattr(kg_builder, "GEMINI_TWO_PASS_EXTRACTION", True)

    calls = {"count": 0}

    def fake_extract_json_with_retry(prompt: str, **_kwargs):
        """提供 `fake_extract_json_with_retry` 測試替身以模擬外部依賴或固定回傳。
        此函式讓測試可注入可預測資料，將驗證焦點集中在流程與邏輯判斷。
        """
        calls["count"] += 1
        return ({"entities": [{"name": "台積電", "type": "Organization"}], "relations": []}, 0)

    monkeypatch.setattr(builder, "_extract_json_with_retry", fake_extract_json_with_retry)

    def _unexpected_fetch(_entities):
        """執行 `_unexpected_fetch` 的內部輔助流程。
        此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
        """
        raise AssertionError("two-pass entity prefill should not run for non-gemini providers")

    monkeypatch.setattr(builder, "_fetch_existing_entity_keys", _unexpected_fetch)

    extracted = builder.extract_entities_relations("台積電", provider="ollama")

    assert calls["count"] == 1
    assert extracted["meta"]["two_pass"] is False
