from __future__ import annotations

import pytest

from genai_project.llm_kg import kg_builder


def _builder_without_driver() -> kg_builder.KnowledgeGraphBuilder:
    return kg_builder.KnowledgeGraphBuilder.__new__(kg_builder.KnowledgeGraphBuilder)


def test_extract_json_retries_until_5th_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    builder = _builder_without_driver()
    monkeypatch.setattr(kg_builder, "EXTRACTION_JSON_MODE", "strict_json")
    monkeypatch.setattr(kg_builder, "EXTRACTION_MAX_JSON_RETRIES", 4)

    calls = {"count": 0}

    def fake_chat_json(**_kwargs):
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
    builder = _builder_without_driver()
    monkeypatch.setattr(kg_builder, "EXTRACTION_JSON_MODE", "strict_json")
    monkeypatch.setattr(kg_builder, "EXTRACTION_MAX_JSON_RETRIES", 4)
    monkeypatch.setattr(kg_builder.llm_client, "chat_json", lambda **_kwargs: {"invalid": True})

    with pytest.raises(ValueError, match="after 5 attempts"):
        builder._extract_json_with_retry("test prompt")
