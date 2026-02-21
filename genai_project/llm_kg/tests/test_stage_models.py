from __future__ import annotations

import pytest

from genai_project.llm_kg import kg_builder, nl2cypher


def _builder_without_driver() -> kg_builder.KnowledgeGraphBuilder:
    return kg_builder.KnowledgeGraphBuilder.__new__(kg_builder.KnowledgeGraphBuilder)


def test_kg_builder_uses_extraction_model_override(monkeypatch: pytest.MonkeyPatch) -> None:
    builder = _builder_without_driver()
    monkeypatch.setenv("EXTRACTION_MODEL", "sam860/deepseek-r1-0528-qwen3:8b")
    monkeypatch.setattr(kg_builder, "EXTRACTION_JSON_MODE", "strict_json")

    captured: dict[str, object] = {}

    def fake_chat_json(**kwargs):
        captured.update(kwargs)
        return {"entities": [{"name": "台積電", "type": "Organization"}], "relations": []}

    monkeypatch.setattr(kg_builder.llm_client, "chat_json", fake_chat_json)

    parsed, retries = builder._extract_json_with_retry("test prompt")

    assert captured["model"] == "sam860/deepseek-r1-0528-qwen3:8b"
    assert retries == 0
    assert parsed["entities"][0]["name"] == "台積電"


def test_kg_builder_gemini_uses_gemini_model_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    builder = _builder_without_driver()
    monkeypatch.setenv("EXTRACTION_MODEL", "sam860/deepseek-r1-0528-qwen3:8b")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3-pro-preview")
    monkeypatch.setattr(kg_builder, "EXTRACTION_JSON_MODE", "strict_json")

    captured: dict[str, object] = {}

    def fake_chat_json(**kwargs):
        captured.update(kwargs)
        return {"entities": [], "relations": []}

    monkeypatch.setattr(kg_builder.llm_client, "chat_json", fake_chat_json)

    parsed, retries = builder._extract_json_with_retry("test prompt", provider="gemini")

    assert captured["provider"] == "gemini"
    assert captured["model"] == "gemini-3-pro-preview"
    assert retries == 0
    assert parsed["entities"] == []


def test_nl2cypher_uses_model_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NL2CYPHER_MODEL", "ministral-3:14b")

    captured: dict[str, object] = {}

    def fake_chat_text(**kwargs):
        captured.update(kwargs)
        return "MATCH (n) RETURN n LIMIT 1"

    monkeypatch.setattr(nl2cypher.llm_client, "chat_text", fake_chat_text)

    cypher = nl2cypher.natural_language_to_cypher(
        question="台積電在哪裡？",
        schema="Node Labels:\nOrganization\n\nRelationships:\n- (Organization)-[:HEADQUARTERED_IN]->(Location)",
    )

    assert captured["model"] == "ministral-3:14b"
    assert cypher == "MATCH (n) RETURN n LIMIT 1"
