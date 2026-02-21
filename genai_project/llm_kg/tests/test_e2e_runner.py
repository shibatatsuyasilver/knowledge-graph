from __future__ import annotations

import pytest

from genai_project.llm_kg import e2e_runner


class _FakeStats:
    def __init__(self, entities: int, relations: int):
        self.entities = entities
        self.relations = relations
        self.merged_entities = 0
        self.dropped_relations = 0
        self.json_retries = 0


class _FakeSession:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, _query: str):
        return None


class _FakeDriver:
    def session(self):
        return _FakeSession()


class _BuilderExtractionFails:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = _FakeDriver()

    def close(self) -> None:
        return None

    def extract_entities_relations(self, _text: str):
        raise RuntimeError("boom")

    def populate_graph(self, payload):
        return _FakeStats(
            entities=len(payload.get("entities", [])),
            relations=len(payload.get("relations", [])),
        )


def test_run_uses_fallback_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("E2E_ALLOW_FALLBACK_EXTRACT", "1")
    monkeypatch.setattr(e2e_runner, "KnowledgeGraphBuilder", _BuilderExtractionFails)
    monkeypatch.setattr(
        e2e_runner,
        "deterministic_fallback_payload",
        lambda: {
            "entities": [{"name": "台積電", "type": "Organization"}, {"name": "張忠謀", "type": "Person"}],
            "relations": [{"source": "台積電", "relation": "FOUNDED_BY", "target": "張忠謀"}],
        },
    )
    monkeypatch.setattr(
        e2e_runner,
        "answer_with_manual_prompt",
        lambda question: {"question": question, "rows": [{"ok": True}], "cypher": "MATCH ...", "attempt": 1},
    )

    result = e2e_runner.run()

    assert result["fallback_extract_used"] is True
    assert result["upserted_entities"] == 2
    assert result["upserted_relations"] == 1
    assert result["qa_success"] == 3


def test_run_uses_qa_fallback_template(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("E2E_ALLOW_FALLBACK_EXTRACT", "1")
    monkeypatch.setenv("E2E_ALLOW_FALLBACK_QA", "1")
    monkeypatch.setattr(e2e_runner, "KnowledgeGraphBuilder", _BuilderExtractionFails)
    monkeypatch.setattr(
        e2e_runner,
        "deterministic_fallback_payload",
        lambda: {
            "entities": [{"name": "台積電", "type": "Organization"}, {"name": "張忠謀", "type": "Person"}],
            "relations": [{"source": "台積電", "relation": "FOUNDED_BY", "target": "張忠謀"}],
        },
    )
    monkeypatch.setattr(
        e2e_runner,
        "answer_with_manual_prompt",
        lambda _question: (_ for _ in ()).throw(RuntimeError("nl2cypher failed")),
    )
    monkeypatch.setattr(
        e2e_runner,
        "_fallback_qa_template",
        lambda **kwargs: {
            "question": kwargs["question"],
            "rows": [{"ok": True}],
            "cypher": "MATCH ...",
            "attempt": 1,
            "fallback": "template",
        },
    )

    result = e2e_runner.run()

    assert result["fallback_extract_used"] is True
    assert result["fallback_qa_used"] is True
    assert result["qa_success"] == 3


def test_run_raises_when_fallback_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("E2E_ALLOW_FALLBACK_EXTRACT", "0")
    monkeypatch.setattr(e2e_runner, "KnowledgeGraphBuilder", _BuilderExtractionFails)

    with pytest.raises(RuntimeError, match="boom"):
        e2e_runner.run()


def test_run_raises_when_qa_fallback_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("E2E_ALLOW_FALLBACK_EXTRACT", "1")
    monkeypatch.setenv("E2E_ALLOW_FALLBACK_QA", "0")
    monkeypatch.setattr(e2e_runner, "KnowledgeGraphBuilder", _BuilderExtractionFails)
    monkeypatch.setattr(
        e2e_runner,
        "deterministic_fallback_payload",
        lambda: {
            "entities": [{"name": "台積電", "type": "Organization"}, {"name": "張忠謀", "type": "Person"}],
            "relations": [{"source": "台積電", "relation": "FOUNDED_BY", "target": "張忠謀"}],
        },
    )
    monkeypatch.setattr(
        e2e_runner,
        "answer_with_manual_prompt",
        lambda _question: (_ for _ in ()).throw(RuntimeError("nl2cypher failed hard")),
    )

    with pytest.raises(RuntimeError, match="nl2cypher failed hard"):
        e2e_runner.run()
