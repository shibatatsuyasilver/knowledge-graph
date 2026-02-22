from __future__ import annotations

import pytest

from backend.llm_kg import e2e_runner


class _FakeStats:
    def __init__(self, entities: int, relations: int):
        """初始化物件狀態並保存後續流程所需依賴。
        此方法會依目前參數設定實例欄位，供其他方法在生命週期內重複使用。
        """
        self.entities = entities
        self.relations = relations
        self.merged_entities = 0
        self.dropped_relations = 0
        self.json_retries = 0


class _FakeSession:
    def __enter__(self):
        """建立並回傳 context manager 進入階段所需資源。
        此方法在 `with` 區塊開始時執行，並維持既有回傳物件與行為契約。
        """
        return self

    def __exit__(self, exc_type, exc, tb):
        """負責 context manager 離開階段的清理與收尾作業。
        此方法會依目前例外傳入參數完成資源釋放，並保持既有錯誤傳遞語意。
        """
        return False

    def run(self, _query: str):
        """執行 `run` 的主要流程。
        函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
        """
        return None


class _FakeDriver:
    def session(self):
        """執行 `session` 的主要流程。
        函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
        """
        return _FakeSession()


class _BuilderExtractionFails:
    def __init__(self, uri: str, user: str, password: str):
        """初始化物件狀態並保存後續流程所需依賴。
        此方法會依目前參數設定實例欄位，供其他方法在生命週期內重複使用。
        """
        self.driver = _FakeDriver()

    def close(self) -> None:
        """執行 `close` 的主要流程。
        函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
        """
        return None

    def extract_entities_relations(self, _text: str):
        """執行 `extract_entities_relations` 的主要流程。
        函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
        """
        raise RuntimeError("boom")

    def populate_graph(self, payload):
        """執行 `populate_graph` 的主要流程。
        函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
        """
        return _FakeStats(
            entities=len(payload.get("entities", [])),
            relations=len(payload.get("relations", [])),
        )


def test_run_uses_fallback_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 `test_run_uses_fallback_when_enabled` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
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
    """驗證 `test_run_uses_qa_fallback_template` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
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
    """驗證 `test_run_raises_when_fallback_disabled` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    monkeypatch.setenv("E2E_ALLOW_FALLBACK_EXTRACT", "0")
    monkeypatch.setattr(e2e_runner, "KnowledgeGraphBuilder", _BuilderExtractionFails)

    with pytest.raises(RuntimeError, match="boom"):
        e2e_runner.run()


def test_run_raises_when_qa_fallback_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 `test_run_raises_when_qa_fallback_disabled` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
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
