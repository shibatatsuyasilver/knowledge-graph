from __future__ import annotations

import pytest

from backend.llm_kg import nl2cypher


def test_prompt_includes_union_alias_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    captured: dict[str, object] = {}

    def fake_chat_text(**kwargs):
        captured.update(kwargs)
        return "MATCH (n) RETURN n LIMIT 1"

    monkeypatch.setattr(nl2cypher.llm_client, "chat_text", fake_chat_text)

    _ = nl2cypher.natural_language_to_cypher(
        question="鴻海的事業有哪些",
        schema="Node Labels:\nOrganization\n\nRelationships:\n- (Organization)-[:INVESTS_IN]->(Industry)",
    )

    prompt = captured["messages"][0]["content"]
    assert "若使用 UNION/UNION ALL，各分支 RETURN 欄位數量、順序、欄位別名必須完全一致。" in prompt
