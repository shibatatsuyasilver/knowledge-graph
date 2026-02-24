from __future__ import annotations

import pytest

from backend.llm_kg import nl2cypher


def test_normalize_union_aliases_aligns_mismatch_when_counts_match() -> None:
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    cypher = """
    MATCH (o:Organization)-[:HAS_FINANCIAL_METRIC]->(m:FinancialMetric)-[:FOR_PERIOD]->(p:FiscalPeriod)
    RETURN o.name AS company, p.period AS quarter
    UNION ALL
    MATCH (o:Organization)-[:HAS_FINANCIAL_METRIC]->(m:FinancialMetric)-[:FOR_PERIOD]->(p:FiscalPeriod)
    RETURN o.name AS organization, p.period AS period
    """.strip()

    normalized = nl2cypher._normalize_union_return_aliases(cypher)
    parts, _ = nl2cypher._split_union_query_parts(normalized)

    assert "AS company" in parts[1]
    assert "AS quarter" in parts[1]
    assert "AS organization" not in parts[1]
    assert "AS period" not in parts[1]


def test_normalize_union_aliases_raises_when_projection_count_differs() -> None:
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    cypher = """
    MATCH (o:Organization)
    RETURN o.name AS company, o.code AS ticker
    UNION ALL
    MATCH (o:Organization)
    RETURN o.name AS company
    """.strip()

    with pytest.raises(ValueError, match=r"UNION branch 2 RETURN projection count mismatch: expected 2, got 1"):
        nl2cypher._normalize_union_return_aliases(cypher)


def test_normalize_union_aliases_raises_when_missing_as_alias() -> None:
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    cypher = """
    MATCH (o:Organization)
    RETURN o.name AS company, o.code AS ticker
    UNION ALL
    MATCH (o:Organization)
    RETURN o.name AS company, o.code
    """.strip()

    with pytest.raises(ValueError, match=r"UNION branch 2 RETURN expression 2 must use explicit AS alias"):
        nl2cypher._normalize_union_return_aliases(cypher)


def test_normalize_union_aliases_noop_for_non_union() -> None:
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    cypher = "MATCH (o:Organization) RETURN o.name AS company"
    assert nl2cypher._normalize_union_return_aliases(cypher) == cypher


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
