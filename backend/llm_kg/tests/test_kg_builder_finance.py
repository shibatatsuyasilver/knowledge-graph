from __future__ import annotations

from backend.llm_kg import kg_builder


def _builder_without_driver() -> kg_builder.KnowledgeGraphBuilder:
    """`_builder_without_driver` 的內部輔助函式。

主要用途：
- 封裝局部步驟，讓主流程維持可讀性。
- 集中處理細節與邊界條件，避免重複邏輯分散。

回傳約定：
- 保持既有輸入/輸出契約，不改變對外行為。
    """
    return kg_builder.KnowledgeGraphBuilder.__new__(kg_builder.KnowledgeGraphBuilder)


def test_sanitize_finance_entities_relations_and_props() -> None:
    """財報實體屬性保留原始文字，不做 canonicalization。"""
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    builder = _builder_without_driver()
    payload = {
        "entities": [
            {"name": "鴻海", "type": "Organization"},
            {"name": "2025 Q2", "type": "FiscalPeriod"},
            {
                "name": "鴻海 2025Q2 營收",
                "type": "FinancialMetric",
                "metric_type": "營收",
                "value": "1.23 兆",
                "unit": "兆",
                "currency": "TWD",
                "period": "2025 Q2",
            },
        ],
        "relations": [
            {"source": "鴻海", "relation": "HAS_FINANCIAL_METRIC", "target": "鴻海 2025Q2 營收"},
            {"source": "鴻海 2025Q2 營收", "relation": "FOR_PERIOD", "target": "2025 Q2"},
        ],
    }

    sanitized = builder._sanitize_extraction(payload)
    entities = sanitized["entities"]
    relations = sanitized["relations"]

    metric = next(e for e in entities if e["type"] == "FinancialMetric")
    period = next(e for e in entities if e["type"] == "FiscalPeriod")

    assert metric["metric_type"] == "營收"
    assert metric["value"] == "1.23 兆"
    assert metric["currency"] == "TWD"
    assert metric["period"] == "2025 Q2"
    assert period["name"] == "2025 Q2"
    assert {"source": "鴻海", "relation": "HAS_FINANCIAL_METRIC", "target": "鴻海 2025Q2 營收"} in relations
    assert {"source": "鴻海 2025Q2 營收", "relation": "FOR_PERIOD", "target": "2025 Q2"} in relations


def test_sanitize_finance_period_mismatch_drops_for_period_relation() -> None:
    """不再因 period mismatch 丟棄 FOR_PERIOD 關係。"""
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    builder = _builder_without_driver()
    payload = {
        "entities": [
            {"name": "鴻海", "type": "Organization"},
            {"name": "2025Q2", "type": "FiscalPeriod"},
            {"name": "2025Q3", "type": "FiscalPeriod"},
            {
                "name": "鴻海 2025Q2 營收",
                "type": "FinancialMetric",
                "metric_type": "REVENUE",
                "value": "100",
                "period": "2025Q2",
            },
        ],
        "relations": [
            {"source": "鴻海", "relation": "HAS_FINANCIAL_METRIC", "target": "鴻海 2025Q2 營收"},
            {"source": "鴻海 2025Q2 營收", "relation": "FOR_PERIOD", "target": "2025Q2"},
            {"source": "鴻海 2025Q2 營收", "relation": "FOR_PERIOD", "target": "2025Q3"},
        ],
    }

    sanitized = builder._sanitize_extraction(payload)
    relations = sanitized["relations"]

    assert {"source": "鴻海 2025Q2 營收", "relation": "FOR_PERIOD", "target": "2025Q2"} in relations
    assert {"source": "鴻海 2025Q2 營收", "relation": "FOR_PERIOD", "target": "2025Q3"} in relations


def test_sanitize_finance_period_inferred_from_for_period_relation() -> None:
    """不再由 FOR_PERIOD 反推並覆蓋 FinancialMetric.period。"""
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    builder = _builder_without_driver()
    payload = {
        "entities": [
            {"name": "鴻海", "type": "Organization"},
            {"name": "2025Q3", "type": "FiscalPeriod"},
            {
                "name": "鴻海 2025Q3 營益率",
                "type": "FinancialMetric",
                "metric_type": "OPERATING_MARGIN",
                "value": "3.2",
            },
        ],
        "relations": [
            {"source": "鴻海", "relation": "HAS_FINANCIAL_METRIC", "target": "鴻海 2025Q3 營益率"},
            {"source": "鴻海 2025Q3 營益率", "relation": "FOR_PERIOD", "target": "2025Q3"},
        ],
    }

    sanitized = builder._sanitize_extraction(payload)
    metric = next(e for e in sanitized["entities"] if e["type"] == "FinancialMetric")

    assert "period" not in metric
