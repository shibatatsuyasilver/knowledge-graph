from __future__ import annotations

from backend.llm_kg import kg_builder


def _builder_without_driver() -> kg_builder.KnowledgeGraphBuilder:
    """執行 `_builder_without_driver` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    return kg_builder.KnowledgeGraphBuilder.__new__(kg_builder.KnowledgeGraphBuilder)


def test_sanitize_finance_entities_relations_and_props() -> None:
    """驗證 `test_sanitize_finance_entities_relations_and_props` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
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

    assert metric["metric_type"] == "REVENUE"
    assert metric["value"] == "1.23 兆"
    assert metric["currency"] == "TWD"
    assert metric["period"] == "2025Q2"
    assert period["name"] == "2025Q2"
    assert period["period"] == "2025Q2"
    assert {"source": "鴻海", "relation": "HAS_FINANCIAL_METRIC", "target": "鴻海 2025Q2 營收"} in relations
    assert {"source": "鴻海 2025Q2 營收", "relation": "FOR_PERIOD", "target": "2025Q2"} in relations


def test_sanitize_finance_period_mismatch_drops_for_period_relation() -> None:
    """驗證 `test_sanitize_finance_period_mismatch_drops_for_period_relation` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
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
    assert {"source": "鴻海 2025Q2 營收", "relation": "FOR_PERIOD", "target": "2025Q3"} not in relations


def test_sanitize_finance_period_inferred_from_for_period_relation() -> None:
    """驗證 `test_sanitize_finance_period_inferred_from_for_period_relation` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
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

    assert metric["period"] == "2025Q3"
