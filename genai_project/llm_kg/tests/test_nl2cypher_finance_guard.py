from __future__ import annotations

from genai_project.llm_kg import nl2cypher


def test_detect_hardcoded_finance_constant_query() -> None:
    cypher = """
    MATCH (o:Organization {name:'鴻海'})
    RETURN '2025Q2' AS quarter, '無資料' AS revenue, '無資料' AS profit_margin
    UNION ALL
    MATCH (o:Organization {name:'鴻海'})
    RETURN '2025Q3' AS quarter, '無資料' AS revenue, '無資料' AS profit_margin
    """.strip()
    assert nl2cypher._cypher_uses_hardcoded_finance_constants(cypher)


def test_finance_path_check_requires_metric_and_period_relations() -> None:
    valid = (
        "MATCH (o:Organization)-[:HAS_FINANCIAL_METRIC]->(m:FinancialMetric)"
        "-[:FOR_PERIOD]->(p:FiscalPeriod) RETURN p.period, m.value"
    )
    invalid = "MATCH (o:Organization) RETURN '2025Q2' AS quarter"
    assert nl2cypher._cypher_has_required_finance_path(valid)
    assert not nl2cypher._cypher_has_required_finance_path(invalid)


def test_finance_template_contains_graph_bound_projection() -> None:
    cypher = nl2cypher._build_finance_template_cypher(
        "2025Q2 與 2025Q3，鴻海營收與營益率各是多少？",
        ["鴻海精密", "台積電"],
    )
    assert "HAS_FINANCIAL_METRIC" in cypher
    assert "FOR_PERIOD" in cypher
    assert "m.value" in cypher
