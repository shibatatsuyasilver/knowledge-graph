"""Natural language to Cypher helpers with schema grounding and repair loop.

Two methods:
1) LangChain GraphCypherQAChain (recommended when dependencies are available)
2) Manual schema-grounded prompting + execution/self-correction
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from neo4j import Driver, GraphDatabase
from neo4j.exceptions import Neo4jError

try:
    from langchain_community.chains import GraphCypherQAChain
    from langchain_community.graphs import Neo4jGraph
    from langchain_community.llms import Ollama
except Exception:  # pragma: no cover - optional imports
    GraphCypherQAChain = None
    Neo4jGraph = None
    Ollama = None

try:
    from . import llm_client
except ImportError:  # pragma: no cover - direct script execution
    import llm_client  # type: ignore[no-redef]

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
CYPHER_REPAIR_RETRIES = int(os.getenv("CYPHER_REPAIR_RETRIES", "2"))
ENTITY_LINK_THRESHOLD = float(os.getenv("ENTITY_LINK_THRESHOLD", "0.82"))
NL2CYPHER_TIMEOUT_SECONDS = int(os.getenv("NL2CYPHER_TIMEOUT_SECONDS", "180"))
NL2CYPHER_NUM_PREDICT = int(os.getenv("NL2CYPHER_NUM_PREDICT", os.getenv("LLM_MAX_TOKENS", "1024")))

ROLE_TITLE_TERMS = (
    "董事長",
    "主席",
    "執行長",
    "總經理",
    "ceo",
    "chairman",
    "chairperson",
    "president",
)
FOUNDER_TERMS = ("創辦人", "創辦", "創立者", "創立", "founder", "founded")
ROLE_TITLE_SCHEMA_HINTS = (
    "chair",
    "chairman",
    "chairperson",
    "ceo",
    "executive",
    "president",
    "director",
    "board",
    "董事長",
    "主席",
    "執行長",
    "總經理",
)
FINANCE_TERMS = (
    "營收",
    "revenue",
    "sales",
    "營益率",
    "營業利益率",
    "operating margin",
    "operating_margin",
    "q1",
    "q2",
    "q3",
    "q4",
    "季度",
    "quarter",
)
FINANCE_RETURN_ALIAS_HINTS = {"quarter", "revenue", "profit_margin", "operating_margin"}
FINANCE_PROPERTY_HINTS = (
    "value",
    "metric_type",
    "period",
    "currency",
    "unit",
)
FINANCE_REQUIRED_REL_TYPES = ("HAS_FINANCIAL_METRIC", "FOR_PERIOD")


@dataclass
class SchemaSnapshot:
    labels: List[str]
    relationship_types: List[str]
    properties: Dict[str, List[str]]
    schema_text: str


def _question_has_any_term(question: str, terms: tuple[str, ...]) -> bool:
    q = question.lower()
    return any(term in q for term in terms)


def _schema_tokens(schema: SchemaSnapshot) -> List[str]:
    tokens: List[str] = []
    tokens.extend([_normalize(rel) for rel in schema.relationship_types])
    tokens.extend([_normalize(label) for label in schema.labels])
    for keys in schema.properties.values():
        tokens.extend([_normalize(key) for key in keys])
    deduped: List[str] = []
    seen = set()
    for token in tokens:
        if not token or token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


def _schema_supports_role_title(schema: SchemaSnapshot) -> bool:
    hints = [_normalize(x) for x in ROLE_TITLE_SCHEMA_HINTS if _normalize(x)]
    schema_terms = _schema_tokens(schema)
    for hint in hints:
        for term in schema_terms:
            if hint in term:
                return True
    return False


def _question_is_finance(question: str) -> bool:
    return _question_has_any_term(question, FINANCE_TERMS)


def _schema_supports_finance_metrics(schema: SchemaSnapshot) -> bool:
    rel_types = set(schema.relationship_types)
    return all(rel in rel_types for rel in FINANCE_REQUIRED_REL_TYPES)


def _cypher_has_required_finance_path(cypher: str) -> bool:
    return all(_cypher_uses_relationship(cypher, rel) for rel in FINANCE_REQUIRED_REL_TYPES)


def _cypher_uses_hardcoded_finance_constants(cypher: str) -> bool:
    aliases = re.findall(r"'[^']*'\s+AS\s+([A-Za-z_][A-Za-z0-9_]*)", cypher, flags=re.IGNORECASE)
    if not aliases:
        return False
    alias_hit = any(alias.lower() in FINANCE_RETURN_ALIAS_HINTS for alias in aliases)
    if not alias_hit:
        return False

    lower = cypher.lower()
    has_finance_field_access = bool(
        re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\.(value|metric_type|period|currency|unit)\b", lower)
    )
    if has_finance_field_access:
        return False

    has_relationship = bool(
        re.search(r"-\s*\[[^\]]*\]\s*->|<-\s*\[[^\]]*\]\s*-|-\s*\[[^\]]*\]-", cypher, flags=re.IGNORECASE)
    )
    has_union = bool(re.search(r"\bUNION(?:\s+ALL)?\b", cypher, flags=re.IGNORECASE))
    return has_union or not has_relationship


def _extract_finance_periods(question: str) -> List[str]:
    periods: List[str] = []
    normalized = question.upper()
    for match in re.finditer(r"(20\d{2})\s*Q\s*([1-4])", normalized):
        periods.append(f"{match.group(1)}Q{match.group(2)}")
    for match in re.finditer(r"(20\d{2})\s*年?\s*第?\s*([1-4])\s*季", question):
        periods.append(f"{match.group(1)}Q{match.group(2)}")
    deduped: List[str] = []
    seen = set()
    for period in periods:
        if period in seen:
            continue
        seen.add(period)
        deduped.append(period)
    return deduped


def _cypher_quote(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _guess_finance_org_hint(question: str, entity_names: List[str]) -> str:
    normalized_q = _normalize(question)
    best = ""
    best_len = 0
    for name in entity_names:
        normalized_name = _normalize(name)
        if not normalized_name:
            continue
        if normalized_name in normalized_q and len(normalized_name) > best_len:
            best = name
            best_len = len(normalized_name)
    return best


def _build_finance_template_cypher(question: str, entity_names: List[str]) -> str:
    periods = _extract_finance_periods(question)
    org_hint = _guess_finance_org_hint(question, entity_names)
    where_parts = ["m.metric_type IN ['REVENUE','OPERATING_MARGIN']"]
    if periods:
        where_parts.append("p.period IN [" + ", ".join(_cypher_quote(period) for period in periods) + "]")
    if org_hint:
        where_parts.append(
            "("
            f"toLower(o.name) CONTAINS toLower({_cypher_quote(org_hint)}) "
            f"OR any(a IN coalesce(o.aliases, []) WHERE toLower(a) CONTAINS toLower({_cypher_quote(org_hint)}))"
            ")"
        )
    elif _question_has_any_term(question, ("鴻海", "honhai", "foxconn")):
        where_parts.append(
            "("
            "toLower(o.name) CONTAINS toLower('鴻海') "
            "OR any(a IN coalesce(o.aliases, []) WHERE toLower(a) CONTAINS toLower('鴻海'))"
            ")"
        )

    return (
        "MATCH (o:Organization)-[:HAS_FINANCIAL_METRIC]->(m:FinancialMetric)-[:FOR_PERIOD]->(p:FiscalPeriod)\n"
        f"WHERE {' AND '.join(where_parts)}\n"
        "RETURN o.name AS organization, p.period AS period, m.metric_type AS metric_type, "
        "m.value AS value, m.unit AS unit, m.currency AS currency\n"
        "ORDER BY period, metric_type"
    )


def _empty_result(question: str, reason: str) -> Dict[str, Any]:
    return {
        "question": question,
        "cypher": "MATCH (n) WHERE false RETURN n LIMIT 0",
        "rows": [],
        "attempt": 0,
        "reason": reason,
    }


def _cypher_uses_relationship(cypher: str, rel_type: str) -> bool:
    pattern = rf":\s*`?{re.escape(rel_type)}`?"
    return bool(re.search(pattern, cypher, flags=re.IGNORECASE))


def _relax_exact_organization_name_match(cypher: str) -> str | None:
    """Relaxes (o:Organization {name:'X'}) to fuzzy name match when exact name returns no rows."""
    pattern = re.compile(
        r"\((?P<var>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*Organization\s*\{\s*name\s*:\s*'(?P<name>[^']+)'\s*\}\)"
    )
    match = pattern.search(cypher)
    if not match:
        return None

    var = match.group("var")
    name = match.group("name")
    relaxed = pattern.sub(f"({var}:Organization)", cypher, count=1)
    filter_expr = f"toLower({var}.name) CONTAINS toLower('{name}')"

    if re.search(r"\bWHERE\b", relaxed, flags=re.IGNORECASE):
        return re.sub(r"\bWHERE\b", f"WHERE {filter_expr} AND ", relaxed, count=1, flags=re.IGNORECASE)

    if re.search(r"\bRETURN\b", relaxed, flags=re.IGNORECASE):
        return re.sub(r"\bRETURN\b", f"WHERE {filter_expr}\nRETURN", relaxed, count=1, flags=re.IGNORECASE)
    return None


def _resolve_nl2cypher_model() -> str | None:
    """Returns stage-specific NL2Cypher model override if configured."""
    model = os.getenv("NL2CYPHER_MODEL", "").strip()
    return model or None


def strip_markdown_fence(content: str) -> str:
    text = content.strip()
    match = re.match(r"^```(?:cypher|sql)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else text).strip()


def _normalize(value: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value.strip().lower())


def load_schema_snapshot(driver: Driver) -> SchemaSnapshot:
    with driver.session() as session:
        labels = sorted([row["label"] for row in session.run("CALL db.labels() YIELD label RETURN label")])
        relationship_types = sorted(
            [row["relationshipType"] for row in session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")]
        )

        properties: Dict[str, List[str]] = {}
        for label in labels:
            rows = session.run(
                f"MATCH (n:`{label}`) WITH n LIMIT 500 "
                "UNWIND keys(n) AS key RETURN DISTINCT key ORDER BY key"
            )
            properties[label] = [row["key"] for row in rows]

        shape_rows = session.run(
            """
            MATCH (a)-[r]->(b)
            UNWIND labels(a) AS fromLabel
            UNWIND labels(b) AS toLabel
            WITH type(r) AS relType, fromLabel, toLabel
            WHERE fromLabel <> 'Entity' AND toLabel <> 'Entity'
            RETURN relType, collect(DISTINCT fromLabel) AS fromLabels, collect(DISTINCT toLabel) AS toLabels
            ORDER BY relType
            """
        )

        relation_shapes: List[str] = []
        for row in shape_rows:
            rel_type = row["relType"]
            for from_label in sorted(row["fromLabels"]):
                for to_label in sorted(row["toLabels"]):
                    relation_shapes.append(f"({from_label})-[:{rel_type}]->({to_label})")

    schema_lines = [
        "Node Labels:",
        ", ".join(labels) if labels else "(empty)",
        "",
        "Properties by label:",
    ]
    for label in labels:
        prop_line = ", ".join(properties[label]) if properties[label] else "(no properties)"
        schema_lines.append(f"- {label}: {prop_line}")

    schema_lines.append("")
    schema_lines.append("Relationships:")
    if relation_shapes:
        for shape in sorted(set(relation_shapes)):
            schema_lines.append(f"- {shape}")
    else:
        for rel_type in relationship_types:
            schema_lines.append(f"- {rel_type}")

    return SchemaSnapshot(
        labels=labels,
        relationship_types=relationship_types,
        properties=properties,
        schema_text="\n".join(schema_lines),
    )


def fetch_entity_names(driver: Driver) -> List[str]:
    with driver.session() as session:
        rows = session.run("MATCH (e:Entity) WHERE e.name IS NOT NULL RETURN DISTINCT e.name AS name")
        return [str(row["name"]) for row in rows]


def fetch_organization_names(driver: Driver) -> List[str]:
    with driver.session() as session:
        rows = session.run("MATCH (o:Organization) WHERE o.name IS NOT NULL RETURN DISTINCT o.name AS name")
        return [str(row["name"]) for row in rows]


def link_entity_literals(cypher: str, entity_names: List[str]) -> str:
    if not entity_names:
        return cypher

    entity_lookup = {_normalize(name): name for name in entity_names if _normalize(name)}

    def replace_literal(match: re.Match[str]) -> str:
        raw_literal = match.group(1)
        normalized = _normalize(raw_literal)
        if not normalized or len(normalized) <= 1:
            return match.group(0)

        if normalized in entity_lookup:
            return f"'{entity_lookup[normalized]}'"

        best_key = ""
        best_score = 0.0
        for candidate in entity_lookup:
            score = SequenceMatcher(None, normalized, candidate).ratio()
            if score > best_score:
                best_key, best_score = candidate, score

        if best_key and best_score >= ENTITY_LINK_THRESHOLD:
            return f"'{entity_lookup[best_key]}'"
        return match.group(0)

    return re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", replace_literal, cypher)


def query_with_graph_chain(question: str) -> Optional[Dict[str, Any]]:
    if not all([GraphCypherQAChain, Neo4jGraph, Ollama]):
        print("LangChain components are not installed. Skip GraphCypherQAChain method.")
        return None
    cfg = llm_client.get_runtime_config()
    if cfg.provider != "ollama":
        print("GraphCypherQAChain(Ollama) is only available when LLM_PROVIDER=ollama.")
        return None

    graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)
    llm = Ollama(model=_resolve_nl2cypher_model() or cfg.model, temperature=0)

    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True,
        return_intermediate_steps=True,
        allow_dangerous_requests=True,
    )

    result = chain.invoke({"query": question})
    generated = None
    if result.get("intermediate_steps"):
        generated = result["intermediate_steps"][0].get("query")

    return {
        "question": question,
        "answer": result.get("result", ""),
        "cypher": generated,
        "raw": result,
    }


def natural_language_to_cypher(
    question: str,
    schema: str,
    previous_cypher: str | None = None,
    error_message: str | None = None,
) -> str:
    repair_block = ""
    if previous_cypher and error_message:
        repair_block = f"""
上一次失敗的 Cypher：
{previous_cypher}

Neo4j 錯誤訊息：
{error_message}

請修正上面的 Cypher。
""".strip()

    prompt = f"""
你是 Neo4j Cypher 查詢專家。
請根據以下 Graph Schema 把使用者問題轉成單一可執行的 Cypher 查詢。

Schema:
{schema}

規則:
1. 只可用 schema 中存在的標籤與關係。
2. 只可用 schema 中看得到的屬性鍵。
3. 使用 MATCH ... RETURN。
4. RETURN 必須回傳可讀欄位（如 p.name、o.name），不要直接 RETURN 整個節點或關係物件。
5. 字串比對優先使用 toLower(x) CONTAINS toLower('關鍵詞')。
6. 問題語意必須與關係類型一致：
   - FOUNDED_BY 只可回答「創辦人/創立者」相關問題。
   - CHAIRED_BY 只可回答「董事長/主席/執行長/CEO」相關問題。
   - 若問題是「董事長/主席/執行長/CEO」但 schema 沒有對應欄位或關係，請回傳空結果查詢：
     MATCH (n) WHERE false RETURN n LIMIT 0
7. 不可用硬編碼常數偽造答案（禁止用 '無資料' AS revenue 這種常值回覆）。
8. 若是財報問題（營收/營益率/季度），優先使用下列圖譜路徑查詢真實資料：
   (o:Organization)-[:HAS_FINANCIAL_METRIC]->(m:FinancialMetric)-[:FOR_PERIOD]->(p:FiscalPeriod)
9. 僅回傳 Cypher，不要其他文字與 markdown。

使用者問題：{question}
{repair_block}
""".strip()

    text = llm_client.chat_text(
        messages=[{"role": "user", "content": prompt}],
        model=_resolve_nl2cypher_model(),
        temperature=0.0,
        max_tokens=NL2CYPHER_NUM_PREDICT,
        timeout_seconds=NL2CYPHER_TIMEOUT_SECONDS,
    )
    return strip_markdown_fence(text)


def validate_cypher_relationships(cypher: str, schema: SchemaSnapshot) -> None:
    if not schema.relationship_types:
        return

    relation_tokens = re.findall(r"\[[^\]]*:(`?[A-Za-z_][A-Za-z0-9_]*`?)", cypher)
    for token in relation_tokens:
        rel_name = token.strip("`")
        if rel_name not in schema.relationship_types:
            raise ValueError(f"Unknown relationship type: {rel_name}")


def execute_cypher(driver: Driver, cypher: str) -> List[Dict[str, Any]]:
    with driver.session() as session:
        result = session.run(cypher)
        return [dict(row) for row in result]


def answer_with_manual_prompt(question: str) -> Dict[str, Any]:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        schema = load_schema_snapshot(driver)
        entity_names = fetch_entity_names(driver)
        organization_names = fetch_organization_names(driver)
        is_finance_question = _question_is_finance(question)

        # Prevent incorrect founder answers for leadership-title questions when schema has no role signal.
        if _question_has_any_term(question, ROLE_TITLE_TERMS) and not _schema_supports_role_title(schema):
            return _empty_result(
                question,
                "Schema does not include chairman/CEO style relationship or property; return empty result.",
            )
        if is_finance_question and not _schema_supports_finance_metrics(schema):
            return _empty_result(
                question,
                "Schema does not include HAS_FINANCIAL_METRIC/FOR_PERIOD; finance question cannot be grounded.",
            )

        cypher = natural_language_to_cypher(question, schema=schema.schema_text)
        last_error = ""
        attempts = CYPHER_REPAIR_RETRIES + 1

        for attempt in range(attempts):
            linked_cypher = link_entity_literals(cypher, entity_names)
            # Leadership title queries must not silently degrade to founder queries.
            if _question_has_any_term(question, ROLE_TITLE_TERMS) and _cypher_uses_relationship(linked_cypher, "FOUNDED_BY"):
                last_error = "Semantic mismatch: leadership-title question cannot use FOUNDED_BY."
                if attempt + 1 >= attempts:
                    return _empty_result(question, last_error)
                cypher = natural_language_to_cypher(
                    question,
                    schema=schema.schema_text,
                    previous_cypher=linked_cypher,
                    error_message=last_error,
                )
                continue

            if _question_has_any_term(question, FOUNDER_TERMS) and _cypher_uses_relationship(linked_cypher, "HEADQUARTERED_IN"):
                last_error = "Semantic mismatch: founder question should prioritize FOUNDED_BY."
                if attempt + 1 >= attempts:
                    return _empty_result(question, last_error)
                cypher = natural_language_to_cypher(
                    question,
                    schema=schema.schema_text,
                    previous_cypher=linked_cypher,
                    error_message=last_error,
                )
                continue
            if is_finance_question and _cypher_uses_hardcoded_finance_constants(linked_cypher):
                last_error = "Finance query used hardcoded constant aliases instead of graph-bound fields."
                if attempt + 1 >= attempts:
                    fallback_cypher = _build_finance_template_cypher(question, organization_names)
                    fallback_rows = execute_cypher(driver, fallback_cypher)
                    return {
                        "question": question,
                        "cypher": fallback_cypher,
                        "rows": fallback_rows,
                        "attempt": attempt + 1,
                        "reason": f"{last_error} Applied finance fallback template.",
                    }
                cypher = natural_language_to_cypher(
                    question,
                    schema=schema.schema_text,
                    previous_cypher=linked_cypher,
                    error_message=last_error,
                )
                continue
            if is_finance_question and not _cypher_has_required_finance_path(linked_cypher):
                last_error = "Finance query must use HAS_FINANCIAL_METRIC and FOR_PERIOD relationships."
                if attempt + 1 >= attempts:
                    fallback_cypher = _build_finance_template_cypher(question, organization_names)
                    fallback_rows = execute_cypher(driver, fallback_cypher)
                    return {
                        "question": question,
                        "cypher": fallback_cypher,
                        "rows": fallback_rows,
                        "attempt": attempt + 1,
                        "reason": f"{last_error} Applied finance fallback template.",
                    }
                cypher = natural_language_to_cypher(
                    question,
                    schema=schema.schema_text,
                    previous_cypher=linked_cypher,
                    error_message=last_error,
                )
                continue

            try:
                validate_cypher_relationships(linked_cypher, schema)
                rows = execute_cypher(driver, linked_cypher)
                if not rows:
                    relaxed_cypher = _relax_exact_organization_name_match(linked_cypher)
                    if relaxed_cypher and relaxed_cypher != linked_cypher:
                        relaxed_rows = execute_cypher(driver, relaxed_cypher)
                        if relaxed_rows:
                            return {
                                "question": question,
                                "cypher": relaxed_cypher,
                                "rows": relaxed_rows,
                                "attempt": attempt + 1,
                            }

                    if attempt + 1 >= attempts:
                        if is_finance_question:
                            fallback_cypher = _build_finance_template_cypher(question, organization_names)
                            fallback_rows = execute_cypher(driver, fallback_cypher)
                            return {
                                "question": question,
                                "cypher": fallback_cypher,
                                "rows": fallback_rows,
                                "attempt": attempt + 1,
                                "reason": "Finance fallback template used after empty result.",
                            }
                        return {
                            "question": question,
                            "cypher": linked_cypher,
                            "rows": [],
                            "attempt": attempt + 1,
                        }

                    last_error = "Cypher executed but returned 0 rows; relax literal matching and retry."
                    cypher = natural_language_to_cypher(
                        question,
                        schema=schema.schema_text,
                        previous_cypher=linked_cypher,
                        error_message=last_error,
                    )
                    continue

                return {
                    "question": question,
                    "cypher": linked_cypher,
                    "rows": rows,
                    "attempt": attempt + 1,
                }
            except (Neo4jError, ValueError) as exc:
                last_error = str(exc)
                if attempt + 1 >= attempts:
                    break
                cypher = natural_language_to_cypher(
                    question,
                    schema=schema.schema_text,
                    previous_cypher=linked_cypher,
                    error_message=last_error,
                )

        if is_finance_question:
            fallback_cypher = _build_finance_template_cypher(question, organization_names)
            fallback_rows = execute_cypher(driver, fallback_cypher)
            return {
                "question": question,
                "cypher": fallback_cypher,
                "rows": fallback_rows,
                "attempt": attempts,
                "reason": f"Finance fallback template used after retries: {last_error}",
            }
        raise RuntimeError(f"Cypher generation failed after retries: {last_error}")
    finally:
        driver.close()


def main() -> None:
    questions = [
        "誰創立了台積電？",
        "台積電的總部在哪裡？",
        "台積電供應晶片給哪些公司？",
    ]

    print("=== Method 1: GraphCypherQAChain ===")
    for q in questions:
        try:
            result = query_with_graph_chain(q)
            if result is None:
                break
            print(f"Q: {q}")
            print(f"Cypher: {result['cypher']}")
            print(f"Answer: {result['answer']}")
            print("-")
        except Exception as exc:
            print(f"Graph chain failed for '{q}': {exc}")

    print("\n=== Method 2: Manual Prompting ===")
    for q in questions:
        try:
            result = answer_with_manual_prompt(q)
            print(f"Q: {q}")
            print(f"Cypher: {result['cypher']}")
            print(f"Rows: {result['rows']}")
            print("-")
        except Exception as exc:
            print(f"Manual prompting failed for '{q}': {exc}")


if __name__ == "__main__":
    main()
