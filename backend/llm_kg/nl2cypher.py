"""Natural language to Cypher helpers with schema grounding and repair loop.

Two methods:
1) LangChain GraphCypherQAChain (recommended when dependencies are available)
2) Manual schema-grounded prompting + execution/self-correction
"""

from __future__ import annotations

import json
import logging
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from neo4j import Driver, GraphDatabase
from neo4j.exceptions import Neo4jError
from backend.config.settings import (
    get_neo4j_settings,
    get_nl2cypher_settings,
    resolve_nl2cypher_provider as resolve_nl2cypher_provider_from_settings,
    resolve_nl2cypher_model as resolve_nl2cypher_model_from_settings,
)

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

_neo4j_settings = get_neo4j_settings()
_nl2cypher_settings = get_nl2cypher_settings()

NEO4J_URI = _neo4j_settings.uri
NEO4J_USER = _neo4j_settings.user
NEO4J_PASSWORD = _neo4j_settings.password
CYPHER_REPAIR_RETRIES = _nl2cypher_settings.cypher_repair_retries
ENTITY_LINK_THRESHOLD = _nl2cypher_settings.entity_link_threshold
NL2CYPHER_TIMEOUT_SECONDS = _nl2cypher_settings.timeout_seconds
NL2CYPHER_NUM_PREDICT = _nl2cypher_settings.num_predict
NL2CYPHER_AGENTIC_MAX_ROUNDS = _nl2cypher_settings.agentic_max_rounds
NL2CYPHER_AGENTIC_PLAN_TOKENS = _nl2cypher_settings.agentic_plan_tokens
NL2CYPHER_AGENTIC_REACT_TOKENS = _nl2cypher_settings.agentic_react_tokens
NL2CYPHER_AGENTIC_CRITIC_TOKENS = _nl2cypher_settings.agentic_critic_tokens
_AGENT_JSON_RETRY_LIMIT = 3

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
_UNION_TOKEN_PATTERN = re.compile(r"UNION(?:\s+ALL)?\b", flags=re.IGNORECASE)
_RETURN_ALIAS_PATTERN = re.compile(r"(?is)\bAS\s+(`[^`]+`|[A-Za-z_][A-Za-z0-9_]*)\s*$")
_RETURN_KEYWORD_PATTERN = re.compile(r"RETURN\b", flags=re.IGNORECASE)
_ORDER_BY_KEYWORD_PATTERN = re.compile(r"ORDER\s+BY\b", flags=re.IGNORECASE)
_SKIP_KEYWORD_PATTERN = re.compile(r"SKIP\b", flags=re.IGNORECASE)
_LIMIT_KEYWORD_PATTERN = re.compile(r"LIMIT\b", flags=re.IGNORECASE)
_ALLOWED_AGENTIC_STRATEGIES = {"single_query", "union", "staged_merge"}
_ALLOWED_CRITIC_VERDICTS = {"accept", "replan", "fail_fast"}
_AGENTIC_MAX_FAILURE_CHAIN = 20
_AGENTIC_REPEAT_CYPHER_LIMIT = 2
_logger = logging.getLogger(__name__)


@dataclass
class SchemaSnapshot:
    labels: List[str]
    relationship_types: List[str]
    properties: Dict[str, List[str]]
    schema_text: str


@dataclass
class DeterministicIssue:
    code: str
    message: str
    severity: str = "high"


def _question_has_any_term(question: str, terms: tuple[str, ...]) -> bool:
    """執行 `_question_has_any_term` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    q = question.lower()
    return any(term in q for term in terms)


def _schema_tokens(schema: SchemaSnapshot) -> List[str]:
    """執行 `_schema_tokens` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
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
    """執行 `_schema_supports_role_title` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    hints = [_normalize(x) for x in ROLE_TITLE_SCHEMA_HINTS if _normalize(x)]
    schema_terms = _schema_tokens(schema)
    for hint in hints:
        for term in schema_terms:
            if hint in term:
                return True
    return False


def _question_is_finance(question: str) -> bool:
    """執行 `_question_is_finance` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    return _question_has_any_term(question, FINANCE_TERMS)


def _schema_supports_finance_metrics(schema: SchemaSnapshot) -> bool:
    """執行 `_schema_supports_finance_metrics` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    rel_types = set(schema.relationship_types)
    return all(rel in rel_types for rel in FINANCE_REQUIRED_REL_TYPES)


def _cypher_has_required_finance_path(cypher: str) -> bool:
    """執行 `_cypher_has_required_finance_path` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    return all(_cypher_uses_relationship(cypher, rel) for rel in FINANCE_REQUIRED_REL_TYPES)


def _cypher_uses_hardcoded_finance_constants(cypher: str) -> bool:
    """執行 `_cypher_uses_hardcoded_finance_constants` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
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
    """執行 `_extract_finance_periods` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
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
    """執行 `_cypher_quote` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _guess_finance_org_hint(question: str, entity_names: List[str]) -> str:
    """執行 `_guess_finance_org_hint` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
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
    """執行 `_build_finance_template_cypher` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
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
    """執行 `_empty_result` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    return {
        "question": question,
        "cypher": "MATCH (n) WHERE false RETURN n LIMIT 0",
        "rows": [],
        "attempt": 0,
        "reason": reason,
    }


def _cypher_uses_relationship(cypher: str, rel_type: str) -> bool:
    """執行 `_cypher_uses_relationship` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    pattern = rf":\s*`?{re.escape(rel_type)}`?"
    return bool(re.search(pattern, cypher, flags=re.IGNORECASE))


def _relax_exact_organization_name_match(cypher: str) -> str | None:
    """執行 `_relax_exact_organization_name_match` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
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


def _resolve_nl2cypher_provider(provider_override: str | None = None) -> str | None:
    """執行 `_resolve_nl2cypher_provider` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    return resolve_nl2cypher_provider_from_settings(provider_override)


def _resolve_nl2cypher_model(*, provider_override: str | None = None, model_override: str | None = None) -> str | None:
    """執行 `_resolve_nl2cypher_model` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    return resolve_nl2cypher_model_from_settings(provider=provider_override, explicit_model=model_override)


def strip_markdown_fence(content: str) -> str:
    """執行 `strip_markdown_fence` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
    text = content.strip()
    match = re.match(r"^```(?:cypher|sql)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else text).strip()


def _normalize(value: str) -> str:
    """執行 `_normalize` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value.strip().lower())


def load_schema_snapshot(driver: Driver) -> SchemaSnapshot:
    """執行 `load_schema_snapshot` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
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
    """執行 `fetch_entity_names` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
    with driver.session() as session:
        rows = session.run("MATCH (e:Entity) WHERE e.name IS NOT NULL RETURN DISTINCT e.name AS name")
        return [str(row["name"]) for row in rows]


def fetch_organization_names(driver: Driver) -> List[str]:
    """執行 `fetch_organization_names` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
    with driver.session() as session:
        rows = session.run("MATCH (o:Organization) WHERE o.name IS NOT NULL RETURN DISTINCT o.name AS name")
        return [str(row["name"]) for row in rows]


def link_entity_literals(cypher: str, entity_names: List[str]) -> str:
    """執行 `link_entity_literals` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
    if not entity_names:
        return cypher

    entity_lookup = {_normalize(name): name for name in entity_names if _normalize(name)}

    def replace_literal(match: re.Match[str]) -> str:
        """執行 `replace_literal` 的主要流程。
        函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
        """
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


def _is_identifier_char(ch: str) -> bool:
    """執行 `_is_identifier_char` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    return ch.isalnum() or ch == "_"


def _split_union_query_parts(cypher: str) -> tuple[List[str], List[str]]:
    """執行 `_split_union_query_parts` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    parts: List[str] = []
    separators: List[str] = []

    i = 0
    start = 0
    n = len(cypher)
    depth_paren = 0
    depth_bracket = 0
    depth_brace = 0
    in_single_quote = False
    in_double_quote = False
    in_backtick = False

    while i < n:
        ch = cypher[i]

        if in_single_quote:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == "'":
                in_single_quote = False
            i += 1
            continue

        if in_double_quote:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == '"':
                in_double_quote = False
            i += 1
            continue

        if in_backtick:
            if ch == "`":
                in_backtick = False
            i += 1
            continue

        if ch == "'":
            in_single_quote = True
            i += 1
            continue
        if ch == '"':
            in_double_quote = True
            i += 1
            continue
        if ch == "`":
            in_backtick = True
            i += 1
            continue
        if ch == "(":
            depth_paren += 1
            i += 1
            continue
        if ch == ")":
            depth_paren = max(0, depth_paren - 1)
            i += 1
            continue
        if ch == "[":
            depth_bracket += 1
            i += 1
            continue
        if ch == "]":
            depth_bracket = max(0, depth_bracket - 1)
            i += 1
            continue
        if ch == "{":
            depth_brace += 1
            i += 1
            continue
        if ch == "}":
            depth_brace = max(0, depth_brace - 1)
            i += 1
            continue

        if depth_paren == 0 and depth_bracket == 0 and depth_brace == 0:
            if i > 0 and _is_identifier_char(cypher[i - 1]):
                i += 1
                continue
            match = _UNION_TOKEN_PATTERN.match(cypher, i)
            if match:
                part = cypher[start:i].strip()
                branch_idx = len(parts) + 1
                if not part:
                    raise ValueError(f"UNION branch {branch_idx} is empty.")
                parts.append(part)
                separators.append(match.group(0))
                i = match.end()
                start = i
                continue

        i += 1

    if not separators:
        return [cypher], []

    tail = cypher[start:].strip()
    branch_idx = len(parts) + 1
    if not tail:
        raise ValueError(f"UNION branch {branch_idx} is empty.")
    parts.append(tail)
    return parts, separators


def _match_keyword_at(text: str, index: int, pattern: re.Pattern[str]) -> re.Match[str] | None:
    """執行 `_match_keyword_at` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    if index > 0 and _is_identifier_char(text[index - 1]):
        return None
    return pattern.match(text, index)


def _find_return_projection_span(part: str) -> tuple[int, int]:
    """執行 `_find_return_projection_span` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    i = 0
    n = len(part)
    depth_paren = 0
    depth_bracket = 0
    depth_brace = 0
    in_single_quote = False
    in_double_quote = False
    in_backtick = False
    return_start: int | None = None

    while i < n:
        ch = part[i]

        if in_single_quote:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == "'":
                in_single_quote = False
            i += 1
            continue

        if in_double_quote:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == '"':
                in_double_quote = False
            i += 1
            continue

        if in_backtick:
            if ch == "`":
                in_backtick = False
            i += 1
            continue

        if ch == "'":
            in_single_quote = True
            i += 1
            continue
        if ch == '"':
            in_double_quote = True
            i += 1
            continue
        if ch == "`":
            in_backtick = True
            i += 1
            continue
        if ch == "(":
            depth_paren += 1
            i += 1
            continue
        if ch == ")":
            depth_paren = max(0, depth_paren - 1)
            i += 1
            continue
        if ch == "[":
            depth_bracket += 1
            i += 1
            continue
        if ch == "]":
            depth_bracket = max(0, depth_bracket - 1)
            i += 1
            continue
        if ch == "{":
            depth_brace += 1
            i += 1
            continue
        if ch == "}":
            depth_brace = max(0, depth_brace - 1)
            i += 1
            continue

        if depth_paren == 0 and depth_bracket == 0 and depth_brace == 0:
            match = _match_keyword_at(part, i, _RETURN_KEYWORD_PATTERN)
            if match:
                return_start = match.end()
                i = return_start
                break
        i += 1

    if return_start is None:
        raise ValueError("Missing RETURN clause.")

    projection_end = n
    i = return_start
    while i < n:
        ch = part[i]

        if in_single_quote:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == "'":
                in_single_quote = False
            i += 1
            continue

        if in_double_quote:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == '"':
                in_double_quote = False
            i += 1
            continue

        if in_backtick:
            if ch == "`":
                in_backtick = False
            i += 1
            continue

        if ch == "'":
            in_single_quote = True
            i += 1
            continue
        if ch == '"':
            in_double_quote = True
            i += 1
            continue
        if ch == "`":
            in_backtick = True
            i += 1
            continue
        if ch == "(":
            depth_paren += 1
            i += 1
            continue
        if ch == ")":
            depth_paren = max(0, depth_paren - 1)
            i += 1
            continue
        if ch == "[":
            depth_bracket += 1
            i += 1
            continue
        if ch == "]":
            depth_bracket = max(0, depth_bracket - 1)
            i += 1
            continue
        if ch == "{":
            depth_brace += 1
            i += 1
            continue
        if ch == "}":
            depth_brace = max(0, depth_brace - 1)
            i += 1
            continue

        if depth_paren == 0 and depth_bracket == 0 and depth_brace == 0:
            if _match_keyword_at(part, i, _ORDER_BY_KEYWORD_PATTERN):
                projection_end = i
                break
            if _match_keyword_at(part, i, _SKIP_KEYWORD_PATTERN):
                projection_end = i
                break
            if _match_keyword_at(part, i, _LIMIT_KEYWORD_PATTERN):
                projection_end = i
                break
        i += 1

    return return_start, projection_end


def _extract_return_projection(part: str) -> str:
    """執行 `_extract_return_projection` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    start, end = _find_return_projection_span(part)
    projection = part[start:end].strip()
    if not projection:
        raise ValueError("RETURN projection is empty.")
    return projection


def _split_top_level_expressions(projection: str) -> List[str]:
    """執行 `_split_top_level_expressions` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    expressions: List[str] = []
    i = 0
    start = 0
    n = len(projection)
    depth_paren = 0
    depth_bracket = 0
    depth_brace = 0
    in_single_quote = False
    in_double_quote = False
    in_backtick = False

    while i < n:
        ch = projection[i]

        if in_single_quote:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == "'":
                in_single_quote = False
            i += 1
            continue

        if in_double_quote:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == '"':
                in_double_quote = False
            i += 1
            continue

        if in_backtick:
            if ch == "`":
                in_backtick = False
            i += 1
            continue

        if ch == "'":
            in_single_quote = True
            i += 1
            continue
        if ch == '"':
            in_double_quote = True
            i += 1
            continue
        if ch == "`":
            in_backtick = True
            i += 1
            continue
        if ch == "(":
            depth_paren += 1
            i += 1
            continue
        if ch == ")":
            depth_paren = max(0, depth_paren - 1)
            i += 1
            continue
        if ch == "[":
            depth_bracket += 1
            i += 1
            continue
        if ch == "]":
            depth_bracket = max(0, depth_bracket - 1)
            i += 1
            continue
        if ch == "{":
            depth_brace += 1
            i += 1
            continue
        if ch == "}":
            depth_brace = max(0, depth_brace - 1)
            i += 1
            continue

        if ch == "," and depth_paren == 0 and depth_bracket == 0 and depth_brace == 0:
            expressions.append(projection[start:i].strip())
            start = i + 1
        i += 1

    expressions.append(projection[start:].strip())
    return expressions


def _extract_return_alias(expr: str) -> str | None:
    """執行 `_extract_return_alias` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    match = _RETURN_ALIAS_PATTERN.search(expr.strip())
    if not match:
        return None
    return match.group(1)


def _replace_return_alias(expr: str, alias: str) -> str:
    """執行 `_replace_return_alias` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    stripped = expr.strip()
    match = _RETURN_ALIAS_PATTERN.search(stripped)
    if not match:
        raise ValueError("RETURN expression must use explicit AS alias.")
    return stripped[: match.start(1)] + alias + stripped[match.end(1) :]


def _normalize_union_return_aliases(cypher: str) -> str:
    """執行 `_normalize_union_return_aliases` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    parts, separators = _split_union_query_parts(cypher)
    if not separators:
        return cypher

    canonical_aliases: List[str] = []
    normalized_parts: List[str] = []
    changed = False

    for branch_idx, part in enumerate(parts, start=1):
        try:
            return_start, return_end = _find_return_projection_span(part)
            projection = _extract_return_projection(part)
        except ValueError as exc:
            raise ValueError(f"UNION branch {branch_idx} is missing RETURN projection.") from exc

        distinct_prefix = ""
        projection_body = projection
        distinct_match = re.match(r"(?is)^\s*(DISTINCT\b)\s*", projection)
        if distinct_match:
            distinct_prefix = distinct_match.group(1)
            projection_body = projection[distinct_match.end() :].strip()

        expressions = _split_top_level_expressions(projection_body)
        if not expressions or any(not expr for expr in expressions):
            raise ValueError(f"UNION branch {branch_idx} has empty RETURN projection expression.")

        aliases: List[str] = []
        for expr_idx, expr in enumerate(expressions, start=1):
            alias = _extract_return_alias(expr)
            if not alias:
                raise ValueError(
                    f"UNION branch {branch_idx} RETURN expression {expr_idx} must use explicit AS alias."
                )
            aliases.append(alias)

        if branch_idx == 1:
            canonical_aliases = aliases
            normalized_parts.append(part)
            continue

        if len(aliases) != len(canonical_aliases):
            raise ValueError(
                f"UNION branch {branch_idx} RETURN projection count mismatch: "
                f"expected {len(canonical_aliases)}, got {len(aliases)}."
            )

        aligned_expressions: List[str] = []
        for expr_idx, expr in enumerate(expressions):
            canonical_alias = canonical_aliases[expr_idx]
            if aliases[expr_idx] != canonical_alias:
                changed = True
            aligned_expressions.append(_replace_return_alias(expr, canonical_alias))

        aligned_projection = ", ".join(aligned_expressions)
        if distinct_prefix:
            aligned_projection = f"{distinct_prefix} {aligned_projection}"

        original_slice = part[return_start:return_end]
        leading_len = len(original_slice) - len(original_slice.lstrip())
        trailing_len = len(original_slice) - len(original_slice.rstrip())
        leading_ws = original_slice[:leading_len]
        trailing_ws = original_slice[len(original_slice) - trailing_len :] if trailing_len else ""
        replacement = f"{leading_ws}{aligned_projection}{trailing_ws}"
        normalized_parts.append(part[:return_start] + replacement + part[return_end:])

    if not changed:
        return cypher

    rebuilt = normalized_parts[0]
    for separator, part in zip(separators, normalized_parts[1:]):
        rebuilt = f"{rebuilt}\n{separator}\n{part}"
    return rebuilt


def query_with_graph_chain(question: str) -> Optional[Dict[str, Any]]:
    """執行 `query_with_graph_chain` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
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
    nl2cypher_provider: str | None = None,
    nl2cypher_model: str | None = None,
) -> str:
    """執行 `natural_language_to_cypher` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
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
9. 若使用 UNION/UNION ALL，各分支 RETURN 欄位數量、順序、欄位別名必須完全一致。
10. 僅回傳 Cypher，不要其他文字與 markdown。

使用者問題：{question}
{repair_block}
""".strip()

    resolved_provider = _resolve_nl2cypher_provider(nl2cypher_provider)
    resolved_model = _resolve_nl2cypher_model(
        provider_override=resolved_provider,
        model_override=nl2cypher_model,
    )
    text = llm_client.chat_text(
        messages=[{"role": "user", "content": prompt}],
        provider=resolved_provider,
        model=resolved_model,
        temperature=0.0,
        max_tokens=NL2CYPHER_NUM_PREDICT,
        timeout_seconds=NL2CYPHER_TIMEOUT_SECONDS,
    )
    return strip_markdown_fence(text)


def validate_cypher_relationships(cypher: str, schema: SchemaSnapshot) -> None:
    """執行 `validate_cypher_relationships` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
    if not schema.relationship_types:
        return

    relation_tokens = re.findall(r"\[[^\]]*:(`?[A-Za-z_][A-Za-z0-9_]*`?)", cypher)
    for token in relation_tokens:
        rel_name = token.strip("`")
        if rel_name not in schema.relationship_types:
            raise ValueError(f"Unknown relationship type: {rel_name}")


def execute_cypher(driver: Driver, cypher: str) -> List[Dict[str, Any]]:
    """執行 `execute_cypher` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
    with driver.session() as session:
        result = session.run(cypher)
        return [dict(row) for row in result]


def _issue_to_dict(issue: DeterministicIssue) -> Dict[str, str]:
    """執行 `_issue_to_dict` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    return {"code": issue.code, "message": issue.message, "severity": issue.severity}


def _append_failure(failure_chain: List[str], message: str) -> None:
    """執行 `_append_failure` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    text = str(message or "").strip()
    if not text:
        return
    failure_chain.append(text[:1024])
    overflow = len(failure_chain) - _AGENTIC_MAX_FAILURE_CHAIN
    if overflow > 0:
        del failure_chain[:overflow]


def _clean_string_list(value: Any, limit: int = 12) -> List[str]:
    """執行 `_clean_string_list` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    if not isinstance(value, list):
        return []
    cleaned: List[str] = []
    for item in value:
        text = str(item).strip()
        if not text:
            continue
        cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def _normalize_strategy(strategy: str) -> str:
    """執行 `_normalize_strategy` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    normalized = strategy.strip().lower()
    if normalized in _ALLOWED_AGENTIC_STRATEGIES:
        return normalized
    return "single_query"


def _normalize_critic_verdict(verdict: str) -> str:
    """執行 `_normalize_critic_verdict` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    normalized = verdict.strip().lower()
    if normalized in _ALLOWED_CRITIC_VERDICTS:
        return normalized
    return "replan"


def _normalize_severity(severity: str) -> str:
    """執行 `_normalize_severity` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    normalized = severity.strip().lower()
    if normalized in {"low", "med", "high"}:
        return normalized
    return "med"


def _default_agentic_plan(question: str, is_finance_question: bool) -> Dict[str, Any]:
    """執行 `_default_agentic_plan` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    must_have_paths: List[str] = []
    if is_finance_question:
        must_have_paths.append(
            "(o:Organization)-[:HAS_FINANCIAL_METRIC]->(m:FinancialMetric)-[:FOR_PERIOD]->(p:FiscalPeriod)"
        )
    return {
        "intent": question.strip() or "unknown",
        "strategy": "single_query",
        "must_have_paths": must_have_paths,
        "forbidden_patterns": ["hardcoded constant aliases for finance answers"],
        "output_contract": {"columns": []},
        "risk_hypotheses": ["schema mismatch", "wrong relationship", "union projection mismatch"],
    }


def _sanitize_planner_payload(payload: Dict[str, Any], *, question: str, is_finance_question: bool) -> Dict[str, Any]:
    """執行 `_sanitize_planner_payload` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    plan = _default_agentic_plan(question, is_finance_question)
    plan["intent"] = str(payload.get("intent") or plan["intent"]).strip() or plan["intent"]
    plan["strategy"] = _normalize_strategy(str(payload.get("strategy") or plan["strategy"]))
    plan["must_have_paths"] = _clean_string_list(payload.get("must_have_paths"))
    plan["forbidden_patterns"] = _clean_string_list(payload.get("forbidden_patterns"))
    plan["risk_hypotheses"] = _clean_string_list(payload.get("risk_hypotheses"))

    output_contract = payload.get("output_contract")
    if isinstance(output_contract, dict):
        plan["output_contract"] = {"columns": _clean_string_list(output_contract.get("columns"))}

    if is_finance_question and not any("HAS_FINANCIAL_METRIC" in p for p in plan["must_have_paths"]):
        plan["must_have_paths"].append(
            "(o:Organization)-[:HAS_FINANCIAL_METRIC]->(m:FinancialMetric)-[:FOR_PERIOD]->(p:FiscalPeriod)"
        )
    return plan


def _sanitize_reactor_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """執行 `_sanitize_reactor_payload` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    cypher = strip_markdown_fence(str(payload.get("cypher") or "")).strip()
    if not cypher:
        raise ValueError("Reactor returned empty cypher.")

    self_checks_raw = payload.get("self_checks")
    self_checks = {"schema_grounded": False, "projection_consistent": False}
    if isinstance(self_checks_raw, dict):
        self_checks["schema_grounded"] = bool(self_checks_raw.get("schema_grounded"))
        self_checks["projection_consistent"] = bool(self_checks_raw.get("projection_consistent"))

    return {
        "cypher": cypher,
        "assumptions": _clean_string_list(payload.get("assumptions")),
        "self_checks": self_checks,
    }


def _sanitize_critic_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """執行 `_sanitize_critic_payload` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    issues: List[Dict[str, str]] = []
    raw_issues = payload.get("issues")
    if isinstance(raw_issues, list):
        for item in raw_issues[:12]:
            if not isinstance(item, dict):
                continue
            message = str(item.get("message") or "").strip()
            if not message:
                continue
            issues.append(
                {
                    "code": str(item.get("code") or "UNKNOWN").strip() or "UNKNOWN",
                    "message": message,
                    "severity": _normalize_severity(str(item.get("severity") or "med")),
                }
            )

    return {
        "verdict": _normalize_critic_verdict(str(payload.get("verdict") or "")),
        "issues": issues,
        "repair_actions": _clean_string_list(payload.get("repair_actions")),
        "next_strategy": _normalize_strategy(str(payload.get("next_strategy") or "single_query")),
    }


def _sanitize_replanner_payload(payload: Dict[str, Any], *, fallback_strategy: str) -> Dict[str, Any]:
    """執行 `_sanitize_replanner_payload` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    return {
        "strategy": _normalize_strategy(str(payload.get("strategy") or fallback_strategy)),
        "delta_actions": _clean_string_list(payload.get("delta_actions")),
        "tightened_constraints": _clean_string_list(payload.get("tightened_constraints")),
        "stop_if": _clean_string_list(payload.get("stop_if")),
    }


def _call_agent_json(
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    nl2cypher_provider: str | None = None,
    nl2cypher_model: str | None = None,
) -> Dict[str, Any]:
    """執行 `_call_agent_json` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    resolved_provider = _resolve_nl2cypher_provider(nl2cypher_provider)
    resolved_model = _resolve_nl2cypher_model(
        provider_override=resolved_provider,
        model_override=nl2cypher_model,
    )
    retry_tokens = max(1, int(max_tokens))
    last_error: Exception | None = None

    for attempt in range(1, _AGENT_JSON_RETRY_LIMIT + 1):
        retry_prompt = user_prompt
        if attempt > 1 and last_error is not None:
            retry_prompt = (
                f"{user_prompt}\n\n"
                "前次輸出無法解析為合法 JSON object。"
                "請只輸出一個 JSON 物件，不要 markdown、不要註解、不要多餘文字。\n"
                f"parse_error={last_error}"
            )

        try:
            parsed = llm_client.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": retry_prompt},
                ],
                provider=resolved_provider,
                model=resolved_model,
                temperature=0.0,
                max_tokens=retry_tokens,
                timeout_seconds=NL2CYPHER_TIMEOUT_SECONDS,
            )
            if not isinstance(parsed, dict):
                raise ValueError("Agent JSON response must be an object.")
            return parsed
        except Exception as exc:
            last_error = exc
            message = str(exc)
            if "GEMINI_API_KEY is required" in message:
                raise
            if attempt >= _AGENT_JSON_RETRY_LIMIT:
                raise
            retry_tokens = min(max(retry_tokens * 2, retry_tokens + 256), 8192)

    raise RuntimeError(f"Agent JSON call failed after retries: {last_error}")


def _run_planner_agent(
    *,
    question: str,
    schema: SchemaSnapshot,
    is_finance_question: bool,
    failure_chain: List[str],
    nl2cypher_provider: str | None = None,
    nl2cypher_model: str | None = None,
) -> Dict[str, Any]:
    """執行 `_run_planner_agent` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    default_plan = _default_agentic_plan(question, is_finance_question)
    system_prompt = (
        "你是 NL2Cypher Planner。"
        "請以 strict JSON 規劃查詢策略，只能回傳 JSON object。"
    )
    user_prompt = (
        "輸出 schema:\n"
        '{"intent":"...", "strategy":"single_query|union|staged_merge",'
        '"must_have_paths":[...], "forbidden_patterns":[...],'
        '"output_contract":{"columns":[...]}, "risk_hypotheses":[...]}\n\n'
        f"Question:\n{question}\n\n"
        f"Schema:\n{schema.schema_text}\n\n"
        f"is_finance_question={is_finance_question}\n"
        f"failure_chain={json.dumps(failure_chain[-5:], ensure_ascii=False)}\n"
        f"default_plan={json.dumps(default_plan, ensure_ascii=False)}"
    )
    payload = _call_agent_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=NL2CYPHER_AGENTIC_PLAN_TOKENS,
        nl2cypher_provider=nl2cypher_provider,
        nl2cypher_model=nl2cypher_model,
    )
    return _sanitize_planner_payload(payload, question=question, is_finance_question=is_finance_question)


def _run_reactor_agent(
    *,
    question: str,
    schema: SchemaSnapshot,
    plan: Dict[str, Any],
    previous_cypher: str,
    last_error: str,
    round_idx: int,
    nl2cypher_provider: str | None = None,
    nl2cypher_model: str | None = None,
) -> Dict[str, Any]:
    """執行 `_run_reactor_agent` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    system_prompt = (
        "你是 NL2Cypher Reactor。"
        "依照規劃產生可執行 Cypher，只能回傳 strict JSON object。"
    )
    user_prompt = (
        "輸出 schema:\n"
        '{"cypher":"...", "assumptions":[...], "self_checks":{"schema_grounded":true, "projection_consistent":true}}\n\n'
        f"Round={round_idx}\n"
        f"Question:\n{question}\n\n"
        f"Plan:\n{json.dumps(plan, ensure_ascii=False)}\n\n"
        f"Schema:\n{schema.schema_text}\n\n"
        f"Previous cypher:\n{previous_cypher or '(none)'}\n\n"
        f"Last error:\n{last_error or '(none)'}\n\n"
        "注意：僅輸出 JSON，不要 markdown。"
    )
    payload = _call_agent_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=NL2CYPHER_AGENTIC_REACT_TOKENS,
        nl2cypher_provider=nl2cypher_provider,
        nl2cypher_model=nl2cypher_model,
    )
    return _sanitize_reactor_payload(payload)


def _fallback_critic_decision(
    *,
    rows: List[Dict[str, Any]],
    deterministic_issues: List[DeterministicIssue],
    runtime_error: str,
    round_idx: int,
    repeated: bool,
) -> Dict[str, Any]:
    """執行 `_fallback_critic_decision` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    if rows and not deterministic_issues and not runtime_error:
        return {"verdict": "accept", "issues": [], "repair_actions": [], "next_strategy": "single_query"}

    issues: List[Dict[str, str]] = [_issue_to_dict(issue) for issue in deterministic_issues]
    if runtime_error:
        issues.append({"code": "RUNTIME_ERROR", "message": runtime_error, "severity": "high"})
    if repeated:
        issues.append(
            {
                "code": "NO_PROGRESS_REPEAT_CYPHER",
                "message": "Agent loop generated repeated Cypher without progress.",
                "severity": "high",
            }
        )
    verdict = "replan"
    if round_idx >= NL2CYPHER_AGENTIC_MAX_ROUNDS or repeated:
        verdict = "fail_fast"
    return {
        "verdict": verdict,
        "issues": issues,
        "repair_actions": ["tighten constraints and avoid previous error pattern"],
        "next_strategy": "single_query",
    }


def _run_critic_agent(
    *,
    question: str,
    plan: Dict[str, Any],
    cypher: str,
    rows: List[Dict[str, Any]],
    deterministic_issues: List[DeterministicIssue],
    runtime_error: str,
    round_idx: int,
    repeated: bool,
    nl2cypher_provider: str | None = None,
    nl2cypher_model: str | None = None,
) -> Dict[str, Any]:
    """執行 `_run_critic_agent` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    fallback = _fallback_critic_decision(
        rows=rows,
        deterministic_issues=deterministic_issues,
        runtime_error=runtime_error,
        round_idx=round_idx,
        repeated=repeated,
    )
    system_prompt = (
        "你是 NL2Cypher Critic。"
        "判斷目前結果要 accept / replan / fail_fast，只能回傳 strict JSON object。"
    )
    user_prompt = (
        "輸出 schema:\n"
        '{"verdict":"accept|replan|fail_fast", "issues":[{"code":"...", "message":"...", "severity":"low|med|high"}],'
        '"repair_actions":[...], "next_strategy":"single_query|union|staged_merge"}\n\n'
        f"Question:\n{question}\n\n"
        f"Plan:\n{json.dumps(plan, ensure_ascii=False)}\n\n"
        f"Cypher:\n{cypher}\n\n"
        f"Rows count: {len(rows)}\n"
        f"Deterministic issues:\n{json.dumps([_issue_to_dict(x) for x in deterministic_issues], ensure_ascii=False)}\n"
        f"Runtime error: {runtime_error or '(none)'}\n"
        f"Repeated cypher: {repeated}\n"
        f"Round: {round_idx}/{NL2CYPHER_AGENTIC_MAX_ROUNDS}\n"
        f"Fallback decision:\n{json.dumps(fallback, ensure_ascii=False)}"
    )
    payload = _call_agent_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=NL2CYPHER_AGENTIC_CRITIC_TOKENS,
        nl2cypher_provider=nl2cypher_provider,
        nl2cypher_model=nl2cypher_model,
    )
    return _sanitize_critic_payload(payload)


def _fallback_replan(plan: Dict[str, Any], last_error: str) -> Dict[str, Any]:
    """執行 `_fallback_replan` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    current = _normalize_strategy(str(plan.get("strategy") or "single_query"))
    strategy = current
    if "union" in (last_error or "").lower():
        strategy = "single_query"
    elif current == "single_query":
        strategy = "staged_merge"
    else:
        strategy = "single_query"
    return {
        "strategy": strategy,
        "delta_actions": ["avoid previous invalid projection and tighten schema grounding"],
        "tightened_constraints": ["must pass deterministic guards"],
        "stop_if": ["same cypher repeated"],
    }


def _run_replanner_agent(
    *,
    question: str,
    plan: Dict[str, Any],
    critic_payload: Dict[str, Any],
    last_error: str,
    failure_chain: List[str],
    nl2cypher_provider: str | None = None,
    nl2cypher_model: str | None = None,
) -> Dict[str, Any]:
    """執行 `_run_replanner_agent` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    fallback = _fallback_replan(plan, last_error)
    system_prompt = (
        "你是 NL2Cypher Replanner。"
        "依 critic 結果調整下一輪策略，只能回傳 strict JSON object。"
    )
    user_prompt = (
        "輸出 schema:\n"
        '{"strategy":"single_query|union|staged_merge", "delta_actions":[...],'
        '"tightened_constraints":[...], "stop_if":[...]}\n\n'
        f"Question:\n{question}\n\n"
        f"Current plan:\n{json.dumps(plan, ensure_ascii=False)}\n\n"
        f"Critic:\n{json.dumps(critic_payload, ensure_ascii=False)}\n\n"
        f"Last error:\n{last_error or '(none)'}\n"
        f"Failure chain:\n{json.dumps(failure_chain[-6:], ensure_ascii=False)}\n"
        f"Fallback:\n{json.dumps(fallback, ensure_ascii=False)}"
    )
    payload = _call_agent_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=NL2CYPHER_AGENTIC_CRITIC_TOKENS,
        nl2cypher_provider=nl2cypher_provider,
        nl2cypher_model=nl2cypher_model,
    )
    return _sanitize_replanner_payload(payload, fallback_strategy=fallback["strategy"])


def _merge_replan_into_plan(plan: Dict[str, Any], replan_payload: Dict[str, Any]) -> Dict[str, Any]:
    """執行 `_merge_replan_into_plan` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    merged = deepcopy(plan)
    merged["strategy"] = _normalize_strategy(str(replan_payload.get("strategy") or merged.get("strategy") or "single_query"))
    merged["risk_hypotheses"] = _clean_string_list(
        list(merged.get("risk_hypotheses", [])) + list(replan_payload.get("tightened_constraints", []))
    )
    merged["forbidden_patterns"] = _clean_string_list(
        list(merged.get("forbidden_patterns", [])) + list(replan_payload.get("stop_if", []))
    )
    return merged


def _issues_to_text(issues: List[DeterministicIssue], runtime_error: str, critic_payload: Dict[str, Any]) -> str:
    """執行 `_issues_to_text` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    messages: List[str] = []
    if runtime_error:
        messages.append(runtime_error)
    messages.extend(issue.message for issue in issues if issue.message)
    for item in critic_payload.get("issues", []):
        if isinstance(item, dict):
            text = str(item.get("message") or "").strip()
            if text:
                messages.append(text)
    deduped = list(dict.fromkeys(messages))
    return "; ".join(deduped)


def _run_deterministic_checks(
    *,
    question: str,
    cypher: str,
    schema: SchemaSnapshot,
    is_finance_question: bool,
) -> tuple[str, List[DeterministicIssue]]:
    """執行 `_run_deterministic_checks` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    issues: List[DeterministicIssue] = []
    normalized_cypher = cypher

    try:
        normalized_cypher = _normalize_union_return_aliases(normalized_cypher)
    except ValueError as exc:
        issues.append(DeterministicIssue(code="UNION_RETURN_SHAPE", message=str(exc), severity="high"))
        return normalized_cypher, issues

    try:
        validate_cypher_relationships(normalized_cypher, schema)
    except ValueError as exc:
        issues.append(DeterministicIssue(code="UNKNOWN_RELATIONSHIP", message=str(exc), severity="high"))

    if _question_has_any_term(question, ROLE_TITLE_TERMS) and _cypher_uses_relationship(normalized_cypher, "FOUNDED_BY"):
        issues.append(
            DeterministicIssue(
                code="SEMANTIC_REL_MISMATCH",
                message="Semantic mismatch: leadership-title question cannot use FOUNDED_BY.",
                severity="high",
            )
        )
    if _question_has_any_term(question, FOUNDER_TERMS) and _cypher_uses_relationship(normalized_cypher, "HEADQUARTERED_IN"):
        issues.append(
            DeterministicIssue(
                code="SEMANTIC_REL_MISMATCH",
                message="Semantic mismatch: founder question should prioritize FOUNDED_BY.",
                severity="med",
            )
        )

    if is_finance_question and _cypher_uses_hardcoded_finance_constants(normalized_cypher):
        issues.append(
            DeterministicIssue(
                code="FINANCE_HARDCODED_CONSTANTS",
                message="Finance query used hardcoded constant aliases instead of graph-bound fields.",
                severity="high",
            )
        )
    if is_finance_question and not _cypher_has_required_finance_path(normalized_cypher):
        issues.append(
            DeterministicIssue(
                code="FINANCE_REQUIRED_PATH_MISSING",
                message="Finance query must use HAS_FINANCIAL_METRIC and FOR_PERIOD relationships.",
                severity="high",
            )
        )

    return normalized_cypher, issues


def _run_agentic_query_loop(
    *,
    question: str,
    schema: SchemaSnapshot,
    driver: Driver,
    entity_names: List[str],
    organization_names: List[str],
    is_finance_question: bool,
    nl2cypher_provider: str | None = None,
    nl2cypher_model: str | None = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """執行 `_run_agentic_query_loop` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    question_id = str(uuid4())
    started = time.perf_counter()
    failure_chain: List[str] = []
    resolved_provider = _resolve_nl2cypher_provider(nl2cypher_provider)
    resolved_model = _resolve_nl2cypher_model(
        provider_override=resolved_provider,
        model_override=nl2cypher_model,
    )
    plan = _default_agentic_plan(question, is_finance_question)
    trace: Dict[str, Any] = {
        "stage": "planner",
        "round_count": 0,
        "replan_count": 0,
        "final_strategy": plan["strategy"],
        "failure_chain": [],
        "llm_provider": resolved_provider,
        "llm_model": resolved_model,
    }

    def _emit_progress(stage: str, detail: str = "") -> None:
        """執行 `_emit_progress` 的內部輔助流程。
        此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
        """
        if not progress_callback:
            return
        payload = {
            "type": "agentic_progress",
            "stage": stage,
            "round_count": int(trace.get("round_count", 0)),
            "replan_count": int(trace.get("replan_count", 0)),
            "final_strategy": str(trace.get("final_strategy", "single_query")),
            "failure_chain": list(failure_chain),
            "detail": detail,
            "llm_provider": resolved_provider,
            "llm_model": resolved_model,
        }
        try:
            progress_callback(payload)
        except Exception:
            return

    _emit_progress("planner", "Planning query strategy")

    try:
        plan = _run_planner_agent(
            question=question,
            schema=schema,
            is_finance_question=is_finance_question,
            failure_chain=failure_chain,
            nl2cypher_provider=resolved_provider,
            nl2cypher_model=resolved_model,
        )
    except Exception as exc:
        _append_failure(failure_chain, f"planner_error: {exc}")
        _emit_progress("planner", "Planner failed, fallback strategy in use")

    trace["final_strategy"] = plan["strategy"]
    planner_detail = f"Planner strategy={trace['final_strategy']}"
    if resolved_provider:
        planner_detail += f", provider={resolved_provider}"
    if resolved_model:
        planner_detail += f", model={resolved_model}"
    _emit_progress("planner", planner_detail)
    previous_cypher = ""
    last_error = ""
    seen_cypher_counts: Dict[str, int] = {}

    for round_idx in range(1, NL2CYPHER_AGENTIC_MAX_ROUNDS + 1):
        trace["stage"] = "react"
        trace["round_count"] = round_idx
        _emit_progress("react", f"Round {round_idx}: generating candidate cypher")
        reactor_payload: Dict[str, Any]
        try:
            reactor_payload = _run_reactor_agent(
                question=question,
                schema=schema,
                plan=plan,
                previous_cypher=previous_cypher,
                last_error=last_error,
                round_idx=round_idx,
                nl2cypher_provider=resolved_provider,
                nl2cypher_model=resolved_model,
            )
        except Exception as exc:
            _append_failure(failure_chain, f"reactor_error(r{round_idx}): {exc}")
            _emit_progress("react", f"Round {round_idx}: reactor failed, using fallback")
            try:
                fallback_cypher = natural_language_to_cypher(
                    question=question,
                    schema=schema.schema_text,
                    previous_cypher=previous_cypher or None,
                    error_message=str(exc),
                    nl2cypher_provider=resolved_provider,
                    nl2cypher_model=resolved_model,
                )
            except Exception as fallback_exc:
                _append_failure(failure_chain, f"reactor_fallback_error(r{round_idx}): {fallback_exc}")
                fallback_cypher = previous_cypher or "MATCH (n) WHERE false RETURN n LIMIT 0"
            reactor_payload = {
                "cypher": fallback_cypher,
                "assumptions": ["fallback via natural_language_to_cypher"],
                "self_checks": {"schema_grounded": False, "projection_consistent": False},
            }

        candidate_cypher = str(reactor_payload.get("cypher") or "").strip()
        linked_cypher = link_entity_literals(candidate_cypher, entity_names)
        checked_cypher, deterministic_issues = _run_deterministic_checks(
            question=question,
            cypher=linked_cypher,
            schema=schema,
            is_finance_question=is_finance_question,
        )

        count = seen_cypher_counts.get(checked_cypher, 0) + 1
        seen_cypher_counts[checked_cypher] = count
        repeated = count > _AGENTIC_REPEAT_CYPHER_LIMIT
        if repeated:
            deterministic_issues.append(
                DeterministicIssue(
                    code="NO_PROGRESS_REPEAT_CYPHER",
                    message="Agent loop generated repeated Cypher without progress.",
                    severity="high",
                )
            )
            _emit_progress("critic", f"Round {round_idx}: repeated cypher detected")

        rows: List[Dict[str, Any]] = []
        runtime_error = ""
        if not deterministic_issues:
            try:
                rows = execute_cypher(driver, checked_cypher)
                if not rows:
                    relaxed_cypher = _relax_exact_organization_name_match(checked_cypher)
                    if relaxed_cypher and relaxed_cypher != checked_cypher:
                        relaxed_rows = execute_cypher(driver, relaxed_cypher)
                        if relaxed_rows:
                            checked_cypher = relaxed_cypher
                            rows = relaxed_rows
            except Neo4jError as exc:
                runtime_error = str(exc)

        trace["stage"] = "critic"
        _emit_progress("critic", f"Round {round_idx}: evaluating candidate")
        critic_payload: Dict[str, Any]
        try:
            critic_payload = _run_critic_agent(
                question=question,
                plan=plan,
                cypher=checked_cypher,
                rows=rows,
                deterministic_issues=deterministic_issues,
                runtime_error=runtime_error,
                round_idx=round_idx,
                repeated=repeated,
                nl2cypher_provider=resolved_provider,
                nl2cypher_model=resolved_model,
            )
        except Exception as exc:
            _append_failure(failure_chain, f"critic_error(r{round_idx}): {exc}")
            _emit_progress("critic", f"Round {round_idx}: critic failed, using fallback decision")
            critic_payload = _fallback_critic_decision(
                rows=rows,
                deterministic_issues=deterministic_issues,
                runtime_error=runtime_error,
                round_idx=round_idx,
                repeated=repeated,
            )

        verdict = str(critic_payload.get("verdict") or "replan")
        round_error = _issues_to_text(deterministic_issues, runtime_error, critic_payload)
        last_error = round_error or "agentic loop requested replan"

        _logger.info(
            "agentic_nl2cypher_round %s",
            json.dumps(
                {
                    "question_id": question_id,
                    "round": round_idx,
                    "verdict": verdict,
                    "error_code": (critic_payload.get("issues") or [{}])[0].get("code", "") if critic_payload.get("issues") else "",
                    "strategy": plan.get("strategy", "single_query"),
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                },
                ensure_ascii=False,
            ),
        )

        if verdict == "accept" and rows:
            trace["stage"] = "done"
            trace["final_strategy"] = plan.get("strategy", "single_query")
            trace["failure_chain"] = list(failure_chain)
            _emit_progress("done", f"Completed in round {round_idx}")
            return {
                "question": question,
                "cypher": checked_cypher,
                "rows": rows,
                "attempt": round_idx,
                "agentic_trace": trace,
            }

        if verdict == "fail_fast":
            trace["stage"] = "fail_fast"
            _emit_progress("fail_fast", f"Stopped at round {round_idx}")
            break

        trace["stage"] = "replan"
        trace["replan_count"] = int(trace.get("replan_count", 0)) + 1
        _emit_progress("replan", f"Round {round_idx}: replanning")
        try:
            replan_payload = _run_replanner_agent(
                question=question,
                plan=plan,
                critic_payload=critic_payload,
                last_error=last_error,
                failure_chain=failure_chain,
                nl2cypher_provider=resolved_provider,
                nl2cypher_model=resolved_model,
            )
        except Exception as exc:
            _append_failure(failure_chain, f"replanner_error(r{round_idx}): {exc}")
            _emit_progress("replan", f"Round {round_idx}: replanner failed, using fallback")
            replan_payload = _fallback_replan(plan, last_error)

        plan = _merge_replan_into_plan(plan, replan_payload)
        trace["final_strategy"] = plan.get("strategy", "single_query")
        previous_cypher = checked_cypher
        _emit_progress("replan", f"Round {round_idx}: next strategy={trace['final_strategy']}")

    trace["stage"] = "exhausted"
    trace["final_strategy"] = plan.get("strategy", "single_query")
    trace["failure_chain"] = list(failure_chain)
    _emit_progress("exhausted", "Reached max rounds without accepted result")

    _logger.info(
        "agentic_nl2cypher_final %s",
        json.dumps(
            {
                "question_id": question_id,
                "round": trace.get("round_count", 0),
                "verdict": "fallback" if is_finance_question else "error",
                "error_code": "RETRY_EXHAUSTED",
                "strategy": trace.get("final_strategy", "single_query"),
                "latency_ms": int((time.perf_counter() - started) * 1000),
            },
            ensure_ascii=False,
        ),
    )

    if is_finance_question:
        fallback_cypher = _build_finance_template_cypher(question, organization_names)
        fallback_rows = execute_cypher(driver, fallback_cypher)
        _emit_progress("done", "Applied finance fallback template")
        return {
            "question": question,
            "cypher": fallback_cypher,
            "rows": fallback_rows,
            "attempt": NL2CYPHER_AGENTIC_MAX_ROUNDS,
            "reason": f"Finance fallback template used after retries: {last_error}",
            "agentic_trace": trace,
        }
    raise RuntimeError(f"Cypher generation failed after retries: {last_error}")


def answer_with_manual_prompt(
    question: str,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    nl2cypher_provider: str | None = None,
    nl2cypher_model: str | None = None,
) -> Dict[str, Any]:
    """執行 `answer_with_manual_prompt` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
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

        return _run_agentic_query_loop(
            question=question,
            schema=schema,
            driver=driver,
            entity_names=entity_names,
            organization_names=organization_names,
            is_finance_question=is_finance_question,
            nl2cypher_provider=nl2cypher_provider,
            nl2cypher_model=nl2cypher_model,
            progress_callback=progress_callback,
        )
    finally:
        driver.close()


def main() -> None:
    """作為模組執行入口，串接並啟動既有主流程。
    此函式會依目前設定呼叫核心邏輯，並維持原本輸入輸出與錯誤行為。
    """
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
