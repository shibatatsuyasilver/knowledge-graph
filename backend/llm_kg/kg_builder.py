"""Knowledge Graph builder for Neo4j with schema-constrained extraction.

Workflow:
1) Text -> LLM JSON extraction (with JSON repair retries)
2) Entity canonicalization + relation direction/type validation
3) Upsert nodes/edges into Neo4j
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Tuple

from neo4j import GraphDatabase, Driver
from backend.config.settings import (
    get_kg_builder_settings,
    get_neo4j_settings,
    resolve_extraction_model as resolve_extraction_model_from_settings,
    resolve_extraction_num_predict as resolve_extraction_num_predict_from_settings,
    resolve_extraction_provider as resolve_extraction_provider_from_settings,
)

try:
    from . import llm_client
except ImportError:  # pragma: no cover - direct script execution
    import llm_client  # type: ignore[no-redef]

_neo4j_settings = get_neo4j_settings()
_kg_builder_settings = get_kg_builder_settings()

NEO4J_URI = _neo4j_settings.uri
NEO4J_USER = _neo4j_settings.user
NEO4J_PASSWORD = _neo4j_settings.password
EXTRACTION_TIMEOUT_SECONDS = _kg_builder_settings.extraction_timeout_seconds
# Default to 5 total attempts (initial try + 4 retries) for robust JSON extraction.
EXTRACTION_MAX_JSON_RETRIES = _kg_builder_settings.extraction_max_json_retries
ENTITY_RESOLVE_THRESHOLD = _kg_builder_settings.entity_resolve_threshold
EXTRACTION_JSON_MODE = _kg_builder_settings.extraction_json_mode
EXTRACTION_ERROR_RAW_MAX_CHARS = _kg_builder_settings.extraction_error_raw_max_chars
GEMINI_TWO_PASS_EXTRACTION = _kg_builder_settings.gemini_two_pass_extraction

ALLOWED_ENTITY_TYPES = {
    "Person",
    "Organization",
    "Location",
    "Technology",
    "Product",
    "FiscalPeriod",
    "FinancialMetric",
}
ALLOWED_RELATION_TYPES = {
    "FOUNDED_BY",
    "CHAIRED_BY",
    "HEADQUARTERED_IN",
    "PRODUCES",
    "SUPPLIES_TO",
    "USES",
    "COMPETES_WITH",
    "HAS_FINANCIAL_METRIC",
    "FOR_PERIOD",
}
FINANCIAL_METRIC_CANONICAL_TYPES = {"REVENUE", "OPERATING_MARGIN"}
FINANCIAL_METRIC_TYPE_ALIASES = {
    "revenue": "REVENUE",
    "sales": "REVENUE",
    "營收": "REVENUE",
    "營業收入": "REVENUE",
    "operatingmargin": "OPERATING_MARGIN",
    "operatingprofitmargin": "OPERATING_MARGIN",
    "operatingprofitratio": "OPERATING_MARGIN",
    "operatingmarginrate": "OPERATING_MARGIN",
    "operating_margin": "OPERATING_MARGIN",
    "營益率": "OPERATING_MARGIN",
    "營業利益率": "OPERATING_MARGIN",
    "營運利潤率": "OPERATING_MARGIN",
}
ENTITY_EXTRA_PROPERTY_ALLOWLIST = {
    "description",
    "metric_type",
    "value",
    "unit",
    "currency",
    "period",
}

RELATION_DIRECTION_RULES: Dict[str, Dict[str, set[str]]] = {
    "FOUNDED_BY": {"source": {"Organization"}, "target": {"Person"}},
    "CHAIRED_BY": {"source": {"Organization"}, "target": {"Person"}},
    "HEADQUARTERED_IN": {"source": {"Organization"}, "target": {"Location"}},
    "PRODUCES": {"source": {"Organization"}, "target": {"Technology", "Product"}},
    "SUPPLIES_TO": {"source": {"Organization"}, "target": {"Organization"}},
    "USES": {"source": {"Organization"}, "target": {"Technology", "Product"}},
    "COMPETES_WITH": {"source": {"Organization"}, "target": {"Organization"}},
    "HAS_FINANCIAL_METRIC": {"source": {"Organization"}, "target": {"FinancialMetric"}},
    "FOR_PERIOD": {"source": {"FinancialMetric"}, "target": {"FiscalPeriod"}},
}

SAMPLE_TEXT = """
台灣積體電路製造股份有限公司（TSMC，簡稱台積電）成立於1987年，是全球最大的專用半導體晶圓代工廠。
總部位於台灣新竹科學園區。創辦人為張忠謀（Morris Chang）。
台積電為蘋果（Apple）、NVIDIA、AMD等無廠半導體公司生產晶片。
其主要競爭對手包括三星電子（Samsung Electronics）和英特爾（Intel）。
台積電在先進製程技術（如3奈米、2奈米）方面處於領先地位。
""".strip()


def _trim_raw_error(raw: str) -> str:
    """執行 `_trim_raw_error` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    text = str(raw or "").strip()
    limit = max(0, EXTRACTION_ERROR_RAW_MAX_CHARS)
    if limit == 0 or len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}... [truncated {omitted} chars]"


def _resolve_extraction_model(
    model: str | None = None,
    *,
    provider: str | None = None,
) -> str | None:
    """執行 `_resolve_extraction_model` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    return resolve_extraction_model_from_settings(provider=provider, explicit_model=model)


def _resolve_extraction_provider(provider: str | None = None) -> str | None:
    """執行 `_resolve_extraction_provider` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    return resolve_extraction_provider_from_settings(provider)


def _resolve_extraction_num_predict(provider: str | None = None) -> int:
    """執行 `_resolve_extraction_num_predict` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    return resolve_extraction_num_predict_from_settings(provider, llm_client.DEFAULT_GEMINI_OUTPUT_TOKEN_LIMIT)


@dataclass
class GraphBuildStats:
    entities: int = 0
    relations: int = 0
    merged_entities: int = 0
    dropped_relations: int = 0
    json_retries: int = 0


def strip_markdown_fence(content: str) -> str:
    """執行 `strip_markdown_fence` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
    text = content.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    return (m.group(1) if m else text).strip()


def _normalize_name(name: str) -> str:
    """執行 `_normalize_name` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    lowered = name.strip().lower()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", lowered)


def _name_variants(name: str) -> List[str]:
    """執行 `_name_variants` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    variants = {name.strip()}
    compact = re.sub(r"[（(].*?[）)]", "", name).strip()
    if compact:
        variants.add(compact)

    for content in re.findall(r"[（(](.*?)[）)]", name):
        for token in re.split(r"[、,，/|;；]", content):
            normalized = token.replace("簡稱", "").replace("简称", "").replace("aka", "").strip()
            if normalized:
                variants.add(normalized)

    return [v for v in variants if v]


def _best_match(name: str, candidates: Iterable[str]) -> Tuple[str, float]:
    """執行 `_best_match` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    best = ""
    best_score = 0.0
    for candidate in candidates:
        score = SequenceMatcher(None, name, candidate).ratio()
        if score > best_score:
            best, best_score = candidate, score
    return best, best_score


def _safe_type(value: str) -> str:
    """執行 `_safe_type` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    if value not in ALLOWED_ENTITY_TYPES:
        raise ValueError(f"Unsupported entity type: {value}")
    return value


def _safe_relation(value: str) -> str:
    """執行 `_safe_relation` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    if value not in ALLOWED_RELATION_TYPES:
        raise ValueError(f"Unsupported relation type: {value}")
    return value


def _canonicalize_period(value: str) -> str:
    """執行 `_canonicalize_period` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    raw = str(value or "").strip()
    if not raw:
        return ""
    normalized = raw.upper().replace(" ", "")
    direct = re.search(r"(20\d{2})Q([1-4])", normalized)
    if direct:
        return f"{direct.group(1)}Q{direct.group(2)}"

    zh = re.search(r"(20\d{2})\s*年?\s*第?\s*([1-4])\s*季", raw)
    if zh:
        return f"{zh.group(1)}Q{zh.group(2)}"
    return ""


def _canonicalize_metric_type(entity: Dict[str, Any], entity_name: str) -> str:
    """執行 `_canonicalize_metric_type` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    candidate = str(entity.get("metric_type", "")).strip()
    if not candidate:
        candidate = entity_name
    key = re.sub(r"[^0-9a-zA-Z_\u4e00-\u9fff]+", "", candidate).lower()
    mapped = FINANCIAL_METRIC_TYPE_ALIASES.get(key)
    if mapped:
        return mapped

    upper_candidate = candidate.upper().strip()
    if upper_candidate in FINANCIAL_METRIC_CANONICAL_TYPES:
        return upper_candidate
    return ""


def _to_optional_text(value: Any) -> Optional[str]:
    """執行 `_to_optional_text` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _extract_entity_extra_props(entity: Dict[str, Any], entity_type: str, canonical_name: str) -> Dict[str, str]:
    """執行 `_extract_entity_extra_props` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    extra: Dict[str, str] = {}
    description = _to_optional_text(entity.get("description"))
    if description:
        extra["description"] = description

    if entity_type == "FinancialMetric":
        metric_type = _canonicalize_metric_type(entity, canonical_name)
        if metric_type:
            extra["metric_type"] = metric_type
        value = _to_optional_text(entity.get("value"))
        if value:
            extra["value"] = value
        unit = _to_optional_text(entity.get("unit"))
        if unit:
            extra["unit"] = unit
        currency = _to_optional_text(entity.get("currency"))
        if currency:
            extra["currency"] = currency
        period = _canonicalize_period(str(entity.get("period", "")))
        if period:
            extra["period"] = period

    if entity_type == "FiscalPeriod":
        period = _canonicalize_period(canonical_name) or _canonicalize_period(str(entity.get("period", "")))
        if period:
            extra["period"] = period

    return {k: v for k, v in extra.items() if k in ENTITY_EXTRA_PROPERTY_ALLOWLIST and v}


class KnowledgeGraphBuilder:
    def __init__(self, uri: str, user: str, password: str):
        """初始化物件狀態並保存後續流程所需依賴。
        此方法會依目前參數設定實例欄位，供其他方法在生命週期內重複使用。
        """
        self.driver: Driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        """執行 `close` 的主要流程。
        函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
        """
        self.driver.close()

    def _should_use_two_pass_extraction(self, provider: str | None) -> bool:
        """執行 `_should_use_two_pass_extraction` 的內部輔助流程。
        此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
        """
        if not GEMINI_TWO_PASS_EXTRACTION:
            return False
        if provider == "gemini":
            return True
        if provider:
            return False
        try:
            return llm_client.get_runtime_config().provider == "gemini"
        except Exception:
            return False

    @staticmethod
    def _build_phase1_entity_inventory_prompt(text: str) -> str:
        """執行 `_build_phase1_entity_inventory_prompt` 的內部輔助流程。
        此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
        """
        return f"""
-Goal-
Given a text document and fixed ontology constraints, identify all entities only.
Do not extract relationships in this phase.
Return JSON only.

-Phase-1 Entity Inventory Rules-
1. Identify all entities. For each entity, extract:
- name: Name of the entity
- type: One of [{", ".join(sorted(ALLOWED_ENTITY_TYPES))}]
- description: Grounded short description from text

2. Financial normalization:
- For FiscalPeriod, normalize quarter as YYYYQ1..YYYYQ4 (example: 2025Q2).
- For FinancialMetric, include:
  - metric_type: REVENUE or OPERATING_MARGIN
  - value: raw numeric value text from source
  - unit: e.g. 億元, %, basis points
  - currency: e.g. TWD, USD
  - period: YYYYQ1..YYYYQ4 if available

3. Return exactly this JSON shape:
{{
  "entities": [
    {{"name": "台積電", "type": "Organization", "description": "..."}}
  ],
  "relations": []
}}

4. If nothing is found, return:
{{"entities": [], "relations": []}}

5. Do not output markdown, explanations, comments, or extra text.

-Real Data-
Text:
\"\"\"
{text}
\"\"\"
Output:
""".strip()

    @staticmethod
    def _render_seed_entities_for_prompt(seed_entities: List[Dict[str, Any]], limit: int = 120) -> str:
        """執行 `_render_seed_entities_for_prompt` 的內部輔助流程。
        此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
        """
        if not seed_entities:
            return "[]"
        lines: List[str] = []
        for entity in seed_entities[:limit]:
            name = str(entity.get("name", "")).strip()
            entity_type = str(entity.get("type", "")).strip()
            if not name or not entity_type:
                continue
            lines.append(f"- {name} ({entity_type})")
        if len(seed_entities) > limit:
            lines.append(f"- ... and {len(seed_entities) - limit} more")
        return "\n".join(lines) if lines else "[]"

    def _build_phase2_relation_prompt(self, text: str, seed_entities: List[Dict[str, Any]]) -> str:
        """執行 `_build_phase2_relation_prompt` 的內部輔助流程。
        此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
        """
        seed_block = self._render_seed_entities_for_prompt(seed_entities)
        # Use GraphRAG's Goal/Steps prompt style, adapted to this project's strict enum-based schema.
        return f"""
-Goal-
Given a text document and fixed ontology constraints, identify all entities of the allowed types and all clearly supported relationships among those entities.
Return JSON only.

-Phase-1 Seed Entities (reuse canonical names whenever applicable)-
{seed_block}

-Steps-
1. Identify all entities. For each entity, extract:
- name: Name of the entity
- type: One of [{", ".join(sorted(ALLOWED_ENTITY_TYPES))}]
- description: Comprehensive description grounded in the text

Financial entity notes:
- For FiscalPeriod, normalize quarter as YYYYQ1..YYYYQ4 (example: 2025Q2).
- For FinancialMetric, include:
  - metric_type: REVENUE or OPERATING_MARGIN
  - value: raw numeric value text from source
  - unit: e.g. 億元, %, basis points
  - currency: e.g. TWD, USD
  - period: YYYYQ1..YYYYQ4 if available

2. From entities in step 1, identify all clearly related pairs. For each relationship, extract:
- source: source entity name
- target: target entity name
- relation: One of [{", ".join(sorted(ALLOWED_RELATION_TYPES))}]
- description: Why source and target are related
- relationship_strength: integer score 1..10

3. Enforce relation direction constraints:
- Organization -[FOUNDED_BY]-> Person
- Organization -[CHAIRED_BY]-> Person
- Organization -[HEADQUARTERED_IN]-> Location
- Organization -[PRODUCES]-> Technology|Product
- Organization -[SUPPLIES_TO]-> Organization
- Organization -[USES]-> Technology|Product
- Organization -[COMPETES_WITH]-> Organization
- Organization -[HAS_FINANCIAL_METRIC]-> FinancialMetric
- FinancialMetric -[FOR_PERIOD]-> FiscalPeriod

Semantic mapping note:
- 董事長 / 主席 / 執行長 / CEO / President -> CHAIRED_BY
- 創辦人 / 創立者 / founded by -> FOUNDED_BY
- 營收 / revenue / sales -> FinancialMetric.metric_type=REVENUE
- 營益率 / 營業利益率 / operating margin -> FinancialMetric.metric_type=OPERATING_MARGIN

4. Return exactly this JSON shape:
{{
  "entities": [
    {{
      "name": "台積電",
      "type": "Organization",
      "description": "..."
    }},
    {{
      "name": "2025Q2",
      "type": "FiscalPeriod",
      "description": "2025 年第二季",
      "period": "2025Q2"
    }},
    {{
      "name": "鴻海 2025Q2 營收",
      "type": "FinancialMetric",
      "description": "鴻海 2025Q2 營收指標",
      "metric_type": "REVENUE",
      "value": "1.23 兆",
      "unit": "兆",
      "currency": "TWD",
      "period": "2025Q2"
    }}
  ],
  "relations": [
    {{
      "source": "台積電",
      "target": "張忠謀",
      "relation": "FOUNDED_BY",
      "description": "...",
      "relationship_strength": 9
    }},
    {{
      "source": "鴻海",
      "target": "鴻海 2025Q2 營收",
      "relation": "HAS_FINANCIAL_METRIC",
      "description": "鴻海擁有營收指標",
      "relationship_strength": 9
    }},
    {{
      "source": "鴻海 2025Q2 營收",
      "target": "2025Q2",
      "relation": "FOR_PERIOD",
      "description": "該指標對應季度",
      "relationship_strength": 9
    }}
  ]
}}

5. If nothing is found, return:
{{"entities": [], "relations": []}}

6. Do not output markdown, explanations, comments, or extra text.

-Real Data-
Text:
\"\"\"
{text}
\"\"\"
Output:
""".strip()

    def _fetch_existing_entity_keys(self, entities: List[Dict[str, Any]]) -> set[Tuple[str, str]]:
        """執行 `_fetch_existing_entity_keys` 的內部輔助流程。
        此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
        """
        probes: List[Dict[str, Any]] = []
        for entity in entities:
            name = str(entity.get("name", "")).strip()
            entity_type = str(entity.get("type", "")).strip()
            if not name or not entity_type:
                continue
            aliases = [str(alias).strip().lower() for alias in entity.get("aliases", []) if str(alias).strip()]
            aliases.append(name.lower())
            probes.append(
                {
                    "name": name,
                    "type": entity_type,
                    "normalized": _normalize_name(name),
                    "name_lower": name.lower(),
                    "aliases_lower": sorted(set(aliases)),
                }
            )

        if not probes:
            return set()

        query = """
UNWIND $entities AS item
OPTIONAL MATCH (e:Entity)
WHERE
  (item.normalized <> '' AND e.normalizedName = item.normalized)
  OR toLower(e.name) = item.name_lower
  OR any(a IN coalesce(e.aliases, []) WHERE toLower(a) IN item.aliases_lower)
WITH item, count(e) > 0 AS exists
RETURN item.name AS name, item.type AS type, exists AS exists
""".strip()

        existing: set[Tuple[str, str]] = set()
        with self.driver.session() as session:
            for row in session.run(query, entities=probes):
                if row["exists"]:
                    existing.add((str(row["name"]), str(row["type"])))
        return existing

    def _prefill_missing_entities(self, entities: List[Dict[str, Any]]) -> int:
        """執行 `_prefill_missing_entities` 的內部輔助流程。
        此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
        """
        inserted = 0
        for entity in entities:
            self._create_entity(
                entity["name"],
                entity["type"],
                aliases=entity.get("aliases"),
                extra_props=entity,
            )
            inserted += 1
        return inserted

    def _extract_entities_relations_two_pass(
        self,
        text: str,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> Dict[str, Any]:
        """執行 `_extract_entities_relations_two_pass` 的內部輔助流程。
        此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
        """
        phase1_prompt = self._build_phase1_entity_inventory_prompt(text)
        phase1_payload, phase1_retries = self._extract_json_with_retry(
            phase1_prompt,
            provider=provider,
            model=model,
        )
        phase1 = self._sanitize_extraction(phase1_payload)
        phase1_entities = list(phase1.get("entities", []))

        existing_keys = self._fetch_existing_entity_keys(phase1_entities)
        missing_entities = [
            entity
            for entity in phase1_entities
            if (str(entity.get("name", "")), str(entity.get("type", ""))) not in existing_keys
        ]
        prefilled_count = self._prefill_missing_entities(missing_entities)

        phase2_prompt = self._build_phase2_relation_prompt(text, phase1_entities)
        phase2_payload, phase2_retries = self._extract_json_with_retry(
            phase2_prompt,
            provider=provider,
            model=model,
        )
        phase2 = self._sanitize_extraction(phase2_payload)
        phase2.setdefault("meta", {})
        phase2["meta"]["two_pass"] = True
        phase2["meta"]["phase1_entities"] = len(phase1_entities)
        phase2["meta"]["prefilled_missing_entities"] = prefilled_count
        phase2["meta"]["phase1_json_retries"] = phase1_retries
        phase2["meta"]["phase2_json_retries"] = phase2_retries
        phase2["meta"]["json_retries"] = phase1_retries + phase2_retries
        return phase2

    def extract_entities_relations(
        self,
        text: str,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> Dict[str, Any]:
        """執行 `extract_entities_relations` 的主要流程。
        函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
        """
        use_provider = _resolve_extraction_provider(provider)
        if self._should_use_two_pass_extraction(use_provider):
            return self._extract_entities_relations_two_pass(
                text,
                provider=use_provider,
                model=model,
            )

        prompt = self._build_phase2_relation_prompt(text, seed_entities=[])
        parsed, retries = self._extract_json_with_retry(prompt, provider=use_provider, model=model)
        sanitized = self._sanitize_extraction(parsed)
        sanitized.setdefault("meta", {})
        sanitized["meta"]["two_pass"] = False
        sanitized["meta"]["json_retries"] = retries
        return sanitized

    def _extract_json_with_retry(
        self,
        prompt: str,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> Tuple[Dict[str, Any], int]:
        """執行 `_extract_json_with_retry` 的內部輔助流程。
        此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
        """
        last_error: Exception | None = None
        raw = ""
        current_prompt = prompt
        mode = EXTRACTION_JSON_MODE if EXTRACTION_JSON_MODE in {"auto", "strict_json", "text"} else "auto"
        use_provider = _resolve_extraction_provider(provider)
        use_model = _resolve_extraction_model(model, provider=provider)
        extraction_num_predict = _resolve_extraction_num_predict(use_provider)

        for attempt in range(EXTRACTION_MAX_JSON_RETRIES + 1):
            if mode in {"auto", "strict_json"}:
                try:
                    parsed = llm_client.chat_json(
                        messages=[{"role": "user", "content": current_prompt}],
                        provider=use_provider,
                        model=use_model,
                        temperature=0.0,
                        max_tokens=extraction_num_predict,
                        timeout_seconds=EXTRACTION_TIMEOUT_SECONDS,
                    )
                    parsed = self._validate_extraction_payload(parsed)
                    return (
                        parsed,
                        attempt,
                    )
                except Exception as exc:
                    last_error = exc
                    if mode == "strict_json":
                        current_prompt = f"""
請把以下內容修正為「合法 JSON」，且保持語意不變。
只輸出 JSON，不可加入其他文字。

原始輸出:
{current_prompt}

解析錯誤:
{exc}
""".strip()
                        continue

            raw = llm_client.chat_text(
                messages=[{"role": "user", "content": current_prompt}],
                provider=use_provider,
                model=use_model,
                temperature=0.0,
                max_tokens=extraction_num_predict,
                timeout_seconds=EXTRACTION_TIMEOUT_SECONDS,
            )
            try:
                parsed = json.loads(strip_markdown_fence(raw))
                parsed = self._validate_extraction_payload(parsed)
                return parsed, attempt
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                current_prompt = f"""
請把以下內容修正為「合法 JSON」，且保持語意不變。
只輸出 JSON，不可加入其他文字。

原始輸出:
{raw}

解析錯誤:
{exc}
""".strip()
                if mode == "strict_json":
                    continue

        total_attempts = EXTRACTION_MAX_JSON_RETRIES + 1
        raise ValueError(
            f"LLM JSON parse failed after {total_attempts} attempts: {last_error}; "
            f"raw={_trim_raw_error(raw)}"
        )

    @staticmethod
    def _validate_extraction_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        """執行 `_validate_extraction_payload` 的內部輔助流程。
        此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
        """
        if not isinstance(payload, dict):
            raise ValueError("Extraction payload must be a JSON object")
        entities = payload.get("entities")
        relations = payload.get("relations")
        if not isinstance(entities, list) or not isinstance(relations, list):
            raise ValueError("Extraction payload must include list fields: entities, relations")
        return payload

    def _sanitize_extraction(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """執行 `_sanitize_extraction` 的內部輔助流程。
        此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
        """
        entities: List[Dict[str, Any]] = []
        relations: List[Dict[str, str]] = []
        merged_entities = 0
        dropped_relations = 0

        entity_store: Dict[Tuple[str, str], set[str]] = {}
        entity_extra_store: Dict[Tuple[str, str], Dict[str, str]] = {}
        alias_index_by_type: Dict[str, Dict[str, str]] = {k: {} for k in ALLOWED_ENTITY_TYPES}
        for entity in payload.get("entities", []):
            name = str(entity.get("name", "")).strip()
            entity_type = str(entity.get("type", "")).strip()
            if not name:
                continue
            if entity_type not in ALLOWED_ENTITY_TYPES:
                continue

            if entity_type == "FiscalPeriod":
                period_name = _canonicalize_period(name) or _canonicalize_period(str(entity.get("period", "")))
                if not period_name:
                    continue
                name = period_name

            if entity_type == "FinancialMetric":
                metric_type = _canonicalize_metric_type(entity, name)
                if not metric_type:
                    continue

            variants = _name_variants(name)
            normalized_variants = [_normalize_name(v) for v in variants if _normalize_name(v)]
            existing = alias_index_by_type[entity_type]

            canonical_name = ""
            for norm in normalized_variants:
                if norm in existing:
                    canonical_name = existing[norm]
                    break

            if not canonical_name and normalized_variants:
                candidate, score = _best_match(normalized_variants[0], existing.keys())
                if candidate and score >= ENTITY_RESOLVE_THRESHOLD:
                    canonical_name = existing[candidate]

            if not canonical_name:
                canonical_name = name
            elif canonical_name != name:
                merged_entities += 1

            entity_key = (canonical_name, entity_type)
            if entity_key not in entity_store:
                entity_store[entity_key] = set()
            entity_store[entity_key].update(variants)
            entity_store[entity_key].add(canonical_name)
            extra_props = _extract_entity_extra_props(entity, entity_type, canonical_name)
            if extra_props:
                current_extra = entity_extra_store.setdefault(entity_key, {})
                for key, value in extra_props.items():
                    if key not in current_extra:
                        current_extra[key] = value

            for alias in entity_store[entity_key]:
                norm = _normalize_name(alias)
                if norm:
                    existing[norm] = canonical_name

        for (canonical_name, entity_type), aliases in entity_store.items():
            entity_row = {
                "name": canonical_name,
                "type": entity_type,
                "aliases": sorted(aliases),
            }
            entity_row.update(entity_extra_store.get((canonical_name, entity_type), {}))
            entities.append(entity_row)

        entity_type_by_name = {entity["name"]: entity["type"] for entity in entities}
        entity_by_name = {entity["name"]: entity for entity in entities}
        financial_period_by_name: Dict[str, str] = {}
        fiscal_period_by_name: Dict[str, str] = {}
        for entity in entities:
            entity_name = entity["name"]
            entity_type = entity["type"]
            if entity_type == "FinancialMetric":
                metric_period = _canonicalize_period(str(entity.get("period", "")))
                if metric_period:
                    financial_period_by_name[entity_name] = metric_period
            elif entity_type == "FiscalPeriod":
                period_value = _canonicalize_period(str(entity.get("period", ""))) or _canonicalize_period(entity_name)
                if period_value:
                    fiscal_period_by_name[entity_name] = period_value
        canonical_lookup: Dict[str, str] = {}
        for entity in entities:
            canonical_lookup[_normalize_name(entity["name"])] = entity["name"]
            for alias in entity.get("aliases", []):
                norm = _normalize_name(alias)
                if norm:
                    canonical_lookup[norm] = entity["name"]

        seen_relations = set()
        for relation in payload.get("relations", []):
            source = str(relation.get("source", "")).strip()
            rel_type = str(relation.get("relation", "")).strip()
            target = str(relation.get("target", "")).strip()
            if not source or not target or rel_type not in ALLOWED_RELATION_TYPES:
                continue

            source = self._resolve_entity_reference(source, canonical_lookup)
            target = self._resolve_entity_reference(target, canonical_lookup)
            source_type = entity_type_by_name.get(source)
            target_type = entity_type_by_name.get(target)

            if not source_type or not target_type:
                dropped_relations += 1
                continue

            rule = RELATION_DIRECTION_RULES[rel_type]
            source_allowed = source_type in rule["source"]
            target_allowed = target_type in rule["target"]

            if not (source_allowed and target_allowed):
                can_reverse = target_type in rule["source"] and source_type in rule["target"]
                if can_reverse:
                    source, target = target, source
                    source_type, target_type = target_type, source_type
                else:
                    dropped_relations += 1
                    continue

            # Keep quarter links coherent: a FinancialMetric can only connect to one period.
            if rel_type == "FOR_PERIOD" and source_type == "FinancialMetric" and target_type == "FiscalPeriod":
                metric_period = financial_period_by_name.get(source, "")
                target_period = fiscal_period_by_name.get(target, "") or _canonicalize_period(target)
                if target_period and metric_period and target_period != metric_period:
                    dropped_relations += 1
                    continue
                if target_period and not metric_period:
                    financial_period_by_name[source] = target_period
                    metric_entity = entity_by_name.get(source)
                    if isinstance(metric_entity, dict):
                        metric_entity["period"] = target_period

            key = (source.lower(), rel_type, target.lower())
            if key in seen_relations:
                continue
            seen_relations.add(key)
            relations.append({"source": source, "relation": rel_type, "target": target})

        return {
            "entities": entities,
            "relations": relations,
            "meta": {"merged_entities": merged_entities, "dropped_relations": dropped_relations},
        }

    def _resolve_entity_reference(self, name: str, canonical_lookup: Dict[str, str]) -> str:
        """執行 `_resolve_entity_reference` 的內部輔助流程。
        此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
        """
        normalized = _normalize_name(name)
        if normalized in canonical_lookup:
            return canonical_lookup[normalized]

        candidate, score = _best_match(normalized, canonical_lookup.keys())
        if candidate and score >= ENTITY_RESOLVE_THRESHOLD:
            return canonical_lookup[candidate]
        return name

    def _ensure_constraints(self) -> None:
        """執行 `_ensure_constraints` 的內部輔助流程。
        此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
        """
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
            session.run("CREATE INDEX entity_normalized_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.normalizedName)")

    def _create_entity(
        self,
        name: str,
        entity_type: str,
        aliases: Iterable[str] | None = None,
        extra_props: Optional[Dict[str, Any]] = None,
    ) -> None:
        """執行 `_create_entity` 的內部輔助流程。
        此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
        """
        label = _safe_type(entity_type)
        alias_values = sorted({a.strip() for a in (aliases or []) if a and a.strip()})
        if name not in alias_values:
            alias_values.append(name)
        query = [
            f"MERGE (e:{label} {{name:$name}}) "
            "SET e:Entity, "
            "    e.type = $entity_type, "
            "    e.normalizedName = $normalized_name, "
            "    e.updatedAt = datetime(), "
            "    e.aliases = reduce(acc = coalesce(e.aliases, []), a IN $aliases | "
            "      CASE WHEN a IN acc THEN acc ELSE acc + a END)"
        ]
        params: Dict[str, Any] = {
            "name": name,
            "entity_type": entity_type,
            "normalized_name": _normalize_name(name),
            "aliases": alias_values,
        }
        for key in ENTITY_EXTRA_PROPERTY_ALLOWLIST:
            if key in {"name", "type", "aliases"}:
                continue
            value = _to_optional_text((extra_props or {}).get(key))
            if value:
                param_name = f"extra_{key}"
                query.append(f"    , e.{key} = ${param_name}")
                params[param_name] = value
        with self.driver.session() as session:
            session.run(" ".join(query), **params)

    def _create_relation(self, source: str, relation: str, target: str) -> None:
        """執行 `_create_relation` 的內部輔助流程。
        此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
        """
        rel = _safe_relation(relation)
        query = (
            "MATCH (a:Entity {name:$source}), (b:Entity {name:$target}) "
            f"MERGE (a)-[r:{rel}]->(b) "
            "SET r.updatedAt = datetime()"
        )
        with self.driver.session() as session:
            session.run(query, source=source, target=target)

    def populate_graph(self, data: Dict[str, Any]) -> GraphBuildStats:
        """執行 `populate_graph` 的主要流程。
        函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
        """
        self._ensure_constraints()
        meta = data.get("meta", {})
        stats = GraphBuildStats(
            merged_entities=int(meta.get("merged_entities", 0)),
            dropped_relations=int(meta.get("dropped_relations", 0)),
            json_retries=int(meta.get("json_retries", 0)),
        )

        for entity in data.get("entities", []):
            self._create_entity(
                entity["name"],
                entity["type"],
                aliases=entity.get("aliases"),
                extra_props=entity,
            )
            stats.entities += 1

        for relation in data.get("relations", []):
            self._create_relation(relation["source"], relation["relation"], relation["target"])
            stats.relations += 1

        return stats

    def build_from_text(self, text: str) -> Tuple[Dict[str, Any], GraphBuildStats]:
        """執行 `build_from_text` 的主要流程。
        函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
        """
        extracted = self.extract_entities_relations(text)
        stats = self.populate_graph(extracted)
        return extracted, stats


def _fallback_data() -> Dict[str, Any]:
    """執行 `_fallback_data` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    return {
        "entities": [
            {"name": "台積電", "type": "Organization"},
            {"name": "張忠謀", "type": "Person"},
            {"name": "新竹科學園區", "type": "Location"},
            {"name": "Apple", "type": "Organization"},
            {"name": "NVIDIA", "type": "Organization"},
            {"name": "3奈米", "type": "Technology"},
            {"name": "2025Q2", "type": "FiscalPeriod", "period": "2025Q2"},
            {
                "name": "台積電 2025Q2 營收",
                "type": "FinancialMetric",
                "metric_type": "REVENUE",
                "value": "8,390 億元",
                "unit": "億元",
                "currency": "TWD",
                "period": "2025Q2",
            },
        ],
        "relations": [
            {"source": "台積電", "relation": "FOUNDED_BY", "target": "張忠謀"},
            {"source": "台積電", "relation": "HEADQUARTERED_IN", "target": "新竹科學園區"},
            {"source": "台積電", "relation": "SUPPLIES_TO", "target": "Apple"},
            {"source": "台積電", "relation": "SUPPLIES_TO", "target": "NVIDIA"},
            {"source": "台積電", "relation": "PRODUCES", "target": "3奈米"},
            {"source": "台積電", "relation": "HAS_FINANCIAL_METRIC", "target": "台積電 2025Q2 營收"},
            {"source": "台積電 2025Q2 營收", "relation": "FOR_PERIOD", "target": "2025Q2"},
        ],
    }


def deterministic_fallback_payload() -> Dict[str, Any]:
    """執行 `deterministic_fallback_payload` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
    return _fallback_data()


def main() -> None:
    """作為模組執行入口，串接並啟動既有主流程。
    此函式會依目前設定呼叫核心邏輯，並維持原本輸入輸出與錯誤行為。
    """
    builder = KnowledgeGraphBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        model = llm_client.get_runtime_config().model
        print(f"[1/2] Extracting entities and relations with {model}...")
        try:
            extracted = builder.extract_entities_relations(SAMPLE_TEXT)
        except Exception as exc:
            print(f"LLM extraction failed, fallback to sample mock data: {exc}")
            extracted = _fallback_data()

        print(f"Extracted entities: {len(extracted['entities'])}")
        print(f"Extracted relations: {len(extracted['relations'])}")

        print("[2/2] Populating Neo4j graph...")
        stats = builder.populate_graph(extracted)
        print(
            "Done. "
            f"Nodes upserted={stats.entities}, relations upserted={stats.relations}, "
            f"entity merges={stats.merged_entities}, dropped relations={stats.dropped_relations}, "
            f"json retries={stats.json_retries}"
        )

    finally:
        builder.close()


if __name__ == "__main__":
    main()
