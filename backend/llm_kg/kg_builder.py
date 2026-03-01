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
    """截斷原始錯誤訊息至 EXTRACTION_ERROR_RAW_MAX_CHARS 字元限制。"""
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
    """根據提供者或明確模型名稱解析實體抽取所用的模型，委派給 settings。"""
    return resolve_extraction_model_from_settings(provider=provider, explicit_model=model)


def _resolve_extraction_provider(provider: str | None = None) -> str | None:
    """解析實體抽取的 LLM 提供者，委派給 settings。"""
    return resolve_extraction_provider_from_settings(provider)


def _resolve_extraction_num_predict(provider: str | None = None) -> int:
    """解析實體抽取的預測 token 上限，委派給 settings。"""
    return resolve_extraction_num_predict_from_settings(provider, llm_client.DEFAULT_GEMINI_OUTPUT_TOKEN_LIMIT)


@dataclass
class GraphBuildStats:
    entities: int = 0
    relations: int = 0
    merged_entities: int = 0
    dropped_relations: int = 0
    json_retries: int = 0


def strip_markdown_fence(content: str) -> str:
    """移除 Markdown 代碼欄標記（```json ... ```），取出其中的純文字內容。"""
    text = content.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    return (m.group(1) if m else text).strip()


def _normalize_name(name: str) -> str:
    """將名稱轉換為小寫並移除所有非字母數字及非中文字元。"""
    lowered = name.strip().lower()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", lowered)


def _name_variants(name: str) -> List[str]:
    """從名稱中提取所有變體：去除括號內容、分割別名（簡稱、aka 等）。"""
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
    """以字串相似度從候選清單中找出最佳匹配名稱，回傳（名稱, 分數）。"""
    best = ""
    best_score = 0.0
    for candidate in candidates:
        score = SequenceMatcher(None, name, candidate).ratio()
        if score > best_score:
            best, best_score = candidate, score
    return best, best_score


def _safe_type(value: str) -> str:
    """驗證實體類型是否在允許清單中，不合法時拋出 ValueError。"""
    if value not in ALLOWED_ENTITY_TYPES:
        raise ValueError(f"Unsupported entity type: {value}")
    return value


def _safe_relation(value: str) -> str:
    """驗證關係類型是否在允許清單中，不合法時拋出 ValueError。"""
    if value not in ALLOWED_RELATION_TYPES:
        raise ValueError(f"Unsupported relation type: {value}")
    return value


def _to_optional_text(value: Any) -> Optional[str]:
    """將值轉換為可選字串：None 或空字串時回傳 None，否則回傳去頭尾空白的字串。"""
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _extract_entity_extra_props(entity: Dict[str, Any], entity_type: str, canonical_name: str) -> Dict[str, str]:
    """從實體字典中提取允許清單內的額外屬性（description、metric_type 等）。"""
    extra: Dict[str, str] = {}
    description = _to_optional_text(entity.get("description"))
    if description:
        extra["description"] = description

    if entity_type in {"FinancialMetric", "FiscalPeriod"}:
        metric_type = _to_optional_text(entity.get("metric_type"))
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
        period = _to_optional_text(entity.get("period"))
        if period:
            extra["period"] = period

    return {k: v for k, v in extra.items() if k in ENTITY_EXTRA_PROPERTY_ALLOWLIST and v}


class KnowledgeGraphBuilder:
    def __init__(self, uri: str, user: str, password: str):
        """初始化 Neo4j Driver 連線。"""
        self.driver: Driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        """關閉 Neo4j Driver 連線。"""
        self.driver.close()

    def _should_use_two_pass_extraction(self, provider: str | None) -> bool:
        """判斷是否應使用 Gemini 雙階段提取模式。

        需先確認環境變數 GEMINI_TWO_PASS_EXTRACTION 為 True；若未啟用，
        無論 provider 為何皆回傳 False。啟用後，再依 provider 決定是否為 gemini。
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
        """建構第一階段提示：要求 LLM 僅識別實體（不提取關係），回傳 JSON。"""
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

2. If grounded in source text, include optional text fields when available:
- metric_type
- value
- unit
- currency
- period

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
        """將種子實體清單格式化為提示用的可讀文字（每行一個實體），超出上限時附加省略提示。"""
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
        """建構第二階段提示：給定種子實體清單，要求 LLM 提取實體與關係，回傳 JSON。"""
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

Optional entity text fields (when available): metric_type, value, unit, currency, period.

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

4. Return exactly this JSON shape:
{{
  "entities": [
    {{
      "name": "台積電",
      "type": "Organization",
      "description": "..."
    }},
    {{
      "name": "張忠謀",
      "type": "Person",
      "description": "..."
    }},
    {{
      "name": "新竹",
      "type": "Location",
      "description": "..."
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
      "source": "台積電",
      "target": "新竹",
      "relation": "HEADQUARTERED_IN",
      "description": "...",
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
        """從 Neo4j 查詢哪些實體已存在（以名稱、規範化名稱或別名比對），回傳（名稱, 類型）集合。"""
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
        """批量建立 Phase 1 中發現但在 Neo4j 中尚不存在的實體，回傳建立數量。"""
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
        """執行雙階段提取：Phase 1 識別實體庫並預填 Neo4j，Phase 2 提取關係。"""
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
        """主流程：根據設定決定使用雙階段或單階段提取實體與關係，回傳正規化結果。"""
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
        """以重試機制呼叫 LLM 進行 JSON 提取，解析失敗時送修復提示重試，回傳（結果, 重試次數）。"""
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
        """驗證 LLM 提取的 JSON 是否為包含 entities 和 relations 清單的字典。"""
        if not isinstance(payload, dict):
            raise ValueError("Extraction payload must be a JSON object")
        entities = payload.get("entities")
        relations = payload.get("relations")
        if not isinstance(entities, list) or not isinstance(relations, list):
            raise ValueError("Extraction payload must include list fields: entities, relations")
        return payload

    def _sanitize_extraction(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """正規化提取結果：合併重複實體、驗證關係方向、丟棄不符 schema 的記錄。"""
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
        """將實體名稱或別名解析為規範名稱，先精確比對，再以相似度比對。"""
        normalized = _normalize_name(name)
        if normalized in canonical_lookup:
            return canonical_lookup[normalized]

        candidate, score = _best_match(normalized, canonical_lookup.keys())
        if candidate and score >= ENTITY_RESOLVE_THRESHOLD:
            return canonical_lookup[candidate]
        return name

    def _ensure_constraints(self) -> None:
        """在 Neo4j 中建立 Entity 節點的唯一性約束與規範化名稱索引。"""
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
        """在 Neo4j 中 upsert 實體節點，包含名稱、類型、規範化名稱、別名與額外屬性。"""
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
        """在 Neo4j 中建立或更新兩個實體節點之間的關係邊。"""
        rel = _safe_relation(relation)
        query = (
            "MATCH (a:Entity {name:$source}), (b:Entity {name:$target}) "
            f"MERGE (a)-[r:{rel}]->(b) "
            "SET r.updatedAt = datetime()"
        )
        with self.driver.session() as session:
            session.run(query, source=source, target=target)

    def populate_graph(self, data: Dict[str, Any]) -> GraphBuildStats:
        """批量將實體與關係寫入 Neo4j，回傳建立/更新的統計資訊。"""
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
        """端到端流程：提取文字中的實體與關係後寫入知識圖譜，回傳提取結果與統計。"""
        extracted = self.extract_entities_relations(text)
        stats = self.populate_graph(extracted)
        return extracted, stats


def _fallback_data() -> Dict[str, Any]:
    """回傳台積電示範資料，作為 LLM 提取失敗時的後備結果。"""
    return {
        "entities": [
            {"name": "台積電", "type": "Organization"},
            {"name": "張忠謀", "type": "Person"},
            {"name": "新竹科學園區", "type": "Location"},
            {"name": "Apple", "type": "Organization"},
            {"name": "NVIDIA", "type": "Organization"},
            {"name": "3奈米", "type": "Technology"},
        ],
        "relations": [
            {"source": "台積電", "relation": "FOUNDED_BY", "target": "張忠謀"},
            {"source": "台積電", "relation": "HEADQUARTERED_IN", "target": "新竹科學園區"},
            {"source": "台積電", "relation": "SUPPLIES_TO", "target": "Apple"},
            {"source": "台積電", "relation": "SUPPLIES_TO", "target": "NVIDIA"},
            {"source": "台積電", "relation": "PRODUCES", "target": "3奈米"},
        ],
    }


def deterministic_fallback_payload() -> Dict[str, Any]:
    """回傳固定的示範提取結果（供測試與 fallback 使用）。"""
    return _fallback_data()


def main() -> None:
    """以台積電示範文字執行端到端 KG 建構流程（CLI 入口）。"""
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
