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

GraphCypherQAChain = None
try:  # pragma: no cover - optional imports
    # Preferred import path on recent langchain-community versions.
    from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain as _GraphCypherQAChain

    GraphCypherQAChain = _GraphCypherQAChain
except Exception:  # pragma: no cover - optional imports
    try:
        # Backward-compatible import path on older versions.
        from langchain_community.chains import GraphCypherQAChain as _GraphCypherQAChain

        GraphCypherQAChain = _GraphCypherQAChain
    except Exception:
        GraphCypherQAChain = None

try:  # pragma: no cover - optional imports
    from langchain_community.graphs import Neo4jGraph
except Exception:
    Neo4jGraph = None

try:  # pragma: no cover - optional imports
    from langchain_community.llms import Ollama
except Exception:
    Ollama = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:  # pragma: no cover - optional imports
    ChatGoogleGenerativeAI = None

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

_ALLOWED_AGENTIC_STRATEGIES = {"single_query", "union", "staged_merge"}
_ALLOWED_CRITIC_VERDICTS = {"accept", "replan", "fail_fast"}
_AGENTIC_MAX_FAILURE_CHAIN = 20
_AGENTIC_REPEAT_CYPHER_LIMIT = 2
_ORG_NAME_PATTERN = re.compile(
    r"\((?P<var>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*Organization\s*\{\s*name\s*:\s*'(?P<name>[^']+)'\s*\}\)"
)
_logger = logging.getLogger(__name__)


@dataclass
class SchemaSnapshot:
    labels: List[str]
    relationship_types: List[str]
    properties: Dict[str, List[str]]
    schema_text: str


def _relax_exact_organization_name_match(cypher: str) -> str | None:
    """
    放寬 Cypher 查詢中對 Organization 名稱的精確匹配。

    主要用途：
    - 將原本精確匹配的 `{name: '...'}` 轉換為使用 `CONTAINS` 和 `toLower` 的模糊匹配，
      以提高查詢的容錯率（例如處理大小寫或部分名稱）。

    回傳約定：
    - 如果成功修改 Cypher 語句，回傳新的 Cypher 字串。
    - 如果沒有找到符合的模式或無法修改，則回傳 None。
    """
    # 尋找匹配 (var:Organization {name: 'name'}) 的模式
    match = _ORG_NAME_PATTERN.search(cypher)
    if not match:
        return None

    var = match.group("var")
    name = match.group("name")

    # 移除原本的屬性過濾，將節點改為單純的 (var:Organization)
    relaxed = _ORG_NAME_PATTERN.sub(f"({var}:Organization)", cypher, count=1)
    
    # 建立不分大小寫的模糊匹配條件
    filter_expr = f"toLower({var}.name) CONTAINS toLower('{name}')"

    # 如果原查詢包含 WHERE，則在 WHERE 後面加上新的過濾條件
    if re.search(r"\bWHERE\b", relaxed, flags=re.IGNORECASE):
        return re.sub(r"\bWHERE\b", f"WHERE {filter_expr} AND ", relaxed, count=1, flags=re.IGNORECASE)

    # 如果原查詢沒有 WHERE 但有 RETURN，則在 RETURN 前面插入 WHERE 條件
    if re.search(r"\bRETURN\b", relaxed, flags=re.IGNORECASE):
        return re.sub(r"\bRETURN\b", f"WHERE {filter_expr}\nRETURN", relaxed, count=1, flags=re.IGNORECASE)
    
    return None


def _resolve_nl2cypher_provider(provider_override: str | None = None) -> str | None:
    """
    解析並決定使用的 NL2Cypher LLM 供應商 (Provider)。

    主要用途：
    - 取得最終要使用的 LLM 供應商，若有傳入 `provider_override` 則優先使用，
      否則依賴系統設定中的預設值。

    回傳約定：
    - 回傳字串代表供應商名稱（例如 'openai', 'ollama' 等），或 None。
    """
    return resolve_nl2cypher_provider_from_settings(provider_override)


def _resolve_nl2cypher_model(*, provider_override: str | None = None, model_override: str | None = None) -> str | None:
    """
    解析並決定使用的 NL2Cypher LLM 模型名稱。

    主要用途：
    - 根據可選的供應商與模型覆蓋參數，決定最終應使用的模型名稱。
      主要用於允許在請求層級覆蓋全域設定。

    回傳約定：
    - 回傳字串代表模型名稱，或 None。
    """
    return resolve_nl2cypher_model_from_settings(provider=provider_override, explicit_model=model_override)


def strip_markdown_fence(content: str) -> str:
    """
    移除字串前後的 Markdown 程式碼區塊標記 (如 ```cypher ... ```)。

    主要用途：
    - 當 LLM 回傳的結果包含 Markdown 格式的程式碼區塊時，清理這些標記，
      以便後續能直接將字串作為 Cypher 語法執行。

    回傳約定：
    - 回傳清理後的純 Cypher 查詢字串。
    """
    text = content.strip()
    match = re.match(r"^```(?:cypher|sql)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else text).strip()


def _normalize(value: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value.strip().lower())


def load_schema_snapshot(driver: Driver) -> SchemaSnapshot:
    """
    從 Neo4j 資料庫中載入並建立 Schema 結構快照。

    主要用途：
    - 查詢資料庫中的所有 Labels、Relationship Types 以及每個 Label 對應的屬性（Properties）。
    - 建立實體之間的關聯形狀（Relation Shapes，例如 `(Person)-[:WORKS_AT]->(Organization)`）。
    - 將這些資訊格式化為純文字，供後續傳遞給 LLM 作為生成 Cypher 查詢的上下文（Schema Grounding）。

    參數：
    - driver: Neo4j 資料庫的連線實例。

    回傳：
    - SchemaSnapshot: 包含解析後的 labels, relationship_types, properties 及格式化 schema_text 的資料物件。
    """
    with driver.session() as session:
        # === 查詢所有 Node Labels 與 Relationship Types ===
        labels = sorted([row["label"] for row in session.run("CALL db.labels() YIELD label RETURN label")])
        relationship_types = sorted(
            [row["relationshipType"] for row in session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")]
        )

        # === 逐標籤提取屬性鍵（每個 Label 取樣前 500 筆節點以兼顧效能）===
        properties: Dict[str, List[str]] = {}
        for label in labels:
            rows = session.run(
                f"MATCH (n:`{label}`) WITH n LIMIT 500 "
                "UNWIND keys(n) AS key RETURN DISTINCT key ORDER BY key"
            )
            properties[label] = [row["key"] for row in rows]

        # === 探索並建立關係形狀（Relation Shapes）===
        # 查詢實際存在的關係端點，組成 (FromLabel)-[:REL_TYPE]->(ToLabel) 形式的說明，
        # 過濾掉通用的 'Entity' 標籤以避免產生無意義的形狀。
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

    # === 組合 Schema 文字，供後續作為 LLM Prompt 的上下文（Schema Grounding）===
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
        # 優先使用具體的關係形狀（含端點節點類型）
        for shape in sorted(set(relation_shapes)):
            schema_lines.append(f"- {shape}")
    else:
        # 若無法取得形狀，退而列出所有關係類型名稱
        for rel_type in relationship_types:
            schema_lines.append(f"- {rel_type}")

    return SchemaSnapshot(
        labels=labels,
        relationship_types=relationship_types,
        properties=properties,
        schema_text="\n".join(schema_lines),
    )


def fetch_entity_names(driver: Driver) -> List[str]:
    """
    從 Neo4j 資料庫中提取所有 Entity 節點的名稱清單。

    主要用途：
    - 在進行實體連結（Entity Linking）時，需要比對使用者提問中的文字與資料庫中實際存在的實體名稱。
    - 透過提取這些名稱，可以避免 LLM 生成不存在的實體名稱，進而提升 Cypher 查詢的成功率。

    參數：
    - driver: Neo4j 資料庫的連線實例。

    回傳：
    - List[str]: 所有具有 name 屬性的 Entity 名稱清單。
    """
    with driver.session() as session:
        rows = session.run("MATCH (e:Entity) WHERE e.name IS NOT NULL RETURN DISTINCT e.name AS name")
        return [str(row["name"]) for row in rows]


def link_entity_literals(cypher: str, entity_names: List[str]) -> str:
    """
    將 Cypher 語句中的引號字串字面量對應到資料庫中實際存在的實體名稱。

    主要用途：
    - LLM 生成的 Cypher 中可能含有拼寫稍有差異的實體名稱（如大小寫、簡稱）。
    - 本函式先進行精確比對，若失敗則以 SequenceMatcher 進行模糊比對，
      分數達到或超過 ENTITY_LINK_THRESHOLD 時才替換，避免過度更正造成語意偏移。

    參數：
    - cypher: 待替換的 Cypher 字串。
    - entity_names: 資料庫中實際存在的實體名稱清單（來自 fetch_entity_names）。

    回傳：
    - str: 實體名稱已修正的 Cypher 字串。
    """
    # === 快速返回：無實體名稱清單時，Cypher 無需修改 ===
    if not entity_names:
        return cypher

    # === 建立正規化實體查詢表（normalized key → 原始名稱）===
    # 正規化規則：移除非字母數字字元並統一小寫，方便模糊比對
    entity_lookup: Dict[str, str] = {}
    for name in entity_names:
        key = _normalize(name)
        if key:
            entity_lookup[key] = name

    # === 定義替換邏輯：先精確比對，再以 SequenceMatcher 模糊比對 ===
    def replace_literal(match: re.Match[str]) -> str:
        raw_literal = match.group(1)
        normalized = _normalize(raw_literal)
        # 過短的字串（≤1 字元）跳過，避免誤替換
        if not normalized or len(normalized) <= 1:
            return match.group(0)

        # 精確比對：正規化後完全相同
        if normalized in entity_lookup:
            return f"'{entity_lookup[normalized]}'"

        # 模糊比對：找相似度最高的候選
        best_key = ""
        best_score = 0.0
        for candidate in entity_lookup:
            score = SequenceMatcher(None, normalized, candidate).ratio()
            if score > best_score:
                best_key, best_score = candidate, score

        if best_key and best_score >= ENTITY_LINK_THRESHOLD:
            return f"'{entity_lookup[best_key]}'"
        return match.group(0)

    # === 用 regex 對 Cypher 中所有引號字串執行替換 ===
    return re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", replace_literal, cypher)


def _is_identifier_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"



def _extract_graph_chain_cypher(intermediate_steps: Any) -> str:
    """從 GraphCypherQAChain intermediate steps 擷取 Cypher。"""
    if not isinstance(intermediate_steps, list):
        return ""

    for step in intermediate_steps:
        if not isinstance(step, dict):
            continue
        query = step.get("query")
        if isinstance(query, str) and query.strip():
            return query
    return ""


def _extract_graph_chain_rows(intermediate_steps: Any) -> List[Dict[str, Any]]:
    """從 GraphCypherQAChain intermediate steps 擷取 context rows。"""
    if not isinstance(intermediate_steps, list):
        return []

    rows: List[Dict[str, Any]] = []
    for step in intermediate_steps:
        if not isinstance(step, dict):
            continue
        context = step.get("context")
        if not isinstance(context, list):
            continue
        for item in context:
            if isinstance(item, dict):
                rows.append(item)
            else:
                rows.append({"value": item})
    return rows


def query_with_graph_chain(
    question: str,
    nl2cypher_provider: str | None = None,
    nl2cypher_model: str | None = None,
) -> Dict[str, Any]:
    """
    使用 LangChain 的 GraphCypherQAChain 進行自然語言到 Cypher 的轉換與查詢。

    主要用途：
    - 作為 Method 1，依賴 LangChain 內建的 GraphQA 邏輯來處理 Neo4j 的查詢。
    - 支援 Ollama 與 Gemini 等多種 LLM 供應商。
    - 自動建立 Schema，產生 Cypher，執行並將結果轉換為自然語言回答。

    參數：
    - question: 使用者的自然語言問題。
    - nl2cypher_provider: LLM 供應商名稱（例如 'gemini' 或 'ollama'）。
    - nl2cypher_model: LLM 模型名稱。

    回傳：
    - Dict[str, Any]: 包含 question, answer, cypher, rows, raw 結果等資訊的字典。
    """
    # === 檢查 LangChain 元件是否安裝（GraphCypherQAChain 與 Neo4jGraph 為可選依賴）===
    if not all([GraphCypherQAChain, Neo4jGraph]):
        raise RuntimeError(
            "LangChain components are not installed. Install langchain-community to enable GraphCypherQAChain."
        )

    # === 解析 Provider / Model 設定（優先使用傳入的覆蓋值，其次使用全域執行時設定）===
    cfg = llm_client.get_runtime_config()
    resolved_provider = _resolve_nl2cypher_provider(nl2cypher_provider) or cfg.provider
    resolved_model = _resolve_nl2cypher_model(
        provider_override=resolved_provider,
        model_override=nl2cypher_model,
    ) or cfg.model

    # === 建立 Neo4j Graph 連線（LangChain 負責 schema 擷取）===
    graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)

    # === 依 Provider 初始化 LLM（目前支援 Ollama 與 Gemini）===
    if resolved_provider == "ollama":
        if Ollama is None:
            raise RuntimeError("langchain-community Ollama integration is not installed.")
        llm = Ollama(
            model=resolved_model,
            temperature=0,
            base_url=cfg.ollama_base_url,
        )
    elif resolved_provider == "gemini":
        if ChatGoogleGenerativeAI is None:
            raise RuntimeError(
                "langchain-google-genai is not installed. Install it to enable GraphCypherQAChain with Gemini."
            )
        if not cfg.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is required for GraphCypherQAChain with provider=gemini.")
        llm = ChatGoogleGenerativeAI(
            model=resolved_model,
            temperature=0,
            google_api_key=cfg.gemini_api_key,
        )
    else:
        raise ValueError(f"Unsupported GraphCypherQAChain provider: {resolved_provider}")

    # === 建立 GraphCypherQAChain 並執行查詢 ===
    # return_intermediate_steps=True 可取得 LLM 產生的 Cypher 與原始查詢結果
    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True,
        return_intermediate_steps=True,
        allow_dangerous_requests=True,
    )

    result = chain.invoke({"query": question})

    # === 從 intermediate_steps 擷取 Cypher 與 rows，組成統一的回傳格式 ===
    intermediate_steps = result.get("intermediate_steps")
    generated = _extract_graph_chain_cypher(intermediate_steps)
    rows = _extract_graph_chain_rows(intermediate_steps)
    answer = result.get("result", "")

    return {
        "question": question,
        "answer": answer if isinstance(answer, str) else str(answer),
        "cypher": generated,
        "rows": rows,
        "raw": result,
        "query_engine": "graph_chain",
        "engine_provider": resolved_provider,
        "engine_model": resolved_model,
    }


def natural_language_to_cypher(
    question: str,
    schema: str,
    previous_cypher: str | None = None,
    error_message: str | None = None,
    nl2cypher_provider: str | None = None,
    nl2cypher_model: str | None = None,
) -> str:
    """
    直接呼叫 LLM，將自然語言問題轉換為可執行的 Neo4j Cypher 查詢語句。

    主要用途：
    - 結合 Graph Schema 與使用者的問題，透過 Prompt 指導 LLM 產生正確的 Cypher 語法。
    - 支援自我修正機制：如果前一次執行失敗，可傳入 `previous_cypher` 與 `error_message`，
      讓 LLM 根據錯誤訊息進行修正並重新產生。

    參數：
    - question: 使用者的自然語言問題。
    - schema: 純文字格式的 Graph Schema。
    - previous_cypher: (選填) 前次產生但執行失敗的 Cypher 語句。
    - error_message: (選填) 前次執行 Cypher 失敗時產生的 Neo4j 錯誤訊息。

    回傳：
    - str: LLM 生成的純 Cypher 查詢字串（已去除了 Markdown 標記）。
    """
    # === 建立錯誤修復區塊（僅在前次 Cypher 執行失敗時才附加，讓 LLM 自我修正）===
    repair_block = ""
    if previous_cypher and error_message:
        repair_block = f"""
上一次失敗的 Cypher：
{previous_cypher}

Neo4j 錯誤訊息：
{error_message}

請修正上面的 Cypher。
""".strip()

    # === 組合 LLM Prompt（角色設定 + Schema 上下文 + 生成規則 + 使用者問題）===
    # 溫度設為 0 以確保輸出確定性；規則 8 特別處理 UNION 欄位一致性問題
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
6. 查詢必須可執行且語法正確。
7. 僅回傳 Cypher，不要其他文字與 markdown。
8. 若使用 UNION/UNION ALL，各分支 RETURN 欄位數量、順序、欄位別名必須完全一致。

使用者問題：{question}
{repair_block}
""".strip()

    # === 解析 Provider/Model，呼叫 LLM（temperature=0 確保確定性輸出）===
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
    # === 移除 LLM 可能附加的 Markdown 程式碼區塊標記，回傳純 Cypher ===
    return strip_markdown_fence(text)


def execute_cypher(driver: Driver, cypher: str) -> List[Dict[str, Any]]:
    with driver.session() as session:
        result = session.run(cypher)
        return [dict(row) for row in result]


def _append_failure(failure_chain: List[str], message: str) -> None:
    text = str(message or "").strip()
    if not text:
        return
    failure_chain.append(text[:1024])
    overflow = len(failure_chain) - _AGENTIC_MAX_FAILURE_CHAIN
    if overflow > 0:
        del failure_chain[:overflow]


def _clean_string_list(value: Any, limit: int = 12) -> List[str]:
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
    normalized = strategy.strip().lower()
    if normalized in _ALLOWED_AGENTIC_STRATEGIES:
        return normalized
    return "single_query"


def _normalize_critic_verdict(verdict: str) -> str:
    normalized = verdict.strip().lower()
    if normalized in _ALLOWED_CRITIC_VERDICTS:
        return normalized
    return "replan"


def _normalize_severity(severity: str) -> str:
    normalized = severity.strip().lower()
    if normalized in {"low", "med", "high"}:
        return normalized
    return "med"


def _default_agentic_plan(question: str) -> Dict[str, Any]:
    """
    建立並回傳一個預設的 Agentic 查詢計畫結構。

    主要用途：
    - 作為 Planner Agent 失敗時的保底計畫，以及傳遞給 Planner prompt 的參考基準。
    - 預設策略為 single_query，並預設幾個常見的 risk_hypotheses。

    回傳：
    - Dict[str, Any]: 含 intent, strategy, must_have_paths, forbidden_patterns,
      output_contract, risk_hypotheses 欄位的計畫字典。
    """
    return {
        "intent": question.strip() or "unknown",
        "strategy": "single_query",
        "must_have_paths": [],
        "forbidden_patterns": [],
        "output_contract": {"columns": []},
        "risk_hypotheses": ["schema mismatch", "wrong relationship", "union projection mismatch"],
    }


def _sanitize_planner_payload(payload: Dict[str, Any], *, question: str) -> Dict[str, Any]:
    """
    清理並正規化 Planner Agent 回傳的 JSON payload。

    主要用途：
    - 以 _default_agentic_plan 為基準，逐欄位覆蓋 LLM 回傳的值，
      確保 strategy 合法、各清單欄位為有效字串列表。
    - 即使 LLM 回傳部分欄位缺失，仍能回傳完整結構。

    回傳：
    - Dict[str, Any]: 清理後的計畫 payload。
    """
    plan = _default_agentic_plan(question)
    plan["intent"] = str(payload.get("intent") or plan["intent"]).strip() or plan["intent"]
    plan["strategy"] = _normalize_strategy(str(payload.get("strategy") or plan["strategy"]))
    plan["must_have_paths"] = _clean_string_list(payload.get("must_have_paths"))
    plan["forbidden_patterns"] = _clean_string_list(payload.get("forbidden_patterns"))
    plan["risk_hypotheses"] = _clean_string_list(payload.get("risk_hypotheses"))

    output_contract = payload.get("output_contract")
    if isinstance(output_contract, dict):
        plan["output_contract"] = {"columns": _clean_string_list(output_contract.get("columns"))}

    return plan


def _sanitize_reactor_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    清理並正規化 Reactor Agent 回傳的 JSON payload。

    主要用途：
    - 確保 cypher 欄位非空（空值直接拋出 ValueError）。
    - 移除 cypher 中可能的 Markdown 標記。
    - 正規化 self_checks 欄位為 bool 型別。

    回傳：
    - Dict[str, Any]: 含 cypher, assumptions, self_checks 的乾淨 payload。
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
    """
    清理並正規化 Critic Agent 回傳的 JSON payload。

    主要用途：
    - 確保 verdict 只能是合法值（accept / replan / fail_fast）。
    - 逐筆清理 issues 列表，保留有效的 code / message / severity，
      並限制最多 12 筆避免 payload 過大。
    - next_strategy 正規化為合法策略名稱。

    回傳：
    - Dict[str, Any]: 含 verdict, issues, repair_actions, next_strategy 的乾淨 payload。
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
    """
    清理並正規化 Replanner Agent 回傳的 JSON payload。

    主要用途：
    - 確保 strategy 為合法值；若缺失則回退至 fallback_strategy。
    - 正規化 delta_actions, tightened_constraints, stop_if 為有效字串列表。

    回傳：
    - Dict[str, Any]: 含 strategy, delta_actions, tightened_constraints, stop_if 的乾淨 payload。
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
    """
    呼叫 LLM 並要求其回傳嚴格的 JSON object，內建重試機制。

    主要用途：
    - 作為所有 Agent（Planner、Reactor、Critic、Replanner）與 LLM 溝通的統一入口。
    - 若 LLM 回傳無法解析的輸出，則在 prompt 中附加錯誤訊息後重試（最多 _AGENT_JSON_RETRY_LIMIT 次）。
    - 每次重試時自動加大 token 上限，應對 LLM 因 token 截斷導致 JSON 不完整的情況。

    參數：
    - system_prompt: Agent 角色定義的 System prompt。
    - user_prompt: 包含輸出 schema 與上下文的 User prompt。
    - max_tokens: 初始 token 上限。
    - nl2cypher_provider / nl2cypher_model: LLM 供應商與模型覆蓋。

    回傳：
    - Dict[str, Any]: 成功解析的 JSON object。
    """
    # === 解析設定、初始化 token 預算 ===
    resolved_provider = _resolve_nl2cypher_provider(nl2cypher_provider)
    resolved_model = _resolve_nl2cypher_model(
        provider_override=resolved_provider,
        model_override=nl2cypher_model,
    )
    retry_tokens = max(1, int(max_tokens))
    last_error: Exception | None = None

    # === 重試迴圈：最多 _AGENT_JSON_RETRY_LIMIT 次 ===
    for attempt in range(1, _AGENT_JSON_RETRY_LIMIT + 1):
        retry_prompt = user_prompt
        # --- 若為重試，在 prompt 末尾附加前次解析錯誤，引導 LLM 修正輸出格式 ---
        if attempt > 1 and last_error is not None:
            retry_prompt = (
                f"{user_prompt}\n\n"
                "前次輸出無法解析為合法 JSON object。"
                "請只輸出一個 JSON 物件，不要 markdown、不要註解、不要多餘文字。\n"
                f"parse_error={last_error}"
            )

        try:
            # --- 呼叫 LLM，要求回傳可解析的 JSON dict ---
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
            # --- 致命錯誤（如 API key 缺失）直接向上拋出，重試無意義 ---
            if "GEMINI_API_KEY is required" in message:
                raise
            if attempt >= _AGENT_JSON_RETRY_LIMIT:
                raise
            # 每次重試加大 token 上限（最大 8192），應對 JSON 因截斷而不完整
            retry_tokens = min(max(retry_tokens * 2, retry_tokens + 256), 8192)

    raise RuntimeError(f"Agent JSON call failed after retries: {last_error}")


def _run_planner_agent(
    *,
    question: str,
    schema: SchemaSnapshot,
    failure_chain: List[str],
    nl2cypher_provider: str | None = None,
    nl2cypher_model: str | None = None,
) -> Dict[str, Any]:
    """
    執行 Planner Agent：分析問題並規劃 Cypher 生成策略。

    主要用途：
    - 在 ReAct 迴圈開始前（或重規劃後）呼叫，讓 LLM 決定查詢策略
      （single_query / union / staged_merge）。
    - 輸入包含問題、Schema、以往失敗記錄與預設計畫，
      有助於 LLM 避免重蹈覆轍。

    回傳：
    - Dict[str, Any]: 清理後的計畫 payload（intent, strategy, must_have_paths 等）。
    """
    # === 準備預設計畫（作為 LLM 無法給出合法輸出時的保底基準）===
    default_plan = _default_agentic_plan(question)
    system_prompt = (
        "你是 NL2Cypher Planner。"
        "請以 strict JSON 規劃查詢策略，只能回傳 JSON object。"
    )
    # === 組合 Planner Prompt（含 Schema 上下文與近期失敗記錄）===
    # failure_chain 只傳最近 5 筆，避免 prompt 過長
    user_prompt = (
        "輸出 schema:\n"
        '{"intent":"...", "strategy":"single_query|union|staged_merge",'
        '"must_have_paths":[...], "forbidden_patterns":[...],'
        '"output_contract":{"columns":[...]}, "risk_hypotheses":[...]}\n\n'
        f"Question:\n{question}\n\n"
        f"Schema:\n{schema.schema_text}\n\n"
        f"failure_chain={json.dumps(failure_chain[-5:], ensure_ascii=False)}\n"
        f"default_plan={json.dumps(default_plan, ensure_ascii=False)}"
    )

    # === 呼叫 Agent，清理並回傳計畫 payload ===
    payload = _call_agent_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=NL2CYPHER_AGENTIC_PLAN_TOKENS,
        nl2cypher_provider=nl2cypher_provider,
        nl2cypher_model=nl2cypher_model,
    )
    return _sanitize_planner_payload(payload, question=question)


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
    """
    執行 Reactor Agent：依據 Planner 的計畫生成候選 Cypher 查詢。

    主要用途：
    - 在每個 ReAct 迴圈輪次中負責實際的 Cypher 生成工作。
    - 接收計畫（strategy、must_have_paths 等）、Schema、
      前次失敗的 Cypher 與錯誤訊息，產出新的候選 Cypher。
    - 同時要求 LLM 輸出自我檢查結果（schema_grounded, projection_consistent）。

    回傳：
    - Dict[str, Any]: 清理後的 reactor payload（cypher, assumptions, self_checks）。
    """
    # === 組合 Reactor Prompt（角色設定、輸出 schema 規格、當前執行上下文）===
    system_prompt = (
        "你是 NL2Cypher Reactor。"
        "依照規劃產生可執行 Cypher，只能回傳 strict JSON object。"
    )
    # previous_cypher 與 last_error 讓 LLM 知道上一輪為何失敗，避免重蹈覆轍
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

    # === 呼叫 Agent，清理並回傳 Cypher payload ===
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
    runtime_error: str,
    round_idx: int,
    repeated: bool,
) -> Dict[str, Any]:
    """
    在 Critic Agent 呼叫失敗時，以規則為基礎產生安全的回退裁決。

    主要用途：
    - 作為 _run_critic_agent 的保底邏輯：當 LLM 無法回傳合法 JSON 時，
      依照以下規則自動決定裁決結果：
      * 有結果且無錯誤 → accept
      * 有執行時錯誤或重複 Cypher → replan 或 fail_fast（視輪次而定）
    - 也作為傳遞給 Critic Agent Prompt 的「fallback decision」參考，
      讓 LLM 可以對照確認自己的判斷。

    回傳：
    - Dict[str, Any]: 含 verdict, issues, repair_actions, next_strategy 的裁決字典。
    """
    if rows and not runtime_error:
        return {"verdict": "accept", "issues": [], "repair_actions": [], "next_strategy": "single_query"}

    issues: List[Dict[str, str]] = []
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
    runtime_error: str,
    round_idx: int,
    repeated: bool,
    nl2cypher_provider: str | None = None,
    nl2cypher_model: str | None = None,
) -> Dict[str, Any]:
    """
    執行 Critic Agent：評估 Cypher 執行結果，決定下一步動作。

    主要用途：
    - 在每個 ReAct 輪次中，於 Reactor 執行後呼叫，負責判斷：
      * accept：結果符合預期，迴圈成功結束。
      * replan：結果不佳或有錯誤，需調整策略後重試。
      * fail_fast：無法修復或重複陷入相同錯誤，立即中止。
    - 將 fallback_decision 一併傳入 prompt，讓 LLM 可對照安全基準做裁決。

    回傳：
    - Dict[str, Any]: 清理後的裁決 payload（verdict, issues, repair_actions, next_strategy）。
    """
    # === 計算安全回退決策（fallback），同時作為 prompt 的參考基準 ===
    fallback = _fallback_critic_decision(
        rows=rows,
        runtime_error=runtime_error,
        round_idx=round_idx,
        repeated=repeated,
    )

    # === 組合 Critic Prompt（角色設定、裁決選項、執行結果的完整上下文）===
    system_prompt = (
        "你是 NL2Cypher Critic。"
        "判斷目前結果要 accept / replan / fail_fast，只能回傳 strict JSON object。"
    )
    # fallback decision 附在 prompt 末尾，供 LLM 參考但不強制採用
    user_prompt = (
        "輸出 schema:\n"
        '{"verdict":"accept|replan|fail_fast", "issues":[{"code":"...", "message":"...", "severity":"low|med|high"}],'
        '"repair_actions":[...], "next_strategy":"single_query|union|staged_merge"}\n\n'
        f"Question:\n{question}\n\n"
        f"Plan:\n{json.dumps(plan, ensure_ascii=False)}\n\n"
        f"Cypher:\n{cypher}\n\n"
        f"Rows count: {len(rows)}\n"
        f"Runtime error: {runtime_error or '(none)'}\n"
        f"Repeated cypher: {repeated}\n"
        f"Round: {round_idx}/{NL2CYPHER_AGENTIC_MAX_ROUNDS}\n"
        f"Fallback decision:\n{json.dumps(fallback, ensure_ascii=False)}"
    )

    # === 呼叫 Agent，清理並回傳裁決 payload ===
    payload = _call_agent_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=NL2CYPHER_AGENTIC_CRITIC_TOKENS,
        nl2cypher_provider=nl2cypher_provider,
        nl2cypher_model=nl2cypher_model,
    )
    return _sanitize_critic_payload(payload)


def _fallback_replan(plan: Dict[str, Any], last_error: str) -> Dict[str, Any]:
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
        "tightened_constraints": ["avoid repeated invalid patterns"],
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
    """
    執行 Replanner Agent：依 Critic 裁決結果調整下一輪查詢策略。

    主要用途：
    - 當 Critic 裁決為 replan 時呼叫，負責根據錯誤訊息、
      Critic 的修復建議與歷史失敗記錄，決定下一輪要採用的策略
      （strategy、delta_actions、tightened_constraints、stop_if）。
    - 回傳的 payload 由 _merge_replan_into_plan 合併回現有計畫。

    回傳：
    - Dict[str, Any]: 清理後的重規劃 payload（strategy, delta_actions 等）。
    """
    # === 計算回退策略（依錯誤類型啟發式地切換 strategy）===
    fallback = _fallback_replan(plan, last_error)

    # === 組合 Replanner Prompt（含 Critic 意見、歷史失敗鏈與回退建議）===
    system_prompt = (
        "你是 NL2Cypher Replanner。"
        "依 critic 結果調整下一輪策略，只能回傳 strict JSON object。"
    )
    # failure_chain 只傳最近 6 筆，幫助 LLM 識別反覆出現的失敗模式
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

    # === 呼叫 Agent，以回退策略為底線清理 payload ===
    payload = _call_agent_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=NL2CYPHER_AGENTIC_CRITIC_TOKENS,
        nl2cypher_provider=nl2cypher_provider,
        nl2cypher_model=nl2cypher_model,
    )
    return _sanitize_replanner_payload(payload, fallback_strategy=fallback["strategy"])


def _merge_replan_into_plan(plan: Dict[str, Any], replan_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    將 Replanner 的調整結果合併回現有計畫，回傳新計畫。

    主要用途：
    - 更新 strategy 為重規劃後的策略。
    - 累積 tightened_constraints 到 risk_hypotheses，stop_if 到 forbidden_patterns，
      讓後續輪次的 Reactor 能避免已知的無效模式。

    回傳：
    - Dict[str, Any]: 合併後的計畫字典（深拷貝自原計畫，不修改原物件）。
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


def _issues_to_text(runtime_error: str, critic_payload: Dict[str, Any]) -> str:
    """
    將 runtime_error 與 Critic payload 中的 issues 合併為單一純文字字串。

    主要用途：
    - 供下一輪 Reactor / Replanner 使用，作為 last_error 的摘要說明。
    - 去除重複訊息（dict.fromkeys 保留順序並去重），以分號串接回傳。

    回傳：
    - str: 去重後以「; 」分隔的錯誤訊息字串。
    """
    messages: List[str] = []
    if runtime_error:
        messages.append(runtime_error)
    for item in critic_payload.get("issues", []):
        if isinstance(item, dict):
            text = str(item.get("message") or "").strip()
            if text:
                messages.append(text)
    deduped = list(dict.fromkeys(messages))
    return "; ".join(deduped)


def _run_agentic_query_loop(
    *,
    question: str,
    schema: SchemaSnapshot,
    driver: Driver,
    entity_names: List[str],
    nl2cypher_provider: str | None = None,
    nl2cypher_model: str | None = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    執行 Agentic Loop（代理迴圈），負責多輪式的 Cypher 生成、執行、評估與重規劃。

    主要用途：
    - 實作 ReAct (Reasoning and Acting) 機制，確保生成的 Cypher 查詢是正確且有結果的。
    - 包含四個階段：
      1. Planner: 分析問題並決定查詢策略（如 single_query, union 等）。
      2. Reactor: 根據策略產生候選的 Cypher，並進行實體名稱連結 (Entity Linking)。
      3. Critic: 執行 Cypher 並檢驗結果，判斷是否成功 (accept)、需要重試 (replan) 或是終止 (fail_fast)。
      4. Replanner: 如果失敗，則根據錯誤訊息調整策略並進入下一輪。

    參數：
    - question: 使用者的自然語言問題。
    - schema: Neo4j Schema 快照。
    - driver: Neo4j 資料庫連線。
    - entity_names: 用於實體連結的實體名稱清單。
    - progress_callback: (選填) 每個 Agent 執行階段結束後的回呼函式，用於回報進度。

    回傳：
    - Dict[str, Any]: 最終結果字典，包含成功產生的 cypher、查詢出的 rows 資料以及執行的嘗試次數。
    """
    # === 初始化：解析 Provider/Model、建立預設計畫與追蹤元資料 ===
    question_id = str(uuid4())
    started = time.perf_counter()
    failure_chain: List[str] = []
    resolved_provider = _resolve_nl2cypher_provider(nl2cypher_provider)
    resolved_model = _resolve_nl2cypher_model(
        provider_override=resolved_provider,
        model_override=nl2cypher_model,
    )
    default_plan = _default_agentic_plan(question)
    plan = deepcopy(default_plan)
    # trace 記錄整個 Agentic Loop 的執行歷史，供除錯與前端進度顯示使用
    trace: Dict[str, Any] = {
        "stage": "planner",
        "round_count": 0,
        "replan_count": 0,
        "final_strategy": plan["strategy"],
        "failure_chain": [],
        "llm_provider": resolved_provider,
        "llm_model": resolved_model,
        "plan_initial": deepcopy(default_plan),
        "planner_plan": deepcopy(plan),
        "plan_final": deepcopy(plan),
        "rounds": [],
    }

    def _emit_progress(stage: str, detail: str = "") -> None:
        if not progress_callback:
            return
        trace["stage"] = stage
        trace["failure_chain"] = list(failure_chain)
        trace["plan_final"] = plan  # deepcopy(trace) below already copies this
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
            "agentic_trace": deepcopy(trace),
        }
        try:
            progress_callback(payload)
        except Exception:
            return

    # === Planner 階段：分析問題、決定查詢策略（在所有 ReAct 輪次開始前執行一次）===
    _emit_progress("planner", "Planning query strategy")

    try:
        plan = _run_planner_agent(
            question=question,
            schema=schema,
            failure_chain=failure_chain,
            nl2cypher_provider=resolved_provider,
            nl2cypher_model=resolved_model,
        )
    except Exception as exc:
        # Planner 失敗時，保留 default_plan 繼續執行（不中止迴圈）
        _append_failure(failure_chain, f"planner_error: {exc}")
        _emit_progress("planner", "Planner failed, fallback strategy in use")

    trace["final_strategy"] = plan["strategy"]
    trace["planner_plan"] = deepcopy(plan)
    trace["plan_final"] = deepcopy(plan)
    planner_detail = f"Planner strategy={trace['final_strategy']}"
    if resolved_provider:
        planner_detail += f", provider={resolved_provider}"
    if resolved_model:
        planner_detail += f", model={resolved_model}"
    _emit_progress("planner", planner_detail)
    previous_cypher = ""
    last_error = ""
    seen_cypher_counts: Dict[str, int] = {}

    # === 主要 ReAct 迴圈（Reactor → Critic → Replanner，最多 NL2CYPHER_AGENTIC_MAX_ROUNDS 輪）===
    for round_idx in range(1, NL2CYPHER_AGENTIC_MAX_ROUNDS + 1):
        # --- Reactor：根據計畫生成候選 Cypher ---
        trace["stage"] = "react"
        trace["round_count"] = round_idx
        _emit_progress("react", f"Round {round_idx}: generating candidate cypher")
        round_entry: Dict[str, Any] = {
            "round": round_idx,
            "strategy_before": str(plan.get("strategy") or "single_query"),
        }
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
            # Reactor 失敗時，以 natural_language_to_cypher 作為簡化回退
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
                # 雙重失敗：使用空查詢佔位，確保流程繼續而非崩潰
                _append_failure(failure_chain, f"reactor_fallback_error(r{round_idx}): {fallback_exc}")
                fallback_cypher = previous_cypher or "MATCH (n) WHERE false RETURN n LIMIT 0"
            reactor_payload = {
                "cypher": fallback_cypher,
                "assumptions": ["fallback via natural_language_to_cypher"],
                "self_checks": {"schema_grounded": False, "projection_consistent": False},
            }

        # --- 實體連結（Entity Linking）：修正 Cypher 中的實體名稱拼寫 ---
        candidate_cypher = str(reactor_payload.get("cypher") or "").strip()
        linked_cypher = link_entity_literals(candidate_cypher, entity_names)
        checked_cypher = linked_cypher
        round_entry["reactor"] = deepcopy(reactor_payload)
        round_entry["candidate_cypher"] = candidate_cypher
        round_entry["linked_cypher"] = linked_cypher

        # --- 重複 Cypher 偵測：避免 Agent 陷入無進展的死循環 ---
        count = seen_cypher_counts.get(checked_cypher, 0) + 1
        seen_cypher_counts[checked_cypher] = count
        repeated = count > _AGENTIC_REPEAT_CYPHER_LIMIT
        if repeated:
            _emit_progress("critic", f"Round {round_idx}: repeated cypher detected")

        # --- 執行 Cypher；若無結果則嘗試放寬 Organization 名稱比對 ---
        round_entry["checked_cypher"] = checked_cypher
        round_entry["deterministic_issues"] = []
        rows: List[Dict[str, Any]] = []
        runtime_error = ""
        try:
            rows = execute_cypher(driver, checked_cypher)
            if not rows:
                # 零結果時，嘗試將精確 Organization 名稱比對放寬為 CONTAINS 模糊比對
                relaxed_cypher = _relax_exact_organization_name_match(checked_cypher)
                if relaxed_cypher and relaxed_cypher != checked_cypher:
                    relaxed_rows = execute_cypher(driver, relaxed_cypher)
                    if relaxed_rows:
                        checked_cypher = relaxed_cypher
                        rows = relaxed_rows
        except Neo4jError as exc:
            runtime_error = str(exc)

        round_entry["checked_cypher"] = checked_cypher
        round_entry["runtime_error"] = runtime_error
        round_entry["rows_count"] = len(rows)

        # --- Critic：評估執行結果，決定 accept / replan / fail_fast ---
        trace["stage"] = "critic"
        _emit_progress("critic", f"Round {round_idx}: evaluating candidate")
        critic_payload: Dict[str, Any]
        try:
            critic_payload = _run_critic_agent(
                question=question,
                plan=plan,
                cypher=checked_cypher,
                rows=rows,
                runtime_error=runtime_error,
                round_idx=round_idx,
                repeated=repeated,
                nl2cypher_provider=resolved_provider,
                nl2cypher_model=resolved_model,
            )
        except Exception as exc:
            # Critic 失敗時，以規則型回退決策繼續（不中止）
            _append_failure(failure_chain, f"critic_error(r{round_idx}): {exc}")
            _emit_progress("critic", f"Round {round_idx}: critic failed, using fallback decision")
            critic_payload = _fallback_critic_decision(
                rows=rows,
                runtime_error=runtime_error,
                round_idx=round_idx,
                repeated=repeated,
            )

        round_entry["critic"] = deepcopy(critic_payload)
        verdict = str(critic_payload.get("verdict") or "replan")
        round_entry["verdict"] = verdict
        round_error = _issues_to_text(runtime_error, critic_payload)
        last_error = round_error or "agentic loop requested replan"

        # 記錄每輪的關鍵指標（verdict、strategy、延遲）供日誌分析
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

        # --- accept：結果有效，回傳成功結果並結束迴圈 ---
        if verdict == "accept" and rows:
            round_entry["strategy_after"] = str(plan.get("strategy") or "single_query")
            trace["rounds"].append(round_entry)
            trace["stage"] = "done"
            trace["final_strategy"] = plan.get("strategy", "single_query")
            trace["failure_chain"] = list(failure_chain)
            trace["plan_final"] = deepcopy(plan)
            _emit_progress("done", f"Completed in round {round_idx}")
            return {
                "question": question,
                "cypher": checked_cypher,
                "rows": rows,
                "attempt": round_idx,
                "agentic_trace": deepcopy(trace),
            }

        # --- fail_fast：無法修復，提前中止迴圈 ---
        if verdict == "fail_fast":
            round_entry["strategy_after"] = str(plan.get("strategy") or "single_query")
            trace["rounds"].append(round_entry)
            trace["stage"] = "fail_fast"
            _emit_progress("fail_fast", f"Stopped at round {round_idx}")
            break

        # --- replan：調整策略，準備下一輪 Reactor ---
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
            # Replanner 失敗時，以規則型回退策略繼續
            _append_failure(failure_chain, f"replanner_error(r{round_idx}): {exc}")
            _emit_progress("replan", f"Round {round_idx}: replanner failed, using fallback")
            replan_payload = _fallback_replan(plan, last_error)

        # 將重規劃結果合併回現有計畫（累積 forbidden_patterns 與 risk_hypotheses）
        round_entry["replan"] = deepcopy(replan_payload)
        plan = _merge_replan_into_plan(plan, replan_payload)
        trace["final_strategy"] = plan.get("strategy", "single_query")
        trace["plan_final"] = deepcopy(plan)
        round_entry["strategy_after"] = str(plan.get("strategy") or "single_query")
        trace["rounds"].append(round_entry)
        previous_cypher = checked_cypher
        _emit_progress("replan", f"Round {round_idx}: next strategy={trace['final_strategy']}")

    # === 迴圈耗盡：達到最大輪次仍無有效結果，記錄錯誤日誌並拋出例外 ===
    trace["stage"] = "exhausted"
    trace["final_strategy"] = plan.get("strategy", "single_query")
    trace["failure_chain"] = list(failure_chain)
    trace["plan_final"] = deepcopy(plan)
    _emit_progress("exhausted", "Reached max rounds without accepted result")

    _logger.info(
        "agentic_nl2cypher_final %s",
        json.dumps(
            {
                "question_id": question_id,
                "round": trace.get("round_count", 0),
                "verdict": "error",
                "error_code": "RETRY_EXHAUSTED",
                "strategy": trace.get("final_strategy", "single_query"),
                "latency_ms": int((time.perf_counter() - started) * 1000),
            },
            ensure_ascii=False,
        ),
    )
    raise RuntimeError(f"Cypher generation failed after retries: {last_error}")


def answer_with_manual_prompt(
    question: str,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    nl2cypher_provider: str | None = None,
    nl2cypher_model: str | None = None,
) -> Dict[str, Any]:
    """
    Method 2 的公開入口：以手動 Prompt 方式執行自然語言查詢（Agentic Loop）。

    主要用途：
    - 建立 Neo4j Driver 連線，載入 Schema 快照與實體名稱，
      再委由 _run_agentic_query_loop 執行完整的 Planner→Reactor→Critic→Replanner 迴圈。
    - 確保 Driver 在函式結束後（無論成功或例外）一定被關閉。

    參數：
    - question: 使用者的自然語言問題。
    - progress_callback: (選填) 每個 Agent 階段完成後的進度回呼，用於即時通知前端。
    - nl2cypher_provider / nl2cypher_model: LLM 供應商與模型覆蓋。

    回傳：
    - Dict[str, Any]: 包含 question, cypher, rows, attempt, agentic_trace 的結果字典。
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        schema = load_schema_snapshot(driver)
        entity_names = fetch_entity_names(driver)

        return _run_agentic_query_loop(
            question=question,
            schema=schema,
            driver=driver,
            entity_names=entity_names,
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
