"""QA and general chat service functions."""

from __future__ import annotations

import inspect
import json
from collections.abc import Mapping
from typing import Any, Callable, Dict, List, Optional

from backend.config.settings import get_general_chat_settings, get_kg_qa_settings
from backend.llm_kg import llm_client


def _load_kg_query_executor() -> Any:
    """延遲載入 (Lazy Load) 手動提示 (Manual Prompt) 模式的 KG 查詢執行器。
    
    此設計是為了避免在一啟動時就強制載入 `backend.llm_kg` 等笨重的模型套件。
    只有當使用者發起 QA 查詢且選用 `manual` 引擎時，才會載入 `answer_with_manual_prompt`。
    
    回傳值:
        Callable: 用來將自然語言查詢轉換為 Cypher 的處理函式。
    """
    try:
        from backend.llm_kg.nl2cypher import answer_with_manual_prompt
    except ImportError as exc:
        raise RuntimeError(
            "Unable to load KG query module. Install backend dependencies before calling KG endpoints."
        ) from exc
    return answer_with_manual_prompt


def _load_graph_chain_query_executor() -> Any:
    """延遲載入 LangChain GraphQA Chain 模式的 KG 查詢執行器。
    
    跟 _load_kg_query_executor 類似，只是這裡載入的是基於 `graph_chain` (例如 LangChain 架構)
    的查詢引擎 `query_with_graph_chain`。
    
    回傳值:
        Callable: 用來執行 GraphQA 流程的函式。
    """
    try:
        from backend.llm_kg.nl2cypher import query_with_graph_chain
    except ImportError as exc:
        raise RuntimeError(
            "Unable to load KG graph-chain module. Install backend dependencies before calling KG endpoints."
        ) from exc
    return query_with_graph_chain


def _invoke_kg_query_executor(
    *,
    executor: Callable[..., Dict[str, Any]],
    question: str,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]],
    nl2cypher_provider: Optional[str],
    nl2cypher_model: Optional[str],
) -> Dict[str, Any]:
    """以相容方式呼叫 KG query executor，支援舊版與新版函式簽名。"""
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    kwargs: Dict[str, Any] = {}
    if progress_callback is not None:
        kwargs["progress_callback"] = progress_callback
    if nl2cypher_provider:
        kwargs["nl2cypher_provider"] = nl2cypher_provider
    if nl2cypher_model:
        kwargs["nl2cypher_model"] = nl2cypher_model

    try:
        signature = inspect.signature(executor)
    except (TypeError, ValueError):
        return executor(question, **kwargs)

    params = signature.parameters
    supports_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
    if supports_kwargs:
        return executor(question, **kwargs)

    filtered_kwargs = {key: value for key, value in kwargs.items() if key in params}
    return executor(question, **filtered_kwargs)


def _normalize_query_rows(rows_value: Any) -> List[Dict[str, Any]]:
    """將任意 rows 輸入轉為 API 相容的 list[dict]。"""
    if not isinstance(rows_value, list):
        return []

    normalized_rows: List[Dict[str, Any]] = []
    for row in rows_value:
        if isinstance(row, dict):
            normalized_rows.append(row)
        else:
            normalized_rows.append({"value": row})
    return normalized_rows


def _stringify_query_value(value: Any) -> str:
    """將從 Neo4j 查出來的各種資料型別格式化為適合閱讀的字串。
    
    圖形資料庫查詢回來的資料可能是字串、整數、陣列、甚至是一個包含多個屬性的 Node 節點。
    此函式負責將它們攤平並轉為字串，方便後續丟給 LLM 總結或是直接顯示。
    
    範例：
    - `None` -> "N/A"
    - `["A", "B", "C", "D", "E"]` -> "A、B、C、D..." (最多顯示 4 個，後面加上 ...)
    - `{"name": "張三", "age": 30}` -> 優先提取 name 欄位，回傳 "張三"
    - `{"job": "工程師", "city": "台北"}` -> "{job:工程師, city:台北}"
    
    參數:
        value (Any): 任意型別的欄位值。
        
    回傳值:
        str: 格式化後的字串表示。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    if value is None:
        return "N/A"
    if isinstance(value, (str, int, float, bool)):
        return str(value)
        
    # 如果是列表 (List)，最多取出前 4 筆並用頓號連接，超過的話在最後加上 "..."
    if isinstance(value, list):
        if not value:
            return "[]"
        items = [_stringify_query_value(item) for item in value[:4]]
        suffix = "..." if len(value) > 4 else ""
        return "、".join(items) + suffix
        
    # 如果是字典 (Mapping) 或物件 (帶有 get 和 keys 方法)，如 Neo4j 的 Node 物件
    if isinstance(value, Mapping) or (hasattr(value, "get") and hasattr(value, "keys")):
        mapping: Dict[str, Any] = {}
        if isinstance(value, Mapping):
            mapping = dict(value)
        else:
            try:
                mapping = {key: value.get(key) for key in value.keys()}
            except Exception:
                # 若直接操作失敗，嘗試讀取隱藏的屬性 _properties (Neo4j driver 常見屬性)
                props = getattr(value, "_properties", None)
                if isinstance(props, dict):
                    mapping = dict(props)

        # 優先回傳字典裡的 name，因為通常這最具代表性
        if isinstance(mapping.get("name"), str) and mapping.get("name"):
            return str(mapping["name"])
            
        # 若沒有 name，就將前 3 個欄位組合成 "{key:value, ...}" 的格式
        parts = []
        for key, inner in list(mapping.items())[:3]:
            parts.append(f"{key}:{_stringify_query_value(inner)}")
        if not parts:
            return "{}"
        suffix = "..." if len(mapping) > 3 else ""
        return "{" + ", ".join(parts) + suffix + "}"
    return str(value)


def _display_query_key(key: str) -> str:
    """清理並取得適合顯示的鍵值名稱。
    
    從 Cypher 查回來的欄位名可能會帶有變數別名，例如 "e.name" 或 "r.type"。
    此函式負責將前綴去除，只保留最後的屬性名稱 "name" 或 "type"。
    
    參數:
        key (str): 原始欄位鍵值。
        
    回傳值:
        str: 清理後的顯示名稱。
    """
    text = str(key)
    if "." in text:
        return text.split(".", 1)[1]
    return text


def _is_metadata_key(key: str) -> bool:
    """判斷欄位是否屬於不適合直接回覆給使用者的技術性欄位。"""
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    raw = _display_query_key(key).strip()
    normalized = raw.replace("_", "").replace(" ", "").lower()
    hidden_exact = {
        "normalizedname",
        "normalized_name",
        "canonicalname",
        "canonical_name",
        "entityid",
        "entity_id",
        "nodeid",
        "node_id",
        "uuid",
        "正規化名稱",
        "正規化名",
        "規範化名稱",
        "规范化名称",
    }
    if raw in hidden_exact or normalized in hidden_exact:
        return True
    hidden_contains = ("normalized", "canonical", "entityid", "nodeid", "uuid", "正規化", "規範化", "规范化")
    return any(token in normalized for token in hidden_contains)


def _filter_user_facing_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """過濾掉技術欄位與空值，保留適合回覆給使用者的資料。"""
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        visible_row: Dict[str, Any] = {}
        for key, value in row.items():
            if _is_metadata_key(key):
                continue
            text = _stringify_query_value(value).strip()
            if text in {"", "N/A", "[]", "{}"}:
                continue
            visible_row[key] = value
        if visible_row:
            filtered.append(visible_row)
    return filtered


def _format_single_value_answer(question: str, value: str) -> str:
    """將單值結果轉為自然語句，避免回覆過度模板化。"""
    cleaned_q = question.strip()
    if "誰" in cleaned_q or "哪裡" in cleaned_q or "哪里" in cleaned_q or "何處" in cleaned_q or "何地" in cleaned_q:
        return f"{value}。"
    return f"{cleaned_q.rstrip('？?')}：{value}。"


def _summarize_query_rows(question: str, rows: List[Dict[str, Any]]) -> str:
    """當不用 LLM 總結或是 LLM 發生錯誤時，透過基於規則的模板生成回答。
    
    這個備用方案 (Fallback) 負責分析查回來的 rows，並組合出一段人類看得懂的回答。
    
    運作邏輯：
    1. 如果完全沒有回傳結果，直接回答「找不到相關資料」。
    2. 如果有結果，會先過濾掉 UUID 等隱藏欄位，只留下有效欄位 (user_rows)。
    3. 如果結果包含名為 "name" 的欄位（例如 {"name": "劉德華"}），會把這些名字抽出來組合。
       - 例如：查到 3 筆，輸出：「使用者詢問的問題：張學友、劉德華、黎明。」
    4. 否則，就把前 5 筆資料的每個欄位都用冒號連起來。
       - 例如：「類型：員工；年齡：30。另有其他結果...」
    
    參數:
        question: 使用者的原始問題。
        rows: 從 Neo4j 取出的多筆資料清單。
        
    回傳值:
        str: 組裝好的人類可讀總結文字。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    if not rows:
        return f"目前在知識圖譜中找不到與「{question}」直接相關的資料。"

    user_rows = _filter_user_facing_rows(rows)
    effective_rows = user_rows or rows
    first_row_keys = list(effective_rows[0].keys())
    
    # 尋找是否有名稱為 "name" 的欄位
    name_keys = [key for key in first_row_keys if _display_query_key(key).lower() == "name"]
    if name_keys:
        key = name_keys[0]
        values: List[str] = []
        for row in effective_rows:
            if key not in row:
                continue
            value = _stringify_query_value(row[key]).strip()
            if value and value not in {"N/A", "[]", "{}"}:
                values.append(value)
                
        # 移除重複出現的值
        unique_values = list(dict.fromkeys(values))
        if unique_values:
            # 取出前 8 個作為代表
            top_values = unique_values[:8]
            suffix = f" 等 {len(unique_values)} 筆" if len(unique_values) > 8 else ""
            if len(top_values) == 1:
                return _format_single_value_answer(question, top_values[0])
            return f"{question}：{'、'.join(top_values)}{suffix}。"

    # 若沒有 name 欄位，但整個 dict 只有 1 個欄位時 (例如 {"age": 20})
    if len(first_row_keys) == 1:
        key = first_row_keys[0]
        values: List[str] = []
        for row in effective_rows:
            if key not in row:
                continue
            value = _stringify_query_value(row[key])
            if value and value not in {"N/A", "[]", "{}"}:
                values.append(value)

        unique_values = list(dict.fromkeys(values))
        if unique_values:
            top_values = unique_values[:8]
            suffix = f" 等 {len(unique_values)} 筆" if len(unique_values) > 8 else ""
            if len(top_values) == 1:
                return _format_single_value_answer(question, top_values[0])
            return f"{question}包含：{'、'.join(top_values)}{suffix}。"

    # 若有多個欄位，則將前 5 筆的各個欄位轉為 "Key: Value" 格式
    highlights: List[str] = []
    for row in effective_rows[:5]:
        cells = [f"{_display_query_key(key)}：{_stringify_query_value(value)}" for key, value in row.items()]
        if cells:
            highlights.append("；".join(cells))

    if not highlights:
        return f"查到 {len(rows)} 筆資料，但欄位內容較空，請展開 Rows 查看。"

    # 用 "。" 將所有筆數串聯
    summary = "；".join(highlights)
    if len(effective_rows) > 5:
        return f"{summary}。另有其他結果可再查看明細。"
    return f"{summary}。"


def _kg_qa_use_llm() -> bool:
    """取得當前是否開啟使用 LLM 來總結答案的設定。"""
    return get_kg_qa_settings().use_llm


def _kg_qa_model() -> Optional[str]:
    """取得 QA 回覆產生使用的 LLM 模型名稱 (例如 'gemini-1.5-pro')。"""
    return get_kg_qa_settings().model


def _kg_qa_temperature() -> float:
    """取得生成回答時的創造力參數 Temperature (0.0~1.0)。"""
    return get_kg_qa_settings().temperature


def _kg_qa_max_tokens() -> int:
    """取得生成回答時的長度限制 Max Tokens。"""
    return get_kg_qa_settings().max_tokens


def _format_rows_for_qa_prompt(rows: List[Dict[str, Any]]) -> str:
    """將 GraphDB 取出的 Rows 限制筆數後轉換為 JSON 字串。
    
    這樣 LLM 的 Prompt 就不會被無數筆相同或過長的資料給撐爆 Context Window。
    
    參數:
        rows: 從資料庫查出的資料列表。
        
    回傳值:
        str: 轉成文字格式的 JSON 內容。
    """
    max_rows = get_kg_qa_settings().max_rows_for_prompt
    limited = rows[: max(1, max_rows)]
    try:
        return json.dumps(limited, ensure_ascii=False, indent=2)
    except Exception:
        return str(limited)


def _generate_kg_answer_with_llm(*, question: str, cypher: str, rows: List[Dict[str, Any]]) -> str:
    """呼叫大型語言模型 (LLM)，將查詢回來的原始資料 (Rows) 總結成自然的回答。
    
    這負責了問答系統的最後一哩路，讓使用者不需要自己看 JSON 資料，而是得到「某某公司的創辦人是郭台銘」
    這種順暢的句型。
    
    運作邏輯：
    1. 組合 System Prompt，限制 LLM 不要透露內部細節 (不說 "從 Cypher 看出...")。
    2. 放入 `question`, `cypher` 以及格式化後的 `rows`。
    3. 發送請求取得 `answer`。
    4. 最後做個安全性檢查：驗證 LLM 回答的內容是否真的有包含 Rows 裡的字詞，防止它憑空捏造 (Hallucination)。
    
    參數:
        question: 使用者問的問題。
        cypher: 生成出的查詢語法，僅供 LLM 參考查詢意圖。
        rows: 在 Neo4j 查到的真實資料。
        
    回傳值:
        str: LLM 產生的人類閱讀友好回答。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    if not _kg_qa_use_llm():
        raise RuntimeError("KG_QA_USE_LLM disabled")

    # 定義嚴格的系統指令，要求 LLM 不要成為說書人，只需轉譯資料
    system_prompt = (
        "你是企業知識圖譜問答助理，請使用繁體中文，語氣自然、精簡。"
        "只回覆給使用者看的最終答案。"
        "Cypher 只用於理解查詢語意；Rows 是回答時的參考資料，且優先以 Rows 為準。"
        "不要提到資料筆數、欄位名稱、Cypher、rows、正規化名稱、normalizedName 或任何技術細節。"
        "若有多筆結果，直接列出關鍵答案即可。"
        "若無資料，明確說明目前缺少哪一類資料。"
        "不得捏造未出現在 rows 的事實。"
    )
    user_prompt = (
        f"問題：{question}\n"
        f"Cypher（查詢語意參考）：\n{cypher}\n\n"
        f"Rows(JSON，以下為回答時的參考資料，請以此為準)：\n{_format_rows_for_qa_prompt(rows)}\n\n"
        "請根據上述參考資料直接輸出自然語句答案。"
    )

    chat_settings = get_general_chat_settings()
    # 呼叫底層 LLM 客戶端
    answer = llm_client.chat_text(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=_kg_qa_model(),
        temperature=_kg_qa_temperature(),
        max_tokens=_kg_qa_max_tokens(),
        timeout_seconds=chat_settings.timeout_seconds,
    )

    normalized = answer.strip()
    if not normalized:
        raise RuntimeError("KG QA LLM returned empty answer")

    # 防幻覺 (Hallucination) 檢查
    # 取出 rows 前 3 筆的 value，確認 LLM 的回答裡至少出現過一次這些關鍵字
    sample_values: List[str] = []
    for row in rows[:3]:
        for value in row.values():
            text = _stringify_query_value(value).strip()
            if text and text not in {"N/A", "[]", "{}"}:
                sample_values.append(text)
    sample_values = list(dict.fromkeys(sample_values))[:6]
    
    # 若 LLM 回覆完全沒有提到我們資料庫撈出來的任何詞彙，可能是自行發揮，拋出例外進入 fallback 模板機制
    if sample_values and not any(value in normalized for value in sample_values):
        raise RuntimeError("KG QA LLM answer does not reference row values")

    return normalized


def query_kg(
    question: str,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    nl2cypher_provider: Optional[str] = None,
    nl2cypher_model: Optional[str] = None,
    query_engine: Optional[str] = None,
) -> Dict[str, Any]:
    """執行知識圖譜問答 (QA) 的總控流程。
    
    支援兩種引擎模式：
    1. "graph_chain": 使用高度抽象的外部工具（如 LangChain 的 GraphCypherQAChain），
       內部會自己完成 Cypher 產生 -> 執行查詢 -> 回答生成的步驟。
    2. "manual": 自行手動組裝 Prompt，更具掌控力。取得資料庫 rows 後再呼叫 `_generate_kg_answer_with_llm`
       來做總結回答。如果 LLM 當掉，會觸發 Fallback 降級為規則模板 (`_summarize_query_rows`)。
       
    參數:
        question (str): 查詢問題。
        progress_callback: 用來傳送處理進度的回呼函式（目前 `graph_chain` 模式不支援非同步回報）。
        nl2cypher_provider, nl2cypher_model: 將自然語言轉 Cypher 時的 LLM 設定。
        query_engine: 指定使用 "manual" 或 "graph_chain" 引擎。
        
    回傳值:
        Dict[str, Any]: 查詢與回答結果組合字典，包含 answer, rows, cypher 等。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("Question cannot be empty")

    resolved_engine = (query_engine or "manual").strip().lower()
    if resolved_engine not in {"manual", "graph_chain"}:
        raise ValueError("Unsupported query_engine. Expected one of: manual, graph_chain")

    # 分支 1：使用 LangChain 等 Graph Chain 工具
    if resolved_engine == "graph_chain":
        if progress_callback is not None:
            raise ValueError("query_engine=graph_chain does not support async progress callbacks")
            
        # 載入模組並執行
        query_with_graph_chain = _load_graph_chain_query_executor()
        raw_result = _invoke_kg_query_executor(
            executor=query_with_graph_chain,
            question=cleaned,
            progress_callback=None,
            nl2cypher_provider=nl2cypher_provider,
            nl2cypher_model=nl2cypher_model,
        )
        if not isinstance(raw_result, dict):
            raise RuntimeError("Unexpected KG graph-chain result format")

        # 正規化圖譜回傳的 rows
        normalized_rows = _normalize_query_rows(raw_result.get("rows"))

        result = dict(raw_result)
        result["question"] = str(result.get("question") or cleaned)
        result["cypher"] = str(result.get("cypher") or "")
        result["rows"] = normalized_rows
        result["query_engine"] = "graph_chain"
        result["engine_provider"] = str(result.get("engine_provider") or nl2cypher_provider or "").strip() or None
        result["engine_model"] = str(result.get("engine_model") or nl2cypher_model or "").strip() or None
        
        # 保存原始資料，供 debug 顯示
        if "graph_chain_raw" not in result:
            raw_blob = result.get("raw")
            if isinstance(raw_blob, dict):
                result["graph_chain_raw"] = raw_blob
            elif raw_blob is None:
                result["graph_chain_raw"] = {}
            else:
                result["graph_chain_raw"] = {"value": raw_blob}

        answer = str(result.get("answer") or "").strip()
        # 萬一 Graph Chain 沒有提供最終 answer，我們用規則模板來補上
        if not answer:
            answer = _summarize_query_rows(cleaned, normalized_rows)
        result["answer"] = answer
        return result

    # 分支 2：使用手刻流程 (Manual)
    answer_with_manual_prompt = _load_kg_query_executor()
    raw_result = _invoke_kg_query_executor(
        executor=answer_with_manual_prompt,
        question=cleaned,
        progress_callback=progress_callback,
        nl2cypher_provider=nl2cypher_provider,
        nl2cypher_model=nl2cypher_model,
    )
    if not isinstance(raw_result, dict):
        raise RuntimeError("Unexpected KG query result format")

    normalized_rows = _normalize_query_rows(raw_result.get("rows"))

    result = dict(raw_result)
    result["question"] = str(result.get("question") or cleaned)
    result["rows"] = normalized_rows
    result["query_engine"] = "manual"

    # 嘗試用 LLM 總結資料
    try:
        answer = _generate_kg_answer_with_llm(
            question=cleaned,
            cypher=str(result.get("cypher") or ""),
            rows=normalized_rows,
        )
        result["answer"] = answer
        result["answer_source"] = "qa_llm"  # 代表是由 LLM 回答的
    except Exception:
        # LLM 掛掉或幻覺失敗時，Fallback 退回使用規則模板
        result["answer"] = _summarize_query_rows(cleaned, normalized_rows)
        result["answer_source"] = "template_fallback"  # 代表這是用備用規則回答的

    return result


def _normalize_chat_history(history: Optional[List[Dict[str, str]]]) -> List[Dict[str, str]]:
    """清理歷史對話紀錄，過濾無效輸入並限制對話深度。
    
    確保送入 LLM 的對話陣列結構都是 {"role": "...", "content": "..."}，
    並且限制最多只送出過去的 20 則訊息，避免過度消耗 Token 與記憶體。
    
    參數:
        history: 前端傳來的舊對話記錄陣列。
        
    回傳值:
        List[Dict]: 乾淨的對話列表。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    if not history:
        return []

    normalized: List[Dict[str, str]] = []
    # 只取最後 20 筆對話，控制 Token 大小
    for message in history[-20:]:
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        # 過濾不合法的角色或是沒有內容的對話
        if role not in {"user", "assistant"} or not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def chat_general(message: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """普通的通用聊天入口 (不經過知識圖譜，純粹和 LLM 對話)。
    
    此處的 System Prompt 被設定為「溫和、支持性的陪伴助理」，不提供醫療診斷，專注於傾聽。
    
    參數:
        message (str): 使用者的最新訊息。
        history: 先前的聊天紀錄。
        
    回傳值:
        Dict[str, Any]: 包含 "answer" (回覆) 以及所使用的 "model" 等資訊。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    cleaned_message = message.strip()
    if not cleaned_message:
        raise ValueError("Message cannot be empty")

    system_prompt = (
        "你是一位溫和、支持性的聊天夥伴，請使用繁體中文。"
        "目標是陪伴失眠或情緒低落的使用者，語氣要穩定、簡潔、無評價。"
        "避免提供醫療診斷或藥物建議；若使用者提到自傷風險或急性危機，"
        "請鼓勵立即聯絡當地緊急資源與可信任的人。"
    )
    # 將 System Prompt、舊紀錄與新對話組裝成發送給 API 的格式
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(_normalize_chat_history(history))
    messages.append({"role": "user", "content": cleaned_message})

    # 取出設定並呼叫 LLM
    chat_settings = get_general_chat_settings()
    answer = llm_client.chat_text(
        messages=messages,
        temperature=chat_settings.temperature,
        max_tokens=chat_settings.num_predict,
        timeout_seconds=chat_settings.timeout_seconds,
    )
    
    cfg = llm_client.get_runtime_config()
    return {"answer": answer, "model": cfg.model, "provider": cfg.provider}
