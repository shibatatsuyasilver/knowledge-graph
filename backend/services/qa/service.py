"""QA and general chat service functions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

from backend.config.settings import get_general_chat_settings, get_kg_qa_settings
from backend.llm_kg import llm_client


def _load_kg_query_executor() -> Any:
    """執行 `_load_kg_query_executor` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    try:
        from backend.llm_kg.nl2cypher import answer_with_manual_prompt
    except ImportError as exc:
        raise RuntimeError(
            "Unable to load KG query module. Install backend dependencies before calling KG endpoints."
        ) from exc
    return answer_with_manual_prompt


def _stringify_query_value(value: Any) -> str:
    """執行 `_stringify_query_value` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    if value is None:
        return "N/A"
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        if not value:
            return "[]"
        items = [_stringify_query_value(item) for item in value[:4]]
        suffix = "..." if len(value) > 4 else ""
        return "、".join(items) + suffix
    if isinstance(value, Mapping) or (hasattr(value, "get") and hasattr(value, "keys")):
        mapping: Dict[str, Any] = {}
        if isinstance(value, Mapping):
            mapping = dict(value)
        else:
            try:
                mapping = {key: value.get(key) for key in value.keys()}
            except Exception:
                props = getattr(value, "_properties", None)
                if isinstance(props, dict):
                    mapping = dict(props)

        if isinstance(mapping.get("name"), str) and mapping.get("name"):
            return str(mapping["name"])
        parts = []
        for key, inner in list(mapping.items())[:3]:
            parts.append(f"{key}:{_stringify_query_value(inner)}")
        if not parts:
            return "{}"
        suffix = "..." if len(mapping) > 3 else ""
        return "{" + ", ".join(parts) + suffix + "}"
    return str(value)


def _display_query_key(key: str) -> str:
    """執行 `_display_query_key` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    text = str(key)
    if "." in text:
        return text.split(".", 1)[1]
    return text


def _is_metadata_key(key: str) -> bool:
    """判斷欄位是否屬於不適合直接回覆給使用者的技術性欄位。"""
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
    """執行 `_summarize_query_rows` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    if not rows:
        return f"目前在知識圖譜中找不到與「{question}」直接相關的資料。"

    user_rows = _filter_user_facing_rows(rows)
    effective_rows = user_rows or rows
    first_row_keys = list(effective_rows[0].keys())
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
        unique_values = list(dict.fromkeys(values))
        if unique_values:
            top_values = unique_values[:8]
            suffix = f" 等 {len(unique_values)} 筆" if len(unique_values) > 8 else ""
            if len(top_values) == 1:
                return _format_single_value_answer(question, top_values[0])
            return f"{question}：{'、'.join(top_values)}{suffix}。"

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

    highlights: List[str] = []
    for row in effective_rows[:5]:
        cells = [f"{_display_query_key(key)}：{_stringify_query_value(value)}" for key, value in row.items()]
        if cells:
            highlights.append("；".join(cells))

    if not highlights:
        return f"查到 {len(rows)} 筆資料，但欄位內容較空，請展開 Rows 查看。"

    summary = "；".join(highlights)
    if len(effective_rows) > 5:
        return f"{summary}。另有其他結果可再查看明細。"
    return f"{summary}。"


def _kg_qa_use_llm() -> bool:
    """執行 `_kg_qa_use_llm` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    return get_kg_qa_settings().use_llm


def _kg_qa_model() -> Optional[str]:
    """執行 `_kg_qa_model` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    return get_kg_qa_settings().model


def _kg_qa_temperature() -> float:
    """執行 `_kg_qa_temperature` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    return get_kg_qa_settings().temperature


def _kg_qa_max_tokens() -> int:
    """執行 `_kg_qa_max_tokens` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    return get_kg_qa_settings().max_tokens


def _format_rows_for_qa_prompt(rows: List[Dict[str, Any]]) -> str:
    """執行 `_format_rows_for_qa_prompt` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    max_rows = get_kg_qa_settings().max_rows_for_prompt
    limited = rows[: max(1, max_rows)]
    try:
        return json.dumps(limited, ensure_ascii=False, indent=2)
    except Exception:
        return str(limited)


def _generate_kg_answer_with_llm(*, question: str, cypher: str, rows: List[Dict[str, Any]]) -> str:
    """執行 `_generate_kg_answer_with_llm` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    if not _kg_qa_use_llm():
        raise RuntimeError("KG_QA_USE_LLM disabled")

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

    sample_values: List[str] = []
    for row in rows[:3]:
        for value in row.values():
            text = _stringify_query_value(value).strip()
            if text and text not in {"N/A", "[]", "{}"}:
                sample_values.append(text)
    sample_values = list(dict.fromkeys(sample_values))[:6]
    if sample_values and not any(value in normalized for value in sample_values):
        raise RuntimeError("KG QA LLM answer does not reference row values")

    return normalized


def query_kg(question: str) -> Dict[str, Any]:
    """執行 `query_kg` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("Question cannot be empty")

    answer_with_manual_prompt = _load_kg_query_executor()
    raw_result = answer_with_manual_prompt(cleaned)
    if not isinstance(raw_result, dict):
        raise RuntimeError("Unexpected KG query result format")

    rows_value = raw_result.get("rows")
    if not isinstance(rows_value, list):
        rows_value = []

    normalized_rows: List[Dict[str, Any]] = []
    for row in rows_value:
        if isinstance(row, dict):
            normalized_rows.append(row)
        else:
            normalized_rows.append({"value": row})

    result = dict(raw_result)
    result["question"] = str(result.get("question") or cleaned)
    result["rows"] = normalized_rows

    try:
        answer = _generate_kg_answer_with_llm(
            question=cleaned,
            cypher=str(result.get("cypher") or ""),
            rows=normalized_rows,
        )
        result["answer"] = answer
        result["answer_source"] = "qa_llm"
    except Exception:
        result["answer"] = _summarize_query_rows(cleaned, normalized_rows)
        result["answer_source"] = "template_fallback"

    return result


def _normalize_chat_history(history: Optional[List[Dict[str, str]]]) -> List[Dict[str, str]]:
    """執行 `_normalize_chat_history` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    if not history:
        return []

    normalized: List[Dict[str, str]] = []
    for message in history[-20:]:
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def chat_general(message: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """執行 `chat_general` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
    cleaned_message = message.strip()
    if not cleaned_message:
        raise ValueError("Message cannot be empty")

    system_prompt = (
        "你是一位溫和、支持性的聊天夥伴，請使用繁體中文。"
        "目標是陪伴失眠或情緒低落的使用者，語氣要穩定、簡潔、無評價。"
        "避免提供醫療診斷或藥物建議；若使用者提到自傷風險或急性危機，"
        "請鼓勵立即聯絡當地緊急資源與可信任的人。"
    )
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(_normalize_chat_history(history))
    messages.append({"role": "user", "content": cleaned_message})

    chat_settings = get_general_chat_settings()
    answer = llm_client.chat_text(
        messages=messages,
        temperature=chat_settings.temperature,
        max_tokens=chat_settings.num_predict,
        timeout_seconds=chat_settings.timeout_seconds,
    )
    cfg = llm_client.get_runtime_config()
    return {"answer": answer, "model": cfg.model, "provider": cfg.provider}
