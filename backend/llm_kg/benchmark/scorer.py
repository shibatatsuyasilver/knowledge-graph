"""Scoring utilities for benchmark pipeline."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple


def normalize_text(value: Any) -> str:
    """`normalize_text` 的主要流程入口。

主要用途：
- 串接此函式負責的核心步驟並回傳既有格式。
- 例外沿用現行錯誤處理策略，避免破壞呼叫端契約。

維護重點：
- 調整流程時需保持 API 欄位、狀態轉移與錯誤語意一致。
    """
    text = unicodedata.normalize("NFKC", str(value or "")).lower().strip()
    text = re.sub(r"[\s\u3000]+", "", text)
    text = re.sub(r"[，。；：！？、,.!?;:\\-_/()\[\]{}'\"`]+", "", text)
    return text


def normalize_entity(value: Any) -> str:
    """`normalize_entity` 的主要流程入口。

主要用途：
- 串接此函式負責的核心步驟並回傳既有格式。
- 例外沿用現行錯誤處理策略，避免破壞呼叫端契約。

維護重點：
- 調整流程時需保持 API 欄位、狀態轉移與錯誤語意一致。
    """
    return normalize_text(value)


def _flatten_values(value: Any) -> List[str]:
    """`_flatten_values` 的內部輔助函式。

主要用途：
- 封裝局部步驟，讓主流程維持可讀性。
- 集中處理細節與邊界條件，避免重複邏輯分散。

回傳約定：
- 保持既有輸入/輸出契約，不改變對外行為。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    if value is None:
        return []
    if isinstance(value, (str, int, float, bool)):
        return [str(value)]
    if isinstance(value, dict):
        out: List[str] = []
        for v in value.values():
            out.extend(_flatten_values(v))
        return out
    if isinstance(value, list):
        out: List[str] = []
        for v in value:
            out.extend(_flatten_values(v))
        return out
    return [str(value)]


def extract_entities_from_rows(rows: Sequence[Dict[str, Any]]) -> Set[str]:
    """`extract_entities_from_rows` 的主要流程入口。

主要用途：
- 串接此函式負責的核心步驟並回傳既有格式。
- 例外沿用現行錯誤處理策略，避免破壞呼叫端契約。

維護重點：
- 調整流程時需保持 API 欄位、狀態轉移與錯誤語意一致。
    """
    entities: Set[str] = set()
    for row in rows:
        for raw in _flatten_values(row):
            norm = normalize_entity(raw)
            if norm:
                entities.add(norm)
    return entities


def rows_to_answer_text(rows: Sequence[Dict[str, Any]]) -> str:
    """`rows_to_answer_text` 的主要流程入口。

主要用途：
- 串接此函式負責的核心步驟並回傳既有格式。
- 例外沿用現行錯誤處理策略，避免破壞呼叫端契約。

維護重點：
- 調整流程時需保持 API 欄位、狀態轉移與錯誤語意一致。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    values: List[str] = []
    seen: Set[str] = set()
    for row in rows:
        for raw in _flatten_values(row):
            text = str(raw).strip()
            normalized = normalize_text(text)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            values.append(text)
    return "、".join(values)


def _bool_from_text(text: str) -> bool | None:
    """`_bool_from_text` 的內部輔助函式。

主要用途：
- 封裝局部步驟，讓主流程維持可讀性。
- 集中處理細節與邊界條件，避免重複邏輯分散。

回傳約定：
- 保持既有輸入/輸出契約，不改變對外行為。
    """
    if any(token in text for token in ("是", "正確", "true", "yes")):
        return True
    if any(token in text for token in ("否", "不是", "false", "no")):
        return False
    return None


def score_qa_accuracy(
    *,
    gold_answer: Dict[str, Any],
    predicted_rows: Sequence[Dict[str, Any]],
    predicted_answer: str,
) -> Tuple[int, str]:
    """`score_qa_accuracy` 的主要流程入口。

主要用途：
- 串接此函式負責的核心步驟並回傳既有格式。
- 例外沿用現行錯誤處理策略，避免破壞呼叫端契約。

維護重點：
- 調整流程時需保持 API 欄位、狀態轉移與錯誤語意一致。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    required = {normalize_entity(x) for x in gold_answer.get("required_entities", []) if normalize_entity(x)}
    row_entities = extract_entities_from_rows(predicted_rows)

    if required:
        if row_entities:
            if required.issubset(row_entities):
                return 1, "rows_match_required_entities"
            return 0, "rows_missing_required_entities"

    answer_norm = normalize_text(predicted_answer)
    if not answer_norm:
        return 0, "empty_answer"

    answer_type = str(gold_answer.get("answer_type", "")).lower().strip()
    canonical = str(gold_answer.get("canonical", ""))
    aliases = [canonical] + list(gold_answer.get("accepted_aliases", []))
    alias_norms = {normalize_text(x) for x in aliases if normalize_text(x)}

    if answer_type == "boolean":
        gold_bool = _bool_from_text(normalize_text(canonical))
        pred_bool = _bool_from_text(answer_norm)
        if gold_bool is not None and pred_bool is not None and gold_bool == pred_bool:
            return 1, "answer_boolean_match"

    if answer_norm in alias_norms:
        return 1, "answer_alias_exact_match"

    if any(alias in answer_norm for alias in alias_norms if alias):
        return 1, "answer_alias_contains_match"

    if required and all(req in answer_norm for req in required):
        return 1, "answer_contains_required_entities"

    return 0, "answer_mismatch"


def normalize_triple(subject: str, relation: str, object_: str) -> Tuple[str, str, str]:
    """`normalize_triple` 的主要流程入口。

主要用途：
- 串接此函式負責的核心步驟並回傳既有格式。
- 例外沿用現行錯誤處理策略，避免破壞呼叫端契約。

維護重點：
- 調整流程時需保持 API 欄位、狀態轉移與錯誤語意一致。
    """
    return (
        normalize_entity(subject),
        normalize_text(relation).upper(),
        normalize_entity(object_),
    )


def extract_triples(payload: Dict[str, Any]) -> Set[Tuple[str, str, str]]:
    """`extract_triples` 的主要流程入口。

主要用途：
- 串接此函式負責的核心步驟並回傳既有格式。
- 例外沿用現行錯誤處理策略，避免破壞呼叫端契約。

維護重點：
- 調整流程時需保持 API 欄位、狀態轉移與錯誤語意一致。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    triples: Set[Tuple[str, str, str]] = set()
    for relation in payload.get("relations", []):
        source = relation.get("source")
        rel = relation.get("relation")
        target = relation.get("target")
        if not source or not rel or not target:
            continue
        triples.add(normalize_triple(str(source), str(rel), str(target)))
    return triples


def gold_triples_set(gold_triples: Iterable[Dict[str, Any]]) -> Set[Tuple[str, str, str]]:
    """`gold_triples_set` 的主要流程入口。

主要用途：
- 串接此函式負責的核心步驟並回傳既有格式。
- 例外沿用現行錯誤處理策略，避免破壞呼叫端契約。

維護重點：
- 調整流程時需保持 API 欄位、狀態轉移與錯誤語意一致。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    triples: Set[Tuple[str, str, str]] = set()
    for triple in gold_triples:
        triples.add(
            normalize_triple(
                str(triple.get("subject", "")),
                str(triple.get("relation", "")),
                str(triple.get("object", "")),
            )
        )
    return {t for t in triples if all(t)}


def triple_prf(
    *,
    predicted: Set[Tuple[str, str, str]],
    gold: Set[Tuple[str, str, str]],
) -> Tuple[float, float, float]:
    """`triple_prf` 的主要流程入口。

主要用途：
- 串接此函式負責的核心步驟並回傳既有格式。
- 例外沿用現行錯誤處理策略，避免破壞呼叫端契約。

維護重點：
- 調整流程時需保持 API 欄位、狀態轉移與錯誤語意一致。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    if not predicted and not gold:
        return 1.0, 1.0, 1.0
    if not predicted:
        return 0.0, 0.0, 0.0
    if not gold:
        return 0.0, 0.0, 0.0

    true_positive = len(predicted.intersection(gold))
    precision = true_positive / len(predicted)
    recall = true_positive / len(gold)
    if precision + recall == 0:
        return precision, recall, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


@dataclass(frozen=True)
class TripleCount:
    true_positive: int
    predicted_total: int
    gold_total: int


def triple_count(
    *,
    predicted: Set[Tuple[str, str, str]],
    gold: Set[Tuple[str, str, str]],
) -> TripleCount:
    """`triple_count` 的主要流程入口。

主要用途：
- 串接此函式負責的核心步驟並回傳既有格式。
- 例外沿用現行錯誤處理策略，避免破壞呼叫端契約。

維護重點：
- 調整流程時需保持 API 欄位、狀態轉移與錯誤語意一致。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    return TripleCount(
        true_positive=len(predicted.intersection(gold)),
        predicted_total=len(predicted),
        gold_total=len(gold),
    )


def micro_prf(counts: Sequence[TripleCount]) -> Tuple[float, float, float]:
    """`micro_prf` 的主要流程入口。

主要用途：
- 串接此函式負責的核心步驟並回傳既有格式。
- 例外沿用現行錯誤處理策略，避免破壞呼叫端契約。

維護重點：
- 調整流程時需保持 API 欄位、狀態轉移與錯誤語意一致。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    if not counts:
        return 0.0, 0.0, 0.0
    tp = sum(c.true_positive for c in counts)
    pred_total = sum(c.predicted_total for c in counts)
    gold_total = sum(c.gold_total for c in counts)

    if pred_total == 0 or gold_total == 0:
        return 0.0, 0.0, 0.0

    precision = tp / pred_total
    recall = tp / gold_total
    if precision + recall == 0:
        return precision, recall, 0.0
    return precision, recall, (2 * precision * recall / (precision + recall))


def mean(values: Sequence[float]) -> float:
    """`mean` 的主要流程入口。

主要用途：
- 串接此函式負責的核心步驟並回傳既有格式。
- 例外沿用現行錯誤處理策略，避免破壞呼叫端契約。

維護重點：
- 調整流程時需保持 API 欄位、狀態轉移與錯誤語意一致。
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


def stddev(values: Sequence[float]) -> float:
    """`stddev` 的主要流程入口。

主要用途：
- 串接此函式負責的核心步驟並回傳既有格式。
- 例外沿用現行錯誤處理策略，避免破壞呼叫端契約。

維護重點：
- 調整流程時需保持 API 欄位、狀態轉移與錯誤語意一致。
    """
    if not values:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((x - avg) ** 2 for x in values) / len(values))
