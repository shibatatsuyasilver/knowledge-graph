from __future__ import annotations

import pytest

from backend.llm_kg.benchmark import schema


GOOD_ITEM = {
    "id": "Q0001",
    "source_type": "graph_seed",
    "question_zh_tw": "台積電的合作夥伴有哪些？",
    "context_text": "台積電供應給Apple與NVIDIA。",
    "gold_triples": [
        {"subject": "台積電", "relation": "SUPPLIES_TO", "object": "Apple"},
        {"subject": "台積電", "relation": "SUPPLIES_TO", "object": "NVIDIA"},
    ],
    "gold_answer": {
        "answer_type": "set",
        "canonical": "Apple、NVIDIA",
        "accepted_aliases": ["NVIDIA、Apple"],
        "required_entities": ["Apple", "NVIDIA"],
    },
    "metadata": {"difficulty": "medium", "question_type": "relation"},
}


def test_validate_dataset_item_ok() -> None:
    """驗證 `test_validate_dataset_item_ok` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    schema.validate_dataset_item(GOOD_ITEM)


def test_validate_dataset_rejects_duplicate_ids() -> None:
    """驗證 `test_validate_dataset_rejects_duplicate_ids` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    with pytest.raises(ValueError, match="Duplicated dataset id"):
        schema.validate_dataset([GOOD_ITEM, GOOD_ITEM])
