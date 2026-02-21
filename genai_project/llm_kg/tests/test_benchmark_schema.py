from __future__ import annotations

import pytest

from genai_project.llm_kg.benchmark import schema


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
    schema.validate_dataset_item(GOOD_ITEM)


def test_validate_dataset_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="Duplicated dataset id"):
        schema.validate_dataset([GOOD_ITEM, GOOD_ITEM])
