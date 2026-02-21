"""Dataset and report schemas for KG benchmark."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal

from jsonschema import Draft202012Validator

SOURCE_TYPES = ("graph_seed", "gemini_synth")
ANSWER_TYPES = ("set", "string", "number", "boolean")
DIFFICULTY_TYPES = ("easy", "medium", "hard")
QUESTION_TYPES = ("fact", "relation", "multi_hop", "comparison", "count", "boolean")

DATASET_ITEM_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": [
        "id",
        "source_type",
        "question_zh_tw",
        "context_text",
        "gold_triples",
        "gold_answer",
        "metadata",
    ],
    "additionalProperties": False,
    "properties": {
        "id": {"type": "string", "pattern": "^Q[0-9]{4}$"},
        "source_type": {"type": "string", "enum": list(SOURCE_TYPES)},
        "question_zh_tw": {"type": "string", "minLength": 2},
        "context_text": {"type": "string", "minLength": 2},
        "gold_triples": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["subject", "relation", "object"],
                "additionalProperties": False,
                "properties": {
                    "subject": {"type": "string", "minLength": 1},
                    "relation": {"type": "string", "minLength": 1},
                    "object": {"type": "string", "minLength": 1},
                },
            },
        },
        "gold_answer": {
            "type": "object",
            "required": ["answer_type", "canonical", "accepted_aliases", "required_entities"],
            "additionalProperties": False,
            "properties": {
                "answer_type": {"type": "string", "enum": list(ANSWER_TYPES)},
                "canonical": {"type": "string", "minLength": 1},
                "accepted_aliases": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "required_entities": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
            },
        },
        "metadata": {
            "type": "object",
            "required": ["difficulty", "question_type"],
            "additionalProperties": False,
            "properties": {
                "difficulty": {"type": "string", "enum": list(DIFFICULTY_TYPES)},
                "question_type": {"type": "string", "enum": list(QUESTION_TYPES)},
            },
        },
    },
}

PER_QUESTION_SCORE_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": [
        "model",
        "run_index",
        "id",
        "qa_correct",
        "triple_precision",
        "triple_recall",
        "triple_f1",
    ],
    "additionalProperties": True,
    "properties": {
        "model": {"type": "string"},
        "run_index": {"type": "integer", "minimum": 1},
        "id": {"type": "string", "pattern": "^Q[0-9]{4}$"},
        "qa_correct": {"type": "integer", "enum": [0, 1]},
        "triple_precision": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "triple_recall": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "triple_f1": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}

SUMMARY_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["winner", "models"],
    "additionalProperties": True,
    "properties": {
        "winner": {"type": "string", "minLength": 1},
        "models": {
            "type": "array",
            "minItems": 2,
            "items": {
                "type": "object",
                "required": ["model", "kg_qa_accuracy_mean", "kg_qa_accuracy_std"],
                "additionalProperties": True,
                "properties": {
                    "model": {"type": "string"},
                    "kg_qa_accuracy_mean": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "kg_qa_accuracy_std": {"type": "number", "minimum": 0.0},
                },
            },
        },
    },
}


@dataclass(frozen=True)
class GoldTriple:
    subject: str
    relation: str
    object: str


@dataclass(frozen=True)
class GoldAnswer:
    answer_type: Literal["set", "string", "number", "boolean"]
    canonical: str
    accepted_aliases: List[str]
    required_entities: List[str]


@dataclass(frozen=True)
class ItemMetadata:
    difficulty: Literal["easy", "medium", "hard"]
    question_type: Literal["fact", "relation", "multi_hop", "comparison", "count", "boolean"]


@dataclass(frozen=True)
class DatasetItem:
    id: str
    source_type: Literal["graph_seed", "gemini_synth"]
    question_zh_tw: str
    context_text: str
    gold_triples: List[GoldTriple]
    gold_answer: GoldAnswer
    metadata: ItemMetadata

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunRecord:
    model: str
    run_index: int
    id: str
    source_type: str
    question_type: str
    difficulty: str
    qa_correct: int
    triple_precision: float
    triple_recall: float
    triple_f1: float
    extracted_entities: int
    extracted_relations: int
    qa_rows_count: int
    question_zh_tw: str
    predicted_answer: str
    gold_canonical_answer: str


def _validator(schema: Dict[str, Any]) -> Draft202012Validator:
    return Draft202012Validator(schema)


def validate_dataset_item(payload: Dict[str, Any]) -> None:
    errors = sorted(_validator(DATASET_ITEM_SCHEMA).iter_errors(payload), key=lambda e: e.path)
    if errors:
        joined = "; ".join(err.message for err in errors)
        raise ValueError(f"Invalid dataset item: {joined}")


def validate_dataset(items: Iterable[Dict[str, Any]]) -> None:
    seen_ids = set()
    for item in items:
        validate_dataset_item(item)
        item_id = item["id"]
        if item_id in seen_ids:
            raise ValueError(f"Duplicated dataset id: {item_id}")
        seen_ids.add(item_id)


def validate_summary(payload: Dict[str, Any]) -> None:
    errors = sorted(_validator(SUMMARY_SCHEMA).iter_errors(payload), key=lambda e: e.path)
    if errors:
        joined = "; ".join(err.message for err in errors)
        raise ValueError(f"Invalid summary payload: {joined}")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def dump_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
