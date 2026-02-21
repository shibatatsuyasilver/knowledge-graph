from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from genai_project.llm_kg.benchmark import runner


class _DummySession:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, _query: str):
        return []


class _DummyDriver:
    def session(self):
        return _DummySession()


class _FakeBuilder:
    def __init__(self, *_args, **_kwargs):
        self.driver = _DummyDriver()

    def close(self) -> None:
        return None

    def extract_entities_relations(self, _text: str):
        return {
            "entities": [{"name": "台積電", "type": "Organization"}, {"name": "Apple", "type": "Organization"}],
            "relations": [{"source": "台積電", "relation": "SUPPLIES_TO", "target": "Apple"}],
        }

    def populate_graph(self, extracted):
        return SimpleNamespace(
            entities=len(extracted.get("entities", [])),
            relations=len(extracted.get("relations", [])),
            merged_entities=0,
            dropped_relations=0,
            json_retries=0,
        )


def _build_dataset(path: Path) -> None:
    rows = []
    for idx in range(1, 11):
        rows.append(
            {
                "id": f"Q{idx:04d}",
                "source_type": "graph_seed",
                "question_zh_tw": "台積電供應給誰？",
                "context_text": "台積電供應給Apple。",
                "gold_triples": [
                    {"subject": "台積電", "relation": "SUPPLIES_TO", "object": "Apple"},
                ],
                "gold_answer": {
                    "answer_type": "set",
                    "canonical": "Apple",
                    "accepted_aliases": ["Apple"],
                    "required_entities": ["Apple"],
                },
                "metadata": {"difficulty": "easy", "question_type": "relation"},
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_runner_outputs_artifacts_for_two_models_three_runs(monkeypatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    _build_dataset(dataset_path)

    config_path = tmp_path / "config.yaml"
    config = {
        "dataset": {"path": str(dataset_path)},
        "benchmark": {
            "provider": "ollama",
            "models": ["deepseek-r1:8b", "gemma3:12b"],
            "runs_per_model": 3,
            "run_output_root": str(tmp_path / "runs"),
            "reset_graph_per_question": True,
        },
        "neo4j": {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "password"},
        "llm": {"ollama_base_url": "http://localhost:11434"},
    }
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    monkeypatch.setattr(runner.kg_builder, "KnowledgeGraphBuilder", _FakeBuilder)
    monkeypatch.setattr(
        runner.nl2cypher,
        "answer_with_manual_prompt",
        lambda _question: {"cypher": "MATCH ...", "rows": [{"partner": "Apple"}]},
    )

    loaded = runner._load_runner_config(config_path)
    run_dir = runner.run_benchmark(loaded)

    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "per_question_scores.csv").exists()
    assert (run_dir / "model_summary.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "report.md").exists()

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["winner"] in {"deepseek-r1:8b", "gemma3:12b"}
    assert len(summary["models"]) == 2
