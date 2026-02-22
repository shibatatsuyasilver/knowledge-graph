from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from backend.llm_kg.benchmark import runner


class _DummySession:
    def __enter__(self):
        """建立並回傳 context manager 進入階段所需資源。
        此方法在 `with` 區塊開始時執行，並維持既有回傳物件與行為契約。
        """
        return self

    def __exit__(self, exc_type, exc, tb):
        """負責 context manager 離開階段的清理與收尾作業。
        此方法會依目前例外傳入參數完成資源釋放，並保持既有錯誤傳遞語意。
        """
        return False

    def run(self, _query: str):
        """執行 `run` 的主要流程。
        函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
        """
        return []


class _DummyDriver:
    def session(self):
        """執行 `session` 的主要流程。
        函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
        """
        return _DummySession()


class _FakeBuilder:
    def __init__(self, *_args, **_kwargs):
        """初始化物件狀態並保存後續流程所需依賴。
        此方法會依目前參數設定實例欄位，供其他方法在生命週期內重複使用。
        """
        self.driver = _DummyDriver()

    def close(self) -> None:
        """執行 `close` 的主要流程。
        函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
        """
        return None

    def extract_entities_relations(self, _text: str):
        """執行 `extract_entities_relations` 的主要流程。
        函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
        """
        return {
            "entities": [{"name": "台積電", "type": "Organization"}, {"name": "Apple", "type": "Organization"}],
            "relations": [{"source": "台積電", "relation": "SUPPLIES_TO", "target": "Apple"}],
        }

    def populate_graph(self, extracted):
        """執行 `populate_graph` 的主要流程。
        函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
        """
        return SimpleNamespace(
            entities=len(extracted.get("entities", [])),
            relations=len(extracted.get("relations", [])),
            merged_entities=0,
            dropped_relations=0,
            json_retries=0,
        )


def _build_dataset(path: Path) -> None:
    """執行 `_build_dataset` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
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
    """驗證 `test_runner_outputs_artifacts_for_two_models_three_runs` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
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
