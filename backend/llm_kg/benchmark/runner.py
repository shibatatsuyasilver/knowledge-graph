"""Benchmark runner: extraction + KG QA for multiple models/runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import yaml

from .. import kg_builder, nl2cypher
from . import reporter
from .schema import load_jsonl, validate_dataset, validate_summary
from .scorer import (
    TripleCount,
    extract_triples,
    gold_triples_set,
    mean,
    micro_prf,
    rows_to_answer_text,
    score_qa_accuracy,
    stddev,
    triple_count,
    triple_prf,
)


@dataclass(frozen=True)
class RunnerConfig:
    config_path: Path
    dataset_path: Path
    run_output_root: Path
    models: List[str]
    runs_per_model: int
    provider: str
    reset_graph_per_question: bool
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    openai_base_url: str
    openai_api_key: str
    ollama_base_url: str
    llm_timeout_seconds: int
    llm_temperature: float
    llm_max_tokens: int
    extraction_max_tokens: int
    nl2cypher_max_tokens: int


def _load_runner_config(path: Path) -> RunnerConfig:
    """`_load_runner_config` 的內部輔助函式。

主要用途：
- 封裝局部步驟，讓主流程維持可讀性。
- 集中處理細節與邊界條件，避免重複邏輯分散。

回傳約定：
- 保持既有輸入/輸出契約，不改變對外行為。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    dataset_cfg = payload.get("dataset", {})
    benchmark_cfg = payload.get("benchmark", {})
    neo4j_cfg = payload.get("neo4j", {})
    llm_cfg = payload.get("llm", {})

    models = benchmark_cfg.get("models", ["deepseek-r1:8b", "gemma3:12b"])
    if not isinstance(models, list) or len(models) < 2:
        raise ValueError("benchmark.models must contain at least 2 models")

    return RunnerConfig(
        config_path=path,
        dataset_path=Path(dataset_cfg.get("path", "backend/llm_kg/benchmark/datasets/kgqa_zh_tw_100_v1.jsonl")),
        run_output_root=Path(benchmark_cfg.get("run_output_root", "backend/llm_kg/benchmark/runs")),
        models=[str(m) for m in models],
        runs_per_model=int(benchmark_cfg.get("runs_per_model", 3)),
        provider=str(benchmark_cfg.get("provider", "ollama")).strip().lower(),
        reset_graph_per_question=bool(benchmark_cfg.get("reset_graph_per_question", True)),
        neo4j_uri=str(neo4j_cfg.get("uri", os.getenv("NEO4J_URI", "bolt://localhost:7687"))),
        neo4j_user=str(neo4j_cfg.get("user", os.getenv("NEO4J_USER", "neo4j"))),
        neo4j_password=str(neo4j_cfg.get("password", os.getenv("NEO4J_PASSWORD", "password"))),
        openai_base_url=str(llm_cfg.get("openai_base_url", os.getenv("OPENAI_BASE_URL", "http://localhost:8080/v1"))),
        openai_api_key=str(llm_cfg.get("openai_api_key", os.getenv("OPENAI_API_KEY", ""))),
        ollama_base_url=str(llm_cfg.get("ollama_base_url", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))),
        llm_timeout_seconds=int(llm_cfg.get("timeout_seconds", os.getenv("LLM_TIMEOUT_SECONDS", 180))),
        llm_temperature=float(llm_cfg.get("temperature", os.getenv("LLM_TEMPERATURE", 0.0))),
        llm_max_tokens=int(llm_cfg.get("max_tokens", os.getenv("LLM_MAX_TOKENS", 2048))),
        extraction_max_tokens=int(llm_cfg.get("extraction_max_tokens", os.getenv("EXTRACTION_NUM_PREDICT", 8192))),
        nl2cypher_max_tokens=int(llm_cfg.get("nl2cypher_max_tokens", os.getenv("NL2CYPHER_NUM_PREDICT", 1024))),
    )


def _ensure_dir(path: Path) -> None:
    """`_ensure_dir` 的內部輔助函式。

主要用途：
- 封裝局部步驟，讓主流程維持可讀性。
- 集中處理細節與邊界條件，避免重複邏輯分散。

回傳約定：
- 保持既有輸入/輸出契約，不改變對外行為。
    """
    path.mkdir(parents=True, exist_ok=True)


def _timestamp() -> str:
    """`_timestamp` 的內部輔助函式。

主要用途：
- 封裝局部步驟，讓主流程維持可讀性。
- 集中處理細節與邊界條件，避免重複邏輯分散。

回傳約定：
- 保持既有輸入/輸出契約，不改變對外行為。
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _reset_graph(builder: kg_builder.KnowledgeGraphBuilder) -> None:
    """`_reset_graph` 的內部輔助函式。

主要用途：
- 封裝局部步驟，讓主流程維持可讀性。
- 集中處理細節與邊界條件，避免重複邏輯分散。

回傳約定：
- 保持既有輸入/輸出契約，不改變對外行為。
    """
    with builder.driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")


def _apply_runtime_settings(config: RunnerConfig, model: str) -> None:
    """`_apply_runtime_settings` 的內部輔助函式。

主要用途：
- 封裝局部步驟，讓主流程維持可讀性。
- 集中處理細節與邊界條件，避免重複邏輯分散。

回傳約定：
- 保持既有輸入/輸出契約，不改變對外行為。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    os.environ["LLM_PROVIDER"] = config.provider
    os.environ["LLM_MODEL"] = model
    os.environ["EXTRACTION_MODEL"] = model
    os.environ["NL2CYPHER_MODEL"] = model
    os.environ["LLM_TIMEOUT_SECONDS"] = str(config.llm_timeout_seconds)
    os.environ["LLM_TEMPERATURE"] = str(config.llm_temperature)
    os.environ["LLM_MAX_TOKENS"] = str(config.llm_max_tokens)
    os.environ["EXTRACTION_NUM_PREDICT"] = str(config.extraction_max_tokens)
    os.environ["NL2CYPHER_NUM_PREDICT"] = str(config.nl2cypher_max_tokens)
    os.environ["NEO4J_URI"] = config.neo4j_uri
    os.environ["NEO4J_USER"] = config.neo4j_user
    os.environ["NEO4J_PASSWORD"] = config.neo4j_password

    if config.provider == "openai":
        os.environ["OPENAI_BASE_URL"] = config.openai_base_url
        os.environ["OPENAI_API_KEY"] = config.openai_api_key
    if config.provider == "ollama":
        os.environ["OLLAMA_BASE_URL"] = config.ollama_base_url

    # nl2cypher and kg_builder constants are module-level; align them at runtime.
    nl2cypher.NEO4J_URI = config.neo4j_uri
    nl2cypher.NEO4J_USER = config.neo4j_user
    nl2cypher.NEO4J_PASSWORD = config.neo4j_password
    nl2cypher.NL2CYPHER_TIMEOUT_SECONDS = config.llm_timeout_seconds
    nl2cypher.NL2CYPHER_NUM_PREDICT = config.nl2cypher_max_tokens

    kg_builder.EXTRACTION_TIMEOUT_SECONDS = config.llm_timeout_seconds
    kg_builder.EXTRACTION_NUM_PREDICT = config.extraction_max_tokens


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    """`_write_json` 的內部輔助函式。

主要用途：
- 封裝局部步驟，讓主流程維持可讀性。
- 集中處理細節與邊界條件，避免重複邏輯分散。

回傳約定：
- 保持既有輸入/輸出契約，不改變對外行為。
    """
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """`_write_csv` 的內部輔助函式。

主要用途：
- 封裝局部步驟，讓主流程維持可讀性。
- 集中處理細節與邊界條件，避免重複邏輯分散。

回傳約定：
- 保持既有輸入/輸出契約，不改變對外行為。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _pick_winner(model_rows: Sequence[Dict[str, Any]]) -> str:
    """`_pick_winner` 的內部輔助函式。

主要用途：
- 封裝局部步驟，讓主流程維持可讀性。
- 集中處理細節與邊界條件，避免重複邏輯分散。

回傳約定：
- 保持既有輸入/輸出契約，不改變對外行為。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    if not model_rows:
        raise ValueError("No model rows to rank")

    best_accuracy = max(float(row["kg_qa_accuracy_mean"]) for row in model_rows)
    contenders = [row for row in model_rows if abs(float(row["kg_qa_accuracy_mean"]) - best_accuracy) < 0.001]

    if len(contenders) == 1:
        return str(contenders[0]["model"])

    min_std = min(float(row["kg_qa_accuracy_std"]) for row in contenders)
    contenders = [row for row in contenders if abs(float(row["kg_qa_accuracy_std"]) - min_std) < 1e-9]
    if len(contenders) == 1:
        return str(contenders[0]["model"])

    max_f1 = max(float(row["triple_micro_f1_mean"]) for row in contenders)
    contenders = [row for row in contenders if abs(float(row["triple_micro_f1_mean"]) - max_f1) < 1e-9]
    contenders.sort(key=lambda row: str(row["model"]))
    return str(contenders[0]["model"])


def run_benchmark(config: RunnerConfig) -> Path:
    """`run_benchmark` 的主要流程入口。

主要用途：
- 串接此函式負責的核心步驟並回傳既有格式。
- 例外沿用現行錯誤處理策略，避免破壞呼叫端契約。

維護重點：
- 調整流程時需保持 API 欄位、狀態轉移與錯誤語意一致。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    dataset_rows = load_jsonl(config.dataset_path)
    validate_dataset(dataset_rows)

    _ensure_dir(config.run_output_root)
    run_dir = config.run_output_root / _timestamp()
    _ensure_dir(run_dir)

    source_counts = Counter(row["source_type"] for row in dataset_rows)
    question_type_counts = Counter(row["metadata"]["question_type"] for row in dataset_rows)

    manifest = {
        "created_at": datetime.now().isoformat(),
        "config_path": str(config.config_path),
        "dataset": {
            "path": str(config.dataset_path),
            "size": len(dataset_rows),
            "source_counts": dict(source_counts),
            "question_type_counts": dict(question_type_counts),
        },
        "models": config.models,
        "runs_per_model": config.runs_per_model,
        "provider": config.provider,
    }
    _write_json(run_dir / "run_manifest.json", manifest)

    per_question_scores: List[Dict[str, Any]] = []
    per_run_metrics: List[Dict[str, Any]] = []

    for model in config.models:
        for run_index in range(1, config.runs_per_model + 1):
            print(f"[benchmark] model={model} run={run_index}/{config.runs_per_model}")
            _apply_runtime_settings(config, model)

            builder = kg_builder.KnowledgeGraphBuilder(config.neo4j_uri, config.neo4j_user, config.neo4j_password)
            run_records: List[Dict[str, Any]] = []
            triple_counts: List[TripleCount] = []
            triple_f1_values: List[float] = []

            try:
                _reset_graph(builder)
                for item in dataset_rows:
                    if config.reset_graph_per_question:
                        _reset_graph(builder)

                    extracted: Dict[str, Any] = {"entities": [], "relations": []}
                    extracted_entities = 0
                    extracted_relations = 0
                    qa_rows: List[Dict[str, Any]] = []
                    qa_reason = "ok"
                    error_message = ""

                    try:
                        extracted = builder.extract_entities_relations(item["context_text"])
                        stats = builder.populate_graph(extracted)
                        extracted_entities = stats.entities
                        extracted_relations = stats.relations

                        qa_result = nl2cypher.answer_with_manual_prompt(item["question_zh_tw"])
                        qa_rows = qa_result.get("rows", []) or []
                    except Exception as exc:  # noqa: BLE001 - keep benchmark running even if one question fails
                        qa_reason = "exception"
                        error_message = str(exc)

                    predicted_answer = rows_to_answer_text(qa_rows)
                    qa_correct, qa_reason_scored = score_qa_accuracy(
                        gold_answer=item["gold_answer"],
                        predicted_rows=qa_rows,
                        predicted_answer=predicted_answer,
                    )
                    if qa_reason == "ok":
                        qa_reason = qa_reason_scored

                    predicted_triples = extract_triples(extracted)
                    gold_triples = gold_triples_set(item["gold_triples"])
                    triple_precision, triple_recall, triple_f1 = triple_prf(predicted=predicted_triples, gold=gold_triples)
                    counts = triple_count(predicted=predicted_triples, gold=gold_triples)
                    triple_counts.append(counts)
                    triple_f1_values.append(triple_f1)

                    record = {
                        "model": model,
                        "run_index": run_index,
                        "id": item["id"],
                        "source_type": item["source_type"],
                        "question_type": item["metadata"]["question_type"],
                        "difficulty": item["metadata"]["difficulty"],
                        "qa_correct": qa_correct,
                        "qa_reason": qa_reason,
                        "triple_precision": round(triple_precision, 6),
                        "triple_recall": round(triple_recall, 6),
                        "triple_f1": round(triple_f1, 6),
                        "extracted_entities": extracted_entities,
                        "extracted_relations": extracted_relations,
                        "qa_rows_count": len(qa_rows),
                        "question_zh_tw": item["question_zh_tw"],
                        "predicted_answer": predicted_answer,
                        "gold_canonical_answer": item["gold_answer"].get("canonical", ""),
                        "error": error_message,
                    }
                    run_records.append(record)
                    per_question_scores.append(record)
            finally:
                builder.close()

            micro_p, micro_r, micro_f1 = micro_prf(triple_counts)
            run_accuracy = mean([float(r["qa_correct"]) for r in run_records])
            run_macro_f1 = mean(triple_f1_values)

            per_run_metrics.append(
                {
                    "model": model,
                    "run_index": run_index,
                    "questions": len(run_records),
                    "kg_qa_accuracy": round(run_accuracy, 6),
                    "triple_micro_precision": round(micro_p, 6),
                    "triple_micro_recall": round(micro_r, 6),
                    "triple_micro_f1": round(micro_f1, 6),
                    "triple_macro_f1": round(run_macro_f1, 6),
                }
            )

    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in per_run_metrics:
        grouped[row["model"]]["kg_qa_accuracy"].append(float(row["kg_qa_accuracy"]))
        grouped[row["model"]]["triple_micro_f1"].append(float(row["triple_micro_f1"]))
        grouped[row["model"]]["triple_macro_f1"].append(float(row["triple_macro_f1"]))

    model_summary: List[Dict[str, Any]] = []
    for model in config.models:
        acc = grouped[model]["kg_qa_accuracy"]
        micro_f1 = grouped[model]["triple_micro_f1"]
        macro_f1 = grouped[model]["triple_macro_f1"]
        model_summary.append(
            {
                "model": model,
                "kg_qa_accuracy_mean": round(mean(acc), 6),
                "kg_qa_accuracy_std": round(stddev(acc), 6),
                "triple_micro_f1_mean": round(mean(micro_f1), 6),
                "triple_micro_f1_std": round(stddev(micro_f1), 6),
                "triple_macro_f1_mean": round(mean(macro_f1), 6),
                "triple_macro_f1_std": round(stddev(macro_f1), 6),
            }
        )

    winner = _pick_winner(model_summary)
    summary = {
        "winner": winner,
        "models": model_summary,
        "per_run": per_run_metrics,
        "dataset_size": len(dataset_rows),
        "runs_per_model": config.runs_per_model,
        "generated_at": datetime.now().isoformat(),
    }
    validate_summary(summary)

    _write_csv(run_dir / "per_question_scores.csv", per_question_scores)
    _write_csv(run_dir / "model_summary.csv", model_summary)
    _write_json(run_dir / "summary.json", summary)

    reporter.write_report(run_dir)
    return run_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令列輸入並建立目前流程所需參數。
    此函式僅負責參數讀取與預設值處理，輸出格式供主流程直接使用。
    """
    parser = argparse.ArgumentParser(description="Run KG benchmark")
    parser.add_argument("--config", type=Path, required=True, help="Path to benchmark config YAML")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """作為模組執行入口，串接並啟動既有主流程。
    此函式會依目前設定呼叫核心邏輯，並維持原本輸入輸出與錯誤行為。
    """
    args = parse_args(argv)
    config = _load_runner_config(args.config)
    run_dir = run_benchmark(config)
    print(f"[benchmark] completed run_dir={run_dir}")


if __name__ == "__main__":
    main()
