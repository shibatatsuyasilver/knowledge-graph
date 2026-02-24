"""Benchmark report generator (CSV/JSON -> Markdown)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


def _load_json(path: Path) -> Dict[str, Any]:
    """`_load_json` 的內部輔助函式。

主要用途：
- 封裝局部步驟，讓主流程維持可讀性。
- 集中處理細節與邊界條件，避免重複邏輯分散。

回傳約定：
- 保持既有輸入/輸出契約，不改變對外行為。
    """
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> List[Dict[str, str]]:
    """`_load_csv` 的內部輔助函式。

主要用途：
- 封裝局部步驟，讓主流程維持可讀性。
- 集中處理細節與邊界條件，避免重複邏輯分散。

回傳約定：
- 保持既有輸入/輸出契約，不改變對外行為。
    """
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return [dict(row) for row in reader]


def _render_model_table(models: Sequence[Dict[str, Any]]) -> str:
    """`_render_model_table` 的內部輔助函式。

主要用途：
- 封裝局部步驟，讓主流程維持可讀性。
- 集中處理細節與邊界條件，避免重複邏輯分散。

回傳約定：
- 保持既有輸入/輸出契約，不改變對外行為。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    header = "| Model | KG QA Acc Mean | KG QA Acc Std | Triple Micro-F1 Mean | Triple Micro-F1 Std |"
    sep = "|---|---:|---:|---:|---:|"
    rows = [header, sep]
    for model in models:
        rows.append(
            "| {model} | {acc_mean:.4f} | {acc_std:.4f} | {f1_mean:.4f} | {f1_std:.4f} |".format(
                model=model["model"],
                acc_mean=float(model.get("kg_qa_accuracy_mean", 0.0)),
                acc_std=float(model.get("kg_qa_accuracy_std", 0.0)),
                f1_mean=float(model.get("triple_micro_f1_mean", 0.0)),
                f1_std=float(model.get("triple_micro_f1_std", 0.0)),
            )
        )
    return "\n".join(rows)


def _render_run_table(per_run: Sequence[Dict[str, Any]]) -> str:
    """`_render_run_table` 的內部輔助函式。

主要用途：
- 封裝局部步驟，讓主流程維持可讀性。
- 集中處理細節與邊界條件，避免重複邏輯分散。

回傳約定：
- 保持既有輸入/輸出契約，不改變對外行為。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    header = "| Model | Run | Accuracy | Triple Micro-F1 | Triple Macro-F1 |"
    sep = "|---|---:|---:|---:|---:|"
    rows = [header, sep]
    ordered = sorted(per_run, key=lambda row: (str(row["model"]), int(row["run_index"])))
    for row in ordered:
        rows.append(
            "| {model} | {run_index} | {acc:.4f} | {micro:.4f} | {macro:.4f} |".format(
                model=row["model"],
                run_index=int(row["run_index"]),
                acc=float(row.get("kg_qa_accuracy", 0.0)),
                micro=float(row.get("triple_micro_f1", 0.0)),
                macro=float(row.get("triple_macro_f1", 0.0)),
            )
        )
    return "\n".join(rows)


def render_report(run_dir: Path) -> str:
    """`render_report` 的主要流程入口。

主要用途：
- 串接此函式負責的核心步驟並回傳既有格式。
- 例外沿用現行錯誤處理策略，避免破壞呼叫端契約。

維護重點：
- 調整流程時需保持 API 欄位、狀態轉移與錯誤語意一致。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    summary = _load_json(run_dir / "summary.json")
    manifest = _load_json(run_dir / "run_manifest.json")

    lines = [
        "# DeepSeek R1 vs Gemma3 Benchmark Report",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Dataset: `{manifest['dataset']['path']}`",
        f"- Dataset size: {manifest['dataset']['size']}",
        f"- Provider: `{manifest['provider']}`",
        f"- Runs per model: {manifest['runs_per_model']}",
        "",
        "## Winner",
        "",
        f"**{summary['winner']}**",
        "",
        "## Model Summary",
        "",
        _render_model_table(summary.get("models", [])),
        "",
        "## Per Run",
        "",
        _render_run_table(summary.get("per_run", [])),
        "",
        "## Rule",
        "",
        "1. 主指標：KG QA Accuracy mean。",
        "2. 同分（差 < 0.1%）時，比 Accuracy std（越低越好）。",
        "3. 再同分時，比 Triple-F1 mean（越高越好）。",
        "",
    ]

    return "\n".join(lines)


def write_report(run_dir: Path) -> Path:
    """`write_report` 的主要流程入口。

主要用途：
- 串接此函式負責的核心步驟並回傳既有格式。
- 例外沿用現行錯誤處理策略，避免破壞呼叫端契約。

維護重點：
- 調整流程時需保持 API 欄位、狀態轉移與錯誤語意一致。
    """
    report = render_report(run_dir)
    report_path = run_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令列輸入並建立目前流程所需參數。
    此函式僅負責參數讀取與預設值處理，輸出格式供主流程直接使用。
    """
    parser = argparse.ArgumentParser(description="Generate benchmark markdown report")
    parser.add_argument("--run-dir", type=Path, required=True, help="Benchmark run directory")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """作為模組執行入口，串接並啟動既有主流程。
    此函式會依目前設定呼叫核心邏輯，並維持原本輸入輸出與錯誤行為。
    """
    args = parse_args(argv)
    report_path = write_report(args.run_dir)
    print(f"[report] generated {report_path}")


if __name__ == "__main__":
    main()
