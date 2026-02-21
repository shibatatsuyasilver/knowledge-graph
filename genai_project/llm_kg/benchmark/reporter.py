"""Benchmark report generator (CSV/JSON -> Markdown)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return [dict(row) for row in reader]


def _render_model_table(models: Sequence[Dict[str, Any]]) -> str:
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
    report = render_report(run_dir)
    report_path = run_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate benchmark markdown report")
    parser.add_argument("--run-dir", type=Path, required=True, help="Benchmark run directory")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report_path = write_report(args.run_dir)
    print(f"[report] generated {report_path}")


if __name__ == "__main__":
    main()
