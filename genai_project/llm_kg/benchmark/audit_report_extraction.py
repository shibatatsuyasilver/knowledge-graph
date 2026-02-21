from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from backend import logic
from genai_project.llm_kg.kg_builder import (
    KnowledgeGraphBuilder,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
)


def _default_report_path() -> Path:
    return Path("/Users/silver/Documents/鴻海/reports_honhai_20260213.txt")


def _build_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path("/Users/silver/Documents/鴻海/genai_project/llm_kg/benchmark/runs") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_audit(
    *,
    input_path: Path,
    provider: str,
    model: str | None,
    chunk_limit: int | None,
) -> Dict[str, Any]:
    raw_text = input_path.read_text(encoding="utf-8")
    chunks = logic.chunk_text(
        raw_text,
        source_url=f"file://{input_path.name}",
        title=input_path.name,
        extraction_provider=provider,
    )
    if chunk_limit is not None and chunk_limit > 0:
        chunks = chunks[:chunk_limit]

    builder = KnowledgeGraphBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    finance_rel_hits = {"HAS_FINANCIAL_METRIC": 0, "FOR_PERIOD": 0}
    chunk_results: List[Dict[str, Any]] = []
    total_entities = 0
    total_relations = 0
    total_json_retries = 0
    failed = 0

    try:
        for order, chunk in enumerate(chunks, start=1):
            row: Dict[str, Any] = {
                "order": order,
                "chunk_id": chunk.chunk_id,
                "chars": len(chunk.text),
                "tokens": int(chunk.tokens or 0),
                "status": "processing",
                "entities": 0,
                "relations": 0,
                "json_retries": 0,
                "finance_relation_hits": {"HAS_FINANCIAL_METRIC": 0, "FOR_PERIOD": 0},
                "error": None,
            }
            try:
                extracted = builder.extract_entities_relations(
                    chunk.text,
                    provider=provider,
                    model=model,
                )
                entities = extracted.get("entities", [])
                relations = extracted.get("relations", [])
                json_retries = int(extracted.get("meta", {}).get("json_retries", 0))
                row["status"] = "processed"
                row["entities"] = len(entities)
                row["relations"] = len(relations)
                row["json_retries"] = json_retries

                rel_types = [str(rel.get("relation", "")) for rel in relations if isinstance(rel, dict)]
                for rel_type in ("HAS_FINANCIAL_METRIC", "FOR_PERIOD"):
                    hit_count = sum(1 for value in rel_types if value == rel_type)
                    row["finance_relation_hits"][rel_type] = hit_count
                    finance_rel_hits[rel_type] += hit_count

                total_entities += len(entities)
                total_relations += len(relations)
                total_json_retries += json_retries
            except Exception as exc:  # pragma: no cover - integration path
                row["status"] = "failed"
                row["error"] = str(exc)
                failed += 1
            chunk_results.append(row)
    finally:
        builder.close()

    return {
        "input_path": str(input_path),
        "provider": provider,
        "model": model,
        "chunks_total": len(chunks),
        "chunks_failed": failed,
        "totals": {
            "entities": total_entities,
            "relations": total_relations,
            "json_retries": total_json_retries,
        },
        "finance_relation_hits": finance_rel_hits,
        "chunk_results": chunk_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Gemini extraction quality on Hon Hai report text.")
    parser.add_argument("--input", type=Path, default=_default_report_path(), help="Path to report text (.txt)")
    parser.add_argument("--provider", default="gemini", choices=["gemini", "ollama", "openai"])
    parser.add_argument("--model", default=None, help="Optional extraction model override")
    parser.add_argument("--chunk-limit", type=int, default=None, help="Optional chunk limit for quick audits")
    args = parser.parse_args()

    audit = run_audit(
        input_path=args.input,
        provider=args.provider,
        model=args.model,
        chunk_limit=args.chunk_limit,
    )
    output_dir = _build_output_dir()
    output_file = output_dir / "report_extraction_audit.json"
    output_file.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_file)


if __name__ == "__main__":
    main()

