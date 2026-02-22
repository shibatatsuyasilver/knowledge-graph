from __future__ import annotations

from collections import Counter
from pathlib import Path

from backend.llm_kg.benchmark.dataset_builder import BuilderConfig, build_dataset


def test_dataset_builder_generates_100_items_with_fixed_ratio(tmp_path: Path) -> None:
    """驗證 `test_dataset_builder_generates_100_items_with_fixed_ratio` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    cfg = BuilderConfig(
        dataset_path=tmp_path / "kgqa_zh_tw_100_v1.jsonl",
        random_seed=42,
        use_gemini_labeling=False,
        allow_gemini_fallback=True,
        gemini_model="gemini-3-pro-preview",
        gemini_timeout_seconds=60,
        gemini_max_retries=3,
        gemini_backoff_seconds=0.1,
        gemini_rate_limit_seconds=0.0,
    )

    rows = build_dataset(cfg)

    assert len(rows) == 100
    source_counts = Counter(row["source_type"] for row in rows)
    assert source_counts["graph_seed"] == 30
    assert source_counts["gemini_synth"] == 70

    qtype_counts = Counter(row["metadata"]["question_type"] for row in rows)
    assert qtype_counts["fact"] == 25
    assert qtype_counts["relation"] == 25
    assert qtype_counts["multi_hop"] == 20
    assert qtype_counts["comparison"] == 10
    assert qtype_counts["count"] == 10
    assert qtype_counts["boolean"] == 10
