from __future__ import annotations

from backend.llm_kg.benchmark import scorer


def test_normalize_text_handles_fullwidth_and_punctuation() -> None:
    """驗證 `test_normalize_text_handles_fullwidth_and_punctuation` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    assert scorer.normalize_text(" 台積電， NVIDIa。") == "台積電nvidia"
    assert scorer.normalize_text("ＡＢＣ　123") == "abc123"


def test_score_qa_prefers_rows_required_entities() -> None:
    """驗證 `test_score_qa_prefers_rows_required_entities` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    gold = {
        "answer_type": "set",
        "canonical": "Apple、NVIDIA",
        "accepted_aliases": ["NVIDIA、Apple"],
        "required_entities": ["Apple", "NVIDIA"],
    }
    rows = [{"partner": "Apple"}, {"partner": "NVIDIA"}]

    score, reason = scorer.score_qa_accuracy(gold_answer=gold, predicted_rows=rows, predicted_answer="")

    assert score == 1
    assert reason == "rows_match_required_entities"


def test_score_qa_fallback_to_answer_alias() -> None:
    """驗證 `test_score_qa_fallback_to_answer_alias` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    gold = {
        "answer_type": "string",
        "canonical": "張忠謀",
        "accepted_aliases": ["台積電創辦人是張忠謀"],
        "required_entities": ["張忠謀"],
    }

    score, reason = scorer.score_qa_accuracy(
        gold_answer=gold,
        predicted_rows=[],
        predicted_answer="台積電創辦人是張忠謀",
    )

    assert score == 1
    assert reason in {"answer_alias_exact_match", "answer_alias_contains_match"}


def test_triple_prf_and_micro_prf() -> None:
    """驗證 `test_triple_prf_and_micro_prf` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    gold = {("台積電", "SUPPLIES_TO", "apple"), ("台積電", "SUPPLIES_TO", "nvidia")}
    predicted = {("台積電", "SUPPLIES_TO", "apple")}

    p, r, f1 = scorer.triple_prf(predicted=predicted, gold=gold)
    assert p == 1.0
    assert r == 0.5
    assert round(f1, 4) == 0.6667

    counts = [scorer.triple_count(predicted=predicted, gold=gold)]
    micro_p, micro_r, micro_f1 = scorer.micro_prf(counts)
    assert micro_p == 1.0
    assert micro_r == 0.5
    assert round(micro_f1, 4) == 0.6667
