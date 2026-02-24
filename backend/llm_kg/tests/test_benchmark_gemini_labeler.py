from __future__ import annotations

import pytest

from backend.llm_kg.benchmark.gemini_labeler import GeminiLabelError, GeminiLabeler, GeminiLabelerConfig


def test_gemini_labeler_retries_on_bad_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 `test_gemini_labeler_retries_on_bad_json` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    labeler = GeminiLabeler(
        config=GeminiLabelerConfig(
            model="gemini-3-pro-preview",
            max_retries=3,
            retry_backoff_seconds=0,
            rate_limit_seconds=0,
        )
    )

    calls = {"count": 0}

    def fake_request(_prompt: str) -> str:
        """提供 `fake_request` 測試替身以模擬外部依賴或固定回傳。
        此函式讓測試可注入可預測資料，將驗證焦點集中在流程與邏輯判斷。
        """
        calls["count"] += 1
        if calls["count"] < 3:
            return "not json"
        return (
            '{"gold_triples":[{"subject":"台積電","relation":"SUPPLIES_TO","object":"Apple"}],'
            '"canonical":"Apple","accepted_aliases":["Apple"],"required_entities":["Apple"]}'
        )

    monkeypatch.setattr(labeler, "_request_gemini", fake_request)

    payload = labeler.label_candidate(
        question_zh_tw="台積電供應給誰？",
        context_text="台積電供應給Apple。",
        answer_type="string",
    )

    assert calls["count"] == 3
    assert payload["gold_answer"]["canonical"] == "Apple"


def test_gemini_labeler_raises_after_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 `test_gemini_labeler_raises_after_retries` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    labeler = GeminiLabeler(
        config=GeminiLabelerConfig(max_retries=2, retry_backoff_seconds=0, rate_limit_seconds=0)
    )

    monkeypatch.setattr(labeler, "_request_gemini", lambda _prompt: "not json")

    with pytest.raises(GeminiLabelError, match="after 2 attempts"):
        labeler.label_candidate(
            question_zh_tw="Q",
            context_text="C",
            answer_type="string",
        )
