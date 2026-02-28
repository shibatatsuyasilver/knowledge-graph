import pytest

from backend.services.qa import qa_service


def test_query_kg_returns_answer_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 `test_query_kg_returns_answer_text` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    def fake_answer_with_manual_prompt(question: str):
        """提供 `fake_answer_with_manual_prompt` 測試替身以模擬外部依賴或固定回傳。
        此函式讓測試可注入可預測資料，將驗證焦點集中在流程與邏輯判斷。
        """
        return {
            "question": question,
            "cypher": "MATCH (n) RETURN n.name AS partner LIMIT 1",
            "rows": [{"partner": "NVIDIA"}],
            "attempt": 1,
        }

    monkeypatch.setattr(qa_service, "_load_kg_query_executor", lambda: fake_answer_with_manual_prompt)

    result = qa_service.query_kg("鴻海公司的合作夥伴")

    assert result["rows"] == [{"partner": "NVIDIA"}]
    assert "answer" in result
    assert "NVIDIA" in result["answer"]
    assert result["answer_source"] in {"qa_llm", "template_fallback"}


def test_query_kg_preserves_agentic_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 query_kg 會保留 agentic_trace 可選欄位。"""
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────

    def fake_answer_with_manual_prompt(question: str):
        return {
            "question": question,
            "cypher": "MATCH (n) RETURN n.name AS partner LIMIT 1",
            "rows": [{"partner": "NVIDIA"}],
            "attempt": 1,
            "agentic_trace": {
                "stage": "done",
                "round_count": 1,
                "replan_count": 0,
                "final_strategy": "single_query",
                "failure_chain": [],
                "plan_initial": {"strategy": "single_query"},
                "planner_plan": {"strategy": "single_query"},
                "plan_final": {"strategy": "single_query"},
                "rounds": [{"round": 1, "verdict": "accept"}],
            },
        }

    monkeypatch.setattr(qa_service, "_load_kg_query_executor", lambda: fake_answer_with_manual_prompt)

    result = qa_service.query_kg("鴻海公司的合作夥伴")

    assert result["agentic_trace"]["stage"] == "done"
    assert result["agentic_trace"]["round_count"] == 1
    assert result["agentic_trace"]["plan_final"]["strategy"] == "single_query"
    assert result["agentic_trace"]["rounds"][0]["verdict"] == "accept"


def test_query_kg_handles_empty_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 `test_query_kg_handles_empty_rows` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    def fake_answer_with_manual_prompt(question: str):
        """提供 `fake_answer_with_manual_prompt` 測試替身以模擬外部依賴或固定回傳。
        此函式讓測試可注入可預測資料，將驗證焦點集中在流程與邏輯判斷。
        """
        return {
            "question": question,
            "cypher": "MATCH (n) WHERE false RETURN n",
            "rows": [],
            "attempt": 1,
        }

    monkeypatch.setattr(qa_service, "_load_kg_query_executor", lambda: fake_answer_with_manual_prompt)

    result = qa_service.query_kg("不存在的問題")

    assert result["rows"] == []
    assert "找不到" in result["answer"]
    assert result["answer_source"] in {"qa_llm", "template_fallback"}


def test_query_kg_prefers_qa_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 `test_query_kg_prefers_qa_llm` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    def fake_answer_with_manual_prompt(question: str):
        """提供 `fake_answer_with_manual_prompt` 測試替身以模擬外部依賴或固定回傳。
        此函式讓測試可注入可預測資料，將驗證焦點集中在流程與邏輯判斷。
        """
        return {
            "question": question,
            "cypher": "MATCH (o:Organization)-[:CHAIRED_BY]->(p:Person) RETURN p.name AS name",
            "rows": [{"name": "劉揚偉"}],
            "attempt": 1,
        }

    monkeypatch.setattr(qa_service, "_load_kg_query_executor", lambda: fake_answer_with_manual_prompt)
    monkeypatch.setattr(qa_service, "_kg_qa_use_llm", lambda: True)
    monkeypatch.setattr(qa_service.llm_client, "chat_text", lambda **kwargs: "鴻海董事長是劉揚偉。")

    result = qa_service.query_kg("鴻海董事長是誰")

    assert result["answer"] == "鴻海董事長是劉揚偉。"
    assert result["answer_source"] == "qa_llm"


def test_query_kg_fallback_when_qa_llm_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 `test_query_kg_fallback_when_qa_llm_fails` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    def fake_answer_with_manual_prompt(question: str):
        """提供 `fake_answer_with_manual_prompt` 測試替身以模擬外部依賴或固定回傳。
        此函式讓測試可注入可預測資料，將驗證焦點集中在流程與邏輯判斷。
        """
        return {
            "question": question,
            "cypher": "MATCH (o:Organization)-[:CHAIRED_BY]->(p:Person) RETURN p.name AS name",
            "rows": [{"name": "劉揚偉"}],
            "attempt": 1,
        }

    monkeypatch.setattr(qa_service, "_load_kg_query_executor", lambda: fake_answer_with_manual_prompt)
    monkeypatch.setattr(qa_service, "_kg_qa_use_llm", lambda: True)

    def raise_chat(**_kwargs):
        raise RuntimeError("llm down")

    monkeypatch.setattr(qa_service.llm_client, "chat_text", raise_chat)

    result = qa_service.query_kg("鴻海董事長是誰")

    assert "劉揚偉" in result["answer"]
    assert result["answer_source"] == "template_fallback"


def test_query_kg_fallback_hides_metadata_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 fallback 回覆不暴露技術欄位，且語氣不使用「我查到」模板。"""
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    def fake_answer_with_manual_prompt(question: str):
        """提供 `fake_answer_with_manual_prompt` 測試替身以模擬外部依賴或固定回傳。"""
        return {
            "question": question,
            "cypher": "MATCH (o:Organization)-[:FOUNDED_BY]->(p:Person) RETURN p.name AS 創辦人, p.name AS 正規化名稱",
            "rows": [{"創辦人": "郭台銘", "正規化名稱": "郭台銘"}],
            "attempt": 1,
        }

    monkeypatch.setattr(qa_service, "_load_kg_query_executor", lambda: fake_answer_with_manual_prompt)
    monkeypatch.setattr(qa_service, "_kg_qa_use_llm", lambda: True)

    def raise_chat(**_kwargs):
        raise RuntimeError("llm down")

    monkeypatch.setattr(qa_service.llm_client, "chat_text", raise_chat)

    result = qa_service.query_kg("誰是鴻海創辦人")

    assert result["answer_source"] == "template_fallback"
    assert "郭台銘" in result["answer"]
    assert "正規化名稱" not in result["answer"]
    assert "我查到" not in result["answer"]


def test_query_kg_forwards_nl2cypher_params(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 qa_service.query_kg 會正確轉送 nl2cypher_provider 與 nl2cypher_model 到 executor。"""
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    captured: dict[str, object] = {}

    def fake_executor(question: str, nl2cypher_provider=None, nl2cypher_model=None, **kwargs):
        captured["question"] = question
        captured["nl2cypher_provider"] = nl2cypher_provider
        captured["nl2cypher_model"] = nl2cypher_model
        return {
            "question": question,
            "cypher": "MATCH (n) RETURN n.name AS name",
            "rows": [{"name": "鴻海"}],
            "attempt": 1,
        }

    monkeypatch.setattr(qa_service, "_load_kg_query_executor", lambda: fake_executor)

    result = qa_service.query_kg(
        "鴻海是什麼",
        nl2cypher_provider="gemini",
        nl2cypher_model="gemini-3-pro-preview",
    )

    assert result["question"] == "鴻海是什麼"
    assert captured["question"] == "鴻海是什麼"
    assert captured["nl2cypher_provider"] == "gemini"
    assert captured["nl2cypher_model"] == "gemini-3-pro-preview"


def test_chat_general_uses_shared_llm_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 `test_chat_general_uses_shared_llm_client` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    monkeypatch.setattr(
        qa_service.llm_client,
        "chat_text",
        lambda **kwargs: "這是一個測試回覆",
    )
    monkeypatch.setattr(
        qa_service.llm_client,
        "get_runtime_config",
        lambda: type("Cfg", (), {"model": "mlx-community/Qwen3-8B-4bit-DWQ-053125", "provider": "openai"})(),
    )

    result = qa_service.chat_general("我今天睡不好")

    assert result["answer"] == "這是一個測試回覆"
    assert result["model"] == "mlx-community/Qwen3-8B-4bit-DWQ-053125"
    assert result["provider"] == "openai"
