import requests
import time
from fastapi.testclient import TestClient

import backend.main as main_module
from backend import logic


client = TestClient(main_module.app)


def test_process_url_route_is_unique() -> None:
    """驗證 `test_process_url_route_is_unique` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    routes = [
        route
        for route in main_module.app.routes
        if getattr(route, "path", None) == "/api/process_url" and "POST" in getattr(route, "methods", set())
    ]
    assert len(routes) == 1


def test_process_url_endpoint_success(monkeypatch) -> None:
    """驗證 `test_process_url_endpoint_success` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    monkeypatch.setattr(
        main_module.logic,
        "process_url_to_kg",
        lambda url, uri, user, pwd, chunk_limit=None, extraction_provider=None, extraction_model=None: {
            "stats": {
                "chunks_processed": 1,
                "entities": 2,
                "relations": 1,
                "merged_entities": 0,
                "dropped_relations": 0,
                "json_retries": 0,
            },
            "summary": [{"chunk_id": "c1", "entities": 2, "relations": 1}],
        },
    )

    response = client.post("/api/process_url", json={"url": "https://example.com"})
    assert response.status_code == 200
    body = response.json()
    assert body["stats"]["chunks_processed"] == 1


def test_process_url_endpoint_forwards_chunk_limit(monkeypatch) -> None:
    """驗證 `test_process_url_endpoint_forwards_chunk_limit` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    captured = {}

    def fake_process_url_to_kg(url, uri, user, pwd, chunk_limit=None, extraction_provider=None, extraction_model=None):
        """提供 `fake_process_url_to_kg` 測試替身以模擬外部依賴或固定回傳。
        此函式讓測試可注入可預測資料，將驗證焦點集中在流程與邏輯判斷。
        """
        captured["chunk_limit"] = chunk_limit
        captured["extraction_provider"] = extraction_provider
        captured["extraction_model"] = extraction_model
        captured["url"] = url
        return {
            "stats": {
                "chunks_processed": 1,
                "entities": 1,
                "relations": 0,
                "merged_entities": 0,
                "dropped_relations": 0,
                "json_retries": 0,
            },
            "summary": [],
        }

    monkeypatch.setattr(main_module.logic, "process_url_to_kg", fake_process_url_to_kg)

    response = client.post(
        "/api/process_url",
        json={
            "url": "https://example.com",
            "chunk_limit": 5,
            "extraction_provider": "gemini",
            "extraction_model": "gemini-3-pro-preview",
        },
    )
    assert response.status_code == 200
    assert captured["url"] == "https://example.com"
    assert captured["chunk_limit"] == 5
    assert captured["extraction_provider"] == "gemini"
    assert captured["extraction_model"] == "gemini-3-pro-preview"


def test_chat_general_timeout_returns_504(monkeypatch) -> None:
    """驗證 `test_chat_general_timeout_returns_504` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    def timeout_chat(message, history=None):
        """`timeout_chat` 的主要流程入口。

主要用途：
- 串接此函式負責的核心步驟並回傳既有格式。
- 例外沿用現行錯誤處理策略，避免破壞呼叫端契約。

維護重點：
- 調整流程時需保持 API 欄位、狀態轉移與錯誤語意一致。
        """
        raise requests.Timeout("timeout")

    monkeypatch.setattr(main_module.logic, "chat_general", timeout_chat)

    response = client.post("/api/chat_general", json={"message": "睡不著"})
    assert response.status_code == 504
    assert response.json()["detail"] == "Upstream service timeout"


def test_query_endpoint_keeps_compat_fields_and_optional_agentic_trace(monkeypatch) -> None:
    """驗證 /api/query 保持既有欄位，並可回傳 agentic_trace。"""
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────

    monkeypatch.setattr(
        main_module.logic,
        "query_kg",
        lambda question, **_kwargs: {
            "question": question,
            "cypher": "MATCH (o:Organization) RETURN o.name AS organization",
            "rows": [{"organization": "鴻海精密"}],
            "attempt": 1,
            "answer": "鴻海精密。",
            "answer_source": "qa_llm",
            "agentic_trace": {
                "stage": "done",
                "round_count": 1,
                "replan_count": 0,
                "final_strategy": "single_query",
                "failure_chain": [],
            },
        },
    )

    response = client.post("/api/query", json={"question": "鴻海的事業有哪些"})
    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "鴻海的事業有哪些"
    assert body["cypher"].startswith("MATCH")
    assert body["rows"] == [{"organization": "鴻海精密"}]
    assert body["attempt"] == 1
    assert body["answer"] == "鴻海精密。"
    assert body["answer_source"] == "qa_llm"
    assert body["agentic_trace"]["stage"] == "done"
    assert body["agentic_trace"]["round_count"] == 1


def test_query_endpoint_forwards_nl2cypher_overrides(monkeypatch) -> None:
    """驗證 /api/query 會轉送 nl2cypher provider/model。"""
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    captured = {}

    def fake_query_kg(
        question,
        progress_callback=None,
        nl2cypher_provider=None,
        nl2cypher_model=None,
        query_engine=None,
    ):
        captured["question"] = question
        captured["progress_callback"] = progress_callback
        captured["nl2cypher_provider"] = nl2cypher_provider
        captured["nl2cypher_model"] = nl2cypher_model
        captured["query_engine"] = query_engine
        return {
            "question": question,
            "cypher": "MATCH (o:Organization) RETURN o.name AS organization",
            "rows": [{"organization": "鴻海精密"}],
            "attempt": 1,
        }

    monkeypatch.setattr(main_module.logic, "query_kg", fake_query_kg)

    response = client.post(
        "/api/query",
        json={
            "question": "鴻海的事業有哪些",
            "nl2cypher_provider": "gemini",
            "nl2cypher_model": "gemini-3-pro-preview",
            "query_engine": "graph_chain",
        },
    )
    assert response.status_code == 200
    assert captured["question"] == "鴻海的事業有哪些"
    assert captured["progress_callback"] is None
    assert captured["nl2cypher_provider"] == "gemini"
    assert captured["nl2cypher_model"] == "gemini-3-pro-preview"
    assert captured["query_engine"] == "graph_chain"


def test_query_endpoint_graph_chain_keeps_compat_fields(monkeypatch) -> None:
    """驗證 /api/query graph_chain 路徑仍回傳相容欄位。"""
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────

    monkeypatch.setattr(
        main_module.logic,
        "query_kg",
        lambda question, **_kwargs: {
            "question": question,
            "cypher": "MATCH (o:Organization) RETURN o.name AS organization",
            "rows": [{"organization": "鴻海精密"}],
            "answer": "鴻海精密。",
            "query_engine": "graph_chain",
            "graph_chain_raw": {"raw": True},
            "engine_provider": "gemini",
            "engine_model": "gemini-3-pro-preview",
        },
    )

    response = client.post(
        "/api/query",
        json={
            "question": "鴻海的事業有哪些",
            "query_engine": "graph_chain",
            "nl2cypher_provider": "gemini",
            "nl2cypher_model": "gemini-3-pro-preview",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "鴻海的事業有哪些"
    assert body["cypher"].startswith("MATCH")
    assert body["rows"] == [{"organization": "鴻海精密"}]
    assert body["answer"] == "鴻海精密。"
    assert body["query_engine"] == "graph_chain"
    assert body["graph_chain_raw"] == {"raw": True}
    assert body["engine_provider"] == "gemini"
    assert body["engine_model"] == "gemini-3-pro-preview"


def test_query_async_start_forwards_nl2cypher_overrides(monkeypatch) -> None:
    """驗證 /api/query_async/start 會轉送 nl2cypher provider/model。"""
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    captured = {}

    def fake_query_kg(
        question,
        progress_callback=None,
        nl2cypher_provider=None,
        nl2cypher_model=None,
        query_engine=None,
    ):
        captured["question"] = question
        captured["nl2cypher_provider"] = nl2cypher_provider
        captured["nl2cypher_model"] = nl2cypher_model
        captured["query_engine"] = query_engine
        captured["progress_callback_called"] = progress_callback is not None
        if progress_callback:
            progress_callback(
                {
                    "type": "agentic_progress",
                    "stage": "planner",
                    "round_count": 0,
                    "replan_count": 0,
                    "final_strategy": "single_query",
                    "failure_chain": [],
                    "detail": "planning",
                    "llm_provider": nl2cypher_provider,
                    "llm_model": nl2cypher_model,
                    "agentic_trace": {
                        "stage": "planner",
                        "round_count": 0,
                        "replan_count": 0,
                        "final_strategy": "single_query",
                        "failure_chain": [],
                        "plan_initial": {"strategy": "single_query"},
                        "planner_plan": {"strategy": "single_query"},
                        "plan_final": {"strategy": "single_query"},
                        "rounds": [],
                    },
                }
            )
        return {
            "question": question,
            "cypher": "MATCH (o:Organization) RETURN o.name AS organization",
            "rows": [{"organization": "鴻海精密"}],
            "attempt": 1,
            "agentic_trace": {
                "stage": "done",
                "round_count": 1,
                "replan_count": 0,
                "final_strategy": "single_query",
                "failure_chain": [],
                "llm_provider": nl2cypher_provider,
                "llm_model": nl2cypher_model,
                "plan_initial": {"strategy": "single_query"},
                "planner_plan": {"strategy": "single_query"},
                "plan_final": {"strategy": "single_query"},
                "rounds": [{"round": 1, "verdict": "accept"}],
            },
        }

    monkeypatch.setattr(main_module.logic, "query_kg", fake_query_kg)

    start_resp = client.post(
        "/api/query_async/start",
        json={
            "question": "鴻海的事業有哪些",
            "nl2cypher_provider": "gemini",
            "nl2cypher_model": "gemini-3-pro-preview",
            "query_engine": "manual",
        },
    )
    assert start_resp.status_code == 200
    job_id = start_resp.json()["job_id"]

    final_body = None
    for _ in range(40):
        poll_resp = client.get(f"/api/query_async/{job_id}")
        assert poll_resp.status_code == 200
        final_body = poll_resp.json()
        if final_body["status"] in {"completed", "failed"}:
            break
        time.sleep(0.01)

    assert final_body is not None
    assert final_body["status"] == "completed"
    assert captured["question"] == "鴻海的事業有哪些"
    assert captured["nl2cypher_provider"] == "gemini"
    assert captured["nl2cypher_model"] == "gemini-3-pro-preview"
    assert captured["query_engine"] == "manual"
    assert captured["progress_callback_called"] is True
    assert final_body["progress"]["llm_provider"] == "gemini"
    assert final_body["progress"]["llm_model"] == "gemini-3-pro-preview"
    assert final_body["progress"]["agentic_trace"]["plan_final"]["strategy"] == "single_query"
    assert final_body["result"]["agentic_trace"]["rounds"][0]["verdict"] == "accept"


def test_query_async_start_rejects_graph_chain_engine() -> None:
    """驗證 /api/query_async/start 不接受 graph_chain。"""
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    response = client.post(
        "/api/query_async/start",
        json={
            "question": "鴻海的事業有哪些",
            "query_engine": "graph_chain",
            "nl2cypher_provider": "gemini",
            "nl2cypher_model": "gemini-3-pro-preview",
        },
    )
    assert response.status_code == 400
    assert "query_engine=graph_chain" in response.json()["detail"]


def test_query_async_failed_job_keeps_agentic_trace(monkeypatch) -> None:
    """驗證 /api/query_async 在 failed 狀態仍保留 progress.agentic_trace。"""
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────

    def fake_query_kg(
        question,
        progress_callback=None,
        nl2cypher_provider=None,
        nl2cypher_model=None,
        query_engine=None,
    ):
        if progress_callback:
            progress_callback(
                {
                    "type": "agentic_progress",
                    "stage": "critic",
                    "round_count": 1,
                    "replan_count": 0,
                    "final_strategy": "single_query",
                    "failure_chain": [],
                    "detail": "Round 1: evaluating candidate",
                    "llm_provider": nl2cypher_provider,
                    "llm_model": nl2cypher_model,
                    "agentic_trace": {
                        "stage": "critic",
                        "round_count": 1,
                        "replan_count": 0,
                        "final_strategy": "single_query",
                        "failure_chain": [],
                        "planner_plan": {"strategy": "single_query"},
                        "plan_final": {"strategy": "single_query"},
                        "rounds": [{"round": 1, "verdict": "replan"}],
                    },
                }
            )
        raise RuntimeError("Cypher generation failed after retries: bad query")

    monkeypatch.setattr(main_module.logic, "query_kg", fake_query_kg)

    start_resp = client.post(
        "/api/query_async/start",
        json={
            "question": "鴻海的事業有哪些",
            "nl2cypher_provider": "gemini",
            "nl2cypher_model": "gemini-3-pro-preview",
            "query_engine": "manual",
        },
    )
    assert start_resp.status_code == 200
    job_id = start_resp.json()["job_id"]

    final_body = None
    for _ in range(40):
        poll_resp = client.get(f"/api/query_async/{job_id}")
        assert poll_resp.status_code == 200
        final_body = poll_resp.json()
        if final_body["status"] in {"completed", "failed"}:
            break
        time.sleep(0.01)

    assert final_body is not None
    assert final_body["status"] == "failed"
    assert "bad query" in final_body["error"]
    assert final_body["progress"]["agentic_trace"]["stage"] == "critic"
    assert final_body["progress"]["agentic_trace"]["rounds"][0]["verdict"] == "replan"


def test_process_keyword_async_job_flow(monkeypatch) -> None:
    """驗證 `test_process_keyword_async_job_flow` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    def fake_process_keyword_to_kg(
        keyword,
        uri,
        user,
        pwd,
        max_results=5,
        language="zh-tw",
        site_allowlist=None,
        chunk_limit=None,
        extraction_provider=None,
        extraction_model=None,
        progress_callback=None,
    ):
        """提供 `fake_process_keyword_to_kg` 測試替身以模擬外部依賴或固定回傳。
        此函式讓測試可注入可預測資料，將驗證焦點集中在流程與邏輯判斷。
        """
        if progress_callback:
            progress_callback(
                {
                    "status": "running",
                    "searched_keyword": keyword,
                    "fetched_urls": [],
                    "failed_urls": [],
                    "stats": {
                        "chunks_processed": 0,
                        "entities": 0,
                        "relations": 0,
                        "merged_entities": 0,
                        "dropped_relations": 0,
                        "json_retries": 0,
                    },
                    "summary": [],
                    "chunk_limit": chunk_limit,
                    "chunks_available": 1,
                    "chunk_progress": [
                        {
                            "order": 1,
                            "chunk_id": "c1",
                            "source_url": "https://example.com/article",
                            "title": "Example",
                            "chars": 100,
                            "status": "processing",
                            "entities": 0,
                            "relations": 0,
                        }
                    ],
                }
            )
        return {
            "searched_keyword": keyword,
            "fetched_urls": ["https://example.com/article"],
            "failed_urls": [],
            "stats": {
                "chunks_processed": 1,
                "entities": 1,
                "relations": 0,
                "merged_entities": 0,
                "dropped_relations": 0,
                "json_retries": 0,
            },
            "summary": [{"chunk_id": "c1", "entities": 1, "relations": 0, "source_url": "https://example.com/article"}],
            "chunk_limit": chunk_limit,
            "chunks_available": 1,
            "chunk_progress": [
                {
                    "order": 1,
                    "chunk_id": "c1",
                    "source_url": "https://example.com/article",
                    "title": "Example",
                    "chars": 100,
                    "status": "processed",
                    "entities": 1,
                    "relations": 0,
                }
            ],
        }

    monkeypatch.setattr(main_module.logic, "process_keyword_to_kg", fake_process_keyword_to_kg)

    start_resp = client.post(
        "/api/process_keyword_async/start",
        json={"keyword": "鴻海", "max_results": 1, "language": "zh-tw", "chunk_limit": 1},
    )
    assert start_resp.status_code == 200
    job_id = start_resp.json()["job_id"]

    final_status = None
    for _ in range(20):
        poll_resp = client.get(f"/api/process_keyword_async/{job_id}")
        assert poll_resp.status_code == 200
        body = poll_resp.json()
        final_status = body["status"]
        if body["status"] == "completed":
            assert body["result"]["searched_keyword"] == "鴻海"
            assert body["result"]["stats"]["chunks_processed"] == 1
            break
        time.sleep(0.01)
    assert final_status == "completed"


def test_process_text_async_job_flow(monkeypatch) -> None:
    """驗證 `test_process_text_async_job_flow` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    def fake_process_text_to_kg(
        text,
        uri,
        user,
        pwd,
        chunk_limit=None,
        extraction_provider=None,
        extraction_model=None,
        progress_callback=None,
    ):
        """提供 `fake_process_text_to_kg` 測試替身以模擬外部依賴或固定回傳。
        此函式讓測試可注入可預測資料，將驗證焦點集中在流程與邏輯判斷。
        """
        if progress_callback:
            progress_callback(
                {
                    "type": "chunk_update",
                    "chunk": {
                        "chunk_id": "c1",
                        "status": "processing",
                        "entities": 0,
                        "relations": 0,
                    },
                    "stats": {
                        "chunks_processed": 0,
                        "entities": 0,
                        "relations": 0,
                        "merged_entities": 0,
                        "dropped_relations": 0,
                        "json_retries": 0,
                    },
                    "chunk_limit": chunk_limit,
                    "chunks_available": 1,
                }
            )
        return {
            "stats": {
                "chunks_processed": 1,
                "entities": 2,
                "relations": 1,
                "merged_entities": 0,
                "dropped_relations": 0,
                "json_retries": 0,
            },
            "summary": [{"chunk_id": "c1", "entities": 2, "relations": 1}],
            "chunk_limit": chunk_limit,
            "chunks_available": 1,
            "chunk_progress": [
                {
                    "order": 1,
                    "chunk_id": "c1",
                    "source_url": "user_input",
                    "title": "User Input",
                    "chars": len(text),
                    "status": "processed",
                    "entities": 2,
                    "relations": 1,
                }
            ],
        }

    monkeypatch.setattr(main_module.logic, "process_text_to_kg", fake_process_text_to_kg)

    start_resp = client.post(
        "/api/process_text_async/start",
        json={"text": "鴻海由郭台銘創立", "chunk_limit": 1, "extraction_provider": "gemini"},
    )
    assert start_resp.status_code == 200
    job_id = start_resp.json()["job_id"]

    final_status = None
    for _ in range(20):
        poll_resp = client.get(f"/api/process_text_async/{job_id}")
        assert poll_resp.status_code == 200
        body = poll_resp.json()
        final_status = body["status"]
        if body["status"] == "completed":
            assert body["result"]["stats"]["chunks_processed"] == 1
            assert body["progress"]["status"] == "completed"
            break
        time.sleep(0.01)
    assert final_status == "completed"


def test_process_url_async_job_flow(monkeypatch) -> None:
    """驗證 `test_process_url_async_job_flow` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    def fake_process_url_to_kg(
        url,
        uri,
        user,
        pwd,
        chunk_limit=None,
        extraction_provider=None,
        extraction_model=None,
        progress_callback=None,
    ):
        """提供 `fake_process_url_to_kg` 測試替身以模擬外部依賴或固定回傳。
        此函式讓測試可注入可預測資料，將驗證焦點集中在流程與邏輯判斷。
        """
        if progress_callback:
            progress_callback(
                {
                    "type": "chunk_update",
                    "chunk": {
                        "chunk_id": "c1",
                        "status": "processing",
                        "entities": 0,
                        "relations": 0,
                    },
                    "stats": {
                        "chunks_processed": 0,
                        "entities": 0,
                        "relations": 0,
                        "merged_entities": 0,
                        "dropped_relations": 0,
                        "json_retries": 0,
                    },
                    "chunk_limit": chunk_limit,
                    "chunks_available": 1,
                }
            )
        return {
            "stats": {
                "chunks_processed": 1,
                "entities": 1,
                "relations": 0,
                "merged_entities": 0,
                "dropped_relations": 0,
                "json_retries": 0,
            },
            "summary": [{"chunk_id": "c1", "entities": 1, "relations": 0}],
            "chunk_limit": chunk_limit,
            "chunks_available": 1,
            "chunk_progress": [
                {
                    "order": 1,
                    "chunk_id": "c1",
                    "source_url": url,
                    "title": "Example",
                    "chars": 120,
                    "status": "processed",
                    "entities": 1,
                    "relations": 0,
                }
            ],
        }

    monkeypatch.setattr(main_module.logic, "process_url_to_kg", fake_process_url_to_kg)

    start_resp = client.post(
        "/api/process_url_async/start",
        json={"url": "https://example.com", "chunk_limit": 1, "extraction_provider": "gemini"},
    )
    assert start_resp.status_code == 200
    job_id = start_resp.json()["job_id"]

    final_status = None
    for _ in range(20):
        poll_resp = client.get(f"/api/process_url_async/{job_id}")
        assert poll_resp.status_code == 200
        body = poll_resp.json()
        final_status = body["status"]
        if body["status"] == "completed":
            assert body["result"]["stats"]["chunks_processed"] == 1
            assert body["progress"]["current_url"] == "https://example.com"
            break
        time.sleep(0.01)
    assert final_status == "completed"


def test_health_compat_endpoint(monkeypatch) -> None:
    """驗證 `test_health_compat_endpoint` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    monkeypatch.setattr(
        main_module.logic.llm_client,
        "get_runtime_config",
        lambda: type("Cfg", (), {"provider": "openai", "model": "mlx-community/Qwen3-8B-4bit-DWQ-053125"})(),
    )
    monkeypatch.setattr(
        main_module.logic.llm_client,
        "health_check",
        lambda timeout_seconds=3.0: {
            "provider": "openai",
            "model": "mlx-community/Qwen3-8B-4bit-DWQ-053125",
            "upstream": "openai-compatible",
            "status": "ok",
            "reachable": True,
        },
    )

    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["provider"] == "openai"
    assert body["upstream"]["status"] == "ok"


def test_chat_compat_endpoint(monkeypatch) -> None:
    """驗證 `test_chat_compat_endpoint` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    monkeypatch.setattr(
        main_module.logic.llm_client,
        "get_runtime_config",
        lambda: type("Cfg", (), {"provider": "openai", "model": "mlx-community/Qwen3-8B-4bit-DWQ-053125"})(),
    )
    monkeypatch.setattr(main_module.logic.llm_client, "chat_text", lambda **kwargs: "這是相容端點測試回覆")

    response = client.post("/api/chat", json={"question": "請介紹知識圖譜"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "這是相容端點測試回覆"
    assert body["model"] == "mlx-community/Qwen3-8B-4bit-DWQ-053125"
