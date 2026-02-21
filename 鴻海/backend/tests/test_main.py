import requests
import time
from fastapi.testclient import TestClient

import backend.main as main_module
from backend import logic


client = TestClient(main_module.app)


def test_process_url_route_is_unique() -> None:
    routes = [
        route
        for route in main_module.app.routes
        if getattr(route, "path", None) == "/api/process_url" and "POST" in getattr(route, "methods", set())
    ]
    assert len(routes) == 1


def test_process_url_endpoint_success(monkeypatch) -> None:
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
    captured = {}

    def fake_process_url_to_kg(url, uri, user, pwd, chunk_limit=None, extraction_provider=None, extraction_model=None):
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
    def timeout_chat(message, history=None):
        raise requests.Timeout("timeout")

    monkeypatch.setattr(main_module.logic, "chat_general", timeout_chat)

    response = client.post("/api/chat_general", json={"message": "睡不著"})
    assert response.status_code == 504
    assert response.json()["detail"] == "Upstream service timeout"


def test_process_keyword_async_job_flow(monkeypatch) -> None:
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
