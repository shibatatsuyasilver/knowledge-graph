import pytest

from backend.services.ingest import ingest_service


BASE_STATS = {
    "chunks_processed": 1,
    "entities": 2,
    "relations": 1,
    "merged_entities": 0,
    "dropped_relations": 0,
    "json_retries": 0,
}


def _fake_chunks(url: str) -> list[ingest_service.Chunk]:
    """提供 `_fake_chunks` 測試替身以模擬外部依賴或固定回傳。
    此函式讓測試可注入可預測資料，將驗證焦點集中在流程與邏輯判斷。
    """
    return [ingest_service.Chunk(chunk_id="c1", text="example chunk", source_url=url, title="Example")]


def _fake_build_result(url: str) -> dict:
    """提供 `_fake_build_result` 測試替身以模擬外部依賴或固定回傳。
    此函式讓測試可注入可預測資料，將驗證焦點集中在流程與邏輯判斷。
    """
    return {
        "stats": dict(BASE_STATS),
        "summary": [{"chunk_id": "c1", "entities": 2, "relations": 1, "source_url": url}],
        "chunks_available": 1,
        "chunk_progress": [
            {
                "order": 1,
                "chunk_id": "c1",
                "source_url": url,
                "title": "Example",
                "chars": 120,
                "status": "processed",
                "entities": 2,
                "relations": 1,
            }
        ],
    }


def test_chunk_text_uses_token_budget_for_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 `test_chunk_text_uses_token_budget_for_gemini` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    monkeypatch.setattr(ingest_service, "CHUNK_SIZE_MODE", "provider")
    monkeypatch.setattr(ingest_service, "DEFAULT_TOKEN_CHUNK_SIZE", 5)
    monkeypatch.setattr(ingest_service, "DEFAULT_TOKEN_CHUNK_MIN_SIZE", 1)

    text = " ".join(f"w{i}" for i in range(12))
    chunks = ingest_service.chunk_text(
        text,
        "https://example.com/article",
        "Example",
        extraction_provider="gemini",
        max_tokens=5,
        min_tokens=1,
    )

    assert len(chunks) == 3
    assert all(chunk.tokens <= 5 for chunk in chunks)


def test_process_keyword_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 `test_process_keyword_success` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    monkeypatch.setattr(
        ingest_service,
        "_search_keyword_urls",
        lambda keyword, max_results, language: ["https://example.com/article"],
    )
    monkeypatch.setattr(ingest_service, "process_url", lambda url, extraction_provider=None: _fake_chunks(url))
    monkeypatch.setattr(
        ingest_service,
        "build_kg_from_chunks",
        lambda chunks, uri, user, pwd, chunk_limit=None, extraction_provider=None, extraction_model=None: _fake_build_result(
            chunks[0].source_url
        ),
    )

    result = ingest_service.process_keyword_to_kg(
        keyword="半導體",
        uri="bolt://localhost:7687",
        user="neo4j",
        pwd="password",
        max_results=5,
        language="zh-tw",
    )

    assert result["searched_keyword"] == "半導體"
    assert result["fetched_urls"] == ["https://example.com/article"]
    assert result["failed_urls"] == []
    assert result["stats"]["entities"] == 2
    assert result["chunks_available"] == 1
    assert len(result["chunk_progress"]) == 1
    assert result["chunk_progress"][0]["status"] == "processed"
    assert result["summary"][0]["source_url"] == "https://example.com/article"


def test_process_keyword_no_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 `test_process_keyword_no_results` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    monkeypatch.setattr(ingest_service, "_search_keyword_urls", lambda keyword, max_results, language: [])

    with pytest.raises(ValueError, match="No results found"):
        ingest_service.process_keyword_to_kg(
            keyword="不存在資料",
            uri="bolt://localhost:7687",
            user="neo4j",
            pwd="password",
            max_results=5,
            language="zh-tw",
        )


def test_process_keyword_partial_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 `test_process_keyword_partial_failures` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    urls = ["https://ok.example.com/1", "https://bad.example.com/2"]
    monkeypatch.setattr(ingest_service, "_search_keyword_urls", lambda keyword, max_results, language: urls)

    def fake_process_url(url: str, extraction_provider: str | None = None) -> list[ingest_service.Chunk]:
        """提供 `fake_process_url` 測試替身以模擬外部依賴或固定回傳。
        此函式讓測試可注入可預測資料，將驗證焦點集中在流程與邏輯判斷。
        """
        if "bad" in url:
            raise ValueError("fetch failed")
        return _fake_chunks(url)

    monkeypatch.setattr(ingest_service, "process_url", fake_process_url)
    monkeypatch.setattr(
        ingest_service,
        "build_kg_from_chunks",
        lambda chunks, uri, user, pwd, chunk_limit=None, extraction_provider=None, extraction_model=None: _fake_build_result(
            chunks[0].source_url
        ),
    )

    result = ingest_service.process_keyword_to_kg(
        keyword="晶片",
        uri="bolt://localhost:7687",
        user="neo4j",
        pwd="password",
        max_results=5,
        language="en",
    )

    assert result["fetched_urls"] == ["https://ok.example.com/1"]
    assert len(result["failed_urls"]) == 1
    assert result["failed_urls"][0]["url"] == "https://bad.example.com/2"
    assert result["stats"]["chunks_processed"] == 1
    assert result["chunks_available"] == 1
    assert len(result["chunk_progress"]) == 1


def test_process_keyword_forwards_chunk_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 `test_process_keyword_forwards_chunk_limit` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    monkeypatch.setattr(
        ingest_service,
        "_search_keyword_urls",
        lambda keyword, max_results, language: ["https://example.com/article"],
    )
    monkeypatch.setattr(
        ingest_service,
        "process_url",
        lambda url, extraction_provider=None: [
            ingest_service.Chunk(chunk_id="c1", text="chunk-1", source_url=url, title="Example"),
            ingest_service.Chunk(chunk_id="c2", text="chunk-2", source_url=url, title="Example"),
        ],
    )

    captured: dict = {}

    def fake_build(chunks, uri, user, pwd, chunk_limit=None, extraction_provider=None, extraction_model=None):
        """提供 `fake_build` 測試替身以模擬外部依賴或固定回傳。
        此函式讓測試可注入可預測資料，將驗證焦點集中在流程與邏輯判斷。
        """
        captured["chunk_limit"] = chunk_limit
        captured["chunk_count"] = len(chunks)
        captured["extraction_provider"] = extraction_provider
        captured["extraction_model"] = extraction_model
        payload = _fake_build_result(chunks[0].source_url)
        payload["chunks_available"] = len(chunks)
        payload["chunk_progress"] = [
            {
                "order": 1,
                "chunk_id": "c1",
                "source_url": chunks[0].source_url,
                "title": "Example",
                "chars": len(chunks[0].text),
                "status": "processed",
                "entities": 2,
                "relations": 1,
            },
            {
                "order": 2,
                "chunk_id": "c2",
                "source_url": chunks[1].source_url,
                "title": "Example",
                "chars": len(chunks[1].text),
                "status": "processed",
                "entities": 0,
                "relations": 0,
            },
        ]
        return payload

    monkeypatch.setattr(ingest_service, "build_kg_from_chunks", fake_build)

    result = ingest_service.process_keyword_to_kg(
        keyword="鴻海",
        uri="bolt://localhost:7687",
        user="neo4j",
        pwd="password",
        max_results=1,
        language="zh-tw",
        chunk_limit=5,
        extraction_provider="gemini",
        extraction_model="gemini-3-pro-preview",
    )

    assert result["chunk_limit"] == 5
    assert result["chunks_available"] == 2
    assert len(result["chunk_progress"]) == 2
    assert captured["chunk_limit"] == 5
    assert captured["chunk_count"] == 2
    assert captured["extraction_provider"] == "gemini"
    assert captured["extraction_model"] == "gemini-3-pro-preview"


def test_process_keyword_emits_progress_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 `test_process_keyword_emits_progress_updates` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    monkeypatch.setattr(
        ingest_service,
        "_search_keyword_urls",
        lambda keyword, max_results, language: ["https://example.com/article"],
    )
    monkeypatch.setattr(
        ingest_service,
        "process_url",
        lambda url, extraction_provider=None: [
            ingest_service.Chunk(chunk_id="c1", text="chunk-1", source_url=url, title="Example"),
            ingest_service.Chunk(chunk_id="c2", text="chunk-2", source_url=url, title="Example"),
        ],
    )

    def fake_build(
        chunks,
        uri,
        user,
        pwd,
        chunk_limit=None,
        extraction_provider=None,
        extraction_model=None,
        progress_callback=None,
    ):
        """提供 `fake_build` 測試替身以模擬外部依賴或固定回傳。
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
                }
            )
            progress_callback(
                {
                    "type": "chunk_update",
                    "chunk": {
                        "chunk_id": "c1",
                        "status": "processed",
                        "entities": 2,
                        "relations": 1,
                    },
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
            "chunks_available": 2,
            "chunk_progress": [
                {
                    "order": 1,
                    "chunk_id": "c1",
                    "source_url": chunks[0].source_url,
                    "title": "Example",
                    "chars": len(chunks[0].text),
                    "status": "processed",
                    "entities": 2,
                    "relations": 1,
                },
                {
                    "order": 2,
                    "chunk_id": "c2",
                    "source_url": chunks[1].source_url,
                    "title": "Example",
                    "chars": len(chunks[1].text),
                    "status": "skipped_by_limit",
                    "entities": 0,
                    "relations": 0,
                },
            ],
        }

    monkeypatch.setattr(ingest_service, "build_kg_from_chunks", fake_build)

    snapshots: list[dict] = []
    result = ingest_service.process_keyword_to_kg(
        keyword="鴻海",
        uri="bolt://localhost:7687",
        user="neo4j",
        pwd="password",
        max_results=1,
        language="zh-tw",
        chunk_limit=1,
        progress_callback=lambda payload: snapshots.append(payload),
    )

    assert result["stats"]["chunks_processed"] == 1
    assert len(snapshots) >= 2
    assert snapshots[-1]["status"] == "completed"
    assert snapshots[-1]["chunk_progress"][0]["status"] == "processed"


def test_search_keyword_urls_falls_back_to_html(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 `test_search_keyword_urls_falls_back_to_html` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    class EmptyDDGS:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def text(self, *args, **kwargs):
            return []

    class FakeResponse:
        text = """
        <html>
          <body>
            <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.honhai.com%2Fen-us">Hon Hai</a>
            <a class="result__a" href="https://example.com/news">News</a>
          </body>
        </html>
        """

        def raise_for_status(self):
            return None

    def fake_get(url, params=None, timeout=None, headers=None):
        """提供 `fake_get` 測試替身以模擬外部依賴或固定回傳。
        此函式讓測試可注入可預測資料，將驗證焦點集中在流程與邏輯判斷。
        """
        assert url == "https://html.duckduckgo.com/html/"
        assert params["q"] == "鴻海"
        return FakeResponse()

    monkeypatch.setattr(ingest_service, "DDGS", EmptyDDGS)
    monkeypatch.setattr(ingest_service.requests, "get", fake_get)

    urls = ingest_service._search_keyword_urls(keyword="鴻海", max_results=5, language="zh-tw")

    assert urls[:2] == ["https://www.honhai.com/en-us", "https://example.com/news"]


def test_search_keyword_urls_falls_back_to_wikipedia_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 `test_search_keyword_urls_falls_back_to_wikipedia_api` 所描述情境是否符合預期行為。
    此測試透過斷言比對輸出與狀態，避免後續修改造成回歸問題。
    """
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    class EmptyDDGS:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def text(self, *args, **kwargs):
            return []

    class HtmlChallengeResponse:
        text = "<html><body>challenge</body></html>"

        def raise_for_status(self):
            return None

    class WikiResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "query": {
                    "search": [
                        {"title": "台灣積體電路製造"},
                        {"title": "張忠謀"},
                    ]
                }
            }

    def fake_get(url, params=None, timeout=None, headers=None):
        """提供 `fake_get` 測試替身以模擬外部依賴或固定回傳。
        此函式讓測試可注入可預測資料，將驗證焦點集中在流程與邏輯判斷。
        """
        if url == "https://html.duckduckgo.com/html/":
            return HtmlChallengeResponse()
        if url == "https://zh.wikipedia.org/w/api.php":
            assert params["srsearch"] == "台積電"
            return WikiResponse()
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(ingest_service, "DDGS", EmptyDDGS)
    monkeypatch.setattr(ingest_service.requests, "get", fake_get)

    urls = ingest_service._search_keyword_urls(keyword="台積電", max_results=5, language="zh-tw")

    assert urls == [
        "https://zh.wikipedia.org/wiki/%E5%8F%B0%E7%81%A3%E7%A9%8D%E9%AB%94%E9%9B%BB%E8%B7%AF%E8%A3%BD%E9%80%A0",
        "https://zh.wikipedia.org/wiki/%E5%BC%B5%E5%BF%A0%E8%AC%80",
    ]
