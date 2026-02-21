import pytest

from backend import logic


BASE_STATS = {
    "chunks_processed": 1,
    "entities": 2,
    "relations": 1,
    "merged_entities": 0,
    "dropped_relations": 0,
    "json_retries": 0,
}


def _fake_chunks(url: str) -> list[logic.Chunk]:
    return [logic.Chunk(chunk_id="c1", text="example chunk", source_url=url, title="Example")]


def _fake_build_result(url: str) -> dict:
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
    monkeypatch.setattr(logic, "CHUNK_SIZE_MODE", "provider")
    monkeypatch.setattr(logic, "DEFAULT_TOKEN_CHUNK_SIZE", 5)
    monkeypatch.setattr(logic, "DEFAULT_TOKEN_CHUNK_MIN_SIZE", 1)

    text = " ".join(f"w{i}" for i in range(12))
    chunks = logic.chunk_text(
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
    monkeypatch.setattr(
        logic,
        "_search_keyword_urls",
        lambda keyword, max_results, language: ["https://example.com/article"],
    )
    monkeypatch.setattr(logic, "process_url", lambda url, extraction_provider=None: _fake_chunks(url))
    monkeypatch.setattr(
        logic,
        "build_kg_from_chunks",
        lambda chunks, uri, user, pwd, chunk_limit=None, extraction_provider=None, extraction_model=None: _fake_build_result(
            chunks[0].source_url
        ),
    )

    result = logic.process_keyword_to_kg(
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
    monkeypatch.setattr(logic, "_search_keyword_urls", lambda keyword, max_results, language: [])

    with pytest.raises(ValueError, match="No results found"):
        logic.process_keyword_to_kg(
            keyword="不存在資料",
            uri="bolt://localhost:7687",
            user="neo4j",
            pwd="password",
            max_results=5,
            language="zh-tw",
        )


def test_process_keyword_partial_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    urls = ["https://ok.example.com/1", "https://bad.example.com/2"]
    monkeypatch.setattr(logic, "_search_keyword_urls", lambda keyword, max_results, language: urls)

    def fake_process_url(url: str, extraction_provider: str | None = None) -> list[logic.Chunk]:
        if "bad" in url:
            raise ValueError("fetch failed")
        return _fake_chunks(url)

    monkeypatch.setattr(logic, "process_url", fake_process_url)
    monkeypatch.setattr(
        logic,
        "build_kg_from_chunks",
        lambda chunks, uri, user, pwd, chunk_limit=None, extraction_provider=None, extraction_model=None: _fake_build_result(
            chunks[0].source_url
        ),
    )

    result = logic.process_keyword_to_kg(
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
    monkeypatch.setattr(
        logic,
        "_search_keyword_urls",
        lambda keyword, max_results, language: ["https://example.com/article"],
    )
    monkeypatch.setattr(
        logic,
        "process_url",
        lambda url, extraction_provider=None: [
            logic.Chunk(chunk_id="c1", text="chunk-1", source_url=url, title="Example"),
            logic.Chunk(chunk_id="c2", text="chunk-2", source_url=url, title="Example"),
        ],
    )

    captured: dict = {}

    def fake_build(chunks, uri, user, pwd, chunk_limit=None, extraction_provider=None, extraction_model=None):
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

    monkeypatch.setattr(logic, "build_kg_from_chunks", fake_build)

    result = logic.process_keyword_to_kg(
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
    monkeypatch.setattr(
        logic,
        "_search_keyword_urls",
        lambda keyword, max_results, language: ["https://example.com/article"],
    )
    monkeypatch.setattr(
        logic,
        "process_url",
        lambda url, extraction_provider=None: [
            logic.Chunk(chunk_id="c1", text="chunk-1", source_url=url, title="Example"),
            logic.Chunk(chunk_id="c2", text="chunk-2", source_url=url, title="Example"),
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

    monkeypatch.setattr(logic, "build_kg_from_chunks", fake_build)

    snapshots: list[dict] = []
    result = logic.process_keyword_to_kg(
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
        assert url == "https://html.duckduckgo.com/html/"
        assert params["q"] == "鴻海"
        return FakeResponse()

    monkeypatch.setattr(logic, "DDGS", EmptyDDGS)
    monkeypatch.setattr(logic.requests, "get", fake_get)

    urls = logic._search_keyword_urls(keyword="鴻海", max_results=5, language="zh-tw")

    assert urls[:2] == ["https://www.honhai.com/en-us", "https://example.com/news"]


def test_search_keyword_urls_falls_back_to_wikipedia_api(monkeypatch: pytest.MonkeyPatch) -> None:
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
        if url == "https://html.duckduckgo.com/html/":
            return HtmlChallengeResponse()
        if url == "https://zh.wikipedia.org/w/api.php":
            assert params["srsearch"] == "台積電"
            return WikiResponse()
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(logic, "DDGS", EmptyDDGS)
    monkeypatch.setattr(logic.requests, "get", fake_get)

    urls = logic._search_keyword_urls(keyword="台積電", max_results=5, language="zh-tw")

    assert urls == [
        "https://zh.wikipedia.org/wiki/%E5%8F%B0%E7%81%A3%E7%A9%8D%E9%AB%94%E9%9B%BB%E8%B7%AF%E8%A3%BD%E9%80%A0",
        "https://zh.wikipedia.org/wiki/%E5%BC%B5%E5%BF%A0%E8%AC%80",
    ]


def test_query_kg_returns_answer_text(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_loader():
        def fake_answer_with_manual_prompt(question: str):
            return {
                "question": question,
                "cypher": "MATCH (n) RETURN n.name AS partner LIMIT 1",
                "rows": [{"partner": "NVIDIA"}],
                "attempt": 1,
            }

        return object, fake_answer_with_manual_prompt

    monkeypatch.setattr(logic, "_load_kg_modules", fake_loader)

    result = logic.query_kg("鴻海公司的合作夥伴")

    assert result["rows"] == [{"partner": "NVIDIA"}]
    assert "answer" in result
    assert "NVIDIA" in result["answer"]
    assert result["answer_source"] in {"qa_llm", "template_fallback"}


def test_query_kg_handles_empty_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_loader():
        def fake_answer_with_manual_prompt(question: str):
            return {
                "question": question,
                "cypher": "MATCH (n) WHERE false RETURN n",
                "rows": [],
                "attempt": 1,
            }

        return object, fake_answer_with_manual_prompt

    monkeypatch.setattr(logic, "_load_kg_modules", fake_loader)

    result = logic.query_kg("不存在的問題")

    assert result["rows"] == []
    assert "找不到" in result["answer"]
    assert result["answer_source"] in {"qa_llm", "template_fallback"}


def test_query_kg_prefers_qa_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_loader():
        def fake_answer_with_manual_prompt(question: str):
            return {
                "question": question,
                "cypher": "MATCH (o:Organization)-[:CHAIRED_BY]->(p:Person) RETURN p.name AS name",
                "rows": [{"name": "劉揚偉"}],
                "attempt": 1,
            }

        return object, fake_answer_with_manual_prompt

    monkeypatch.setattr(logic, "_load_kg_modules", fake_loader)
    monkeypatch.setattr(logic, "_kg_qa_use_llm", lambda: True)
    monkeypatch.setattr(logic.llm_client, "chat_text", lambda **kwargs: "鴻海董事長是劉揚偉。")

    result = logic.query_kg("鴻海董事長是誰")

    assert result["answer"] == "鴻海董事長是劉揚偉。"
    assert result["answer_source"] == "qa_llm"


def test_query_kg_fallback_when_qa_llm_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_loader():
        def fake_answer_with_manual_prompt(question: str):
            return {
                "question": question,
                "cypher": "MATCH (o:Organization)-[:CHAIRED_BY]->(p:Person) RETURN p.name AS name",
                "rows": [{"name": "劉揚偉"}],
                "attempt": 1,
            }

        return object, fake_answer_with_manual_prompt

    monkeypatch.setattr(logic, "_load_kg_modules", fake_loader)
    monkeypatch.setattr(logic, "_kg_qa_use_llm", lambda: True)

    def raise_chat(**_kwargs):
        raise RuntimeError("llm down")

    monkeypatch.setattr(logic.llm_client, "chat_text", raise_chat)

    result = logic.query_kg("鴻海董事長是誰")

    assert "劉揚偉" in result["answer"]
    assert result["answer_source"] == "template_fallback"


def test_chat_general_uses_shared_llm_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        logic.llm_client,
        "chat_text",
        lambda **kwargs: "這是一個測試回覆",
    )
    monkeypatch.setattr(
        logic.llm_client,
        "get_runtime_config",
        lambda: type("Cfg", (), {"model": "mlx-community/Qwen3-8B-4bit-DWQ-053125", "provider": "openai"})(),
    )

    result = logic.chat_general("我今天睡不好")

    assert result["answer"] == "這是一個測試回覆"
    assert result["model"] == "mlx-community/Qwen3-8B-4bit-DWQ-053125"
    assert result["provider"] == "openai"
