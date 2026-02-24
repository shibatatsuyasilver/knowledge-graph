from __future__ import annotations

import pytest

from backend.llm_kg import nl2cypher


def _runtime_cfg(*, provider: str = "openai", model: str = "mlx-community/Qwen3-8B-4bit-DWQ-053125", gemini_api_key: str = "test-key"):
    return type(
        "Cfg",
        (),
        {
            "provider": provider,
            "model": model,
            "gemini_api_key": gemini_api_key,
        },
    )()


def test_query_with_graph_chain_raises_when_langchain_components_missing() -> None:
    """驗證缺少 LangChain 元件時會明確拋錯。"""
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    original_chain = nl2cypher.GraphCypherQAChain
    original_graph = nl2cypher.Neo4jGraph
    try:
        nl2cypher.GraphCypherQAChain = None
        nl2cypher.Neo4jGraph = None
        with pytest.raises(RuntimeError, match="LangChain components are not installed"):
            nl2cypher.query_with_graph_chain("鴻海是誰")
    finally:
        nl2cypher.GraphCypherQAChain = original_chain
        nl2cypher.Neo4jGraph = original_graph


def test_query_with_graph_chain_rejects_unsupported_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 GraphChain provider 僅接受 ollama/gemini。"""
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────

    class FakeChain:
        @classmethod
        def from_llm(cls, **_kwargs):
            return cls()

        def invoke(self, _payload):
            return {"result": "ok", "intermediate_steps": []}

    class FakeGraph:
        def __init__(self, **_kwargs):
            pass

    monkeypatch.setattr(nl2cypher, "GraphCypherQAChain", FakeChain)
    monkeypatch.setattr(nl2cypher, "Neo4jGraph", FakeGraph)
    monkeypatch.setattr(nl2cypher, "Ollama", object())
    monkeypatch.setattr(nl2cypher, "ChatGoogleGenerativeAI", object())
    monkeypatch.setattr(nl2cypher.llm_client, "get_runtime_config", lambda: _runtime_cfg(provider="openai"))

    with pytest.raises(ValueError, match="Unsupported GraphCypherQAChain provider"):
        nl2cypher.query_with_graph_chain("鴻海是誰", nl2cypher_provider="openai")


def test_query_with_graph_chain_gemini_parses_intermediate_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 GraphChain + Gemini 會解析 intermediate steps 的 query/context。"""
    # ─── Arrange：準備測試輸入、替身與前置狀態 ─────────────────────
    # ─── Act：呼叫被測流程，收集實際輸出與副作用 ─────────────────
    # ─── Assert：驗證關鍵結果，確保行為契約不回歸 ─────────────────
    captured: dict[str, object] = {}

    class FakeGraph:
        def __init__(self, **kwargs):
            captured["graph_kwargs"] = kwargs

    class FakeGemini:
        def __init__(self, **kwargs):
            captured["gemini_kwargs"] = kwargs

    class FakeChain:
        @classmethod
        def from_llm(
            cls,
            llm,
            graph,
            verbose: bool,
            return_intermediate_steps: bool,
            allow_dangerous_requests: bool,
        ):
            captured["from_llm"] = {
                "llm": llm,
                "graph": graph,
                "verbose": verbose,
                "return_intermediate_steps": return_intermediate_steps,
                "allow_dangerous_requests": allow_dangerous_requests,
            }
            return cls()

        def invoke(self, payload):
            captured["invoke_payload"] = payload
            return {
                "result": "鴻海是全球電子代工公司。",
                "intermediate_steps": [
                    {
                        "query": "MATCH (o:Organization) RETURN o.name AS organization",
                        "context": [{"organization": "鴻海精密"}, {"organization": "Foxconn"}],
                    }
                ],
            }

    monkeypatch.setattr(nl2cypher, "GraphCypherQAChain", FakeChain)
    monkeypatch.setattr(nl2cypher, "Neo4jGraph", FakeGraph)
    monkeypatch.setattr(nl2cypher, "ChatGoogleGenerativeAI", FakeGemini)
    monkeypatch.setattr(nl2cypher.llm_client, "get_runtime_config", lambda: _runtime_cfg(gemini_api_key="gem-key"))

    result = nl2cypher.query_with_graph_chain(
        "鴻海是誰",
        nl2cypher_provider="gemini",
        nl2cypher_model="gemini-3-pro-preview",
    )

    assert captured["invoke_payload"] == {"query": "鴻海是誰"}
    assert captured["gemini_kwargs"] == {
        "model": "gemini-3-pro-preview",
        "temperature": 0,
        "google_api_key": "gem-key",
    }
    assert result["cypher"] == "MATCH (o:Organization) RETURN o.name AS organization"
    assert result["rows"] == [{"organization": "鴻海精密"}, {"organization": "Foxconn"}]
    assert result["answer"] == "鴻海是全球電子代工公司。"
    assert result["engine_provider"] == "gemini"
    assert result["engine_model"] == "gemini-3-pro-preview"
    assert result["query_engine"] == "graph_chain"
