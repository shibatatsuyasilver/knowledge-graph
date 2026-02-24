#!/usr/bin/env python3
"""A tiny Ollama-compatible mock server for deterministic e2e tests."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict


EXTRACTION_PAYLOAD: Dict[str, Any] = {
    "entities": [
        {"name": "台積電", "type": "Organization"},
        {"name": "張忠謀", "type": "Person"},
        {"name": "新竹科學園區", "type": "Location"},
        {"name": "Apple", "type": "Organization"},
        {"name": "NVIDIA", "type": "Organization"},
        {"name": "3奈米", "type": "Technology"},
    ],
    "relations": [
        {"source": "台積電", "relation": "FOUNDED_BY", "target": "張忠謀"},
        {"source": "台積電", "relation": "HEADQUARTERED_IN", "target": "新竹科學園區"},
        {"source": "台積電", "relation": "SUPPLIES_TO", "target": "Apple"},
        {"source": "台積電", "relation": "SUPPLIES_TO", "target": "NVIDIA"},
        {"source": "台積電", "relation": "PRODUCES", "target": "3奈米"},
    ],
}


def pick_cypher(prompt: str) -> str:
    """`pick_cypher` 的主要流程入口。

主要用途：
- 串接此函式負責的核心步驟並回傳既有格式。
- 例外沿用現行錯誤處理策略，避免破壞呼叫端契約。

維護重點：
- 調整流程時需保持 API 欄位、狀態轉移與錯誤語意一致。
    """
    if "誰創立了台積電" in prompt or "創立" in prompt:
        return (
            "MATCH (o:Organization {name:'台積電'})-[:FOUNDED_BY]->(p:Person) "
            "RETURN p.name AS founder"
        )
    if "總部" in prompt:
        return (
            "MATCH (o:Organization {name:'台積電'})-[:HEADQUARTERED_IN]->(l:Location) "
            "RETURN l.name AS location"
        )
    if "供應" in prompt:
        return (
            "MATCH (o:Organization {name:'台積電'})-[:SUPPLIES_TO]->(c:Organization) "
            "RETURN c.name AS customer ORDER BY customer"
        )
    return "MATCH (e:Entity) RETURN e.name AS name LIMIT 10"


def pick_content(payload: Dict[str, Any]) -> str:
    """`pick_content` 的主要流程入口。

主要用途：
- 串接此函式負責的核心步驟並回傳既有格式。
- 例外沿用現行錯誤處理策略，避免破壞呼叫端契約。

維護重點：
- 調整流程時需保持 API 欄位、狀態轉移與錯誤語意一致。
    """
    messages = payload.get("messages", [])
    prompt = "\n".join(str(message.get("content", "")) for message in messages if isinstance(message, dict))

    if "知識圖譜抽取器" in prompt or "合法 JSON" in prompt:
        return json.dumps(EXTRACTION_PAYLOAD, ensure_ascii=False)
    if "Neo4j Cypher 查詢專家" in prompt:
        return pick_cypher(prompt)
    return "mock-ollama-ok"


class Handler(BaseHTTPRequestHandler):
    server_version = "MockOllama/1.0"

    def _reply(self, status_code: int, data: Dict[str, Any]) -> None:
        """`_reply` 的內部輔助函式。

主要用途：
- 封裝局部步驟，讓主流程維持可讀性。
- 集中處理細節與邊界條件，避免重複邏輯分散。

回傳約定：
- 保持既有輸入/輸出契約，不改變對外行為。
        """
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        """`do_GET` 的主要流程入口。

主要用途：
- 串接此函式負責的核心步驟並回傳既有格式。
- 例外沿用現行錯誤處理策略，避免破壞呼叫端契約。

維護重點：
- 調整流程時需保持 API 欄位、狀態轉移與錯誤語意一致。
        """
        if self.path == "/api/tags":
            self._reply(
                200,
                {
                    "models": [
                        {"name": "deepseek-r1:8b", "model": "deepseek-r1:8b"},
                    ]
                },
            )
            return
        self._reply(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        """`do_POST` 的主要流程入口。

主要用途：
- 串接此函式負責的核心步驟並回傳既有格式。
- 例外沿用現行錯誤處理策略，避免破壞呼叫端契約。

維護重點：
- 調整流程時需保持 API 欄位、狀態轉移與錯誤語意一致。
        """
        if self.path != "/api/chat":
            self._reply(404, {"error": "not_found"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._reply(400, {"error": "invalid_json"})
            return

        content = pick_content(payload)
        self._reply(
            200,
            {
                "model": payload.get("model", "deepseek-r1:8b"),
                "message": {"role": "assistant", "content": content},
                "done": True,
            },
        )


def main() -> None:
    """作為模組執行入口，串接並啟動既有主流程。
    此函式會依目前設定呼叫核心邏輯，並維持原本輸入輸出與錯誤行為。
    """
    server = HTTPServer(("0.0.0.0", 11434), Handler)
    print("mock ollama listening on 0.0.0.0:11434")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
