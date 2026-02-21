#!/usr/bin/env python3
"""A tiny OpenAI-compatible mock server for deterministic tests."""

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
    messages = payload.get("messages", [])
    prompt = "\n".join(str(message.get("content", "")) for message in messages if isinstance(message, dict))

    if "知識圖譜抽取器" in prompt or "合法 JSON" in prompt:
        return json.dumps(EXTRACTION_PAYLOAD, ensure_ascii=False)
    if "Neo4j Cypher 查詢專家" in prompt:
        return pick_cypher(prompt)
    if "strict JSON planner" in prompt:
        return '{"intent":"list_treatments","entityType":"Disease","entityName":"Diabetes"}'
    return "mock-openai-ok"


class Handler(BaseHTTPRequestHandler):
    server_version = "MockOpenAICompatible/1.0"

    def _reply(self, status_code: int, data: Dict[str, Any]) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/v1/models":
            self._reply(
                200,
                {
                    "object": "list",
                    "data": [{"id": "mlx-community/Qwen3-8B-4bit-DWQ-053125", "object": "model"}],
                },
            )
            return
        self._reply(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
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
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "model": payload.get("model", "mlx-community/Qwen3-8B-4bit-DWQ-053125"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            },
        )


def main() -> None:
    server = HTTPServer(("0.0.0.0", 8080), Handler)
    print("mock openai-compatible server listening on 0.0.0.0:8080")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
