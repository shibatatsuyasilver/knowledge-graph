"""Integration tests for the full KG extraction -> Neo4j -> QA pipeline.

Requires real services:
  - Neo4j reachable at NEO4J_URI (default bolt://localhost:7687)
  - LLM endpoint reachable (OPENAI_BASE_URL / OLLAMA_BASE_URL / GEMINI_API_KEY)

Run:
    uv run pytest -m e2e -v

Skip in CI (no infra):
    uv run pytest -m "not e2e"
"""

from __future__ import annotations

import os
import socket
import urllib.request
from urllib.parse import urlparse

import pytest

from backend.llm_kg import e2e_runner


# ---------------------------------------------------------------------------
# Availability checks — called once at module import time.
# ---------------------------------------------------------------------------

def _neo4j_reachable() -> bool:
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    parsed = urlparse(uri)
    host = parsed.hostname or "localhost"
    port = parsed.port or 7687
    sock = socket.socket()
    sock.settimeout(1.5)
    try:
        sock.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _llm_reachable() -> bool:
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if provider == "gemini":
        return True  # no public health endpoint; test will fail with auth error if key missing
    if provider == "openai":
        base = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1").rstrip("/")
        url = f"{base}/models"
    else:  # ollama
        base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        url = f"{base}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


_NO_NEO4J = not _neo4j_reachable()
_NO_LLM = not _llm_reachable()

skip_no_neo4j = pytest.mark.skipif(
    _NO_NEO4J,
    reason=f"Neo4j not reachable at {os.getenv('NEO4J_URI', 'bolt://localhost:7687')}",
)
skip_no_llm = pytest.mark.skipif(
    _NO_LLM,
    reason="LLM endpoint not reachable (check OPENAI_BASE_URL / OLLAMA_BASE_URL / GEMINI_API_KEY)",
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@skip_no_neo4j
@skip_no_llm
def test_e2e_full_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Full pipeline with no fallbacks: LLM extraction -> Neo4j write -> NL2Cypher QA."""
    monkeypatch.setenv("E2E_ALLOW_FALLBACK_EXTRACT", "0")
    monkeypatch.setenv("E2E_ALLOW_FALLBACK_QA", "0")

    result = e2e_runner.run()

    min_qa = int(os.getenv("MIN_QA_SUCCESS", "2"))
    assert result["extracted_entities"] > 0, f"LLM returned no entities: {result}"
    assert result["upserted_entities"] > 0, f"Neo4j upsert failed: {result}"
    assert result["qa_success"] >= min_qa, (
        f"QA passed {result['qa_success']}/{result['qa_total']}, need {min_qa}: {result['qa']}"
    )
    assert result["fallback_extract_used"] is False
    assert result["fallback_qa_used"] is False


@pytest.mark.e2e
@skip_no_neo4j
def test_e2e_pipeline_with_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pipeline with E2E_ALLOW_FALLBACK_EXTRACT=1.

    Validates Neo4j write + QA even when LLM is unavailable, using the
    deterministic fallback payload. Does not require a real LLM endpoint.
    """
    monkeypatch.setenv("E2E_ALLOW_FALLBACK_EXTRACT", "1")
    monkeypatch.setenv("E2E_ALLOW_FALLBACK_QA", "1")

    result = e2e_runner.run()

    min_qa = int(os.getenv("MIN_QA_SUCCESS", "2"))
    assert result["upserted_entities"] > 0, f"Neo4j upsert failed even with fallback: {result}"
    assert result["qa_success"] >= min_qa, (
        f"QA passed {result['qa_success']}/{result['qa_total']}, need {min_qa}: {result['qa']}"
    )
