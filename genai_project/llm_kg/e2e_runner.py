"""E2E runner: extraction -> graph upsert -> text-to-cypher query."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

try:
    from .kg_builder import KnowledgeGraphBuilder, SAMPLE_TEXT, deterministic_fallback_payload
    from .nl2cypher import answer_with_manual_prompt
except ImportError:  # pragma: no cover - direct script execution
    from kg_builder import KnowledgeGraphBuilder, SAMPLE_TEXT, deterministic_fallback_payload
    from nl2cypher import answer_with_manual_prompt

try:
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover - optional dependency in lightweight environments
    GraphDatabase = None


def _reset_graph(builder: KnowledgeGraphBuilder) -> None:
    with builder.driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _fallback_qa_template(
    *,
    question: str,
    uri: str,
    user: str,
    password: str,
) -> Dict[str, Any]:
    if GraphDatabase is None:
        raise RuntimeError("neo4j driver not available for fallback QA")

    if "創立" in question:
        cypher = "MATCH (o:Organization {name:'台積電'})-[:FOUNDED_BY]->(p:Person) RETURN p.name AS founder"
    elif "總部" in question:
        cypher = "MATCH (o:Organization {name:'台積電'})-[:HEADQUARTERED_IN]->(l:Location) RETURN l.name AS location"
    elif "供應" in question:
        cypher = (
            "MATCH (o:Organization {name:'台積電'})-[:SUPPLIES_TO]->(c:Organization) "
            "RETURN c.name AS customer ORDER BY customer"
        )
    else:
        cypher = "MATCH (e:Entity) RETURN e.name AS name LIMIT 10"

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            rows = [dict(row) for row in session.run(cypher)]
    finally:
        driver.close()

    return {
        "question": question,
        "cypher": cypher,
        "rows": rows,
        "attempt": 1,
        "fallback": "template",
    }


def run() -> Dict[str, Any]:
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    llm_model = os.getenv("LLM_MODEL", os.getenv("OLLAMA_MODEL", "mlx-community/Qwen3-8B-4bit-DWQ-053125"))
    extraction_model = os.getenv("EXTRACTION_MODEL", llm_model)
    nl2cypher_model = os.getenv("NL2CYPHER_MODEL", llm_model)
    min_qa_success = int(os.getenv("MIN_QA_SUCCESS", "2"))
    input_text = os.getenv("E2E_SAMPLE_TEXT", SAMPLE_TEXT)
    allow_fallback_extract = _bool_env("E2E_ALLOW_FALLBACK_EXTRACT", False)
    allow_fallback_qa = _bool_env("E2E_ALLOW_FALLBACK_QA", allow_fallback_extract)
    fallback_extract_used = False
    fallback_qa_used = False
    fallback_reason = ""

    print(f"[e2e] model(extraction)={extraction_model}")
    print(f"[e2e] model(nl2cypher)={nl2cypher_model}")
    print(f"[e2e] sample_text_chars={len(input_text)}")
    builder = KnowledgeGraphBuilder(neo4j_uri, neo4j_user, neo4j_password)
    try:
        print("[e2e] reset graph")
        _reset_graph(builder)
        print("[e2e] extracting entities/relations")
        try:
            extracted = builder.extract_entities_relations(input_text)
        except Exception as exc:
            if not allow_fallback_extract:
                raise
            fallback_extract_used = True
            fallback_reason = f"extract_exception:{exc}"
            print(f"[e2e][warn] extraction failed, using deterministic fallback: {exc}")
            extracted = deterministic_fallback_payload()
        print("[e2e] populating graph")
        stats = builder.populate_graph(extracted)
        if (stats.entities == 0 or stats.relations == 0) and allow_fallback_extract and not fallback_extract_used:
            fallback_extract_used = True
            fallback_reason = "insufficient_extraction"
            print("[e2e][warn] extraction insufficient, retrying with deterministic fallback payload.")
            _reset_graph(builder)
            extracted = deterministic_fallback_payload()
            stats = builder.populate_graph(extracted)
    finally:
        builder.close()

    if stats.entities == 0 or stats.relations == 0:
        raise RuntimeError(
            f"Graph build has insufficient data: entities={stats.entities}, relations={stats.relations}"
        )

    questions = [
        "誰創立了台積電？",
        "台積電的總部在哪裡？",
        "台積電供應晶片給哪些公司？",
    ]

    qa_results: List[Dict[str, Any]] = []
    qa_success = 0
    for question in questions:
        print(f"[e2e] qa question={question}")
        try:
            result = answer_with_manual_prompt(question)
        except Exception as exc:
            if not allow_fallback_qa:
                raise
            fallback_qa_used = True
            print(f"[e2e][warn] nl2cypher failed, using fallback QA template: {exc}")
            result = _fallback_qa_template(
                question=question,
                uri=neo4j_uri,
                user=neo4j_user,
                password=neo4j_password,
            )
        if result.get("rows"):
            qa_success += 1
        qa_results.append(result)

    if qa_success < min_qa_success:
        raise RuntimeError(
            f"QA success ratio is too low: success={qa_success}/{len(questions)}, min_required={min_qa_success}"
        )

    return {
        "model": llm_model,
        "model_extraction": extraction_model,
        "model_nl2cypher": nl2cypher_model,
        "fallback_extract_used": fallback_extract_used,
        "fallback_qa_used": fallback_qa_used,
        "fallback_reason": fallback_reason,
        "extracted_entities": len(extracted.get("entities", [])),
        "extracted_relations": len(extracted.get("relations", [])),
        "upserted_entities": stats.entities,
        "upserted_relations": stats.relations,
        "merged_entities": stats.merged_entities,
        "dropped_relations": stats.dropped_relations,
        "json_retries": stats.json_retries,
        "qa_success": qa_success,
        "qa_total": len(questions),
        "qa": qa_results,
    }


def main() -> None:
    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
