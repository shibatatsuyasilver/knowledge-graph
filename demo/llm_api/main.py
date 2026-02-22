"""FastAPI prototype for LLM + Graph DB knowledge QA.

Endpoints:
- GET /health
- POST /kg/extract
- POST /kg/query
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover - optional dependency for demo
    GraphDatabase = None

# Setup path to include project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.llm_kg import llm_client


APP_TITLE = "KG Chatbot Prototype API"
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")

ALLOWED_NODE_TYPES = {"Disease", "Symptom", "RiskFactor", "Treatment", "SourceDoc"}
ALLOWED_RELATIONS = {"HAS_SYMPTOM", "HAS_RISK", "RECOMMENDED_TREATMENT", "MENTIONED_IN"}
ALLOWED_INTENTS = {"list_symptoms", "list_risks", "list_treatments"}

FORBIDDEN_INPUT = re.compile(r"(?i)\b(create|delete|merge|set|drop|apoc|dbms|load\s+csv)\b")
FORBIDDEN_CYPHER = re.compile(r"(?i)\b(create|delete|merge|set|drop|apoc|dbms|load\s+csv)\b")


class ExtractRequest(BaseModel):
    text: str = Field(min_length=20)
    source_url: Optional[str] = None
    chunk_id: Optional[str] = None


class ExtractEntity(BaseModel):
    type: str
    name: str
    aliases: List[str] = Field(default_factory=list)


class ExtractRelation(BaseModel):
    source_type: str
    source_name: str
    relation: str
    target_type: str
    target_name: str


class ExtractResponse(BaseModel):
    entities: List[ExtractEntity]
    relations: List[ExtractRelation]
    trace: Dict[str, Any]


class QueryRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(default=20, ge=1, le=100)


class QueryResponse(BaseModel):
    answer: str
    cypher: str
    rows: List[Dict[str, Any]]
    trace: Dict[str, Any]


app = FastAPI(title=APP_TITLE)


QUERY_TEMPLATES = {
    "list_symptoms": (
        "MATCH (d:Disease {name:$disease})-[:HAS_SYMPTOM]->(s:Symptom) "
        "RETURN d.name AS disease, collect(s.name)[0..$top_k] AS symptoms"
    ),
    "list_risks": (
        "MATCH (d:Disease {name:$disease})-[:HAS_RISK]->(r:RiskFactor) "
        "RETURN d.name AS disease, collect(r.name)[0..$top_k] AS risks"
    ),
    "list_treatments": (
        "MATCH (d:Disease {name:$disease})-[:RECOMMENDED_TREATMENT]->(t:Treatment) "
        "RETURN d.name AS disease, collect(t.name)[0..$top_k] AS treatments"
    ),
}


def call_llm_json(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    return llm_client.chat_json(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        timeout_seconds=80,
    )


def sanitize_extract_payload(data: Dict[str, Any]) -> Tuple[List[ExtractEntity], List[ExtractRelation]]:
    raw_entities = data.get("entities") or []
    raw_relations = data.get("relations") or []

    entities: List[ExtractEntity] = []
    relations: List[ExtractRelation] = []

    seen_entities = set()
    for item in raw_entities:
        node_type = str(item.get("type", "")).strip()
        name = str(item.get("name", "")).strip()
        aliases = [str(x).strip() for x in (item.get("aliases") or []) if str(x).strip()]
        if node_type not in ALLOWED_NODE_TYPES or not name:
            continue
        key = (node_type, name.lower())
        if key in seen_entities:
            continue
        seen_entities.add(key)
        entities.append(ExtractEntity(type=node_type, name=name, aliases=aliases[:8]))

    seen_rel = set()
    for item in raw_relations:
        src_t = str(item.get("source_type", "")).strip()
        src_n = str(item.get("source_name", "")).strip()
        rel = str(item.get("relation", "")).strip()
        tgt_t = str(item.get("target_type", "")).strip()
        tgt_n = str(item.get("target_name", "")).strip()

        if src_t not in ALLOWED_NODE_TYPES or tgt_t not in ALLOWED_NODE_TYPES:
            continue
        if rel not in ALLOWED_RELATIONS or not src_n or not tgt_n:
            continue

        key = (src_t, src_n.lower(), rel, tgt_t, tgt_n.lower())
        if key in seen_rel:
            continue
        seen_rel.add(key)

        relations.append(
            ExtractRelation(
                source_type=src_t,
                source_name=src_n,
                relation=rel,
                target_type=tgt_t,
                target_name=tgt_n,
            )
        )

    return entities, relations


def fallback_extract(text: str) -> Tuple[List[ExtractEntity], List[ExtractRelation]]:
    entities: List[ExtractEntity] = []
    relations: List[ExtractRelation] = []

    known_diseases = ["Diabetes", "Hypertension", "Obesity"]
    known_symptoms = ["excessive thirst", "frequent urination", "headache", "fatigue"]
    known_risks = ["cardiovascular disease", "smoking", "high blood sugar"]
    known_treatments = ["exercise", "healthy diet", "insulin therapy", "blood pressure control"]

    lower = text.lower()
    for disease in known_diseases:
        if disease.lower() in lower:
            entities.append(ExtractEntity(type="Disease", name=disease))
    for symptom in known_symptoms:
        if symptom.lower() in lower:
            entities.append(ExtractEntity(type="Symptom", name=symptom))
    for risk in known_risks:
        if risk.lower() in lower:
            entities.append(ExtractEntity(type="RiskFactor", name=risk))
    for treatment in known_treatments:
        if treatment.lower() in lower:
            entities.append(ExtractEntity(type="Treatment", name=treatment))

    diseases = [e for e in entities if e.type == "Disease"]
    symptoms = [e for e in entities if e.type == "Symptom"]
    risks = [e for e in entities if e.type == "RiskFactor"]
    treatments = [e for e in entities if e.type == "Treatment"]

    for d in diseases:
        for s in symptoms:
            relations.append(
                ExtractRelation(
                    source_type="Disease",
                    source_name=d.name,
                    relation="HAS_SYMPTOM",
                    target_type="Symptom",
                    target_name=s.name,
                )
            )
        for r in risks:
            relations.append(
                ExtractRelation(
                    source_type="Disease",
                    source_name=d.name,
                    relation="HAS_RISK",
                    target_type="RiskFactor",
                    target_name=r.name,
                )
            )
        for t in treatments:
            relations.append(
                ExtractRelation(
                    source_type="Disease",
                    source_name=d.name,
                    relation="RECOMMENDED_TREATMENT",
                    target_type="Treatment",
                    target_name=t.name,
                )
            )

    return entities, relations


def infer_intent_heuristic(question: str) -> Dict[str, str]:
    q = question.strip()
    ql = q.lower()
    if any(k in ql for k in ["symptom", "症狀"]):
        intent = "list_symptoms"
    elif any(k in ql for k in ["risk", "風險"]):
        intent = "list_risks"
    else:
        intent = "list_treatments"

    disease = "Diabetes"
    for candidate in ["Diabetes", "Hypertension", "Obesity"]:
        if candidate.lower() in ql:
            disease = candidate
            break

    return {"intent": intent, "entityType": "Disease", "entityName": disease}


def validate_cypher(cypher: str) -> None:
    if FORBIDDEN_CYPHER.search(cypher):
        raise ValueError("Unsafe cypher token detected")

    labels = re.findall(r":([A-Za-z][A-Za-z0-9_]*)", cypher)
    for label in labels:
        if label in ALLOWED_RELATIONS:
            continue
        if label not in ALLOWED_NODE_TYPES:
            raise ValueError(f"Unknown label in cypher: {label}")


def run_neo4j(cypher: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    if GraphDatabase is None:
        return []

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    rows: List[Dict[str, Any]] = []
    try:
        with driver.session() as session:
            result = session.run(cypher, params)
            rows = [record.data() for record in result]
    finally:
        driver.close()
    return rows


def answer_from_rows(intent: str, disease: str, rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return f"查無 {disease} 的圖譜資料，請先完成 KG 建置或提供更明確疾病名稱。"

    row = rows[0]
    if intent == "list_symptoms":
        return f"{disease} 常見症狀：{', '.join(row.get('symptoms', [])) or '無資料'}"
    if intent == "list_risks":
        return f"{disease} 相關風險因子：{', '.join(row.get('risks', [])) or '無資料'}"
    return f"{disease} 常見建議處置：{', '.join(row.get('treatments', [])) or '無資料'}"


@app.get("/health")
def health() -> Dict[str, Any]:
    upstream = llm_client.health_check(timeout_seconds=5)
    neo4j_ok = False

    if GraphDatabase is not None:
        try:
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            with driver.session() as session:
                _ = session.run("RETURN 1 AS ok").single()
            neo4j_ok = True
        except Exception:
            neo4j_ok = False
        finally:
            try:
                driver.close()
            except Exception:
                pass

    return {
        "status": "ok",
        "service": APP_TITLE,
        "provider": upstream["provider"],
        "model": upstream["model"],
        "upstream": {"type": upstream["upstream"], "status": upstream["status"], "reachable": upstream["reachable"]},
        # Legacy field for backward compatibility.
        "ollama": upstream["reachable"] if upstream["provider"] == "ollama" else "legacy-n/a",
        "neo4j": neo4j_ok,
    }


@app.post("/kg/extract", response_model=ExtractResponse)
def kg_extract(req: ExtractRequest) -> ExtractResponse:
    system_prompt = (
        "Extract medical entities and relations. "
        "Return strict JSON only with keys: entities, relations."
    )
    user_prompt = (
        "Text:\n"
        f"{req.text}\n\n"
        "Schema:\n"
        "entities: [{type,name,aliases[]}]\n"
        "relations: [{source_type,source_name,relation,target_type,target_name}]\n"
        "Allowed types: Disease, Symptom, RiskFactor, Treatment, SourceDoc\n"
        "Allowed relations: HAS_SYMPTOM, HAS_RISK, RECOMMENDED_TREATMENT, MENTIONED_IN"
    )

    method = llm_client.get_runtime_config().provider
    try:
        raw = call_llm_json(system_prompt, user_prompt)
        entities, relations = sanitize_extract_payload(raw)
    except Exception:
        method = "fallback"
        entities, relations = fallback_extract(req.text)

    return ExtractResponse(
        entities=entities,
        relations=relations,
        trace={
            "method": method,
            "model": llm_client.get_runtime_config().model,
            "entityCount": len(entities),
            "relationCount": len(relations),
            "chunkId": req.chunk_id,
            "sourceUrl": req.source_url,
        },
    )


@app.post("/kg/query", response_model=QueryResponse)
def kg_query(req: QueryRequest) -> QueryResponse:
    if FORBIDDEN_INPUT.search(req.question):
        raise HTTPException(status_code=400, detail="Blocked: suspicious query pattern")

    prompt = (
        "Convert user question into strict JSON with keys: intent, entityType, entityName. "
        "Allowed intent: list_symptoms, list_risks, list_treatments. "
        "entityType must be Disease."
        f"\nQuestion: {req.question}"
    )

    source = llm_client.get_runtime_config().provider
    try:
        parsed = call_llm_json("You are a strict JSON planner.", prompt)
        intent = str(parsed.get("intent", "")).strip()
        entity_type = str(parsed.get("entityType", "")).strip()
        entity_name = str(parsed.get("entityName", "")).strip()
    except Exception:
        source = "heuristic"
        parsed = infer_intent_heuristic(req.question)
        intent = parsed["intent"]
        entity_type = parsed["entityType"]
        entity_name = parsed["entityName"]

    if intent not in ALLOWED_INTENTS:
        raise HTTPException(status_code=400, detail=f"Unsupported intent: {intent}")
    if entity_type != "Disease" or not entity_name:
        raise HTTPException(status_code=400, detail="Only Disease entityType is supported")

    cypher = QUERY_TEMPLATES[intent]
    try:
        validate_cypher(cypher)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err

    params = {"disease": entity_name, "top_k": req.top_k}

    try:
        rows = run_neo4j(cypher, params)
    except Exception as err:
        rows = []
        source = f"{source}+neo4j_error"
        neo_err = str(err)
    else:
        neo_err = None

    answer = answer_from_rows(intent, entity_name, rows)

    trace = {
        "planner": source,
        "intent": intent,
        "entityType": entity_type,
        "entityName": entity_name,
        "model": llm_client.get_runtime_config().model,
        "topK": req.top_k,
    }
    if neo_err:
        trace["neo4jError"] = neo_err

    return QueryResponse(answer=answer, cypher=cypher, rows=rows, trace=trace)
