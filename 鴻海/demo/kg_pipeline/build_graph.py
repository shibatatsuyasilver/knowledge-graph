"""Build a domain knowledge graph from WHO pages via the LLM extraction API.

Usage:
  python build_graph.py --api-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List

import requests
from bs4 import BeautifulSoup

try:
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover
    GraphDatabase = None


WHO_URLS = [
    "https://www.who.int/news-room/fact-sheets/detail/diabetes",
    "https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)",
    "https://www.who.int/news-room/fact-sheets/detail/hypertension",
]

ALLOWED_NODE_TYPES = {"Disease", "Symptom", "RiskFactor", "Treatment", "SourceDoc"}
ALLOWED_RELATIONS = {"HAS_SYMPTOM", "HAS_RISK", "RECOMMENDED_TREATMENT", "MENTIONED_IN"}


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_url: str
    title: str


@dataclass
class Stats:
    chunks: int = 0
    entities: int = 0
    relations: int = 0


def fetch_clean_text(url: str, timeout: int = 25) -> Dict[str, str]:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    title = (soup.title.string or "WHO Document").strip() if soup.title else "WHO Document"

    blocks: List[str] = []
    for tag in soup.find_all(["h1", "h2", "h3", "p", "li"]):
        text = " ".join(tag.get_text(" ", strip=True).split())
        if len(text) < 30:
            continue
        if "cookie" in text.lower() or "privacy" in text.lower():
            continue
        blocks.append(text)

    merged = "\n".join(dict.fromkeys(blocks))
    merged = re.sub(r"\n{2,}", "\n", merged)
    return {"title": title, "text": merged}


def chunk_text(text: str, source_url: str, title: str, max_chars: int = 900, min_chars: int = 120) -> List[Chunk]:
    pieces = [x.strip() for x in text.split("\n") if x.strip()]
    chunks: List[Chunk] = []
    bucket: List[str] = []
    size = 0

    for piece in pieces:
        if size + len(piece) + 1 > max_chars and bucket:
            joined = "\n".join(bucket).strip()
            if len(joined) >= min_chars:
                chunk_id = hashlib.sha1(f"{source_url}|{joined}".encode("utf-8")).hexdigest()[:16]
                chunks.append(Chunk(chunk_id=chunk_id, text=joined, source_url=source_url, title=title))
            bucket = []
            size = 0

        bucket.append(piece)
        size += len(piece) + 1

    if bucket:
        joined = "\n".join(bucket).strip()
        if len(joined) >= min_chars:
            chunk_id = hashlib.sha1(f"{source_url}|{joined}".encode("utf-8")).hexdigest()[:16]
            chunks.append(Chunk(chunk_id=chunk_id, text=joined, source_url=source_url, title=title))

    return chunks


def call_extract_api(api_url: str, chunk: Chunk) -> Dict:
    payload = {
        "text": chunk.text,
        "source_url": chunk.source_url,
        "chunk_id": chunk.chunk_id,
    }
    response = requests.post(f"{api_url.rstrip('/')}/kg/extract", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def safe_label(value: str) -> str:
    if value not in ALLOWED_NODE_TYPES:
        raise ValueError(f"Disallowed label: {value}")
    return value


def safe_relation(value: str) -> str:
    if value not in ALLOWED_RELATIONS:
        raise ValueError(f"Disallowed relation: {value}")
    return value


def iter_entities(extract_payload: Dict) -> Iterable[Dict]:
    for entity in extract_payload.get("entities", []):
        node_type = entity.get("type")
        name = str(entity.get("name", "")).strip()
        aliases = [str(x).strip() for x in entity.get("aliases", []) if str(x).strip()]
        if node_type in ALLOWED_NODE_TYPES and name:
            yield {"type": node_type, "name": name, "aliases": aliases[:10]}


def iter_relations(extract_payload: Dict) -> Iterable[Dict]:
    for relation in extract_payload.get("relations", []):
        src_t = relation.get("source_type")
        src_n = str(relation.get("source_name", "")).strip()
        rel = relation.get("relation")
        tgt_t = relation.get("target_type")
        tgt_n = str(relation.get("target_name", "")).strip()
        if (
            src_t in ALLOWED_NODE_TYPES
            and tgt_t in ALLOWED_NODE_TYPES
            and rel in ALLOWED_RELATIONS
            and src_n
            and tgt_n
        ):
            yield {
                "source_type": src_t,
                "source_name": src_n,
                "relation": rel,
                "target_type": tgt_t,
                "target_name": tgt_n,
            }


def upsert_chunk_payload(tx, chunk: Chunk, extract_payload: Dict) -> Stats:
    stats = Stats(chunks=1)

    tx.run(
        "MERGE (d:SourceDoc {docId:$doc_id}) "
        "ON CREATE SET d.url=$url, d.title=$title, d.createdAt=timestamp() "
        "SET d.updatedAt=timestamp()",
        {"doc_id": chunk.chunk_id, "url": chunk.source_url, "title": chunk.title},
    )

    for entity in iter_entities(extract_payload):
        label = safe_label(entity["type"])
        tx.run(
            f"MERGE (n:{label} {{name:$name}}) "
            "ON CREATE SET n.createdAt=timestamp() "
            "SET n.updatedAt=timestamp(), n.aliases=$aliases",
            {"name": entity["name"], "aliases": entity["aliases"]},
        )
        tx.run(
            f"MATCH (n:{label} {{name:$name}}), (d:SourceDoc {{docId:$doc_id}}) "
            "MERGE (n)-[:MENTIONED_IN]->(d)",
            {"name": entity["name"], "doc_id": chunk.chunk_id},
        )
        stats.entities += 1

    for relation in iter_relations(extract_payload):
        src_label = safe_label(relation["source_type"])
        tgt_label = safe_label(relation["target_type"])
        rel_name = safe_relation(relation["relation"])

        tx.run(
            f"MATCH (s:{src_label} {{name:$source_name}}), (t:{tgt_label} {{name:$target_name}}) "
            f"MERGE (s)-[r:{rel_name}]->(t) "
            "ON CREATE SET r.createdAt=timestamp() "
            "SET r.updatedAt=timestamp()",
            {
                "source_name": relation["source_name"],
                "target_name": relation["target_name"],
            },
        )
        stats.relations += 1

    return stats


def build_graph(
    api_url: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    max_chunks_per_url: int,
) -> None:
    if GraphDatabase is None:
        raise RuntimeError("neo4j driver is not installed. Run: pip install neo4j")

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    total = Stats()
    try:
        for url in WHO_URLS:
            print(f"[INFO] Fetching: {url}")
            page = fetch_clean_text(url)
            chunks = chunk_text(page["text"], source_url=url, title=page["title"])[:max_chunks_per_url]

            print(f"[INFO] Chunks ready: {len(chunks)}")
            for chunk in chunks:
                payload = call_extract_api(api_url, chunk)
                with driver.session() as session:
                    stats = session.execute_write(upsert_chunk_payload, chunk, payload)
                total.chunks += stats.chunks
                total.entities += stats.entities
                total.relations += stats.relations
                print(
                    f"  - chunk={chunk.chunk_id} entities={stats.entities} relations={stats.relations}"
                )
    finally:
        driver.close()

    print("\n[SUMMARY]")
    print(f"chunks={total.chunks}")
    print(f"entities={total.entities}")
    print(f"relations={total.relations}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build WHO health knowledge graph via LLM extraction API")
    parser.add_argument("--api-url", default=os.getenv("KG_API_URL", "http://localhost:8000"))
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", "neo4j"))
    parser.add_argument("--max-chunks-per-url", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_graph(
        api_url=args.api_url,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        max_chunks_per_url=args.max_chunks_per_url,
    )
