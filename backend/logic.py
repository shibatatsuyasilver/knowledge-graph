import hashlib
import json
import os
import re
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional
from urllib.parse import parse_qs, quote, unquote, urlparse, urljoin

import requests
from bs4 import BeautifulSoup

try:
    from duckduckgo_search import DDGS
except Exception:  # pragma: no cover - dependency can be optional in test env
    DDGS = None

# Ensure project root is in sys.path to allow imports from genai_project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from genai_project.llm_kg import llm_client

GENERAL_CHAT_TIMEOUT_SECONDS = float(os.getenv("GENERAL_CHAT_TIMEOUT_SECONDS", os.getenv("LLM_TIMEOUT_SECONDS", "120")))
GENERAL_CHAT_NUM_PREDICT = int(os.getenv("GENERAL_CHAT_NUM_PREDICT", os.getenv("LLM_MAX_TOKENS", "512")))
GENERAL_CHAT_TEMPERATURE = float(os.getenv("GENERAL_CHAT_TEMPERATURE", os.getenv("LLM_TEMPERATURE", "0.7")))
KG_QA_MAX_ROWS_FOR_PROMPT = int(os.getenv("KG_QA_MAX_ROWS_FOR_PROMPT", "20"))
KEYWORD_SEARCH_MODE = os.getenv("KEYWORD_SEARCH_MODE", "html_only").strip().lower()
DEFAULT_CHUNK_LIMIT = int(os.getenv("INGEST_CHUNK_LIMIT", "0"))
CHUNK_SIZE_MODE = os.getenv("CHUNK_SIZE_MODE", "provider").strip().lower()
TOKEN_ESTIMATE_PATTERN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]", flags=re.UNICODE)
DEFAULT_TOKEN_CHUNK_SIZE = int(
    os.getenv(
        "CHUNK_SIZE_TOKENS",
        os.getenv("GEMINI_INPUT_TOKEN_LIMIT", "1048576"),
    )
)
DEFAULT_TOKEN_CHUNK_MIN_SIZE = int(os.getenv("CHUNK_MIN_TOKENS", "120"))
DEFAULT_CHAR_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE_CHARS", "900"))
DEFAULT_CHAR_CHUNK_MIN_SIZE = int(os.getenv("CHUNK_MIN_CHARS", "120"))

@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_url: str
    title: str
    tokens: int = 0


def _estimate_token_count(text: str) -> int:
    if not text:
        return 0
    return len(TOKEN_ESTIMATE_PATTERN.findall(text))


def _build_chunk(text: str, source_url: str, title: str) -> Optional[Chunk]:
    normalized = text.strip()
    if not normalized:
        return None
    chunk_id = hashlib.sha1(f"{source_url}|{normalized}".encode("utf-8")).hexdigest()[:16]
    return Chunk(
        chunk_id=chunk_id,
        text=normalized,
        source_url=source_url,
        title=title,
        tokens=_estimate_token_count(normalized),
    )


def _split_text_by_token_limit(text: str, max_tokens: int) -> List[str]:
    if max_tokens <= 0:
        return [text]

    matches = list(TOKEN_ESTIMATE_PATTERN.finditer(text))
    if len(matches) <= max_tokens:
        return [text]

    parts: List[str] = []
    start = 0
    while start < len(matches):
        end = min(start + max_tokens, len(matches))
        start_pos = matches[start].start()
        end_pos = matches[end - 1].end()
        segment = text[start_pos:end_pos].strip()
        if segment:
            parts.append(segment)
        start = end
    return parts


def _use_token_chunking(extraction_provider: Optional[str]) -> bool:
    mode = CHUNK_SIZE_MODE
    if mode == "token":
        return True
    if mode == "char":
        return False
    return (extraction_provider or "").strip().lower() == "gemini"


def _chunk_text_by_char_budget(
    *,
    text: str,
    source_url: str,
    title: str,
    max_chars: int,
    min_chars: int,
) -> List[Chunk]:
    pieces = [x.strip() for x in text.split("\n") if x.strip()]
    chunks: List[Chunk] = []
    bucket: List[str] = []
    size = 0

    for piece in pieces:
        if size + len(piece) + 1 > max_chars and bucket:
            joined = "\n".join(bucket).strip()
            if len(joined) >= min_chars:
                chunk = _build_chunk(joined, source_url, title)
                if chunk:
                    chunks.append(chunk)
            bucket = []
            size = 0

        bucket.append(piece)
        size += len(piece) + 1

    if bucket:
        joined = "\n".join(bucket).strip()
        if len(joined) >= min_chars:
            chunk = _build_chunk(joined, source_url, title)
            if chunk:
                chunks.append(chunk)
    return chunks


def _chunk_text_by_token_budget(
    *,
    text: str,
    source_url: str,
    title: str,
    max_tokens: int,
    min_tokens: int,
) -> List[Chunk]:
    pieces = [x.strip() for x in text.split("\n") if x.strip()]
    chunks: List[Chunk] = []
    bucket: List[str] = []
    token_count = 0

    def flush_bucket() -> None:
        nonlocal bucket, token_count
        if not bucket:
            return
        joined = "\n".join(bucket).strip()
        if _estimate_token_count(joined) >= min_tokens:
            chunk = _build_chunk(joined, source_url, title)
            if chunk:
                chunks.append(chunk)
        bucket = []
        token_count = 0

    for piece in pieces:
        piece_tokens = _estimate_token_count(piece)
        if piece_tokens > max_tokens:
            flush_bucket()
            for segment in _split_text_by_token_limit(piece, max_tokens):
                if _estimate_token_count(segment) < min_tokens:
                    continue
                chunk = _build_chunk(segment, source_url, title)
                if chunk:
                    chunks.append(chunk)
            continue

        if token_count + piece_tokens > max_tokens and bucket:
            flush_bucket()

        bucket.append(piece)
        token_count += piece_tokens

    flush_bucket()
    return chunks


def _normalize_http_url(value: str) -> str:
    raw = value.strip()
    if not raw:
        raise ValueError("URL cannot be empty")

    parsed = urlparse(raw)
    if parsed.scheme == "":
        raw = f"https://{raw}"
        parsed = urlparse(raw)

    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid URL: {value}")
    return parsed.geturl()


def _load_kg_modules() -> tuple[Any, Any]:
    try:
        from genai_project.llm_kg.kg_builder import KnowledgeGraphBuilder
        from genai_project.llm_kg.nl2cypher import answer_with_manual_prompt
    except ImportError as exc:
        raise RuntimeError(
            "Unable to load KG modules. Install backend dependencies before calling KG endpoints."
        ) from exc
    return KnowledgeGraphBuilder, answer_with_manual_prompt


def fetch_clean_text(url: str, timeout: int = 25) -> Dict[str, str]:
    """Fetches and cleans text from a URL."""
    try:
        normalized_url = _normalize_http_url(url)
        response = requests.get(
            normalized_url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; GenAI-KG-Bot/1.0)"},
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        title = (soup.title.string or "Document").strip() if soup.title else "Document"

        blocks: List[str] = []
        # Extract text from common content tags
        for tag in soup.find_all(["h1", "h2", "h3", "p", "li", "article", "section"]):
            text = " ".join(tag.get_text(" ", strip=True).split())
            if len(text) < 30:
                continue
            if "cookie" in text.lower() or "privacy" in text.lower():
                continue
            blocks.append(text)

        merged = "\n".join(dict.fromkeys(blocks))  # Remove duplicates while preserving order
        merged = re.sub(r"\n{2,}", "\n", merged)
        return {"title": title, "text": merged, "url": normalized_url}
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return {"title": "Error", "text": "", "url": url}


def chunk_text(
    text: str,
    source_url: str,
    title: str,
    max_chars: int = DEFAULT_CHAR_CHUNK_SIZE,
    min_chars: int = DEFAULT_CHAR_CHUNK_MIN_SIZE,
    *,
    extraction_provider: Optional[str] = None,
    max_tokens: int = DEFAULT_TOKEN_CHUNK_SIZE,
    min_tokens: int = DEFAULT_TOKEN_CHUNK_MIN_SIZE,
) -> List[Chunk]:
    """Splits text into chunks with provider-aware sizing.

    - Gemini: token-based chunking by default (CHUNK_SIZE_MODE=provider|token)
    - Others: char-based chunking by default
    """
    if not text:
        return []

    if _use_token_chunking(extraction_provider):
        return _chunk_text_by_token_budget(
            text=text,
            source_url=source_url,
            title=title,
            max_tokens=max(1, int(max_tokens)),
            min_tokens=max(1, int(min_tokens)),
        )

    return _chunk_text_by_char_budget(
        text=text,
        source_url=source_url,
        title=title,
        max_chars=max(1, int(max_chars)),
        min_chars=max(1, int(min_chars)),
    )


def process_url(url: str, extraction_provider: Optional[str] = None) -> List[Chunk]:
    """Fetches content from URL and returns chunks."""
    data = fetch_clean_text(url)
    if not data["text"]:
        return []
    return chunk_text(
        data["text"],
        data["url"],
        data["title"],
        extraction_provider=extraction_provider,
    )


def _resolve_chunk_limit(chunk_limit: Optional[int]) -> Optional[int]:
    if chunk_limit is None:
        chunk_limit = DEFAULT_CHUNK_LIMIT
    if chunk_limit is None:
        return None

    value = int(chunk_limit)
    if value <= 0:
        return None
    if value > 200:
        raise ValueError("chunk_limit must be between 1 and 200")
    return value


def build_kg_from_chunks(
    chunks: List[Chunk],
    uri: str,
    user: str,
    pwd: str,
    chunk_limit: Optional[int] = None,
    extraction_provider: Optional[str] = None,
    extraction_model: Optional[str] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Builds Knowledge Graph from text chunks using KnowledgeGraphBuilder.
    """
    KnowledgeGraphBuilder, _ = _load_kg_modules()
    builder = KnowledgeGraphBuilder(uri, user, pwd)
    resolved_chunk_limit = _resolve_chunk_limit(chunk_limit)
    chunks_to_process = chunks[:resolved_chunk_limit] if resolved_chunk_limit else chunks
    chunk_progress: List[Dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        chunk_progress.append(
            {
                "order": index + 1,
                "chunk_id": chunk.chunk_id,
                "source_url": chunk.source_url,
                "title": chunk.title,
                "chars": len(chunk.text),
                "tokens": int(chunk.tokens or _estimate_token_count(chunk.text)),
                "status": "queued"
                if (not resolved_chunk_limit or index < resolved_chunk_limit)
                else "skipped_by_limit",
                "entities": 0,
                "relations": 0,
            }
        )
    total_stats = {
        "chunks_processed": 0,
        "entities": 0,
        "relations": 0,
        "merged_entities": 0,
        "dropped_relations": 0,
        "json_retries": 0
    }
    
    extracted_data_summary = []

    def _emit_chunk_progress(index: int, stats_delta: Optional[Dict[str, int]] = None) -> None:
        if not progress_callback:
            return
        payload = {
            "type": "chunk_update",
            "chunk": dict(chunk_progress[index]),
            "stats": dict(total_stats),
            "stats_delta": dict(stats_delta or {}),
            "chunks_available": len(chunks),
            "chunk_limit": resolved_chunk_limit,
        }
        try:
            progress_callback(payload)
        except Exception:
            # Progress reporting must not break main processing.
            pass

    try:
        for index, chunk in enumerate(chunks_to_process):
            print(f"Processing chunk {chunk.chunk_id}...")
            chunk_progress[index]["status"] = "processing"
            _emit_chunk_progress(index)
            # 1. Extract
            try:
                extracted = builder.extract_entities_relations(
                    chunk.text,
                    provider=extraction_provider,
                    model=extraction_model,
                )
                entity_count = len(extracted.get("entities", []))
                relation_count = len(extracted.get("relations", []))

                # Keep a summary for display
                if entity_count or relation_count:
                    extracted_data_summary.append({
                        "chunk_id": chunk.chunk_id,
                        "entities": entity_count,
                        "relations": relation_count,
                    })
                chunk_progress[index]["entities"] = entity_count
                chunk_progress[index]["relations"] = relation_count

                # 2. Populate
                stats = builder.populate_graph(extracted)
            except Exception as exc:
                chunk_progress[index]["status"] = "failed"
                chunk_progress[index]["error"] = str(exc)
                _emit_chunk_progress(index)
                raise

            stats_delta = {
                "chunks_processed": 1,
                "entities": int(stats.entities),
                "relations": int(stats.relations),
                "merged_entities": int(stats.merged_entities),
                "dropped_relations": int(stats.dropped_relations),
                "json_retries": int(stats.json_retries),
            }

            # Aggregate stats
            total_stats["chunks_processed"] += stats_delta["chunks_processed"]
            total_stats["entities"] += stats_delta["entities"]
            total_stats["relations"] += stats_delta["relations"]
            total_stats["merged_entities"] += stats_delta["merged_entities"]
            total_stats["dropped_relations"] += stats_delta["dropped_relations"]
            total_stats["json_retries"] += stats_delta["json_retries"]
            chunk_progress[index]["status"] = "processed"
            _emit_chunk_progress(index, stats_delta=stats_delta)

    except Exception as exc:
        print(f"Error building KG: {exc}")
        raise
    finally:
        builder.close()

    return {
        "stats": total_stats,
        "summary": extracted_data_summary,
        "chunk_limit": resolved_chunk_limit,
        "chunks_available": len(chunks),
        "chunk_progress": chunk_progress,
    }


def build_kg_from_text_content(
    text: str,
    uri: str,
    user: str,
    pwd: str,
    chunk_limit: Optional[int] = None,
    extraction_provider: Optional[str] = None,
    extraction_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Builds KG from raw text content (not URL).
    """
    # Treat as a single chunk or split if too large?
    # For simplicity, let's chunk it as if it came from a "User Input" source
    chunks = chunk_text(
        text,
        "user_input",
        "User Input",
        extraction_provider=extraction_provider,
    )
    return build_kg_from_chunks(
        chunks,
        uri,
        user,
        pwd,
        chunk_limit=chunk_limit,
        extraction_provider=extraction_provider,
        extraction_model=extraction_model,
    )


def process_text_to_kg(
    text: str,
    uri: str,
    user: str,
    pwd: str,
    *,
    chunk_limit: Optional[int] = None,
    extraction_provider: Optional[str] = None,
    extraction_model: Optional[str] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    chunks = chunk_text(
        text.strip(),
        "user_input",
        "User Input",
        extraction_provider=extraction_provider,
    )
    if not chunks:
        raise ValueError("No content found")
    return build_kg_from_chunks(
        chunks,
        uri,
        user,
        pwd,
        chunk_limit=chunk_limit,
        extraction_provider=extraction_provider,
        extraction_model=extraction_model,
        progress_callback=progress_callback,
    )


def process_url_to_kg(
    url: str,
    uri: str,
    user: str,
    pwd: str,
    *,
    chunk_limit: Optional[int] = None,
    extraction_provider: Optional[str] = None,
    extraction_model: Optional[str] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    chunks = process_url(url, extraction_provider=extraction_provider)
    if not chunks:
        raise ValueError("No content found or empty content")
    return build_kg_from_chunks(
        chunks,
        uri,
        user,
        pwd,
        chunk_limit=chunk_limit,
        extraction_provider=extraction_provider,
        extraction_model=extraction_model,
        progress_callback=progress_callback,
    )


def _stringify_query_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        if not value:
            return "[]"
        items = [_stringify_query_value(item) for item in value[:4]]
        suffix = "..." if len(value) > 4 else ""
        return "、".join(items) + suffix
    if isinstance(value, Mapping) or (hasattr(value, "get") and hasattr(value, "keys")):
        mapping: Dict[str, Any] = {}
        if isinstance(value, Mapping):
            mapping = dict(value)
        else:
            try:
                mapping = {key: value.get(key) for key in value.keys()}
            except Exception:
                props = getattr(value, "_properties", None)
                if isinstance(props, dict):
                    mapping = dict(props)

        if isinstance(mapping.get("name"), str) and mapping.get("name"):
            return str(mapping["name"])
        parts = []
        for key, inner in list(mapping.items())[:3]:
            parts.append(f"{key}:{_stringify_query_value(inner)}")
        if not parts:
            return "{}"
        suffix = "..." if len(mapping) > 3 else ""
        return "{" + ", ".join(parts) + suffix + "}"
    return str(value)


def _display_query_key(key: str) -> str:
    text = str(key)
    if "." in text:
        return text.split(".", 1)[1]
    return text


def _summarize_query_rows(question: str, rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return f"目前在知識圖譜中找不到與「{question}」直接相關的資料。"

    first_row_keys = list(rows[0].keys())
    # If rows contain explicit name columns, summarize by names first.
    name_keys = [key for key in first_row_keys if _display_query_key(key).lower() == "name"]
    if name_keys:
        key = name_keys[0]
        values: List[str] = []
        for row in rows:
            if key not in row:
                continue
            value = _stringify_query_value(row[key]).strip()
            if value and value not in {"N/A", "[]", "{}"}:
                values.append(value)
        unique_values = list(dict.fromkeys(values))
        if unique_values:
            top_values = unique_values[:8]
            suffix = f" 等 {len(unique_values)} 筆" if len(unique_values) > 8 else ""
            if len(top_values) == 1:
                return f"{question}：{top_values[0]}。"
            return f"{question}：{'、'.join(top_values)}{suffix}。"

    if len(first_row_keys) == 1:
        key = first_row_keys[0]
        values: List[str] = []
        for row in rows:
            if key not in row:
                continue
            value = _stringify_query_value(row[key])
            if value and value not in {"N/A", "[]", "{}"}:
                values.append(value)

        unique_values = list(dict.fromkeys(values))
        if unique_values:
            top_values = unique_values[:8]
            suffix = f" 等 {len(unique_values)} 筆" if len(unique_values) > 8 else ""
            if len(top_values) == 1:
                return f"{question}：{top_values[0]}。"
            return f"{question}包含：{'、'.join(top_values)}{suffix}。"

    highlights: List[str] = []
    for row in rows[:5]:
        cells = [f"{_display_query_key(key)}：{_stringify_query_value(value)}" for key, value in row.items()]
        if cells:
            highlights.append("；".join(cells))

    if not highlights:
        return f"查到 {len(rows)} 筆資料，但欄位內容較空，請展開 Rows 查看。"

    summary = "；".join(highlights)
    if len(rows) > 5:
        return f"我查到 {len(rows)} 筆資料，先列出前 5 筆重點：{summary}。"
    return f"我查到 {len(rows)} 筆資料：{summary}。"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _kg_qa_use_llm() -> bool:
    return _env_bool("KG_QA_USE_LLM", True)


def _kg_qa_model() -> Optional[str]:
    value = os.getenv("KG_QA_MODEL", "").strip()
    return value or None


def _kg_qa_temperature() -> float:
    try:
        return float(os.getenv("KG_QA_TEMPERATURE", "0.1"))
    except ValueError:
        return 0.1


def _kg_qa_max_tokens() -> int:
    try:
        return max(1, int(os.getenv("KG_QA_MAX_TOKENS", "1024")))
    except ValueError:
        return 1024


def _format_rows_for_qa_prompt(rows: List[Dict[str, Any]]) -> str:
    limited = rows[: max(1, KG_QA_MAX_ROWS_FOR_PROMPT)]
    try:
        return json.dumps(limited, ensure_ascii=False, indent=2)
    except Exception:
        return str(limited)


def _generate_kg_answer_with_llm(
    *,
    question: str,
    cypher: str,
    rows: List[Dict[str, Any]],
) -> str:
    if not _kg_qa_use_llm():
        raise RuntimeError("KG_QA_USE_LLM disabled")

    system_prompt = (
        "你是企業知識圖譜問答助理，請使用繁體中文、直接回答結論。"
        "禁止使用「根據知識圖譜」這種前綴。"
        "若有多筆結果，請簡潔列出。"
        "若無資料，明確說明缺少哪一類資料（例如缺少財報季度、營收、營益率節點）。"
        "不要捏造未出現在 rows 的事實。"
        "只輸出最終回答，不要輸出步驟、編號、思考過程或提示詞內容。"
    )
    user_prompt = (
        f"問題：{question}\n"
        f"Cypher：\n{cypher}\n\n"
        f"Rows(JSON)：\n{_format_rows_for_qa_prompt(rows)}\n\n"
        "請產生最終回答。"
    )
    answer = llm_client.chat_text(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=_kg_qa_model(),
        temperature=_kg_qa_temperature(),
        max_tokens=_kg_qa_max_tokens(),
        timeout_seconds=GENERAL_CHAT_TIMEOUT_SECONDS,
    )
    normalized = answer.strip()
    if not normalized:
        raise RuntimeError("KG QA LLM returned empty answer")
    if len(normalized) < 8:
        raise RuntimeError("KG QA LLM returned too-short answer")
    if len(rows) > 1 and len(normalized) < 14:
        raise RuntimeError("KG QA LLM returned too-short answer")
    lower = normalized.lower()
    if "final output" in lower or "step " in lower or "步驟" in normalized:
        raise RuntimeError("KG QA LLM returned instruction-like output")
    if not re.search(r"[\u4e00-\u9fff]", normalized):
        raise RuntimeError("KG QA LLM answer missing Traditional Chinese output")
    if normalized.rstrip().endswith((":", "：", "-", "*")):
        raise RuntimeError("KG QA LLM answer looks truncated")
    sample_values: List[str] = []
    for row in rows[:3]:
        for value in row.values():
            text = _stringify_query_value(value).strip()
            if text and text not in {"N/A", "[]", "{}"}:
                sample_values.append(text)
    sample_values = list(dict.fromkeys(sample_values))[:6]
    if sample_values and not any(value in normalized for value in sample_values):
        raise RuntimeError("KG QA LLM answer does not reference row values")
    return normalized


def query_kg(question: str) -> Dict[str, Any]:
    """
    Queries the Knowledge Graph using natural language.
    """
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("Question cannot be empty")

    _, answer_with_manual_prompt = _load_kg_modules()
    raw_result = answer_with_manual_prompt(cleaned)
    if not isinstance(raw_result, dict):
        raise RuntimeError("Unexpected KG query result format")

    rows_value = raw_result.get("rows")
    if not isinstance(rows_value, list):
        rows_value = []

    normalized_rows: List[Dict[str, Any]] = []
    for row in rows_value:
        if isinstance(row, dict):
            normalized_rows.append(row)
        else:
            normalized_rows.append({"value": row})

    result = dict(raw_result)
    result["question"] = str(result.get("question") or cleaned)
    result["rows"] = normalized_rows
    try:
        answer = _generate_kg_answer_with_llm(
            question=cleaned,
            cypher=str(result.get("cypher") or ""),
            rows=normalized_rows,
        )
        result["answer"] = answer
        result["answer_source"] = "qa_llm"
    except Exception:
        result["answer"] = _summarize_query_rows(cleaned, normalized_rows)
        result["answer_source"] = "template_fallback"
    return result


def _to_domain_allowlist(values: Optional[Iterable[str]]) -> List[str]:
    if not values:
        return []

    domains: List[str] = []
    for value in values:
        normalized = value.strip().lower()
        if not normalized:
            continue
        if normalized.startswith("http://") or normalized.startswith("https://"):
            parsed = urlparse(normalized)
            normalized = parsed.netloc.lower()
        normalized = normalized.lstrip(".")
        if normalized:
            domains.append(normalized)
    return sorted(set(domains))


def _is_allowed_domain(url: str, allowlist: List[str]) -> bool:
    if not allowlist:
        return True

    host = (urlparse(url).netloc or "").lower()
    for domain in allowlist:
        if host == domain or host.endswith(f".{domain}"):
            return True
    return False


def _unwrap_duckduckgo_redirect_url(candidate: str) -> str:
    absolute = urljoin("https://duckduckgo.com", candidate)
    parsed = urlparse(absolute)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        redirect_url = parse_qs(parsed.query).get("uddg", [None])[0]
        if redirect_url:
            return unquote(redirect_url)
    return absolute


def _search_keyword_urls_via_html(keyword: str, max_results: int, language: str) -> List[str]:
    region_map = {"zh-tw": "tw-tzh", "en": "us-en"}
    params = {"q": keyword}
    if language in region_map:
        params["kl"] = region_map[language]

    response = requests.get(
        "https://html.duckduckgo.com/html/",
        params=params,
        timeout=20,
        headers={"User-Agent": "Mozilla/5.0 (compatible; GenAI-KG-Bot/1.0)"},
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    urls: List[str] = []
    seen = set()

    for selector in ("a.result__a", "a[data-testid='result-title-a']", "a.result-link"):
        for anchor in soup.select(selector):
            candidate = str(anchor.get("href") or "").strip()
            if not candidate:
                continue
            candidate = _unwrap_duckduckgo_redirect_url(candidate)
            try:
                normalized = _normalize_http_url(candidate)
            except ValueError:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            urls.append(normalized)
            if len(urls) >= max_results:
                return urls
    return urls


def _search_keyword_urls_via_wikipedia_api(keyword: str, max_results: int, language: str) -> List[str]:
    lang_code = "zh" if language == "zh-tw" else "en"
    api_url = f"https://{lang_code}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": keyword,
        "srlimit": max_results,
        "format": "json",
        "utf8": 1,
    }

    try:
        response = requests.get(
            api_url,
            params=params,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0 (compatible; GenAI-KG-Bot/1.0)"},
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []

    urls: List[str] = []
    seen = set()
    for item in payload.get("query", {}).get("search", []):
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        encoded = quote(title.replace(" ", "_"), safe=":_()")
        url = f"https://{lang_code}.wikipedia.org/wiki/{encoded}"
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)
        if len(urls) >= max_results:
            break
    return urls


def _search_keyword_urls(keyword: str, max_results: int, language: str) -> List[str]:
    region_map = {"zh-tw": "tw-tzh", "en": "us-en"}
    preferred_region = region_map.get(language)
    candidate_regions = [preferred_region, None, "wt-wt"]
    regions = [value for idx, value in enumerate(candidate_regions) if value not in candidate_regions[:idx]]

    # In containerized environments DDGS can block for a long time.
    # Prefer deterministic HTML search first, then fallback to DDGS when needed.
    if KEYWORD_SEARCH_MODE in {"html_first", "html_only"}:
        html_urls = _search_keyword_urls_via_html(keyword, max_results=max_results, language=language)
        if html_urls:
            return html_urls

    if DDGS is not None and KEYWORD_SEARCH_MODE != "html_only":
        for backend in ("html", "lite", "auto"):
            for region in regions:
                urls: List[str] = []
                seen = set()
                try:
                    with DDGS() as ddgs:
                        # ddgs.text returns iterator-like search results.
                        results = ddgs.text(
                            keyword,
                            region=region,
                            safesearch="moderate",
                            backend=backend,
                            max_results=max_results,
                        )
                        for item in results:
                            candidate = str(item.get("href") or item.get("url") or item.get("link") or "").strip()
                            if not candidate:
                                continue
                            try:
                                normalized = _normalize_http_url(candidate)
                            except ValueError:
                                continue
                            if normalized in seen:
                                continue
                            seen.add(normalized)
                            urls.append(normalized)
                            if len(urls) >= max_results:
                                break
                except Exception:
                    continue
                if urls:
                    return urls

    html_urls = _search_keyword_urls_via_html(keyword, max_results=max_results, language=language)
    if html_urls:
        return html_urls

    # Final fallback for anti-bot/search API instability: Wikipedia search API.
    return _search_keyword_urls_via_wikipedia_api(keyword, max_results=max_results, language=language)


def _search_keyword_urls_resilient(keyword: str, max_results: int, language: str) -> List[str]:
    search_terms = [keyword]
    max_attempts = 3
    if KEYWORD_SEARCH_MODE == "html_only":
        # Keep html_only mode deterministic and fast.
        max_attempts = 1
    elif language == "zh-tw":
        search_terms.extend([f"{keyword} 新聞", f"{keyword} 官網"])
    else:
        search_terms.extend([f"{keyword} news", f"{keyword} official website"])

    urls: List[str] = []
    seen = set()

    # Search engines can occasionally return empty results; retry a few short attempts.
    for attempt in range(max_attempts):
        for term in search_terms:
            try:
                candidates = _search_keyword_urls(term, max_results=max_results, language=language)
            except Exception:
                continue

            for candidate in candidates:
                if candidate in seen:
                    continue
                seen.add(candidate)
                urls.append(candidate)
                if len(urls) >= max_results:
                    return urls

        if urls:
            return urls
        if attempt < max_attempts - 1:
            time.sleep(1.0)

    return urls


def process_keyword_to_kg(
    keyword: str,
    uri: str,
    user: str,
    pwd: str,
    max_results: int = 5,
    language: str = "zh-tw",
    site_allowlist: Optional[Iterable[str]] = None,
    chunk_limit: Optional[int] = None,
    extraction_provider: Optional[str] = None,
    extraction_model: Optional[str] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    cleaned_keyword = keyword.strip()
    if len(cleaned_keyword) < 2:
        raise ValueError("Keyword must contain at least 2 characters")
    if max_results < 1 or max_results > 10:
        raise ValueError("max_results must be between 1 and 10")
    resolved_chunk_limit = _resolve_chunk_limit(chunk_limit)

    searched_urls = _search_keyword_urls_resilient(cleaned_keyword, max_results=max_results, language=language)
    allowlist = _to_domain_allowlist(site_allowlist)
    target_urls = [url for url in searched_urls if _is_allowed_domain(url, allowlist)]

    if not target_urls:
        raise ValueError("No results found for the keyword")

    aggregated_stats = {
        "chunks_processed": 0,
        "entities": 0,
        "relations": 0,
        "merged_entities": 0,
        "dropped_relations": 0,
        "json_retries": 0,
    }
    aggregated_summary: List[Dict[str, Any]] = []
    fetched_urls: List[str] = []
    failed_urls: List[Dict[str, str]] = []
    aggregated_chunks_available = 0
    aggregated_chunk_progress: List[Dict[str, Any]] = []

    def _emit_keyword_progress(*, status: str = "running", current_url: Optional[str] = None, error: Optional[str] = None) -> None:
        if not progress_callback:
            return
        payload: Dict[str, Any] = {
            "status": status,
            "searched_keyword": cleaned_keyword,
            "fetched_urls": list(fetched_urls),
            "failed_urls": [dict(item) for item in failed_urls],
            "stats": dict(aggregated_stats),
            "summary": [dict(item) for item in aggregated_summary],
            "chunk_limit": resolved_chunk_limit,
            "chunks_available": aggregated_chunks_available,
            "chunk_progress": [dict(item) for item in aggregated_chunk_progress],
        }
        if current_url:
            payload["current_url"] = current_url
        if error:
            payload["error"] = error
        try:
            progress_callback(payload)
        except Exception:
            # Progress reporting should never crash request execution.
            pass

    _emit_keyword_progress(status="running")

    for url in target_urls:
        row_idx_by_chunk_id: Dict[str, int] = {}
        try:
            _emit_keyword_progress(status="running", current_url=url)
            chunks = process_url(url, extraction_provider=extraction_provider)
            if not chunks:
                raise ValueError("No extractable content")

            processable_count = resolved_chunk_limit or len(chunks)
            for idx, chunk in enumerate(chunks):
                aggregated_chunk_progress.append(
                    {
                        "order": len(aggregated_chunk_progress) + 1,
                        "chunk_id": chunk.chunk_id,
                        "source_url": chunk.source_url,
                        "title": chunk.title,
                        "chars": len(chunk.text),
                        "tokens": int(chunk.tokens or _estimate_token_count(chunk.text)),
                        "status": "queued" if idx < processable_count else "skipped_by_limit",
                        "entities": 0,
                        "relations": 0,
                    }
                )
                row_idx_by_chunk_id[chunk.chunk_id] = len(aggregated_chunk_progress) - 1
            aggregated_chunks_available += len(chunks)
            _emit_keyword_progress(status="running", current_url=url)

            def _on_build_chunk_progress(event: Dict[str, Any]) -> None:
                if event.get("type") != "chunk_update":
                    return
                chunk_row = event.get("chunk")
                if not isinstance(chunk_row, dict):
                    return
                chunk_id = str(chunk_row.get("chunk_id", ""))
                row_idx = row_idx_by_chunk_id.get(chunk_id)
                if row_idx is None:
                    return
                row = aggregated_chunk_progress[row_idx]
                row["status"] = str(chunk_row.get("status", row["status"]))
                row["entities"] = int(chunk_row.get("entities", row["entities"]))
                row["relations"] = int(chunk_row.get("relations", row["relations"]))
                if chunk_row.get("error"):
                    row["error"] = str(chunk_row["error"])
                _emit_keyword_progress(status="running", current_url=url)

            build_kwargs: Dict[str, Any] = {
                "chunk_limit": resolved_chunk_limit,
                "extraction_provider": extraction_provider,
                "extraction_model": extraction_model,
            }
            if progress_callback:
                build_kwargs["progress_callback"] = _on_build_chunk_progress

            result = build_kg_from_chunks(
                chunks,
                uri,
                user,
                pwd,
                **build_kwargs,
            )
            fetched_urls.append(url)

            stats = result.get("stats", {})
            for key in aggregated_stats:
                aggregated_stats[key] += int(stats.get(key, 0))

            for item in result.get("summary", []):
                row = dict(item)
                row["source_url"] = url
                aggregated_summary.append(row)
            for item in result.get("chunk_progress", []):
                chunk_id = str(item.get("chunk_id", ""))
                row_idx = row_idx_by_chunk_id.get(chunk_id)
                if row_idx is None:
                    continue
                row = aggregated_chunk_progress[row_idx]
                row["status"] = str(item.get("status", row["status"]))
                row["entities"] = int(item.get("entities", row["entities"]))
                row["relations"] = int(item.get("relations", row["relations"]))
            _emit_keyword_progress(status="running", current_url=url)
        except Exception as exc:
            for row_idx in row_idx_by_chunk_id.values():
                row = aggregated_chunk_progress[row_idx]
                if row.get("status") in {"queued", "processing"}:
                    row["status"] = "failed"
                    row["error"] = str(exc)
            failed_urls.append({"url": url, "error": str(exc)})
            _emit_keyword_progress(status="running", current_url=url, error=str(exc))

    if not fetched_urls:
        first_reason = failed_urls[0]["error"] if failed_urls else "unknown error"
        error_msg = f"All crawled pages failed to process: {first_reason}"
        _emit_keyword_progress(status="failed", error=error_msg)
        raise ValueError(error_msg)

    result_payload = {
        "searched_keyword": cleaned_keyword,
        "fetched_urls": fetched_urls,
        "failed_urls": failed_urls,
        "stats": aggregated_stats,
        "summary": aggregated_summary,
        "chunk_limit": resolved_chunk_limit,
        "chunks_available": aggregated_chunks_available,
        "chunk_progress": aggregated_chunk_progress,
    }
    _emit_keyword_progress(status="completed")
    return result_payload


def _normalize_chat_history(history: Optional[List[Dict[str, str]]]) -> List[Dict[str, str]]:
    if not history:
        return []

    normalized: List[Dict[str, str]] = []
    for message in history[-20:]:
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def chat_general(message: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    cleaned_message = message.strip()
    if not cleaned_message:
        raise ValueError("Message cannot be empty")

    system_prompt = (
        "你是一位溫和、支持性的聊天夥伴，請使用繁體中文。"
        "目標是陪伴失眠或情緒低落的使用者，語氣要穩定、簡潔、無評價。"
        "避免提供醫療診斷或藥物建議；若使用者提到自傷風險或急性危機，"
        "請鼓勵立即聯絡當地緊急資源與可信任的人。"
    )
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(_normalize_chat_history(history))
    messages.append({"role": "user", "content": cleaned_message})

    answer = llm_client.chat_text(
        messages=messages,
        temperature=GENERAL_CHAT_TEMPERATURE,
        max_tokens=GENERAL_CHAT_NUM_PREDICT,
        timeout_seconds=GENERAL_CHAT_TIMEOUT_SECONDS,
    )
    cfg = llm_client.get_runtime_config()
    return {"answer": answer, "model": cfg.model, "provider": cfg.provider}
