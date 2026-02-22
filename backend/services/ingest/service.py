import hashlib
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional
from urllib.parse import parse_qs, quote, unquote, urlparse, urljoin

import requests
from bs4 import BeautifulSoup
from backend.config.settings import get_ingest_chunk_settings

try:
    from duckduckgo_search import DDGS
except Exception:  # pragma: no cover - dependency can be optional in test env
    DDGS = None

_chunk_settings = get_ingest_chunk_settings()
KEYWORD_SEARCH_MODE = _chunk_settings.keyword_search_mode
DEFAULT_CHUNK_LIMIT = _chunk_settings.default_chunk_limit
CHUNK_SIZE_MODE = _chunk_settings.chunk_size_mode
TOKEN_ESTIMATE_PATTERN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]", flags=re.UNICODE)
DEFAULT_TOKEN_CHUNK_SIZE = _chunk_settings.default_token_chunk_size
DEFAULT_TOKEN_CHUNK_MIN_SIZE = _chunk_settings.default_token_chunk_min_size
DEFAULT_CHAR_CHUNK_SIZE = _chunk_settings.default_char_chunk_size
DEFAULT_CHAR_CHUNK_MIN_SIZE = _chunk_settings.default_char_chunk_min_size

@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_url: str
    title: str
    tokens: int = 0


def _estimate_token_count(text: str) -> int:
    """執行 `_estimate_token_count` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    if not text:
        return 0
    return len(TOKEN_ESTIMATE_PATTERN.findall(text))


def _build_chunk(text: str, source_url: str, title: str) -> Optional[Chunk]:
    """執行 `_build_chunk` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
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
    """執行 `_split_text_by_token_limit` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
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
    """執行 `_use_token_chunking` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
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
    """執行 `_chunk_text_by_char_budget` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
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
    """執行 `_chunk_text_by_token_budget` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    pieces = [x.strip() for x in text.split("\n") if x.strip()]
    chunks: List[Chunk] = []
    bucket: List[str] = []
    token_count = 0

    def flush_bucket() -> None:
        """執行 `flush_bucket` 的主要流程。
        函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
        """
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
    """執行 `_normalize_http_url` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
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


def _load_kg_builder() -> Any:
    """執行 `_load_kg_builder` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    try:
        from backend.llm_kg.kg_builder import KnowledgeGraphBuilder
    except ImportError as exc:
        raise RuntimeError(
            "Unable to load KG modules. Install backend dependencies before calling KG endpoints."
        ) from exc
    return KnowledgeGraphBuilder


def fetch_clean_text(url: str, timeout: int = 25) -> Dict[str, str]:
    """執行 `fetch_clean_text` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
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
    """執行 `chunk_text` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
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
    """執行 `process_url` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
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
    """執行 `_resolve_chunk_limit` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
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
    """執行 `build_kg_from_chunks` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
    KnowledgeGraphBuilder = _load_kg_builder()
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
        """執行 `_emit_chunk_progress` 的內部輔助流程。
        此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
        """
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
    """執行 `build_kg_from_text_content` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
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
    """執行 `process_text_to_kg` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
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
    """執行 `process_url_to_kg` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
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


def _to_domain_allowlist(values: Optional[Iterable[str]]) -> List[str]:
    """執行 `_to_domain_allowlist` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
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
    """執行 `_is_allowed_domain` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    if not allowlist:
        return True

    host = (urlparse(url).netloc or "").lower()
    for domain in allowlist:
        if host == domain or host.endswith(f".{domain}"):
            return True
    return False


def _unwrap_duckduckgo_redirect_url(candidate: str) -> str:
    """執行 `_unwrap_duckduckgo_redirect_url` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    absolute = urljoin("https://duckduckgo.com", candidate)
    parsed = urlparse(absolute)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        redirect_url = parse_qs(parsed.query).get("uddg", [None])[0]
        if redirect_url:
            return unquote(redirect_url)
    return absolute


def _search_keyword_urls_via_html(keyword: str, max_results: int, language: str) -> List[str]:
    """執行 `_search_keyword_urls_via_html` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
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
    """執行 `_search_keyword_urls_via_wikipedia_api` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
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
    """執行 `_search_keyword_urls` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
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
    """執行 `_search_keyword_urls_resilient` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
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
    """執行 `process_keyword_to_kg` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
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
        """執行 `_emit_keyword_progress` 的內部輔助流程。
        此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
        """
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
                """執行 `_on_build_chunk_progress` 的內部輔助流程。
                此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
                """
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
