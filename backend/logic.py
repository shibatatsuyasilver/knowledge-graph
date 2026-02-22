"""Thin facade over backend services for backward compatibility."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional

from backend.llm_kg import llm_client
from backend.services.ingest import service as ingest_service
from backend.services.qa import service as qa_service

# Re-export commonly used service symbols for tests/backward compatibility.
Chunk = ingest_service.Chunk
TOKEN_ESTIMATE_PATTERN = ingest_service.TOKEN_ESTIMATE_PATTERN
DDGS = ingest_service.DDGS
requests = ingest_service.requests

# Runtime knobs (kept mutable so tests can monkeypatch).
KEYWORD_SEARCH_MODE = ingest_service.KEYWORD_SEARCH_MODE
DEFAULT_CHUNK_LIMIT = ingest_service.DEFAULT_CHUNK_LIMIT
CHUNK_SIZE_MODE = ingest_service.CHUNK_SIZE_MODE
DEFAULT_TOKEN_CHUNK_SIZE = ingest_service.DEFAULT_TOKEN_CHUNK_SIZE
DEFAULT_TOKEN_CHUNK_MIN_SIZE = ingest_service.DEFAULT_TOKEN_CHUNK_MIN_SIZE
DEFAULT_CHAR_CHUNK_SIZE = ingest_service.DEFAULT_CHAR_CHUNK_SIZE
DEFAULT_CHAR_CHUNK_MIN_SIZE = ingest_service.DEFAULT_CHAR_CHUNK_MIN_SIZE

_BASE_INGEST_SEARCH_KEYWORD_URLS = ingest_service._search_keyword_urls
_BASE_INGEST_PROCESS_URL = ingest_service.process_url
_BASE_INGEST_BUILD_KG_FROM_CHUNKS = ingest_service.build_kg_from_chunks
_BASE_QA_USE_LLM = qa_service._kg_qa_use_llm
_BASE_QA_MODEL = qa_service._kg_qa_model
_BASE_QA_TEMPERATURE = qa_service._kg_qa_temperature
_BASE_QA_MAX_TOKENS = qa_service._kg_qa_max_tokens

_LOGIC_SEARCH_KEYWORD_URLS_WRAPPER = None
_LOGIC_PROCESS_URL_WRAPPER = None
_LOGIC_BUILD_KG_FROM_CHUNKS_WRAPPER = None
_LOGIC_KG_QA_USE_LLM_WRAPPER = None
_LOGIC_KG_QA_MODEL_WRAPPER = None
_LOGIC_KG_QA_TEMPERATURE_WRAPPER = None
_LOGIC_KG_QA_MAX_TOKENS_WRAPPER = None


# QA hooks kept mutable for compatibility with existing tests.
def _load_kg_modules() -> tuple[Any, Any]:
    """執行 `_load_kg_modules` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    try:
        from backend.llm_kg.kg_builder import KnowledgeGraphBuilder
        from backend.llm_kg.nl2cypher import answer_with_manual_prompt
    except ImportError as exc:
        raise RuntimeError(
            "Unable to load KG modules. Install backend dependencies before calling KG endpoints."
        ) from exc
    return KnowledgeGraphBuilder, answer_with_manual_prompt


def _kg_qa_use_llm() -> bool:
    """執行 `_kg_qa_use_llm` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    return _BASE_QA_USE_LLM()


def _kg_qa_model() -> Optional[str]:
    """執行 `_kg_qa_model` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    return _BASE_QA_MODEL()


def _kg_qa_temperature() -> float:
    """執行 `_kg_qa_temperature` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    return _BASE_QA_TEMPERATURE()


def _kg_qa_max_tokens() -> int:
    """執行 `_kg_qa_max_tokens` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    return _BASE_QA_MAX_TOKENS()


def _sync_ingest_runtime() -> None:
    """執行 `_sync_ingest_runtime` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    ingest_service.KEYWORD_SEARCH_MODE = KEYWORD_SEARCH_MODE
    ingest_service.DEFAULT_CHUNK_LIMIT = DEFAULT_CHUNK_LIMIT
    ingest_service.CHUNK_SIZE_MODE = CHUNK_SIZE_MODE
    ingest_service.DEFAULT_TOKEN_CHUNK_SIZE = DEFAULT_TOKEN_CHUNK_SIZE
    ingest_service.DEFAULT_TOKEN_CHUNK_MIN_SIZE = DEFAULT_TOKEN_CHUNK_MIN_SIZE
    ingest_service.DEFAULT_CHAR_CHUNK_SIZE = DEFAULT_CHAR_CHUNK_SIZE
    ingest_service.DEFAULT_CHAR_CHUNK_MIN_SIZE = DEFAULT_CHAR_CHUNK_MIN_SIZE
    ingest_service.DDGS = DDGS

    # Forward monkeypatched helpers while avoiding recursion for default wrappers.
    current_search = globals().get("_search_keyword_urls")
    if current_search is _LOGIC_SEARCH_KEYWORD_URLS_WRAPPER:
        ingest_service._search_keyword_urls = _BASE_INGEST_SEARCH_KEYWORD_URLS
    elif callable(current_search):
        ingest_service._search_keyword_urls = current_search  # type: ignore[assignment]

    current_process_url = globals().get("process_url")
    if current_process_url is _LOGIC_PROCESS_URL_WRAPPER:
        ingest_service.process_url = _BASE_INGEST_PROCESS_URL
    elif callable(current_process_url):
        ingest_service.process_url = current_process_url  # type: ignore[assignment]

    current_build = globals().get("build_kg_from_chunks")
    if current_build is _LOGIC_BUILD_KG_FROM_CHUNKS_WRAPPER:
        ingest_service.build_kg_from_chunks = _BASE_INGEST_BUILD_KG_FROM_CHUNKS
    elif callable(current_build):
        ingest_service.build_kg_from_chunks = current_build  # type: ignore[assignment]


def _sync_qa_runtime() -> None:
    """執行 `_sync_qa_runtime` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    qa_service.llm_client = llm_client

    current_use_llm = globals().get("_kg_qa_use_llm")
    if current_use_llm is _LOGIC_KG_QA_USE_LLM_WRAPPER:
        qa_service._kg_qa_use_llm = _BASE_QA_USE_LLM
    elif callable(current_use_llm):
        qa_service._kg_qa_use_llm = current_use_llm  # type: ignore[assignment]

    current_model = globals().get("_kg_qa_model")
    if current_model is _LOGIC_KG_QA_MODEL_WRAPPER:
        qa_service._kg_qa_model = _BASE_QA_MODEL
    elif callable(current_model):
        qa_service._kg_qa_model = current_model  # type: ignore[assignment]

    current_temperature = globals().get("_kg_qa_temperature")
    if current_temperature is _LOGIC_KG_QA_TEMPERATURE_WRAPPER:
        qa_service._kg_qa_temperature = _BASE_QA_TEMPERATURE
    elif callable(current_temperature):
        qa_service._kg_qa_temperature = current_temperature  # type: ignore[assignment]

    current_max_tokens = globals().get("_kg_qa_max_tokens")
    if current_max_tokens is _LOGIC_KG_QA_MAX_TOKENS_WRAPPER:
        qa_service._kg_qa_max_tokens = _BASE_QA_MAX_TOKENS
    elif callable(current_max_tokens):
        qa_service._kg_qa_max_tokens = current_max_tokens  # type: ignore[assignment]

    qa_service._load_kg_query_executor = lambda: _load_kg_modules()[1]


def _estimate_token_count(text: str) -> int:
    """執行 `_estimate_token_count` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    return ingest_service._estimate_token_count(text)


def _build_chunk(text: str, source_url: str, title: str) -> Optional[Chunk]:
    """執行 `_build_chunk` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    return ingest_service._build_chunk(text, source_url, title)


def _split_text_by_token_limit(text: str, max_tokens: int) -> List[str]:
    """執行 `_split_text_by_token_limit` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    return ingest_service._split_text_by_token_limit(text, max_tokens)


def _use_token_chunking(extraction_provider: Optional[str]) -> bool:
    """執行 `_use_token_chunking` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    _sync_ingest_runtime()
    return ingest_service._use_token_chunking(extraction_provider)


def _normalize_http_url(value: str) -> str:
    """執行 `_normalize_http_url` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    return ingest_service._normalize_http_url(value)


def fetch_clean_text(url: str, timeout: int = 25) -> Dict[str, str]:
    """執行 `fetch_clean_text` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
    return ingest_service.fetch_clean_text(url, timeout=timeout)


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
    _sync_ingest_runtime()
    return ingest_service.chunk_text(
        text=text,
        source_url=source_url,
        title=title,
        max_chars=max_chars,
        min_chars=min_chars,
        extraction_provider=extraction_provider,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
    )


def process_url(url: str, extraction_provider: Optional[str] = None) -> List[Chunk]:
    """執行 `process_url` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
    _sync_ingest_runtime()
    return ingest_service.process_url(url, extraction_provider=extraction_provider)


def _resolve_chunk_limit(chunk_limit: Optional[int]) -> Optional[int]:
    """執行 `_resolve_chunk_limit` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    _sync_ingest_runtime()
    return ingest_service._resolve_chunk_limit(chunk_limit)


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
    _sync_ingest_runtime()
    return ingest_service.build_kg_from_chunks(
        chunks=chunks,
        uri=uri,
        user=user,
        pwd=pwd,
        chunk_limit=chunk_limit,
        extraction_provider=extraction_provider,
        extraction_model=extraction_model,
        progress_callback=progress_callback,
    )


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
    _sync_ingest_runtime()
    return ingest_service.build_kg_from_text_content(
        text=text,
        uri=uri,
        user=user,
        pwd=pwd,
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
    _sync_ingest_runtime()
    return ingest_service.process_text_to_kg(
        text=text,
        uri=uri,
        user=user,
        pwd=pwd,
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
    _sync_ingest_runtime()
    return ingest_service.process_url_to_kg(
        url=url,
        uri=uri,
        user=user,
        pwd=pwd,
        chunk_limit=chunk_limit,
        extraction_provider=extraction_provider,
        extraction_model=extraction_model,
        progress_callback=progress_callback,
    )


def _to_domain_allowlist(values: Optional[Iterable[str]]) -> List[str]:
    """執行 `_to_domain_allowlist` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    return ingest_service._to_domain_allowlist(values)


def _is_allowed_domain(url: str, allowlist: List[str]) -> bool:
    """執行 `_is_allowed_domain` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    return ingest_service._is_allowed_domain(url, allowlist)


def _unwrap_duckduckgo_redirect_url(candidate: str) -> str:
    """執行 `_unwrap_duckduckgo_redirect_url` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    return ingest_service._unwrap_duckduckgo_redirect_url(candidate)


def _search_keyword_urls_via_html(keyword: str, max_results: int, language: str) -> List[str]:
    """執行 `_search_keyword_urls_via_html` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    return ingest_service._search_keyword_urls_via_html(keyword, max_results, language)


def _search_keyword_urls_via_wikipedia_api(keyword: str, max_results: int, language: str) -> List[str]:
    """執行 `_search_keyword_urls_via_wikipedia_api` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    return ingest_service._search_keyword_urls_via_wikipedia_api(keyword, max_results, language)


def _search_keyword_urls(keyword: str, max_results: int, language: str) -> List[str]:
    """執行 `_search_keyword_urls` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    _sync_ingest_runtime()
    return ingest_service._search_keyword_urls(keyword, max_results, language)


def _search_keyword_urls_resilient(keyword: str, max_results: int, language: str) -> List[str]:
    """執行 `_search_keyword_urls_resilient` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    _sync_ingest_runtime()
    return ingest_service._search_keyword_urls_resilient(keyword, max_results, language)


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
    _sync_ingest_runtime()
    return ingest_service.process_keyword_to_kg(
        keyword=keyword,
        uri=uri,
        user=user,
        pwd=pwd,
        max_results=max_results,
        language=language,
        site_allowlist=site_allowlist,
        chunk_limit=chunk_limit,
        extraction_provider=extraction_provider,
        extraction_model=extraction_model,
        progress_callback=progress_callback,
    )


def _normalize_chat_history(history: Optional[List[Dict[str, str]]]) -> List[Dict[str, str]]:
    """執行 `_normalize_chat_history` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    return qa_service._normalize_chat_history(history)


def query_kg(question: str) -> Dict[str, Any]:
    """執行 `query_kg` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
    _sync_qa_runtime()
    return qa_service.query_kg(question)


def chat_general(message: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """執行 `chat_general` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
    _sync_qa_runtime()
    return qa_service.chat_general(message, history=history)


# Capture wrapper references so runtime sync can detect monkeypatch overrides.
_LOGIC_SEARCH_KEYWORD_URLS_WRAPPER = _search_keyword_urls
_LOGIC_PROCESS_URL_WRAPPER = process_url
_LOGIC_BUILD_KG_FROM_CHUNKS_WRAPPER = build_kg_from_chunks
_LOGIC_KG_QA_USE_LLM_WRAPPER = _kg_qa_use_llm
_LOGIC_KG_QA_MODEL_WRAPPER = _kg_qa_model
_LOGIC_KG_QA_TEMPERATURE_WRAPPER = _kg_qa_temperature
_LOGIC_KG_QA_MAX_TOKENS_WRAPPER = _kg_qa_max_tokens
