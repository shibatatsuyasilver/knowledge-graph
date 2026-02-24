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
    """延遲載入 (Lazy Load) 知識圖譜 (KG) 相關的模組。
    
    為了避免在不支援 KG 的環境下（或是相依套件尚未安裝時）發生啟動錯誤，
    此函式將 `KnowledgeGraphBuilder` 和 `answer_with_manual_prompt` 的載入
    推遲到真正需要使用時才執行。
    
    回傳值:
        一個 tuple，包含兩個元素：
        1. KnowledgeGraphBuilder: 用來將文字區塊 (Chunks) 轉換為知識圖譜的類別。
        2. answer_with_manual_prompt: 用來將自然語言查詢轉換為 Cypher 語法並執行的函式。
        
    例外:
        RuntimeError: 當載入失敗（例如缺少 `backend.llm_kg` 相依模組）時拋出，並提示使用者安裝相依。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    try:
        # 嘗試載入實際建立知識圖譜的 Builder 類別
        from backend.llm_kg.kg_builder import KnowledgeGraphBuilder
        
        # 嘗試載入用來執行圖譜查詢 (QA) 的函式
        from backend.llm_kg.nl2cypher import answer_with_manual_prompt
    except ImportError as exc:
        # 如果捕獲到 ImportError（找不到模組），轉換為更明確的 RuntimeError 錯誤訊息
        raise RuntimeError(
            "Unable to load KG modules. Install backend dependencies before calling KG endpoints."
        ) from exc
    
    # 成功載入後，回傳這些模組供上層函式使用
    return KnowledgeGraphBuilder, answer_with_manual_prompt


def _kg_qa_use_llm() -> bool:
    """取得當前環境是否設定要在 QA 流程中使用 LLM（大型語言模型）。
    
    此函式作為一個 getter 介面，封裝了底層服務 `qa_service` 的狀態。
    透過函式呼叫而不是直接存取變數，可以在測試期間更容易地進行 Mock 或 Monkeypatch。
    
    回傳值:
        bool: 如果為 True，代表查詢時會呼叫 LLM 進行處理（例如生成 Cypher）；否則不使用。
    """
    return _BASE_QA_USE_LLM()


def _kg_qa_model() -> Optional[str]:
    """取得 QA 流程中預設使用的 LLM 模型名稱。
    
    回傳值:
        Optional[str]: 模型名稱字串（例如 "gpt-4" 或 "gemini-1.5-pro"），若未設定則回傳 None。
    """
    return _BASE_QA_MODEL()


def _kg_qa_temperature() -> float:
    """取得 QA 流程中 LLM 呼叫的預設 Temperature 參數。
    
    Temperature 決定了模型生成文字的隨機性。數值越高（如 0.8）生成的內容越具創造力；
    數值越低（如 0.0）則越固定且具決定性。在此預設通常為 0。
    
    回傳值:
        float: LLM 的 Temperature 設定值。
    """
    return _BASE_QA_TEMPERATURE()


def _kg_qa_max_tokens() -> int:
    """取得 QA 流程中 LLM 呼叫的預設 Max Tokens 參數。
    
    這限制了模型每次回覆時所能生成的最大 Token 數量，防止過度消耗資源或超時。
    
    回傳值:
        int: 最大 Token 數的上限值（例如 1024）。
    """
    return _BASE_QA_MAX_TOKENS()


def _sync_ingest_runtime() -> None:
    """同步資料攝取 (Ingest) 服務的執行階段設定與函式綁定。
    
    此函式確保當前模組 (logic.py) 中的組態設定（如切割大小、搜尋模式等），
    能正確地同步到底層的 `ingest_service` 模組中。
    這在測試環境進行 Monkeypatch (例如修改 logic.py 中的變數) 時特別重要，
    確保底層服務也能讀取到修改後的值。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    # 1. 同步基本參數設定 (常數與執行時期設定)
    ingest_service.KEYWORD_SEARCH_MODE = KEYWORD_SEARCH_MODE
    ingest_service.DEFAULT_CHUNK_LIMIT = DEFAULT_CHUNK_LIMIT
    ingest_service.CHUNK_SIZE_MODE = CHUNK_SIZE_MODE
    ingest_service.DEFAULT_TOKEN_CHUNK_SIZE = DEFAULT_TOKEN_CHUNK_SIZE
    ingest_service.DEFAULT_TOKEN_CHUNK_MIN_SIZE = DEFAULT_TOKEN_CHUNK_MIN_SIZE
    ingest_service.DEFAULT_CHAR_CHUNK_SIZE = DEFAULT_CHAR_CHUNK_SIZE
    ingest_service.DEFAULT_CHAR_CHUNK_MIN_SIZE = DEFAULT_CHAR_CHUNK_MIN_SIZE
    ingest_service.DDGS = DDGS

    # 2. 處理被 Monkeypatch 的輔助函式，避免無限遞迴迴圈
    # 如果 logic.py 中的函式被測試程式替換了，我們也將其同步到 ingest_service 中
    
    # 檢查並同步 _search_keyword_urls
    current_search = globals().get("_search_keyword_urls")
    if current_search is _LOGIC_SEARCH_KEYWORD_URLS_WRAPPER:
        # 若為預設包裝函式，則直接恢復指向底層最原始的函式
        ingest_service._search_keyword_urls = _BASE_INGEST_SEARCH_KEYWORD_URLS
    elif callable(current_search):
        # 如果是被測試程式取代的自定義函式，則將 ingest_service 指向此函式
        ingest_service._search_keyword_urls = current_search  # type: ignore[assignment]

    # 檢查並同步 process_url
    current_process_url = globals().get("process_url")
    if current_process_url is _LOGIC_PROCESS_URL_WRAPPER:
        ingest_service.process_url = _BASE_INGEST_PROCESS_URL
    elif callable(current_process_url):
        ingest_service.process_url = current_process_url  # type: ignore[assignment]

    # 檢查並同步 build_kg_from_chunks
    current_build = globals().get("build_kg_from_chunks")
    if current_build is _LOGIC_BUILD_KG_FROM_CHUNKS_WRAPPER:
        ingest_service.build_kg_from_chunks = _BASE_INGEST_BUILD_KG_FROM_CHUNKS
    elif callable(current_build):
        ingest_service.build_kg_from_chunks = current_build  # type: ignore[assignment]


def _sync_qa_runtime() -> None:
    """同步問答 (QA) 服務的執行階段設定與函式綁定。
    
    與 `_sync_ingest_runtime` 類似，此函式用來確保 QA 相關的設定（如模型參數）
    和函式介面能正確映射到底層的 `qa_service` 中，確保測試或動態修改能生效。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    # 同步 LLM 客戶端模組
    qa_service.llm_client = llm_client

    # 檢查並同步 _kg_qa_use_llm 參數取得函式
    current_use_llm = globals().get("_kg_qa_use_llm")
    if current_use_llm is _LOGIC_KG_QA_USE_LLM_WRAPPER:
        qa_service._kg_qa_use_llm = _BASE_QA_USE_LLM
    elif callable(current_use_llm):
        qa_service._kg_qa_use_llm = current_use_llm  # type: ignore[assignment]

    # 檢查並同步 _kg_qa_model 參數取得函式
    current_model = globals().get("_kg_qa_model")
    if current_model is _LOGIC_KG_QA_MODEL_WRAPPER:
        qa_service._kg_qa_model = _BASE_QA_MODEL
    elif callable(current_model):
        qa_service._kg_qa_model = current_model  # type: ignore[assignment]

    # 檢查並同步 _kg_qa_temperature 參數取得函式
    current_temperature = globals().get("_kg_qa_temperature")
    if current_temperature is _LOGIC_KG_QA_TEMPERATURE_WRAPPER:
        qa_service._kg_qa_temperature = _BASE_QA_TEMPERATURE
    elif callable(current_temperature):
        qa_service._kg_qa_temperature = current_temperature  # type: ignore[assignment]

    # 檢查並同步 _kg_qa_max_tokens 參數取得函式
    current_max_tokens = globals().get("_kg_qa_max_tokens")
    if current_max_tokens is _LOGIC_KG_QA_MAX_TOKENS_WRAPPER:
        qa_service._kg_qa_max_tokens = _BASE_QA_MAX_TOKENS
    elif callable(current_max_tokens):
        qa_service._kg_qa_max_tokens = current_max_tokens  # type: ignore[assignment]

    # 設定載入 KG 查詢執行器的 Lazy hook
    # 透過 lambda 呼叫 _load_kg_modules，並取其第二個回傳值 (answer_with_manual_prompt)
    qa_service._load_kg_query_executor = lambda: _load_kg_modules()[1]


def _estimate_token_count(text: str) -> int:
    """估算給定字串的 Token 數量。
    
    此函式主要用來決定文本是否超過 LLM 的處理長度上限。由於不同模型使用的 Tokenizer 不同，
    這裡直接委派給底層 `ingest_service` 進行正則表達式或近似長度估算。
    
    參數:
        text: 準備送給 LLM 處理的文本 (例如 "這是一個測試字串")
        
    回傳值:
        int: 估算的 Token 數量 (例如計算得到 8)
    """
    return ingest_service._estimate_token_count(text)


def _build_chunk(text: str, source_url: str, title: str) -> Optional[Chunk]:
    """建立一個資料區塊 (Chunk) 的資料結構。
    
    在處理大型文本時，會將文本切割成多個 Chunk。此函式用來包裝切割後的文字與其來源中繼資料。
    
    參數:
        text: 區塊的純文字內容 (例如 "台灣位於東亞...")
        source_url: 該段落來源的網址 (例如 "https://zh.wikipedia.org/wiki/Taiwan")
        title: 該段落來源的標題 (例如 "台灣")
        
    回傳值:
        Optional[Chunk]: 成功建立則回傳 Chunk 物件，若文本過空則可能回傳 None。
    """
    return ingest_service._build_chunk(text, source_url, title)


def _split_text_by_token_limit(text: str, max_tokens: int) -> List[str]:
    """將過長的文本依據 Token 數量限制切割成多個較短的字串。
    
    當一段文字超過我們設定的 Token 上限時（例如 LLM 處理不了過長的 prompt），
    需要將它切斷。此函式會委派給底層實作進行精細的文字切割（例如盡量在句號或段落處切開）。
    
    參數:
        text: 要切割的長文本。
        max_tokens: 每個片段允許的最大 Token 數量 (例如 1000)。
        
    回傳值:
        List[str]: 切割後的字串列表，每個字串長度皆未超過 max_tokens。
    """
    return ingest_service._split_text_by_token_limit(text, max_tokens)


def _use_token_chunking(extraction_provider: Optional[str]) -> bool:
    """判斷當前的 Chunking 模式是否應該基於 Token 而非字元數。
    
    有些 LLM 提供商或模型更適合使用 Token 為單位的切割方式。這會依賴於環境變數
    `CHUNK_SIZE_MODE` 的設定，或是傳入的 provider 判斷。
    
    參數:
        extraction_provider: 目前準備使用的 LLM 提供商 (例如 "openai", "gemini")
        
    回傳值:
        bool: 如果應該用 Token 切割回傳 True，否則回傳 False（使用字元切割）。
    """
    _sync_ingest_runtime()
    return ingest_service._use_token_chunking(extraction_provider)


def _normalize_http_url(value: str) -> str:
    """正規化 HTTP URL，確保網址擁有正確的通訊協定前綴 (http/https)。
    
    若傳入的網址缺少 "http://" 或 "https://"，函式預設會加上 "http://"。
    
    參數:
        value: 使用者輸入的網址 (例如 "www.example.com")
        
    回傳值:
        str: 加上通訊協定後的完整網址 (例如 "http://www.example.com")
    """
    return ingest_service._normalize_http_url(value)


def fetch_clean_text(url: str, timeout: int = 25) -> Dict[str, str]:
    """從給定的 URL 爬取網頁內容，並清理出純文字與標題。
    
    此函式內部委派給 `ingest_service.fetch_clean_text`。它會發送 HTTP 請求，
    取得網頁原始碼後，利用 BeautifulSoup 等工具去除 HTML 標籤、腳本與樣式，只保留純文字。
    
    參數:
        url: 目標網頁的網址 (例如: "https://zh.wikipedia.org/wiki/Taiwan")
        timeout: 連線超時時間（秒數），預設為 25 秒。
        
    回傳值:
        Dict[str, str]: 包含處理後的文本與標題的字典。
        例如: {"text": "這是一段純文字內容...", "title": "台灣 - 維基百科"}
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
    """將一段長文本切割為多個大小適中的 Chunk，方便後續丟給 LLM 處理。
    
    根據當下的設定（使用字元長度切割，或是 Token 數量切割），將文字切分成一系列段落。
    這有助於在建立知識圖譜或存入向量資料庫時，保持每筆資料長度在 LLM 的 Context Window 內。
    
    參數:
        text: 待切割的完整文章。
        source_url: 該文章的來源網址，用來標記在每一個產生的 Chunk 內。
        title: 文章標題，也是用來標記中繼資料。
        max_chars: 字元模式下的最大長度。
        min_chars: 字元模式下的最小長度（避免切出過短無意義的片段）。
        extraction_provider: 目前的 LLM 提供商，用以決定切分策略。
        max_tokens: Token 模式下的最大 Token 長度。
        min_tokens: Token 模式下的最小 Token 長度。
        
    回傳值:
        List[Chunk]: 切割後的文字片段列表。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
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
    """將指定網址爬取、清理為純文字後，並切割為多個 Chunk 的一條龍流程。
    
    這是整合了 `fetch_clean_text` 和 `chunk_text` 的高階操作。
    主要用於當使用者丟入一個連結時，系統自動將其轉化為一系列可供分析的文字片段。
    
    參數:
        url: 欲處理的網頁連結。
        extraction_provider: (可選) 後續預計用哪個 LLM 來萃取知識，會影響文字切割模式。
        
    回傳值:
        List[Chunk]: 網頁內容萃取並切割完成後的 Chunk 列表。
    """
    _sync_ingest_runtime()
    return ingest_service.process_url(url, extraction_provider=extraction_provider)


def _resolve_chunk_limit(chunk_limit: Optional[int]) -> Optional[int]:
    """解析並決定最終生效的 Chunk 數量限制。
    
    如果在呼叫 API 時未指定限制（傳入 None），系統可能會使用預設的全域限制
    `DEFAULT_CHUNK_LIMIT`，避免一次處理太多資料造成服務掛掉。
    
    參數:
        chunk_limit: 使用者傳入的限制數值（可能為 None）。
        
    回傳值:
        Optional[int]: 決定後真正的限制數量（例如 10）。
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
    """從給定的 Chunk 列表中提取實體與關聯，並建構寫入知識圖譜 (Neo4j)。
    
    此流程會逐一或批次將 Chunk 餵給 LLM，讓 LLM 進行資訊萃取（例如抽取出「張三」、「工作於」、「台積電」）。
    接著再把這些節點與邊寫入 Graph Database。
    
    參數:
        chunks: 已切割好的文字片段清單。
        uri: 圖形資料庫 (Neo4j) 的連接 URI (例如 "bolt://localhost:7687")。
        user: 資料庫使用者名稱。
        pwd: 資料庫密碼。
        chunk_limit: 最多只處理幾筆 Chunk (用來限制處理時間)。
        extraction_provider: 負責執行資訊萃取的 LLM 提供商 (例如 "gemini")。
        extraction_model: 具體使用的 LLM 模型 (例如 "gemini-1.5-pro")。
        progress_callback: 一個可回呼的函式，用來即時回報處理進度給前端 (如 Server-Sent Events)。
        
    回傳值:
        Dict[str, Any]: 處理結果統計，包含成功處理的 Chunk 數與新建的實體、關聯數量。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
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
    """將任意純文字直接轉換並建構寫入知識圖譜。
    
    這是一個便捷操作：它會將傳入的 `text` 先自動切割成 Chunk，然後再呼叫 `build_kg_from_chunks`。
    來源標題與 URL 會被標記為 "User Input" 以區別網路文章。
    
    參數:
        text: 使用者直接輸入的一大段文字內容。
        uri, user, pwd: 資料庫連線資訊。
        chunk_limit: 處理的 Chunk 數量上限。
        extraction_provider, extraction_model: LLM 供應商與模型選擇。
        
    回傳值:
        Dict[str, Any]: 處理統計與結果（同 build_kg_from_chunks）。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
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
    """將一段純文字處理為實體與關聯，並建構為知識圖譜的主入口（帶進度回報）。
    
    這是使用者手動輸入文件建立圖譜時，後端所呼叫的主要函式。
    
    流程包含：
    1. 將整段文字切割成多個 Chunk (`chunk_text`)
    2. 將這些 Chunk 送交 LLM 萃取知識並存入資料庫 (`build_kg_from_chunks`)
    3. 在各階段透過 `progress_callback` 傳送狀態進度。
    
    回傳值:
        Dict[str, Any]: 處理完畢後的摘要資料，如處理耗時與節點數。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
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
    """從單一 URL 爬取網頁內容，並自動建構為知識圖譜的主入口。
    
    這是從網址匯入文章時的標準流程。
    
    流程包含：
    1. 爬取並淨化該網址的 HTML (`process_url`)
    2. 切割文字為多個 Chunk
    3. 呼叫 LLM 萃取圖譜並寫入 Neo4j (`build_kg_from_chunks`)
    4. 透過 `progress_callback` 非同步或同步回傳進度資訊。
    
    回傳值:
        Dict[str, Any]: 處理結果摘要字典。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
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
    """將輸入的允許網域列表轉換為標準化格式。
    
    用來確保搜尋引擎回傳的結果只限制在特定的網域內。這會自動清除空白並轉換小寫。
    
    參數:
        values: 原始輸入的網域清單 (例如: [" Wikipedia.org ", "Example.com"])
        
    回傳值:
        List[str]: 正規化後的網域清單 (例如: ["wikipedia.org", "example.com"])
    """
    return ingest_service._to_domain_allowlist(values)


def _is_allowed_domain(url: str, allowlist: List[str]) -> bool:
    """檢查給定網址的網域是否包含在允許的白名單內。
    
    參數:
        url: 欲檢查的網址 (例如: "https://zh.wikipedia.org/wiki/Taiwan")
        allowlist: 允許的網域白名單 (例如: ["wikipedia.org"])
        
    回傳值:
        bool: 如果 url 的網域存在於 allowlist 內，回傳 True；否則回傳 False。
    """
    return ingest_service._is_allowed_domain(url, allowlist)


def _unwrap_duckduckgo_redirect_url(candidate: str) -> str:
    """解開 DuckDuckGo 搜尋結果中的重新導向網址，取得真實的目標網址。
    
    當從 DuckDuckGo HTML 結果抓取時，連結經常是跳轉網址，此函式用來萃取其 url 參數。
    
    參數:
        candidate: DuckDuckGo 的跳轉網址。
        
    回傳值:
        str: 真實的目的地網址。
    """
    return ingest_service._unwrap_duckduckgo_redirect_url(candidate)


def _search_keyword_urls_via_html(keyword: str, max_results: int, language: str) -> List[str]:
    """透過解析搜尋引擎 HTML 頁面的方式取得關鍵字關聯的網址。
    
    作為爬蟲的底層方法，當無法使用官方 API 時，會模擬瀏覽器請求並解析 HTML DOM。
    
    參數:
        keyword: 搜尋關鍵字。
        max_results: 預計取得的最大網址數量。
        language: 指定搜尋語言。
        
    回傳值:
        List[str]: 解析出的網址列表。
    """
    return ingest_service._search_keyword_urls_via_html(keyword, max_results, language)


def _search_keyword_urls_via_wikipedia_api(keyword: str, max_results: int, language: str) -> List[str]:
    """透過維基百科官方 API 搜尋關鍵字，並取得相關條目的網址。
    
    相較於爬網頁，使用官方 API 更為穩定且不會被封鎖。
    
    參數:
        keyword: 欲查詢的關鍵字 (例如 "鴻海")。
        max_results: 回傳的最大連結數量。
        language: 維基百科語系 (例如 "zh" 代表中文)。
        
    回傳值:
        List[str]: 維基百科條目的完整網址清單。
    """
    return ingest_service._search_keyword_urls_via_wikipedia_api(keyword, max_results, language)


def _search_keyword_urls(keyword: str, max_results: int, language: str) -> List[str]:
    """根據當前設定的搜尋模式 (如 'html' 或是 'wikipedia') 來搜尋並回傳網址列表。
    
    此函式是一個 Factory Method 的概念，根據環境變數 `KEYWORD_SEARCH_MODE` 決定
    要呼叫 `_search_keyword_urls_via_html` 或是 `_search_keyword_urls_via_wikipedia_api`。
    
    參數:
        keyword, max_results, language: 同上。
        
    回傳值:
        List[str]: 搜尋結果的網址列表。
    """
    _sync_ingest_runtime()
    return ingest_service._search_keyword_urls(keyword, max_results, language)


def _search_keyword_urls_resilient(keyword: str, max_results: int, language: str) -> List[str]:
    """帶有容錯機制的關鍵字網址搜尋流程。
    
    為了增加系統穩定性，如果主要的搜尋方式（例如 DuckDuckGo HTML 爬蟲）失敗了，
    它會自動降級 (fallback) 到其他方式（例如改用 Wikipedia API），確保一定有結果產出。
    
    參數:
        keyword, max_results, language: 同上。
        
    回傳值:
        List[str]: 容錯執行後取得的網址列表。
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
    """給定一個關鍵字，自動搜尋網頁、爬取內容並轉換為知識圖譜的主流程。
    
    這是一個高度整合的工作流 (Workflow)，流程如下：
    1. 針對 `keyword` 在網路上搜尋 (透過 Search Engine 或 API)，取得前 `max_results` 個相關網址。
    2. 過濾掉不在 `site_allowlist` 內的網域。
    3. 對每個網址呼叫爬蟲 (`process_url`) 抓取內文並切分成 Chunk。
    4. 收集所有 Chunk，並呼叫 `build_kg_from_chunks` 透過 LLM 提取圖譜結構。
    5. 整個過程會透過 `progress_callback` 即時通知進度。
    
    這也是系統「透過關鍵字匯入知識」功能的核心入口。
    
    回傳值:
        Dict[str, Any]: 處理完畢後的整體統計與結果字典。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
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
    """正規化使用者的對話紀錄結構，以符合後端預期的格式。
    
    前端傳來的對話紀錄可能有時帶有 Null 或是多餘的欄位，此函式用來進行清理與格式校正。
    例如確保每個歷史訊息都有 "role" 和 "content" 欄位。
    
    參數:
        history: 包含舊有對話的字典清單 (例如: [{"role": "user", "content": "你好"}])
        
    回傳值:
        List[Dict[str, str]]: 經過清理、可安全餵給 LLM 的歷史對話紀錄。
    """
    return qa_service._normalize_chat_history(history)


def query_kg(
    question: str,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    nl2cypher_provider: Optional[str] = None,
    nl2cypher_model: Optional[str] = None,
    query_engine: Optional[str] = None,
) -> Dict[str, Any]:
    """針對目前已建立的知識圖譜，使用自然語言進行查詢 (QA)。
    
    這是系統「圖譜問答」功能的核心。
    流程為：
    1. 接收使用者的問題 `question` (例如 "台灣的首都為何？")。
    2. 呼叫 NL2Cypher 模組 (結合 LLM)，將這段自然語言轉換成 Neo4j 的 Cypher 查詢語法。
    3. 在 Neo4j 資料庫執行該語法，撈出相關的實體與關聯資料。
    4. 再請 LLM 將這些圖譜資料總結，回答給使用者。
    
    回傳值:
        Dict[str, Any]: 包含回答內容與過程相關資料的字典 (例如 {"answer": "台北", "cypher": "MATCH..."})。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    _sync_qa_runtime()
    return qa_service.query_kg(
        question,
        progress_callback=progress_callback,
        nl2cypher_provider=nl2cypher_provider,
        nl2cypher_model=nl2cypher_model,
        query_engine=query_engine,
    )


def chat_general(message: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """不依賴知識圖譜的純文字通用聊天功能。
    
    作為普通 LLM 聊天的入口，會把使用者的新訊息 `message` 與舊有的 `history` 串接後，
    送交大語言模型取得回覆。
    
    參數:
        message: 當前使用者的提問。
        history: 過去的對話紀錄。
        
    回傳值:
        Dict[str, Any]: LLM 的回覆內容 (例如 {"reply": "我能為您做什麼？"})。
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
