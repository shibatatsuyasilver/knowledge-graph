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
    """粗略估算給定文字的 Token 數量。
    
    使用正則表達式 `TOKEN_ESTIMATE_PATTERN` 找出中文字元或英數字詞，以此估計 LLM 會如何對這段文字進行 Tokenize。
    例如輸入 "Hello 世界"，可能會被計算成 "Hello", "世", "界" 共 3 個 Tokens。
    
    參數:
        text (str): 待估算的原始字串。
        
    回傳值:
        int: 估算的 Token 數量。若字串為空，回傳 0。
    """
    if not text:
        return 0
    return len(TOKEN_ESTIMATE_PATTERN.findall(text))


def _build_chunk(text: str, source_url: str, title: str) -> Optional[Chunk]:
    """將一段純文字包裝成 `Chunk` 資料類別，並產生雜湊值 ID。
    
    當長篇文章被切割成小段落後，每一段都需要一個唯一 ID 和中繼資料（URL、標題）以便後續寫入圖譜。
    雜湊值透過 `source_url` 與文字內容產生，確保相同的來源與內容會有一致的 `chunk_id`。
    
    參數:
        text (str): 此區塊的文字內容。
        source_url (str): 內容來源的原始網址。
        title (str): 內容來源的標題。
        
    回傳值:
        Optional[Chunk]: 包含所有中繼資料的 Chunk 物件。若文字為空，則回傳 None。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    normalized = text.strip()
    if not normalized:
        return None
    # 產生一組長度 16 的 SHA1 雜湊碼作為唯一識別
    chunk_id = hashlib.sha1(f"{source_url}|{normalized}".encode("utf-8")).hexdigest()[:16]
    return Chunk(
        chunk_id=chunk_id,
        text=normalized,
        source_url=source_url,
        title=title,
        tokens=_estimate_token_count(normalized),
    )


def _split_text_by_token_limit(text: str, max_tokens: int) -> List[str]:
    """將一段文字依照最大 Token 限制強行切成多段。
    
    當單一段落本身就超過 `max_tokens`（例如超長且沒有換行的句子），
    會使用正則尋找 Token 邊界，將其生硬地切成符合上限的數個字串。
    
    參數:
        text (str): 待切割的文字。
        max_tokens (int): 每個片段的 Token 上限。
        
    回傳值:
        List[str]: 切割後的文字片段陣列。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    if max_tokens <= 0:
        return [text]

    matches = list(TOKEN_ESTIMATE_PATTERN.finditer(text))
    # 如果總 Token 數未超過上限，直接回傳原字串
    if len(matches) <= max_tokens:
        return [text]

    parts: List[str] = []
    start = 0
    # 透過迴圈每次取 max_tokens 數量的配對，進行切割
    # 舉例：如果 max_tokens 為 100，則取第 0~99 個 token、100~199 個 token... 以此類推
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
    """決定當前環境下，是否應該以 Token 為單位來進行文字切割。
    
    系統可設定為 "token" 或 "char" 模式。如果未強制指定模式，則依賴 `extraction_provider` 判斷。
    例如，Gemini 模型對於 Token 數量的敏感度較高，因此如果是 "gemini"，會預設啟用 token 切割。
    
    參數:
        extraction_provider (Optional[str]): 預計使用的 LLM 供應商，例如 "gemini" 或 "openai"。
        
    回傳值:
        bool: True 代表使用 Token 為單位切割，False 代表使用字元數切割。
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
    """以字元數量為基準 (Char Budget) 將文本切割為多個區塊。
    
    運作邏輯：
    1. 將文本依照換行符號切割成多行 (pieces)。
    2. 逐行將文字放入暫存區 (bucket)，並累計字元數 (size)。
    3. 若加入下一行會超過 `max_chars`，則將目前的暫存區結算成一個 Chunk，並清空暫存區。
    4. 最後確認若結算出來的 Chunk 大於 `min_chars` 才保留，過短的片段捨棄。
    
    參數:
        text, source_url, title: 文字內容與中繼資料。
        max_chars: 單個 Chunk 允許的最大字元數 (例如 2000)。
        min_chars: 單個 Chunk 的最小字元數 (例如 50)。
        
    回傳值:
        List[Chunk]: 切割完畢的 Chunk 物件列表。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    # 將文本按行拆分，過濾掉空白行
    pieces = [x.strip() for x in text.split("\n") if x.strip()]
    chunks: List[Chunk] = []
    
    # bucket 用來暫存尚未結算的連續行
    bucket: List[str] = []
    size = 0

    for piece in pieces:
        # 如果加入這一行會超過限制，且 bucket 裡面已經有東西，就先結算 bucket
        if size + len(piece) + 1 > max_chars and bucket:
            joined = "\n".join(bucket).strip()
            # 只有當組合起來的長度大於下限時，才真正建立 Chunk
            if len(joined) >= min_chars:
                chunk = _build_chunk(joined, source_url, title)
                if chunk:
                    chunks.append(chunk)
            bucket = []
            size = 0

        # 將當前行放入 bucket，並更新累積字數 (+1 代表換行符號)
        bucket.append(piece)
        size += len(piece) + 1

    # 處理最後一批剩餘在 bucket 的文字
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
    """以 Token 數量為基準 (Token Budget) 將文本切割為多個區塊。
    
    運作邏輯類似於 `_chunk_text_by_char_budget`，但評估標準改為 Token 數：
    1. 將文本依照換行符號切割成多行。
    2. 若某單獨一行就超過 `max_tokens`，則呼叫 `_split_text_by_token_limit` 將其強制切斷。
    3. 否則，將短行不斷累積至 `bucket`，直到累積 Token 數超過上限，再進行結算。
    
    參數:
        text, source_url, title: 文字內容與中繼資料。
        max_tokens: 單個 Chunk 允許的最大 Token 數 (例如 1000)。
        min_tokens: 單個 Chunk 的最小 Token 數 (例如 20)。
        
    回傳值:
        List[Chunk]: 切割完畢的 Chunk 物件列表。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    pieces = [x.strip() for x in text.split("\n") if x.strip()]
    chunks: List[Chunk] = []
    bucket: List[str] = []
    token_count = 0

    def flush_bucket() -> None:
        """內部函式：負責將目前 bucket 內的文字合併，若符合最小限制則封裝成 Chunk 並清空暫存。"""
        nonlocal bucket, token_count
        if not bucket:
            return
        joined = "\n".join(bucket).strip()
        # 結算時若超過最小 Token 數才建立 Chunk
        if _estimate_token_count(joined) >= min_tokens:
            chunk = _build_chunk(joined, source_url, title)
            if chunk:
                chunks.append(chunk)
        bucket = []
        token_count = 0

    for piece in pieces:
        piece_tokens = _estimate_token_count(piece)
        # 邊界處理：如果單行本身就超過 max_tokens，必須強制把它切開
        if piece_tokens > max_tokens:
            flush_bucket() # 先把之前的清掉
            # 針對超長行進行強制 Token 切割
            for segment in _split_text_by_token_limit(piece, max_tokens):
                if _estimate_token_count(segment) < min_tokens:
                    continue
                chunk = _build_chunk(segment, source_url, title)
                if chunk:
                    chunks.append(chunk)
            continue

        # 如果加上當前行會超過 Token 上限，先將舊的 bucket 結算
        if token_count + piece_tokens > max_tokens and bucket:
            flush_bucket()

        # 放入 bucket 累積
        bucket.append(piece)
        token_count += piece_tokens

    # 確保迴圈結束後，剩餘的 bucket 也有被結算
    flush_bucket()
    return chunks


def _normalize_http_url(value: str) -> str:
    """正規化 HTTP URL，確保網址正確且包含通訊協定。
    
    例如輸入 "example.com"，會被自動補齊為 "https://example.com"。
    
    參數:
        value (str): 使用者提供或搜尋引擎抓到的網址。
        
    回傳值:
        str: 具有正確 scheme (http/https) 的網址。
        
    例外:
        ValueError: 若 URL 為空，或解析後不具備合法的 netloc，拋出此錯誤。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    raw = value.strip()
    if not raw:
        raise ValueError("URL cannot be empty")

    parsed = urlparse(raw)
    # 若沒有提供 scheme (如 www.apple.com)，預設補上 https://
    if parsed.scheme == "":
        raw = f"https://{raw}"
        parsed = urlparse(raw)

    # 確保 URL 合法，且屬於 web 連結
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid URL: {value}")
    return parsed.geturl()


def _load_kg_builder() -> Any:
    """延遲載入 (Lazy Load) KnowledgeGraphBuilder。
    
    這用來避免在系統啟動時就強制載入龐大的 KG 套件。
    只有當執行到需要建立圖譜的流程時，才會嘗試引用 `KnowledgeGraphBuilder`。
    
    回傳值:
        Any: KnowledgeGraphBuilder 類別。
        
    例外:
        RuntimeError: 當無法引用時，提示套件未安裝。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    try:
        from backend.llm_kg.kg_builder import KnowledgeGraphBuilder
    except ImportError as exc:
        raise RuntimeError(
            "Unable to load KG modules. Install backend dependencies before calling KG endpoints."
        ) from exc
    return KnowledgeGraphBuilder


def fetch_clean_text(url: str, timeout: int = 25) -> Dict[str, str]:
    """發送 HTTP 請求並抓取指定 URL 的 HTML，萃取成純文字內容。
    
    這是網頁爬蟲的核心函式。利用 `requests` 發出 GET 請求，成功後再交由 `BeautifulSoup`
    解析 HTML，只保留有實質內容的標籤 (`h1`, `p`, `article`...)。
    過程中會過濾掉太短的片段 (小於 30 字元) 以及可能是 cookie 警告等干擾訊息。
    
    參數:
        url (str): 目標網站連結 (例如 "https://zh.wikipedia.org/wiki/Taiwan")。
        timeout (int): 請求超時時間，預設 25 秒。
        
    回傳值:
        Dict[str, str]: 包含 title、text 與正規化後的 url，若發生錯誤則回傳空內容的預設值。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    try:
        normalized_url = _normalize_http_url(url)
        # 設定基本的 User-Agent 避免被擋
        response = requests.get(
            normalized_url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; GenAI-KG-Bot/1.0)"},
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # 擷取網頁標題
        title = (soup.title.string or "Document").strip() if soup.title else "Document"

        blocks: List[str] = []
        # 從特定的 HTML 標籤中提取純文字
        for tag in soup.find_all(["h1", "h2", "h3", "p", "li", "article", "section"]):
            text = " ".join(tag.get_text(" ", strip=True).split())
            
            # 濾除太短的文字，或是包含 "cookie", "privacy" 這些可能是隱私權同意視窗的句子
            if len(text) < 30:
                continue
            if "cookie" in text.lower() or "privacy" in text.lower():
                continue
            blocks.append(text)

        # 去除重複的字串段落，但仍保持它出現的先後順序
        merged = "\n".join(dict.fromkeys(blocks))
        # 將多餘的換行壓縮為單一換行
        merged = re.sub(r"\n{2,}", "\n", merged)
        return {"title": title, "text": merged, "url": normalized_url}
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        # 若抓取失敗，回傳一個包含 Error 標籤且內容為空的字典
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
    """將一段長篇文章自動判定切割策略，切分為多個大小適中的 Chunk。
    
    此為主入口函式，根據設定或 `extraction_provider` 判斷當前該用 Token Budget
    還是 Char Budget 切割。
    
    參數:
        text, source_url, title: 文章內容與中繼資料。
        max_chars, min_chars: Char 模式下，每個 Chunk 的最大/小字元數。
        extraction_provider: 決定切割策略的依據 (如 "gemini" 預設使用 token 模式)。
        max_tokens, min_tokens: Token 模式下，每個 Chunk 的最大/小 Token 數。
        
    回傳值:
        List[Chunk]: 切割完畢的 Chunk 列表。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    if not text:
        return []

    # 判斷是否需要用 Token 模式處理 (例如 Gemini)
    if _use_token_chunking(extraction_provider):
        return _chunk_text_by_token_budget(
            text=text,
            source_url=source_url,
            title=title,
            max_tokens=max(1, int(max_tokens)),
            min_tokens=max(1, int(min_tokens)),
        )

    # 否則退回預設的字元切割模式
    return _chunk_text_by_char_budget(
        text=text,
        source_url=source_url,
        title=title,
        max_chars=max(1, int(max_chars)),
        min_chars=max(1, int(min_chars)),
    )


def process_url(url: str, extraction_provider: Optional[str] = None) -> List[Chunk]:
    """組合技：爬取 URL 的純文字內容，隨後自動執行 Chunking。
    
    主要步驟：
    1. 呼叫 `fetch_clean_text` 取得清洗好的純文字。
    2. 若有文字，呼叫 `chunk_text` 將其切分為小段落。
    
    參數:
        url (str): 目標網址。
        extraction_provider (str): 影響 chunk 切割模式的依據。
        
    回傳值:
        List[Chunk]: 切分完成的 Chunk 列表。若爬取失敗則回傳空列表。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
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
    """決定當前批次處理可以消化多少個 Chunk。
    
    為避免單次處理載入太多 Chunk（導致 LLM 超時或耗用太多資源），我們允許使用者
    傳入 `chunk_limit`，或是從全域變數 `DEFAULT_CHUNK_LIMIT` 獲取預設值。
    
    參數:
        chunk_limit: 傳入的最大處理數量上限。
        
    回傳值:
        Optional[int]: 解析後實際套用的限制值，若為 None 代表無限制。
        
    例外:
        ValueError: 如果限制值為負數，或是設定得太大 (大於 200)，則報錯。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    if chunk_limit is None:
        chunk_limit = DEFAULT_CHUNK_LIMIT
    if chunk_limit is None:
        return None

    value = int(chunk_limit)
    if value <= 0:
        return None
    # 限制每次處理的 chunk 最大不能超過 200 個
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
    """將給定的 Chunk 序列逐一透過 LLM 提取實體與關聯，並寫入知識圖譜。
    
    這個函式是系統將「純文字」轉換為「結構化知識」的核心。
    它會迭代每一筆 Chunk：
    1. 呼叫 `builder.extract_entities_relations`（依賴 LLM）從內文抽出 {"entities": [...], "relations": [...]}。
    2. 將抽出的 JSON 資料呼叫 `builder.populate_graph` 寫入 Neo4j 資料庫。
    3. 每完成一個 Chunk，觸發 `progress_callback`，通知前端目前的進度狀態（如處理成功、找到幾筆實體）。
    
    如果 Chunk 數量超過 `chunk_limit`，則超出的部分會被標記為 'skipped_by_limit'。
    
    參數:
        chunks (List[Chunk]): 待處理的文字區塊陣列。
        uri, user, pwd: 連接 Graph DB 的參數。
        chunk_limit (Optional[int]): 最大處理數量限制。
        extraction_provider, extraction_model: 使用的 LLM 提供商與具體模型。
        progress_callback: 傳遞進度封包的回呼函式。
        
    回傳值:
        Dict[str, Any]: 彙總了總共處理了幾個 Chunk，建立了幾個節點 (entities)、幾條關聯 (relations) 的統計資料。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    KnowledgeGraphBuilder = _load_kg_builder()
    builder = KnowledgeGraphBuilder(uri, user, pwd)
    resolved_chunk_limit = _resolve_chunk_limit(chunk_limit)
    
    # 切出真正要處理的那些 Chunks（受限於 chunk_limit）
    chunks_to_process = chunks[:resolved_chunk_limit] if resolved_chunk_limit else chunks
    
    # 初始化一個代表每個 Chunk 當前處理狀態的清單
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
                # 若超過 limit 則一開始就標為 skipped
                "status": "queued"
                if (not resolved_chunk_limit or index < resolved_chunk_limit)
                else "skipped_by_limit",
                "entities": 0,
                "relations": 0,
            }
        )
        
    # 用來存放總計的圖譜寫入統計數據
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
        """協助推播目前的 Chunk 處理進度給上層的回呼函式 (progress_callback)。
        
        會把當前的整體統計 (total_stats)、單次增加量 (stats_delta) 以及當下 Chunk 的狀態
        包裝成 dict 傳遞出去，就算回傳失敗也不會中斷主流程。
        """
        # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
        # ─── 階段 2：核心處理流程 ─────────────────────────────────
        # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
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
            # 確保推播事件時如果出錯（如連線中斷），不會導致整個圖譜建立流程當機。
            pass

    try:
        # 開始逐個處理 Chunks
        for index, chunk in enumerate(chunks_to_process):
            print(f"Processing chunk {chunk.chunk_id}...")
            chunk_progress[index]["status"] = "processing"
            _emit_chunk_progress(index)
            
            # 階段一：要求 LLM 解析實體與關聯
            try:
                extracted = builder.extract_entities_relations(
                    chunk.text,
                    provider=extraction_provider,
                    model=extraction_model,
                )
                entity_count = len(extracted.get("entities", []))
                relation_count = len(extracted.get("relations", []))

                # 如果有成功抓到資料，就存入摘要陣列中以備稍後統整
                if entity_count or relation_count:
                    extracted_data_summary.append({
                        "chunk_id": chunk.chunk_id,
                        "entities": entity_count,
                        "relations": relation_count,
                    })
                chunk_progress[index]["entities"] = entity_count
                chunk_progress[index]["relations"] = relation_count

                # 階段二：將萃取出的 JSON 資料轉寫入 Neo4j Graph DB 中
                stats = builder.populate_graph(extracted)
            except Exception as exc:
                # 發生任何例外（例如 LLM 逾時），標記為失敗並回報錯誤訊息
                chunk_progress[index]["status"] = "failed"
                chunk_progress[index]["error"] = str(exc)
                _emit_chunk_progress(index)
                raise

            # 統計本次 Chunk 新增或合併的節點與邊數
            stats_delta = {
                "chunks_processed": 1,
                "entities": int(stats.entities),
                "relations": int(stats.relations),
                "merged_entities": int(stats.merged_entities),
                "dropped_relations": int(stats.dropped_relations),
                "json_retries": int(stats.json_retries),
            }

            # 更新整體統計數據
            total_stats["chunks_processed"] += stats_delta["chunks_processed"]
            total_stats["entities"] += stats_delta["entities"]
            total_stats["relations"] += stats_delta["relations"]
            total_stats["merged_entities"] += stats_delta["merged_entities"]
            total_stats["dropped_relations"] += stats_delta["dropped_relations"]
            total_stats["json_retries"] += stats_delta["json_retries"]
            
            # 此 Chunk 處理完成
            chunk_progress[index]["status"] = "processed"
            _emit_chunk_progress(index, stats_delta=stats_delta)

    except Exception as exc:
        print(f"Error building KG: {exc}")
        raise
    finally:
        # 確保在結束時釋放 Neo4j 連線資源
        builder.close()

    # 回傳最終建立結果
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
    """將純文字轉換為知識圖譜 (簡易版)。
    
    這是 `process_text_to_kg` 的簡易版本，不處理 progress callback 狀態。
    它主要用在腳本或是 CLI 這些不需要與前端互動更新進度的情境下。
    
    參數:
        text (str): 準備輸入建構的字串。
        
    回傳值:
        Dict[str, Any]: 處理完的圖譜統計與分析資料。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    # 把它視為 "User Input" 來切割
    chunks = chunk_text(
        text,
        "user_input",
        "User Input",
        extraction_provider=extraction_provider,
    )
    # 直接交給圖譜建立函式
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
    """將一段給定的純文字，切塊後分析寫入知識圖譜。
    
    這個方法通常被後端 API 直接呼叫（例如使用者直接貼上文章內容進行處理時）。
    會給予 "User Input" 的來源標籤，並進行完整的切塊 (chunk_text) 與建圖 (build_kg_from_chunks) 流程。
    
    參數:
        text (str): 使用者輸入的內容。
        uri, user, pwd: Graph DB 的連線參數。
        
    回傳值:
        Dict[str, Any]: 建構圖譜的統計結果。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    # 將整段文字切分為小的 Chunk (以便不超過 LLM 的 Token 上限)
    chunks = chunk_text(
        text.strip(),
        "user_input",
        "User Input",
        extraction_provider=extraction_provider,
    )
    if not chunks:
        raise ValueError("No content found")
        
    # 送交 LLM 並寫入 Neo4j
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
    """從給定的 URL 網址抓取網頁內文，並自動分析寫入知識圖譜。
    
    這是一個方便使用的高階介面。它將 `process_url` (爬蟲+切塊) 和
    `build_kg_from_chunks` (LLM處理+寫入資料庫) 結合在一起。
    
    參數:
        url (str): 預期處理的目標網站網址。
        uri, user, pwd: Graph DB 的連線參數。
        
    回傳值:
        Dict[str, Any]: 建構圖譜的統計結果。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
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
    """將任意格式的網域列表正規化。
    
    這個方法會過濾不合規的格式（例如去除空白、去除 http 首碼、提取 host），
    幫助過濾出搜尋引擎能處理或符合安全原則的網域名稱清單。
    
    參數:
        values: 使用者提供的清單，例如 [" https://www.Foxconn.com/", "wikipedia.org "]。
        
    回傳值:
        List[str]: 乾淨、小寫的 host 名稱列表，如 ["www.foxconn.com", "wikipedia.org"]。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    if not values:
        return []

    domains: List[str] = []
    for value in values:
        normalized = value.strip().lower()
        if not normalized:
            continue
        
        # 若是提供網址格式，使用 urlparse 取得 netloc (即網域部分)
        if normalized.startswith("http://") or normalized.startswith("https://"):
            parsed = urlparse(normalized)
            normalized = parsed.netloc.lower()
            
        # 移除開頭多餘的句點 (例如 .wikipedia.org => wikipedia.org)
        normalized = normalized.lstrip(".")
        if normalized:
            domains.append(normalized)
            
    # 去重並排序
    return sorted(set(domains))


def _is_allowed_domain(url: str, allowlist: List[str]) -> bool:
    """檢查某個爬蟲 URL 的網域是否位於白名單之中。
    
    這個功能避免使用者透過 API 惡意爬取內部 IP 或是系統外的網頁（例如 "localhost", "192.168.1.1"）。
    
    參數:
        url (str): 準備檢查的完整網址，如 "https://zh.wikipedia.org/wiki/Taiwan"
        allowlist (List[str]): 允許的網域，如 ["wikipedia.org"]。
        
    回傳值:
        bool: 符合白名單時為 True（若 allowlist 為空則預設全開，回傳 True）。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    if not allowlist:
        return True

    host = (urlparse(url).netloc or "").lower()
    for domain in allowlist:
        # 當符合完整名稱，或是它是某個 allowlist 網域的子網域 (例如 host 為 zh.wikipedia.org 且 domain 為 wikipedia.org)
        if host == domain or host.endswith(f".{domain}"):
            return True
    return False


def _unwrap_duckduckgo_redirect_url(candidate: str) -> str:
    """解析 DuckDuckGo 搜尋結果中的防追蹤重新導向網址。
    
    當直接去抓 DuckDuckGo 的 HTML 版時，所有的搜尋結果連結會被包成跳轉連結：
    "https://duckduckgo.com/l/?uddg=https://zh.wikipedia.org/..."
    此函式會解開這層包裝，拿出 `uddg` 參數背後真正的目標網址。
    
    參數:
        candidate (str): HTML href 抓到的連結。
        
    回傳值:
        str: 真正的原始網址。如果無法解開，則回傳原本的。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    absolute = urljoin("https://duckduckgo.com", candidate)
    parsed = urlparse(absolute)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        # 取出 query 內的 uddg 參數
        redirect_url = parse_qs(parsed.query).get("uddg", [None])[0]
        if redirect_url:
            return unquote(redirect_url)
    return absolute


def _search_keyword_urls_via_html(keyword: str, max_results: int, language: str) -> List[str]:
    """利用 DuckDuckGo HTML 版模擬網頁請求，以傳統爬蟲抓取搜尋結果網址。
    
    這是一個作為 "DuckDuckGo API" 替代方案的做法（尤其是當 DDGS 庫被封鎖或是遇到 IP 限流時）。
    我們會抓取輕量版的 https://html.duckduckgo.com/html/，然後用 BeautifulSoup 尋找所有 <a> 標籤。
    
    參數:
        keyword (str): 要搜尋的詞彙，如 "人工智慧"。
        max_results (int): 預期回傳的連結數量上限。
        language (str): 搜尋引擎的偏好語系。
        
    回傳值:
        List[str]: 解析並解開跳轉後的網址清單。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    region_map = {"zh-tw": "tw-tzh", "en": "us-en"}
    params = {"q": keyword}
    if language in region_map:
        params["kl"] = region_map[language]

    # 直接對 DDG 的 HTML 搜尋端點發出請求
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

    # 嘗試不同的 CSS 選擇器，這些是 DDG 搜尋結果最常見的 <a> 標籤 class
    for selector in ("a.result__a", "a[data-testid='result-title-a']", "a.result-link"):
        for anchor in soup.select(selector):
            candidate = str(anchor.get("href") or "").strip()
            if not candidate:
                continue
            # 解開 DDG 的追蹤重導向
            candidate = _unwrap_duckduckgo_redirect_url(candidate)
            try:
                normalized = _normalize_http_url(candidate)
            except ValueError:
                continue
                
            if normalized in seen:
                continue
            seen.add(normalized)
            urls.append(normalized)
            
            # 若收集到指定數量則提早結束並回傳
            if len(urls) >= max_results:
                return urls
    return urls


def _search_keyword_urls_via_wikipedia_api(keyword: str, max_results: int, language: str) -> List[str]:
    """使用 Wikipedia 原生 API 搜尋相關條目連結。
    
    這個方法是最穩定且完全不會被阻擋 (封鎖 IP) 的方式，但缺點是它只能找到維基百科的資料，
    所以這通常作為搜尋爬蟲全部失敗時的最後防線 (Fallback)。
    
    參數:
        keyword (str): 想搜尋的條目。
        max_results (int): 條目數限制。
        language (str): 用來決定使用 zh 還是 en 網域的維基百科。
        
    回傳值:
        List[str]: 從 API 組合出的維基條目完整連結清單。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
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
        # API 調用發生錯誤，直接回傳空陣列
        return []

    urls: List[str] = []
    seen = set()
    for item in payload.get("query", {}).get("search", []):
        title = str(item.get("title") or "").strip()
        if not title:
            continue
            
        # 由於條目標題可能包含空白等特殊字元，必須進行 URL Encoding (把空白轉為底線 _ 是維基慣例)
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
    """進行網站查詢取得對應的網址清單 (多重容錯/降級機制)。
    
    由於網路爬蟲 (尤其是搜尋引擎) 經常遇到限流 (Rate Limit) 或驗證碼 (Captcha) 等阻擋，
    這個函式實作了一套循序漸進的 Fallback 機制：
    1. 首先看 `KEYWORD_SEARCH_MODE` 是否設定為 `html_first` 或 `html_only`，是的話先試 HTML 爬蟲。
    2. 若允許使用 `duckduckgo_search` 套件 (DDGS)，則嘗試呼叫該套件取得搜尋結果。
    3. 若 DDGS 也失敗，才嘗試回頭呼叫 HTML 爬取搜尋結果。
    4. 所有搜尋引擎方法都失敗時，使用最穩定的 Wikipedia API 作為最後解答。
    
    參數:
        keyword, max_results, language: 搜尋請求參數。
        
    回傳值:
        List[str]: 第一個成功找到資料的渠道所返回的網址清單。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    region_map = {"zh-tw": "tw-tzh", "en": "us-en"}
    preferred_region = region_map.get(language)
    candidate_regions = [preferred_region, None, "wt-wt"]
    # 保留唯一值，確保不重複測試相同區域
    regions = [value for idx, value in enumerate(candidate_regions) if value not in candidate_regions[:idx]]

    # 1. 優先嘗試純 HTML 爬取 (如果設定為 HTML 優先)
    # 在 docker 中 DDGS 很容易因為 headless request 被限流導致長期阻塞，HTML 模式相對單純。
    if KEYWORD_SEARCH_MODE in {"html_first", "html_only"}:
        html_urls = _search_keyword_urls_via_html(keyword, max_results=max_results, language=language)
        if html_urls:
            return html_urls

    # 2. 嘗試呼叫 duckduckgo_search 套件
    if DDGS is not None and KEYWORD_SEARCH_MODE != "html_only":
        for backend in ("html", "lite", "auto"):
            for region in regions:
                urls: List[str] = []
                seen = set()
                try:
                    with DDGS() as ddgs:
                        results = ddgs.text(
                            keyword,
                            region=region,
                            safesearch="moderate",
                            backend=backend,
                            max_results=max_results,
                        )
                        for item in results:
                            # 找出套件回傳的 url
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
                    # 如果該 backend 失敗，跳過並換下一個
                    continue
                if urls:
                    return urls

    # 3. 再試一次 HTML 爬取（若是之前沒有優先執行過的話）
    html_urls = _search_keyword_urls_via_html(keyword, max_results=max_results, language=language)
    if html_urls:
        return html_urls

    # 4. 終極備用方案: 維基百科 API
    # 當外部搜尋全盤崩潰時，至少還有維基百科條目可抓
    return _search_keyword_urls_via_wikipedia_api(keyword, max_results=max_results, language=language)


def _search_keyword_urls_resilient(keyword: str, max_results: int, language: str) -> List[str]:
    """擴展關鍵字容錯機制的搜尋網址函式。
    
    有時單純的關鍵字在某些搜尋引擎會搜不到東西，此函式會透過擴充多組同義或相關的關鍵字
    (例如在後面加上 "新聞" 或 "官網")，並進行重試。
    
    例如輸入 "鴻海"，會依序測試：
    1. "鴻海"
    2. "鴻海 新聞"
    3. "鴻海 官網"
    
    只要其中一項拿到了滿足 `max_results` 的連結數量就會立刻中斷並回傳。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    search_terms = [keyword]
    max_attempts = 3
    if KEYWORD_SEARCH_MODE == "html_only":
        # HTML 爬取通常沒有重試的必要，為了維持快速直接設為 1 次。
        max_attempts = 1
    elif language == "zh-tw":
        # 對中文增加輔助字
        search_terms.extend([f"{keyword} 新聞", f"{keyword} 官網"])
    else:
        search_terms.extend([f"{keyword} news", f"{keyword} official website"])

    urls: List[str] = []
    seen = set()

    # 外層控制嘗試次數 (例如最多 3 次嘗試，以防短暫的 API 錯誤)
    for attempt in range(max_attempts):
        # 內層逐一測試不同的關鍵字組合
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
                # 湊滿要求數量，提早收工
                if len(urls) >= max_results:
                    return urls

        if urls:
            # 有找到結果即刻回傳
            return urls
        # 如果這一輪任何組合都沒搜到，可能是搜尋引擎封鎖，休息一秒後重試
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
    """完整實作「利用關鍵字建立知識圖譜」的高階商務邏輯。
    
    這是系統提供最複雜的功能之一，其流程包含：
    1. 前置作業：驗證關鍵字與 max_results 數量。
    2. 搜尋階段：呼叫 `_search_keyword_urls_resilient` 去網路上尋找符合 `keyword` 的網頁，
       並透過 `site_allowlist` 進行黑白名單過濾，選定要爬取的 `target_urls`。
    3. 圖譜建立階段：對每個 `url` 發起爬取、切割 Chunk，再將它們逐一交由 LLM 轉成圖譜。
    4. 進度回報與資料彙整：在每一篇文章、每一個 Chunk 處理前後，都會觸發回報事件通知前端。
       將所有結果加總至 `aggregated_stats` 等彙總欄位。
       
    參數:
        keyword (str): 使用者輸入的關鍵字 (例如 "鴻海科技集團")。
        uri, user, pwd: Graph DB 連接認證。
        max_results (int): 最多爬取幾個網址。
        language (str): 限定搜尋語系。
        site_allowlist: 限制只爬取哪些網址的清單。
        chunk_limit: 所有網頁加總起來最多處理的 Chunk 數限制。
        
    回傳值:
        Dict[str, Any]: 總結性的處理報表，包含總計成功數、失敗網址、各 Chunk 處理狀態等。
    """
    # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
    # ─── 階段 2：核心處理流程 ─────────────────────────────────
    # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
    cleaned_keyword = keyword.strip()
    if len(cleaned_keyword) < 2:
        raise ValueError("Keyword must contain at least 2 characters")
    if max_results < 1 or max_results > 10:
        raise ValueError("max_results must be between 1 and 10")
    resolved_chunk_limit = _resolve_chunk_limit(chunk_limit)

    # 1. 執行搜尋並過濾 URL
    searched_urls = _search_keyword_urls_resilient(cleaned_keyword, max_results=max_results, language=language)
    allowlist = _to_domain_allowlist(site_allowlist)
    target_urls = [url for url in searched_urls if _is_allowed_domain(url, allowlist)]

    if not target_urls:
        raise ValueError("No results found for the keyword")

    # 彙總用統計變數
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
        """內嵌進度回報函式：通知外部目前的「搜尋與建圖進度」。"""
        # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
        # ─── 階段 2：核心處理流程 ─────────────────────────────────
        # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
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
            pass

    # 通知前端開始運行
    _emit_keyword_progress(status="running")

    # 2. 針對每個找到的 URL，開始爬取與建圖
    for url in target_urls:
        # 用一個 map 來紀錄各 chunk 對應到 aggregated_chunk_progress 陣列的索引，加速後續狀態更新
        row_idx_by_chunk_id: Dict[str, int] = {}
        try:
            _emit_keyword_progress(status="running", current_url=url)
            # 取得該網頁的所有段落
            chunks = process_url(url, extraction_provider=extraction_provider)
            if not chunks:
                raise ValueError("No extractable content")

            # 準備初始的 Chunk 進度列
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
                """攔截 `build_kg_from_chunks` 底層傳來的 Chunk 更新事件，將它整合到外層的彙總報表中。"""
                # ─── 階段 1：輸入正規化與前置檢查 ─────────────────────────
                # ─── 階段 2：核心處理流程 ─────────────────────────────────
                # ─── 階段 3：整理回傳與錯誤傳遞 ───────────────────────────
                if event.get("type") != "chunk_update":
                    return
                chunk_row = event.get("chunk")
                if not isinstance(chunk_row, dict):
                    return
                chunk_id = str(chunk_row.get("chunk_id", ""))
                row_idx = row_idx_by_chunk_id.get(chunk_id)
                if row_idx is None:
                    return
                # 更新在匯總表格裡的狀態
                row = aggregated_chunk_progress[row_idx]
                row["status"] = str(chunk_row.get("status", row["status"]))
                row["entities"] = int(chunk_row.get("entities", row["entities"]))
                row["relations"] = int(chunk_row.get("relations", row["relations"]))
                if chunk_row.get("error"):
                    row["error"] = str(chunk_row["error"])
                # 將狀態回傳給前端
                _emit_keyword_progress(status="running", current_url=url)

            build_kwargs: Dict[str, Any] = {
                "chunk_limit": resolved_chunk_limit,
                "extraction_provider": extraction_provider,
                "extraction_model": extraction_model,
            }
            if progress_callback:
                build_kwargs["progress_callback"] = _on_build_chunk_progress

            # 開始實質的圖譜建構呼叫
            result = build_kg_from_chunks(
                chunks,
                uri,
                user,
                pwd,
                **build_kwargs,
            )
            fetched_urls.append(url)

            # 將這一個 URL 處理完的統計資料，累積到整體的 aggregated_stats 中
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
            # 單一網址處理失敗時，將尚未完成的 Chunk 標記為失敗
            for row_idx in row_idx_by_chunk_id.values():
                row = aggregated_chunk_progress[row_idx]
                if row.get("status") in {"queued", "processing"}:
                    row["status"] = "failed"
                    row["error"] = str(exc)
            failed_urls.append({"url": url, "error": str(exc)})
            _emit_keyword_progress(status="running", current_url=url, error=str(exc))

    # 3. 所有的 URL 都已經迭代完畢，檢查是否全都失敗了
    if not fetched_urls:
        first_reason = failed_urls[0]["error"] if failed_urls else "unknown error"
        error_msg = f"All crawled pages failed to process: {first_reason}"
        _emit_keyword_progress(status="failed", error=error_msg)
        raise ValueError(error_msg)

    # 4. 準備最終報表回傳
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
