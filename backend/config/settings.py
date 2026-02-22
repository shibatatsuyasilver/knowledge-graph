"""Centralized runtime settings loader for backend and llm_kg modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .parsing import get_env, get_env_bool, get_env_float, get_env_int, get_env_str

DEFAULT_MODEL = "mlx-community/Qwen3-8B-4bit-DWQ-053125"
DEFAULT_PROVIDER = "openai"
DEFAULT_OPENAI_BASE_URL = "http://localhost:8080/v1"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_GEMINI_MODEL = "gemini-3-pro-preview"
DEFAULT_GEMINI_INPUT_TOKEN_LIMIT = 1_048_576
DEFAULT_GEMINI_OUTPUT_TOKEN_LIMIT = 65_536
DEFAULT_ERROR_DETAIL_MAX_CHARS = 4000


@dataclass(frozen=True)
class Neo4jSettings:
    uri: str
    user: str
    password: str


@dataclass(frozen=True)
class JobTTLSettings:
    keyword_job_ttl_seconds: int
    ingest_job_ttl_seconds: int


@dataclass(frozen=True)
class IngestChunkSettings:
    keyword_search_mode: str
    default_chunk_limit: int
    chunk_size_mode: str
    default_token_chunk_size: int
    default_token_chunk_min_size: int
    default_char_chunk_size: int
    default_char_chunk_min_size: int


@dataclass(frozen=True)
class GeneralChatSettings:
    timeout_seconds: float
    num_predict: int
    temperature: float


@dataclass(frozen=True)
class KgQASettings:
    use_llm: bool
    model: Optional[str]
    temperature: float
    max_tokens: int
    max_rows_for_prompt: int


@dataclass(frozen=True)
class LLMRuntimeSettings:
    provider: str
    model: str
    timeout_seconds: float
    temperature: float
    max_tokens: int
    openai_base_url: str
    openai_api_key: str
    ollama_base_url: str
    gemini_base_url: str
    gemini_api_key: str
    gemini_input_token_limit: int
    gemini_output_token_limit: int
    ollama_think: bool
    ollama_think_json: bool


@dataclass(frozen=True)
class KgBuilderSettings:
    extraction_timeout_seconds: int
    extraction_max_json_retries: int
    entity_resolve_threshold: float
    extraction_json_mode: str
    extraction_error_raw_max_chars: int
    gemini_two_pass_extraction: bool


@dataclass(frozen=True)
class Nl2CypherSettings:
    cypher_repair_retries: int
    entity_link_threshold: float
    timeout_seconds: int
    num_predict: int
    agentic_max_rounds: int
    agentic_plan_tokens: int
    agentic_react_tokens: int
    agentic_critic_tokens: int


@dataclass(frozen=True)
class LlmMiscSettings:
    error_detail_max_chars: int
    think_log_enabled: bool
    think_log_path: str


def get_neo4j_settings() -> Neo4jSettings:
    """執行 `get_neo4j_settings` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
    return Neo4jSettings(
        uri=get_env_str("NEO4J_URI", "bolt://localhost:7687"),
        user=get_env_str("NEO4J_USER", "neo4j"),
        password=get_env_str("NEO4J_PASSWORD", "password"),
    )


def get_job_ttl_settings() -> JobTTLSettings:
    """執行 `get_job_ttl_settings` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
    return JobTTLSettings(
        keyword_job_ttl_seconds=max(1, get_env_int("KEYWORD_JOB_TTL_SECONDS", 3600)),
        ingest_job_ttl_seconds=max(1, get_env_int("INGEST_JOB_TTL_SECONDS", 3600)),
    )


def get_ingest_chunk_settings() -> IngestChunkSettings:
    """執行 `get_ingest_chunk_settings` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
    token_chunk_size_default = get_env_int(
        "CHUNK_SIZE_TOKENS",
        get_env_int("GEMINI_INPUT_TOKEN_LIMIT", DEFAULT_GEMINI_INPUT_TOKEN_LIMIT),
    )
    return IngestChunkSettings(
        keyword_search_mode=get_env_str("KEYWORD_SEARCH_MODE", "html_only").strip().lower(),
        default_chunk_limit=get_env_int("INGEST_CHUNK_LIMIT", 0),
        chunk_size_mode=get_env_str("CHUNK_SIZE_MODE", "provider").strip().lower(),
        default_token_chunk_size=max(1, token_chunk_size_default),
        default_token_chunk_min_size=max(1, get_env_int("CHUNK_MIN_TOKENS", 120)),
        default_char_chunk_size=max(1, get_env_int("CHUNK_SIZE_CHARS", 900)),
        default_char_chunk_min_size=max(1, get_env_int("CHUNK_MIN_CHARS", 120)),
    )


def get_general_chat_settings() -> GeneralChatSettings:
    """執行 `get_general_chat_settings` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
    return GeneralChatSettings(
        timeout_seconds=max(
            1.0,
            get_env_float(
                "GENERAL_CHAT_TIMEOUT_SECONDS",
                get_env_float("LLM_TIMEOUT_SECONDS", 120.0),
            ),
        ),
        num_predict=max(
            1,
            get_env_int(
                "GENERAL_CHAT_NUM_PREDICT",
                get_env_int("LLM_MAX_TOKENS", 512),
            ),
        ),
        temperature=get_env_float(
            "GENERAL_CHAT_TEMPERATURE",
            get_env_float("LLM_TEMPERATURE", 0.7),
        ),
    )


def _env_bool(name: str, default: bool) -> bool:
    """執行 `_env_bool` 的內部輔助流程。
    此函式封裝局部邏輯以提升可讀性，並維持既有輸入輸出與邊界行為。
    """
    return get_env_bool(name, default)


def get_kg_qa_settings() -> KgQASettings:
    """執行 `get_kg_qa_settings` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
    model = get_env_str("KG_QA_MODEL", "").strip() or None
    return KgQASettings(
        use_llm=_env_bool("KG_QA_USE_LLM", True),
        model=model,
        temperature=get_env_float("KG_QA_TEMPERATURE", 0.1),
        max_tokens=max(1, get_env_int("KG_QA_MAX_TOKENS", 1024)),
        max_rows_for_prompt=max(1, get_env_int("KG_QA_MAX_ROWS_FOR_PROMPT", 20)),
    )


def resolve_llm_provider() -> str:
    """執行 `resolve_llm_provider` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
    explicit = get_env_str("LLM_PROVIDER", "").strip().lower()
    if explicit in {"openai", "ollama", "gemini"}:
        return explicit
    if explicit:
        raise ValueError(f"Unsupported LLM_PROVIDER: {explicit}")

    # Backward compatibility: if legacy Ollama env exists, use Ollama.
    if get_env("OLLAMA_BASE_URL"):
        return "ollama"
    return DEFAULT_PROVIDER


def resolve_llm_model(provider: str) -> str:
    """執行 `resolve_llm_model` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
    if provider == "ollama":
        return get_env_str("OLLAMA_MODEL", "") or get_env_str("LLM_MODEL", "") or DEFAULT_MODEL
    if provider == "gemini":
        return get_env_str("GEMINI_MODEL", "") or get_env_str("LLM_MODEL", "") or DEFAULT_GEMINI_MODEL
    return get_env_str("LLM_MODEL", "") or get_env_str("OLLAMA_MODEL", "") or DEFAULT_MODEL


def get_llm_runtime_settings() -> LLMRuntimeSettings:
    """執行 `get_llm_runtime_settings` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
    provider = resolve_llm_provider()
    model = resolve_llm_model(provider)

    timeout_seconds = get_env_float(
        "LLM_TIMEOUT_SECONDS",
        get_env_float("OLLAMA_TIMEOUT_SECONDS", 180.0),
    )
    temperature = get_env_float(
        "LLM_TEMPERATURE",
        get_env_float("OLLAMA_TEMPERATURE", 0.1),
    )
    gemini_input_token_limit = get_env_int(
        "GEMINI_INPUT_TOKEN_LIMIT",
        DEFAULT_GEMINI_INPUT_TOKEN_LIMIT,
    )
    gemini_output_token_limit = get_env_int(
        "GEMINI_OUTPUT_TOKEN_LIMIT",
        DEFAULT_GEMINI_OUTPUT_TOKEN_LIMIT,
    )
    max_tokens_env = get_env("LLM_MAX_TOKENS") or get_env("OLLAMA_NUM_PREDICT")
    if max_tokens_env is None and provider == "gemini":
        max_tokens = gemini_output_token_limit
    else:
        try:
            max_tokens = int(max_tokens_env) if max_tokens_env is not None else 512
        except (TypeError, ValueError):
            max_tokens = 512

    ollama_think = get_env_bool("OLLAMA_THINK", False)
    ollama_think_json = get_env_bool("OLLAMA_THINK_JSON", ollama_think)

    return LLMRuntimeSettings(
        provider=provider,
        model=model,
        timeout_seconds=max(1.0, timeout_seconds),
        temperature=temperature,
        max_tokens=max(1, max_tokens),
        openai_base_url=get_env_str("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL).rstrip("/"),
        openai_api_key=get_env_str("OPENAI_API_KEY", "").strip(),
        ollama_base_url=get_env_str("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).rstrip("/"),
        gemini_base_url=get_env_str("GEMINI_BASE_URL", DEFAULT_GEMINI_BASE_URL).rstrip("/"),
        gemini_api_key=get_env_str("GEMINI_API_KEY", "").strip(),
        gemini_input_token_limit=max(1, gemini_input_token_limit),
        gemini_output_token_limit=max(1, gemini_output_token_limit),
        ollama_think=ollama_think,
        ollama_think_json=ollama_think_json,
    )


def get_kg_builder_settings() -> KgBuilderSettings:
    """執行 `get_kg_builder_settings` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
    return KgBuilderSettings(
        extraction_timeout_seconds=max(1, get_env_int("EXTRACTION_TIMEOUT_SECONDS", 60)),
        extraction_max_json_retries=max(0, get_env_int("EXTRACTION_MAX_JSON_RETRIES", 4)),
        entity_resolve_threshold=get_env_float("ENTITY_RESOLVE_THRESHOLD", 0.9),
        extraction_json_mode=get_env_str("EXTRACTION_JSON_MODE", "auto").strip().lower(),
        extraction_error_raw_max_chars=max(0, get_env_int("EXTRACTION_ERROR_RAW_MAX_CHARS", 4000)),
        gemini_two_pass_extraction=get_env_bool("GEMINI_TWO_PASS_EXTRACTION", True),
    )


def get_nl2cypher_settings() -> Nl2CypherSettings:
    """執行 `get_nl2cypher_settings` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
    base_tokens = max(
        1,
        get_env_int(
            "NL2CYPHER_NUM_PREDICT",
            get_env_int("LLM_MAX_TOKENS", 1024),
        ),
    )
    return Nl2CypherSettings(
        cypher_repair_retries=max(0, get_env_int("CYPHER_REPAIR_RETRIES", 2)),
        entity_link_threshold=get_env_float("ENTITY_LINK_THRESHOLD", 0.82),
        timeout_seconds=max(1, get_env_int("NL2CYPHER_TIMEOUT_SECONDS", 180)),
        num_predict=base_tokens,
        agentic_max_rounds=max(1, get_env_int("NL2CYPHER_AGENTIC_MAX_ROUNDS", 5)),
        agentic_plan_tokens=max(1, get_env_int("NL2CYPHER_AGENTIC_PLAN_TOKENS", min(1024, base_tokens))),
        agentic_react_tokens=max(1, get_env_int("NL2CYPHER_AGENTIC_REACT_TOKENS", base_tokens)),
        agentic_critic_tokens=max(1, get_env_int("NL2CYPHER_AGENTIC_CRITIC_TOKENS", min(1024, base_tokens))),
    )


def get_llm_misc_settings() -> LlmMiscSettings:
    """執行 `get_llm_misc_settings` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
    think_log_enabled = get_env_bool("LLM_THINK_LOG_ENABLED", get_env_bool("OLLAMA_THINK_LOG_ENABLED", False))
    think_log_path = (
        get_env("LLM_THINK_LOG_PATH")
        or get_env("OLLAMA_THINK_LOG_PATH")
        or "/tmp/ollama_think.log"
    )
    return LlmMiscSettings(
        error_detail_max_chars=max(0, get_env_int("LLM_ERROR_DETAIL_MAX_CHARS", DEFAULT_ERROR_DETAIL_MAX_CHARS)),
        think_log_enabled=think_log_enabled,
        think_log_path=think_log_path,
    )


def resolve_extraction_model(provider: Optional[str], explicit_model: Optional[str]) -> Optional[str]:
    """執行 `resolve_extraction_model` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
    explicit = (explicit_model or "").strip()
    if explicit:
        return explicit

    resolved_provider = (provider or get_env_str("EXTRACTION_PROVIDER", "")).strip().lower()
    if resolved_provider == "gemini":
        return get_env_str("GEMINI_MODEL", "").strip() or DEFAULT_GEMINI_MODEL

    value = get_env_str("EXTRACTION_MODEL", "").strip()
    return value or None


def resolve_nl2cypher_provider(provider: Optional[str]) -> Optional[str]:
    """執行 `resolve_nl2cypher_provider` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
    value = (provider or get_env_str("NL2CYPHER_PROVIDER", "")).strip().lower()
    if not value:
        return None
    if value not in {"openai", "ollama", "gemini"}:
        raise ValueError(f"Unsupported NL2CYPHER provider: {value}")
    return value


def resolve_nl2cypher_model(provider: Optional[str] = None, explicit_model: Optional[str] = None) -> Optional[str]:
    """執行 `resolve_nl2cypher_model` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
    explicit = (explicit_model or "").strip()
    if explicit:
        return explicit

    value = get_env_str("NL2CYPHER_MODEL", "").strip()
    if value:
        return value

    resolved_provider = resolve_nl2cypher_provider(provider)
    if resolved_provider == "gemini":
        return get_env_str("GEMINI_MODEL", "").strip() or DEFAULT_GEMINI_MODEL
    return None


def resolve_extraction_provider(provider: Optional[str]) -> Optional[str]:
    """執行 `resolve_extraction_provider` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
    value = (provider or get_env_str("EXTRACTION_PROVIDER", "")).strip().lower()
    if not value:
        return None
    if value not in {"openai", "ollama", "gemini"}:
        raise ValueError(f"Unsupported extraction provider: {value}")
    return value


def resolve_extraction_num_predict(provider: Optional[str], gemini_output_default: int) -> int:
    """執行 `resolve_extraction_num_predict` 的主要流程。
    函式會依參數完成資料處理並回傳結果，必要時沿用目前例外處理機制。
    """
    resolved_provider = resolve_extraction_provider(provider)
    if resolved_provider == "gemini":
        return max(1, get_env_int("GEMINI_OUTPUT_TOKEN_LIMIT", gemini_output_default))

    explicit = get_env("EXTRACTION_NUM_PREDICT")
    if explicit:
        try:
            return max(1, int(explicit))
        except (TypeError, ValueError):
            pass
    return max(1, get_env_int("LLM_MAX_TOKENS", 1024))
