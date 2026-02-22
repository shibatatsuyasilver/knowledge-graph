"""Request/response models for backend routers."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class UrlRequest(BaseModel):
    url: str
    chunk_limit: Optional[int] = None
    extraction_provider: Optional[Literal["ollama", "gemini"]] = None
    extraction_model: Optional[str] = None


class TextRequest(BaseModel):
    text: str
    chunk_limit: Optional[int] = None
    extraction_provider: Optional[Literal["ollama", "gemini"]] = None
    extraction_model: Optional[str] = None


class QueryRequest(BaseModel):
    question: str


class KeywordRequest(BaseModel):
    keyword: str
    max_results: int = 5
    language: Literal["zh-tw", "en"] = "zh-tw"
    site_allowlist: Optional[List[str]] = None
    chunk_limit: Optional[int] = None
    extraction_provider: Optional[Literal["ollama", "gemini"]] = None
    extraction_model: Optional[str] = None


class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class GeneralChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatHistoryMessage]] = None


class LlmCompatChatRequest(BaseModel):
    question: str = Field(min_length=1, description="User question")
    model: Optional[str] = Field(default=None, description="Override model name")


class LlmCompatChatResponse(BaseModel):
    answer: str
    model: str


class JobStateResponse(BaseModel):
    job_id: str
    status: str
    progress: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
