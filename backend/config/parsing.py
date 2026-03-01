"""Shared environment parsing helpers."""

from __future__ import annotations

import os
from typing import Optional


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    """從環境變數讀取字串值，若不存在回傳預設值。"""
    return os.getenv(name, default)


def get_env_str(name: str, default: str = "") -> str:
    """從環境變數讀取字串值，值為 None 時回傳預設值。"""
    value = os.getenv(name)
    if value is None:
        return default
    return str(value)


def get_env_int(name: str, default: int) -> int:
    """從環境變數讀取整數值，格式錯誤時回傳預設值。"""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_env_float(name: str, default: float) -> float:
    """從環境變數讀取浮點數值，格式錯誤時回傳預設值。"""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_env_bool(name: str, default: bool) -> bool:
    """從環境變數讀取布林值，支援 1/true/yes/on（真）與 0/false/no/off（假）。"""
    value = os.getenv(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default
