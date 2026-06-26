"""Модели данных приложения.

Все dataclass-ы и type alias-ы собраны в одном месте.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Category(StrEnum):
    FAQ = "faq"
    PROBLEM = "problem"


@dataclass(slots=True)
class SessionStats:
    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_tokens: int = 0
    llm_calls: int = 0


@dataclass(slots=True)
class LLMResult:
    text: str
    tokens: int
    provider: str
    model: str
    used_fallback: bool


@dataclass(slots=True)
class AssistantResponse:
    text: str
    category: Category
    from_cache: bool
    latency_seconds: float
    provider: str
    model: str
    used_fallback: bool