"""Конфигурация приложения.

Загружает настройки из переменных окружения (.env).
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

if os.getenv("HTTP_PROXY"):
    os.environ["HTTP_PROXY"] = os.getenv("HTTP_PROXY", "")
if os.getenv("HTTPS_PROXY"):
    os.environ["HTTPS_PROXY"] = os.getenv("HTTPS_PROXY", "")
os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0,localhost:11434,openrouter.ai,openrouter.ai/api/v1"

@dataclass(slots=True)
class Settings:
    service_name: str
    # Основной провайдер (OpenAI-совместимый API)
    api_key: str | None
    base_url: str | None
    primary_model: str
    classifier_model: str
    
    # NEW: OpenRouter (Резервный)
    openrouter_api_key: str | None
    openrouter_base_url: str | None
    openrouter_model: str | None
    
    # Fallback-провайдер (OpenAI-совместимый API)
    fallback_api_key: str | None
    fallback_base_url: str | None
    fallback_model: str | None
    # Общие настройки
    request_timeout_seconds: float
    retry_attempts: int
    history_limit: int
    log_path: Path

    @classmethod
    def from_env(cls) -> "Settings":
        base_dir = Path(__file__).resolve().parent.parent
        return cls(
            service_name=os.getenv("BAG_SERVICE_NAME", "EVA"),
            api_key=os.getenv("API_KEY") or "not_needed_for_this_proxy",
            base_url=os.getenv("OPENAI_BASE_URL"),
            primary_model=os.getenv("BAG_PRIMARY_MODEL", "gpt-4o-mini"),
            classifier_model=os.getenv("BAG_CLASSIFIER_MODEL", "gpt-4o-mini"),
            fallback_api_key=os.getenv("FALLBACK_API_KEY", "ollama"),
            fallback_base_url=os.getenv("FALLBACK_BASE_URL", "http://localhost:11434/v1"),
            openrouter_model=os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free"), # gpt-oss-20b:free
            openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            fallback_model=os.getenv("FALLBACK_MODEL", "qwen3:1.7b"),
            request_timeout_seconds=float(os.getenv("BAG_TIMEOUT_SECONDS", "30")),
            retry_attempts=int(os.getenv("BAG_RETRY_ATTEMPTS", "3")),
            history_limit=int(os.getenv("BAG_HISTORY_LIMIT", "10")),
            log_path=Path(os.getenv("BAG_LOG_PATH", base_dir / "assistant.log")),
        )