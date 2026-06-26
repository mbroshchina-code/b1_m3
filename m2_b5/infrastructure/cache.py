import os
import time
import logging
import json
import hashlib
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class LLMCache:
    """In-memory кэш для ответов LLM с поддержкой TTL на SHA-256 и статистикой."""

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._cache: dict[str, tuple[str, float]] = {}
        self.ttl = ttl_seconds 
        self.hits = 0
        self.misses = 0
        self.cache_file = "cache.json"  # Имя файла, где будет храниться копия кэша
        self._load_cache()               # Автоматически загружаем кэш при старте программы
        
    def _load_cache(self) -> None:
        """Загружает кеш из локального файла, если он существует."""
        if hasattr(self, "cache_file") and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                logger.info(f"[Cache] Загружено {len(self._cache)} записей из кеша.")
            except Exception as e:
                logger.warning(f"[Cache] Не удалось прочитать файл кеша: {e}")

    def _make_key(
        self, model: str, messages: list[dict], temperature: float = 0
    ) -> str:
        """Ключ = хеш(модель + параметры + промпт)."""
        data = json.dumps(
            {"model": model, "messages": messages, "temperature": temperature},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(data.encode()).hexdigest()

    def get(self, model: str, messages: list[dict], temperature: float = 0) -> str | None:
        # ИСПРАВЛЕНО: добавляем генерацию ключа, чтобы избежать NameError
        key = self._make_key(model, messages, temperature)
        # print(f"   [DEBUG CACHE GET] Ключ: {key}") 
        
        if key in self._cache:
            value, created_at = self._cache[key]
            if time.time() - created_at < self.ttl:
                self.hits += 1
                logger.info("   [LLMCache] !!! HIT !!! Ответ успешно взят из памяти.")
                return value
            # TTL истёк — удаляем запись, чтобы не засорять RAM
            logger.info("   [LLMCache] Запись найдена, но просрочена по TTL. Удаляю.")
            del self._cache[key]  # TTL истёк
            
        self.misses += 1
        logger.info("   [LLMCache] !!! MISS !!! Запись отсутствует в памяти.")
        return None

    def set(self, model: str, messages: list[dict], temperature: float, response: str) -> None:
        key = self._make_key(model, messages, temperature)
        self._cache[key] = (response, time.time())
        # print(f"   [DEBUG CACHE SET] Ключ: {key}") 
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.warning(f"[Cache] Не удалось сохранить кэш в файл: {e}")
            
    def clear(self) -> int:  # ИСПРАВЛЕНО: теперь возвращает количество удаленных записей
        """Полная очистка кэша и сброс метрик."""
        count = len(self._cache)
        self._cache.clear()
        self.hits = 0
        self.misses = 0
        logger.info("   [LLMCache] Память кэша полностью очищена.")
        return count
    def reset_stats(self) -> None:
        """Сбрасывает счетчики попаданий и промахов."""
        self.hits = 0
        self.misses = 0
        logger.info("   [LLMCache] Статистика кэша сброшена.")

    def stats(self) -> dict[str, Any]:
        """Возвращает структурированную статистику для команды /stats."""
        return {
            "keys": len(self._cache),
            "hit_rate": f"{self.hit_rate:.1f}%",
            "hits": self.hits,
            "misses": self.misses
        }
        
    @property
    def hit_rate(self) -> float:
        """Вычисляет эффективность кэша в процентах."""
        total = self.hits + self.misses
        return self.hits / total * 100 if total > 0 else 0.0