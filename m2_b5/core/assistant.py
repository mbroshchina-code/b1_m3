"""Главный модуль бизнес-логики ассистента поддержки.

Класс ``SupportAssistantApp`` оркестрирует весь цикл обработки запроса:
классификация → проверка кеша → вызов LLM (с retry/fallback) → ведение истории → логирование.
"""

from __future__ import annotations
import logging
import time
from loguru import logger

from m2_b5.config import Settings
from m2_b5.models import AssistantResponse, SessionStats
from m2_b5.infrastructure.cache import LLMCache
from m2_b5.infrastructure.llm import FALLBACK_ANSWER, FAQ_ANSWER, RobustLLMClient
from m2_b5.prompts.loader import build_answer_messages, build_classifier_messages, build_system_prompt


class SupportAssistantApp:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.system_prompt = build_system_prompt(settings.service_name)
        self.history: list[dict[str, str]] = []
        self.failed_attempts = 0
        self.stats = SessionStats()
        self.cache = LLMCache()  # Локальный in-memory кеш
        self.client = RobustLLMClient(settings)
        # 1. Удаляем стандартный вывод loguru в терминал
        logger.remove()
        # 2. Настраиваем запись логов строго в файл
        logger.add(
            settings.log_path, 
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", 
            rotation="10 MB",
            encoding="utf-8"
        )
        # 3. Перехватчик для стандартного модуля logging (httpx, openai)
        class InterceptHandler(logging.Handler):
            def emit(self, record):
                try:
                    level = logger.level(record.levelname).name
                except ValueError:
                    level = record.levelno
                frame, depth = logging.currentframe(), 2
                while frame.f_code.co_filename == logging.__file__:
                    frame = frame.f_back
                    depth += 1
                logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

        # Настраиваем корневой логгер Python на наш перехватчик
        logging.root.handlers = [InterceptHandler()]
        logging.root.setLevel(logging.INFO)

        # Принудительно убираем вывод сетевых библиотек из консоли в файл
        logging.getLogger("httpx").handlers = [InterceptHandler()]
        logging.getLogger("httpx").setLevel(logging.INFO)
        logging.getLogger("openai").handlers = [InterceptHandler()]
        logging.getLogger("openai").setLevel(logging.INFO)
        # ───────────────────────────────────────────────────────
        
    def handle_command(self, command: str) -> str | None:
        if command == "/clear":
            self.history.clear()
            self.failed_attempts = 0
            return "История очищена."
            
        if command == "/clear_cache":
            deleted = self.cache.clear()
            return f"Локальный кеш очищен. Удалено записей: {deleted}."
            
        if command == "/reset_stats":
            self.cache.reset_stats()
            return "Статистика локального кеша сброшена."
            
        if command == "/stats":
            cache_info = self.cache.stats()
            
            return (
                f"Запросов: {self.stats.total_queries} | "
                f"LLM вызовов: {self.stats.llm_calls} | "
                f"Токенов: {self.stats.total_tokens} | "
                f"Кеш (память): {cache_info['keys']} ключей, "
                f"hit rate: {cache_info['hit_rate']} "
                f"({cache_info['hits']}/{cache_info['hits'] + cache_info['misses']})"
            )
            
        if command == "/quit":
            return None
        return "Доступные команды: /clear, /clear_cache, /reset_stats, /stats, /quit"

    def respond(self, user_message: str, image_path: str | None = None) -> AssistantResponse:
        started_at = time.perf_counter()
        self.stats.total_queries += 1

        # 1. Классифицируем запрос
        category = self.client.classify(build_classifier_messages(user_message))
        
        # --- ПРОВЕРКА НА FAQ ---
        # Если категория FAQ — мгновенно прерываем цепочку и отдаем вашу заглушку
        if category == "faq" or (hasattr(category, "value") and category.value == "faq"):
            latency = time.perf_counter() - started_at
            self._remember_turn(user_message, FAQ_ANSWER)
            self._log(user_message, str(category), FAQ_ANSWER, 0, latency, False, "stub", "faq_stub")
            return AssistantResponse(
                text=FAQ_ANSWER,
                category=category,
                from_cache=False,
                latency_seconds=latency,
                provider="stub",
                model="faq_stub",
                used_fallback=False
            )
            
        # 2. Подготовка сообщений диалога, если запрос относится к категории problem
        messages = build_answer_messages(self.system_prompt, self.history, user_message, image_path)
        model_name = self.settings.primary_model
        
        # --- ПРОВЕРКА КЭША (Пропускаем, если отправлено изображение) ---
        if not image_path:
            try:
                cached = self.cache.get(model=model_name, messages=messages, temperature=0.2)
                if cached is not None:
                    self.stats.cache_hits += 1
                    self._remember_turn(user_message, cached)
                    latency = time.perf_counter() - started_at
                    self._log(user_message, str(category), cached, 0, latency, True, "cache", "cache")
                    return AssistantResponse(cached, category, True, latency, "cache", "cache", False)
            except Exception as e:
                logger.warning(f"Ошибка при чтении из кэша: {e}")
        else:
            logger.info("Обнаружено изображение в запросе. Пропускаю проверку кэша.")

        # 3. ИЗОЛИРОВАННЫЙ ВЫЗОВ СИСТЕМЫ LLM 
        self.stats.cache_misses += 1
        self.stats.llm_calls += 1
        
        # ИСПРАВЛЕНО: Передаем строку категории, как требует обновленный llm.py для tool_calling
        result = self.client.answer(messages, category=str(category))
        
        # --- СОХРАНЕНИЕ В КЭШ (Записываем только текстовые ответы без картинок) ---
        if not image_path:
            try:
                self.cache.set(model=model_name, messages=messages, temperature=0.2, response=result.text)
            except Exception as e:
                logger.warning(f"Ошибка при записи в кэш: {e}")
                
        # 4. Обработка результатов, подсчет неудачных попыток и токенизация
        if result.text.strip() == FALLBACK_ANSWER:
            self.failed_attempts += 1
        else:
            self.failed_attempts = 0

        latency = time.perf_counter() - started_at
        self._remember_turn(user_message, result.text)
        self.stats.total_tokens += result.tokens
        
        self._log(
            user_message, str(category), result.text,
            result.tokens, latency, False, result.provider, result.model,
        )
        return AssistantResponse(
            result.text, category, False, latency,
            result.provider, result.model, result.used_fallback,
        )

    def _remember_turn(self, user_message: str, answer: str) -> None:
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": answer})
        if len(self.history) > self.settings.history_limit:
            self.history = self.history[-self.settings.history_limit :]

    def _log(
        self,
        user_message: str,
        category: str,
        answer: str,
        tokens: int,
        latency_seconds: float,
        from_cache: bool,
        provider: str,
        model: str,
    ) -> None:
        logger.info(
            "{cat} | {prov}/{mod} | {tok} tok | {lat:.3f}s | cache={cache} | Q: {msg} | A: {ans}",
            cat=category,
            prov=provider,
            mod=model,
            tok=tokens,
            lat=latency_seconds,
            cache=from_cache,
            msg=user_message[:100],
            ans=answer[:100],
        )
