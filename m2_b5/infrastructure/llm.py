"""Клиент LLM с retry (tenacity) и fallback.

Все провайдеры работают через OpenAI-совместимый API.
Цепочка: primary → (если faq: заглушка) -> (если problem: primary → openrouter → fallback → заглушка)
"""

from __future__ import annotations

from collections.abc import Iterator
import json
from loguru import logger
from openai import OpenAI
from m2_b5.config import Settings
from m2_b5.core.classification import heuristic_classify
from m2_b5.infrastructure.tools import get_tools_schema
from m2_b5.models import Category, LLMResult


# Ответ-заглушка, когда ни один провайдер не дал полезный ответ и заглушка для FAQ
FALLBACK_ANSWER = "Извините, что-то пошло не по плану. Пожалуйста, повторите попытку или напишите чуть позже."
FAQ_ANSWER = "Запрос не относится к проблемам. Найдите ответ в базе знаний"

def _build_client(api_key: str | None, base_url: str | None) -> OpenAI | None:
    """Создаёт чистый стандартный OpenAI клиент."""
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url=base_url)


class RobustLLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.primary = _build_client(settings.api_key, settings.base_url)
        self.openrouter = _build_client(settings.openrouter_api_key, settings.openrouter_base_url)
        self.fallback = _build_client(settings.fallback_api_key, settings.fallback_base_url)

    # ── Цепочка провайдеров ───────────────────────────────────────────

    def _provider_chain(self) -> Iterator[tuple[OpenAI, str, bool]]:
        """Отдаёт (client, model, name, used_fallback) для каждого доступного провайдера."""
        # 1. Сначала пробуем основную модель через прокси
        if self.primary is not None:
            yield self.primary, self.settings.primary_model, "primary", False
        # 2. Если упала — идем в OpenRouter через интернет
        if self.openrouter is not None and self.settings.openrouter_model:
            yield self.openrouter, self.settings.openrouter_model, "openrouter", True
        # 3. Если и OpenRouter лег — задействуем локальную Ollama    
        if self.fallback is not None and self.settings.fallback_model:
            yield self.fallback, self.settings.fallback_model, "fallback", True

    # ── Публичные методы ──────────────────────────────────────────────

    def classify(self, messages: list[dict[str, any]]) -> Category:
        """Классифицирует запрос пользователя по цепочке моделей или эвристике."""
        for client, model, name, _ in self._provider_chain():
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0,  # Строго 0.0 для стабильности классификации
                    max_tokens=8,     # Экономим токены, ответ будет очень коротким
                    timeout=self.settings.request_timeout_seconds,
                )
                
                raw = (response.choices[0].message.content or "").strip().lower()
                return Category(raw)
                
            except Exception as e:
                logger.warning(f"Провайдер классификации {name} ({model}) недоступен: {e}")
                continue

        # Если все ИИ в сети упали, срабатывает наша локальная питоновская эвристика по ключевым словам
        return heuristic_classify(messages[-1]["content"])

    def answer(self, messages: list[dict[str, any]], category: str) -> LLMResult:
        """Получает ответ: primary → openrouter → fallback → заглушка."""
        
        for client, model, name, used_fallback in self._provider_chain():
            try:
                if used_fallback:
                    logger.info(f"Переключаюсь на резервный канал ({name}): {model}")
                text, tokens = self._answer_from(client, model, messages, category)
                return LLMResult(
                    text, tokens,
                    provider=name,
                    model=model,
                    used_fallback=used_fallback
                )
            except Exception as e:
                logger.warning(f"Провайдер {name} ({model}) недоступен: {e}")

        # Все провайдеры недоступны — даем заглушку
        return LLMResult(FALLBACK_ANSWER, 0, "none", "none", True)

    # ── Внутренние методы ─────────────────────────────────────────────

    def _answer_from(
        self, client: OpenAI, model: str, messages: list[dict[str, any]], category: str
    ) -> tuple[str, int]:
        """Один ответ от провайдера. Возвращает (текст, токены)."""
        text = self._call(client, model, messages, category)
        return (text or FALLBACK_ANSWER), 0

    def _call(
        self,
        client: OpenAI,
        model: str,
        messages: list[dict[str, any]],
        category: str,
        temperature: float = 0.2,
        max_tokens: int = 250,
    ) -> str:
        """Вызов LLM с жесткой активацией tool_choice в Strict Mode для категории 'problem'."""
        is_problem = str(category).lower().strip() == "problem"
        is_not_ollama = "localhost" not in str(client.base_url)

        # Подключаем схему инструментов и активируем принудительный tool_choice
        if is_problem and is_not_ollama:
            current_tools = get_tools_schema()
            # ТОП ошибок фикс: жестко обязываем модель вызвать наш инструмент поиска багов
            current_tool_choice = {
                "type": "function",
                "function": {"name": "search_bug_database"}
            }
            logger.info(f"[LLM] Для категории 'problem' активирован принудительный вызов базы багов.")
        else:
            current_tools = None
            current_tool_choice = None

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=current_tools,
            tool_choice=current_tool_choice,
            timeout=self.settings.request_timeout_seconds,
        )

        response_message = response.choices[0].message
        # Проверяем наличие вызовов инструментов (Tool Calls)
        if hasattr(response_message, "tool_calls") and response_message.tool_calls:
            # Берём первый вызов из списка
            tool_call = response_message.tool_calls[0]
            
            if tool_call.function.name == "search_bug_database":
                # ТОП ошибок фикс: JSON-парсинг обернут в безопасный try/except с логированием
                try:
                    arguments = json.loads(tool_call.function.arguments)
                    queries_arg = arguments.get("queries", [])
                except (json.JSONDecodeError, TypeError, KeyError) as e:
                    logger.error(f"[Tool Call] Ошибка парсинга аргументов от модели: {e}")
                    queries_arg = []

                # Запускаем локальный питоновский поиск по нашей базе JSON
                tool_result = self._execute_bug_search(queries=queries_arg)

                # ТОП ошибок фикс: Сначала добавляем ассистентский ход в историю!
                messages.append(response_message.model_dump())

                # ТОП ошибок фикс: Возвращаем результат строго по id вызова с ролью "tool"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,  # Стыкуем строго по ID
                    "name": "search_bug_database",
                    "content": tool_result
                })

                # Финальный запрос к ИИ для суммаризации отчета на основе найденных багов
                second_response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return (second_response.choices[0].message.content or "").strip()

        return (response_message.content or "").strip()

    def _execute_bug_search(self, queries: list[str]) -> str:
        """Ищет баги по тексту полей внутри локальной базы данных prompts/bugs_database.json."""
        logger.info(f"[Bug DB Search] ИИ затребовал расширенный поиск по синонимам: {queries}")
        from pathlib import Path
        import json
        
        db_path = Path(__file__).resolve().parent.parent / "prompts" / "bugs_database.json"
        if not db_path.exists():
            return "Ошибка инфраструктуры: Локальная база данных багов отсутствует."
            
        try:
            with open(db_path, "r", encoding="utf-8") as f:
                bugs = json.load(f)
                
            if not queries:
                return "Поисковый запрос пуст. Совпадений не найдено."

            # Собираем СЕТ (уникальное множество) всех слов из ВСЕХ трех фраз-синонимов
            unique_search_words = set()
            for query in queries:
                words = [word.lower().strip() for word in query.split() if len(word) > 2]
                unique_search_words.update(words)

            logger.info(f"[Bug DB Search] Сформирован уникальный поисковый алфавит: {list(unique_search_words)}")

            found_bugs = []
            for bug in bugs:
                name = bug.get("name", "").lower()
                theme = bug.get("theme", "").lower()
                body = bug.get("content", {}).get("body", "").lower()
                
                # Считаем, сколько уникальных семантических слов попало в текст текущего бага
                matches_count = 0
                for word in unique_search_words:
                    if (word in name) or (word in theme) or (word in body):
                        matches_count += 1
                
                # Если совпало хотя бы 2 важных слова из расширенного облака синонимов — баг релевантен!
                if matches_count >= 2:
                    found_bugs.append(bug)
                    
            if found_bugs:
                logger.info(f"[Tool Search] Найдено {len(found_bugs)} багов через алгоритм Query Expansion.")
                return json.dumps(found_bugs, ensure_ascii=False, indent=2)
                
            return f"В базе данных не найдено багов, подходящих под облако синонимов: {list(unique_search_words)}"
        except Exception as e:
            logger.error(f"[Tool Search] Критическая ошибка при чтении файла БД: {e}")
            return f"Ошибка при обращении к базе багов: {e}"