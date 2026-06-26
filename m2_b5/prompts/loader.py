"""Загрузка и сборка промптов для LLM.

Читает шаблоны промптов и few-shot примеры из файлов пакета
(``classifier_system_prompt.txt``, ``system_prompt.txt``, ``classifier_category_few_shots.json``, ``tool_search_bugs_description.txt``) и предоставляет
функции для формирования готовых списков сообщений для ассистента.
"""

from __future__ import annotations
import base64
import json
from importlib import resources
from string import Template
from typing import Any
from pathlib import Path
import copy
import os

def _read_prompt_file(filename: str) -> str:
    return resources.files(__package__).joinpath(filename).read_text(encoding="utf-8").strip()


SYSTEM_PROMPT_TEMPLATE = Template(_read_prompt_file("system_prompt.txt"))
CLASSIFIER_SYSTEM_PROMPT = _read_prompt_file("classifier_system_prompt.txt")
CLASSIFIER_CATEGORY_FEW_SHOTS = json.loads(_read_prompt_file("classifier_category_few_shots.json"))
TOOL_SEARCH_BUGS_DESCRIPTION = _read_prompt_file("tool_search_bugs_description.txt")

def build_system_prompt(service_name: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.safe_substitute(
        service_name=service_name,
    )

def classifier_few_shot_messages() -> list[dict[str, str]]:
    return copy.deepcopy(CLASSIFIER_CATEGORY_FEW_SHOTS)


def encode_image(image_path: str) -> tuple[str, str]:
    """Кодирует изображение в base64. Возвращает (base64_data, расширение)."""
    p = Path(image_path)
    ext = p.suffix.lstrip(".").lower()
    if ext == "jpg":
        ext = "jpeg"
    return base64.b64encode(p.read_bytes()).decode("utf-8"), ext

def build_answer_messages(
    system_prompt: str, 
    history: list[dict[str, any]], 
    user_message: str,
    image_path: str | None = None  # Добавили новый необязательный аргумент
) -> list[dict[str, any]]:  # Тип Any, так как структура контента усложняется
    """Собирает полный список сообщений с возможностью передачи изображения в Base64."""
    
    # 1. Формируем базовый системный промпт и фьюшоты
    messages: list[dict[str, any]] = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    
    # 2. Добавляем реплику пользователя (мультимодальную или обычную текстовую)
    # Проверяем, что путь передан и такой файл физически существует на диске
    if image_path and os.path.exists(image_path):
        b64, ext = encode_image(image_path)
        content: list[dict[str, any]] = [
            {"type": "text", "text": user_message},
            {"type": "image_url", "image_url": {
                "url": f"data:image/{ext};base64,{b64}",
                "detail": "high",  # важно для распознавания скриншотов ошибок
            }},
        ]
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": user_message})

    return messages

def build_classifier_messages(user_message: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT}]
    messages.extend(classifier_few_shot_messages())
    messages.append({"role": "user", "content": user_message})
    return messages
