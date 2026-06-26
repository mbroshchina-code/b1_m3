"""Интерактивный CLI (REPL) для общения с ассистентом поддержки.

Принимает пользовательский ввод, обрабатывает команды CLI
и выводит ответ ассистента с метаданными (категория, источник, задержка).
"""

from __future__ import annotations
import os
from m2_b5.config import Settings
from m2_b5.core.assistant import SupportAssistantApp


def main():
    settings = Settings.from_env()
    assistant = SupportAssistantApp(settings)

    print(f"=== {settings.service_name} Support CLI ===")
    print("Команды: /clear, /clear_cache, /reset_stats, /stats, /quit")
    print("Мультимодальные: /analyze <путь> [вопрос]")

    while True:
        try:
            user_input = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо свидания!")
            return None

        if not user_input:
            continue

        if user_input.startswith("/") and not user_input.startswith("/analyze"):
            command_result = assistant.handle_command(user_input)
            if command_result is None:
                print("До свидания!")
                return None
            print(command_result)
            continue
        
        # ── /analyze — анализ изображения через Vision API ──────────
        if user_input.startswith("/analyze "):
            parts = user_input[9:].strip().split(maxsplit=1)
            image_path = parts[0] if parts else ""
            question = parts[1] if len(parts) > 1 else ""
            if not image_path or not os.path.isfile(image_path):
                print("Использование: /analyze <путь к изображению> [вопрос]")
                if image_path and not os.path.isfile(image_path):
                    print(f"Файл не найден: {image_path}")
                continue
            prompt = question if question else (
                "Пользователь прислал скриншот. "
                "Определи проблему и предложи решение. "
                "Если видишь код ошибки — укажи его."
            )
            print("Анализирую изображение...")
            response = assistant.respond(prompt, image_path=image_path)
            source = "cache" if response.from_cache else response.provider
            print(f"[{response.category} | {source} | {response.latency_seconds:.2f}с]")
            print(response.text)
            continue

        response = assistant.respond(user_input)
        source = "cache" if response.from_cache else response.provider
        print(f"[{response.category} | {source} | {response.latency_seconds:.2f}с]")
        print(response.text)