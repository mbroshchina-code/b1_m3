from typing import Any
from m2_b5.prompts.loader import TOOL_SEARCH_BUGS_DESCRIPTION

def get_tools_schema() -> list[dict[str, Any]]:
    """  
    Описание подтягивается динамически из именованной константы промптов.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "search_bug_database",
                "description": TOOL_SEARCH_BUGS_DESCRIPTION,
                # Включаем Strict Mode на уровне схемы функции
                "strict": True,  # False, <-- ИСПРАВЛЕНО: Меняем True на False, чтобы убрать сетевой завис!
                "parameters": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "description": "Сделай СТРОГО 3 разные текстовые переформулировки (синонима), одинаковые по смыслу с запросом пользователя, для расширенного поиска по ключевым словам в БД багов.",
                            "items": {
                                "type": "string",
                                "description": "Короткая емкая техническая фраза-синоним (например, 'ошибка борис банк', 'сбой эквайринга', 'не проходит оплата')."
                            },
                            "minItems": 3,
                            "maxItems": 3,
                            "additionalProperties": False  # Требование Strict Mode для массивов
                        }
                    },
                    "required": ["queries"],
                    "additionalProperties": False  # Требование Strict Mode для корня объекта
                }
            }
        }
    ]