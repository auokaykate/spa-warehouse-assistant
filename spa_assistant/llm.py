"""Необязательный LLM-слой для Части 3 (Anthropic Claude).

LLM используется только для разбора вопроса (интент и сущности) и для
переформулирования готового ответа. Числа LLM не считает: после
переформулирования каждое число в тексте сверяется с числами черновика,
рассчитанного детерминированно. При любой ошибке, отсутствии ключа или
пакета ``anthropic`` функции возвращают None, и работает fallback на правилах.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
from typing import Any

from spa_assistant import config as cfg

_SPACES = "[ \u00a0\u202f]"
_NUMBER = re.compile(r"(?<![\w.,-])-?\d+(?:" + _SPACES + r"\d{3})*(?:[.,]\d+)?")

PARSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent_scores": {
            "type": "object",
            "properties": {name: {"type": "number"} for name in (*cfg.INTENTS, "out_of_scope")},
            "required": [*cfg.INTENTS, "out_of_scope"],
            "additionalProperties": False,
        },
        "sku": {"type": ["string", "null"]},
        "location": {"type": ["string", "null"]},
    },
    "required": ["intent_scores", "sku", "location"],
    "additionalProperties": False,
}

PARSE_SYSTEM = (
    "Ты разбираешь вопросы управляющего спа-сетью к складскому помощнику. "
    "Оцени, насколько вопрос соответствует каждому интенту, числами от 0 до 1 "
    "(сумма не больше 1): forecast_purchase — сколько закупить или израсходовать "
    "конкретной позиции за период; reorder_list — что нужно заказать сейчас; "
    "budget — общий бюджет закупок за период; deficit_risk — что закончится, "
    "остатки, риск дефицита; expiry_risk — сроки годности, списание партий; "
    "price_dynamics — изменение цен; out_of_scope — всё остальное, включая "
    "вопросы о датах доставки, сравнение с прошлыми планами и темы вне склада. "
    "sku — код позиции из справочника, только если позиция однозначно названа, "
    "иначе null. location — филиал, если назван, иначе null. Не угадывай."
)

RENDER_SYSTEM = (
    "Перепиши ответ складского помощника понятным деловым русским языком. "
    "Нельзя добавлять, удалять, округлять или пересчитывать числа: используй "
    "только числа из исходного текста, в том же виде. Не добавляй фактов."
)


def llm_enabled(context: dict[str, Any] | None = None) -> bool:
    """Проверить, можно ли обращаться к LLM.

    LLM включён, если установлен пакет ``anthropic``, задан ключ
    ``ANTHROPIC_API_KEY``, переменная ``SPA_USE_LLM`` не равна ``0`` и
    в контексте не передано ``use_llm=False``.

    Args:
        context: Контекст вызова ``answer_question``.

    Returns:
        True, если LLM доступен.
    """
    if context is not None and context.get("use_llm") is False:
        return False
    if os.environ.get(cfg.LLM_ENABLE_ENV, "1") == "0" or not os.environ.get(cfg.LLM_API_KEY_ENV):
        return False
    return importlib.util.find_spec("anthropic") is not None


def _call(system: str, user: str, schema: dict[str, Any] | None) -> str | None:
    """Выполнить один запрос к Claude и вернуть текст ответа или None."""
    try:
        import anthropic

        client = anthropic.Anthropic(timeout=cfg.LLM_TIMEOUT_SECONDS)
        output_config: dict[str, Any] = {"effort": cfg.LLM_EFFORT}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        response = client.beta.messages.create(
            model=cfg.LLM_MODEL,
            max_tokens=cfg.LLM_MAX_TOKENS,
            betas=[cfg.LLM_FALLBACK_BETA],
            fallbacks="default",
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config=output_config,
        )
        if response.stop_reason in ("refusal", "max_tokens"):
            return None
        return next((b.text for b in response.content if b.type == "text"), None)
    except Exception:
        return None


def llm_parse_question(question: str, catalog: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Разобрать вопрос с помощью LLM в структурированный JSON.

    Args:
        question: Вопрос пользователя.
        catalog: Справочник ``{sku: карточка}``, передаётся модели для выбора SKU.

    Returns:
        Словарь ``{"intent_scores", "sku", "location"}`` или None при ошибке.
        SKU, отсутствующий в справочнике, заменяется на None.
    """
    listing = "\n".join(f"{sku}: {item['name']}" for sku, item in catalog.items())
    raw = _call(PARSE_SYSTEM, f"Справочник:\n{listing}\n\nВопрос: {question}", PARSE_SCHEMA)
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if parsed.get("sku") not in catalog:
        parsed["sku"] = None
    return parsed


def _numbers(text: str) -> set[float]:
    """Множество чисел, встречающихся в тексте."""
    result: set[float] = set()
    for token in _NUMBER.findall(text):
        clean = re.sub(_SPACES, "", token).replace(",", ".")
        try:
            result.add(float(clean))
        except ValueError:
            continue
    return result


def numbers_are_grounded(candidate: str, source: str) -> bool:
    """Проверить, что все числа ответа LLM есть в детерминированном черновике.

    Args:
        candidate: Текст, полученный от LLM.
        source: Черновик ответа, собранный из результатов Части 2.

    Returns:
        True, если каждое число кандидата совпадает с каким-либо числом
        источника с точностью ``LLM_NUMBER_TOLERANCE``.
    """
    allowed = _numbers(source)
    return all(
        any(abs(value - ok) <= cfg.LLM_NUMBER_TOLERANCE for ok in allowed)
        for value in _numbers(candidate)
    )


def llm_render_answer(draft: str) -> str | None:
    """Переформулировать черновик ответа, не меняя чисел.

    Args:
        draft: Детерминированный текст ответа.

    Returns:
        Переформулированный текст или None, если LLM недоступен либо
        в ответе появилось число, которого нет в черновике.
    """
    text = _call(RENDER_SYSTEM, draft, None)
    if text is None or not numbers_are_grounded(text, draft):
        return None
    return text
