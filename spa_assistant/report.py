"""Сборка таблиц результатов пайплайна в Markdown."""

from __future__ import annotations

from typing import Any

from spa_assistant import config as cfg
from spa_assistant.normalize import RESULT_KEYS

MARK_OK = "✅"
MARK_ABSENT = "— (нет в записи)"
MARK_MISSED = "❌ пропущено"
MARK_WRONG = "⚠️ неверно"


def _cell(value: Any) -> str:
    """Представить значение в ячейке Markdown-таблицы."""
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(header: list[str], rows: list[list[Any]]) -> str:
    """Собрать Markdown-таблицу."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def check_field(actual: Any, expected: Any) -> str:
    """Сравнить извлечённое поле с ручной разметкой.

    Args:
        actual: Значение, которое вернула ``normalize_movement``.
        expected: Эталонное значение из ручной разметки.

    Returns:
        Отметка: корректно, нет в записи, пропущено или неверно.
    """
    if expected is None:
        return MARK_ABSENT if actual is None else MARK_WRONG
    if actual is None:
        return MARK_MISSED
    if isinstance(expected, float):
        return MARK_OK if abs(float(actual) - expected) <= cfg.ROUNDING_EPS else MARK_WRONG
    return MARK_OK if actual == expected else MARK_WRONG


def part1_markdown(rows: list[dict[str, Any]], expected: dict[str, dict[str, Any]]) -> str:
    """Таблицы Части 1: извлечённые значения и проверка по разметке.

    Args:
        rows: Список ``{"id", "text", "result"}``.
        expected: Ручная разметка ``{id: {поле: значение}}``.

    Returns:
        Markdown с двумя таблицами и итоговой статистикой.
    """
    values = _table(
        ["ID", "Исходный текст", *RESULT_KEYS],
        [[r["id"], r["text"], *(r["result"][k] for k in RESULT_KEYS)] for r in rows],
    )
    checks: list[list[Any]] = []
    stats = {MARK_OK: 0, MARK_ABSENT: 0, MARK_MISSED: 0, MARK_WRONG: 0}
    for r in rows:
        marks = [check_field(r["result"][k], expected[r["id"]][k]) for k in RESULT_KEYS]
        for mark in marks:
            stats[mark] += 1
        checks.append([r["id"], *marks])
    summary = (
        f"Итого по {len(rows) * len(RESULT_KEYS)} полям: извлечено корректно — {stats[MARK_OK]}, "
        f"верно возвращён None (поля нет в записи) — {stats[MARK_ABSENT]}, "
        f"пропущено — {stats[MARK_MISSED]}, извлечено неверно — {stats[MARK_WRONG]}."
    )
    return values + "\n\n" + _table(["ID", *RESULT_KEYS], checks) + "\n\n" + summary


def part2_markdown(results: list[dict[str, Any]]) -> str:
    """Таблица Части 2 по позициям и горизонтам."""
    header = [
        "SKU", "Горизонт, дн", "Ед.", "avg_daily_consumption", "forecast_demand", "current_stock",
        "incoming_qty", "safety_stock", "reorder_point", "recommended_qty", "estimated_cost, руб",
        "stockout_date", "stockout (только остаток)", "order_by_date", "confidence", "Низкая уверенность прогноза",
    ]
    rows = [
        [
            r["sku"], r["horizon_days"], r["unit"], r["avg_daily_consumption"], r["forecast_demand"],
            r["current_stock"], r["incoming_qty"], r["safety_stock"], r["reorder_point"],
            r["recommended_qty"], f"{r['estimated_cost']:.2f}", r["stockout_date"],
            r["stockout_date_on_hand"], r["order_by_date"], r["confidence"],
            "да" if r["low_confidence"] else "нет",
        ]
        for r in results
    ]
    diag = _table(
        ["SKU", "Горизонт", "Уровень, ед/нед", "Тренд, ед/нед", "Сдвиг уровня с недели",
         "Пропуски (недели)", "Выбросы (недели)", "Факторы confidence"],
        [
            [r["sku"], r["horizon_days"], r["level_weekly"], r["trend_weekly"], r["level_shift_week"],
             r["missing_weeks"] or "—", r["outlier_weeks"] or "—", r["confidence_factors"]]
            for r in results
        ],
    )
    explanations = "\n".join(f"- **{r['sku']}, {r['horizon_days']} дн.** {r['explanation']}" for r in results)
    return _table(header, rows) + "\n\n" + diag + "\n\nОбъяснения (поле `explanation`):\n\n" + explanations


SHIFT_POLICY_TITLES: dict[str, str] = {
    "new_level": "новый уровень спроса (принято)",
    "outlier": "выброс: хвост заменён прежним уровнем",
    "none": "без обработки сдвига",
}


def sensitivity_markdown(rows: list[dict[str, Any]]) -> str:
    """Сравнение трактовок сдвига уровня спроса для позиций, где он найден."""
    return _table(
        ["SKU", "Горизонт, дн", "Трактовка", "Уровень, ед/нед", "Тренд, ед/нед", "forecast_demand",
         "recommended_qty", "estimated_cost, руб", "stockout_date", "confidence"],
        [
            [row["sku"], row["horizon_days"], SHIFT_POLICY_TITLES[policy], r["level_weekly"], r["trend_weekly"],
             r["forecast_demand"], r["recommended_qty"], f"{r['estimated_cost']:.2f}", r["stockout_date"],
             r["confidence"]]
            for row in rows
            for policy, r in row["variants"].items()
        ],
    )


def part3_markdown(answers: list[dict[str, Any]]) -> str:
    """Таблица Части 3: разобранные параметры, уверенность, ответ."""
    header = ["#", "Вопрос", "intent", "sku", "Объект", "Период, дн", "Лимит, руб", "Загрузка",
              "confidence", "Требуется уточнение вопроса", "Причина уточнения", "Низкая уверенность прогноза",
              "Источник разбора", "Ответ"]
    rows = []
    for i, a in enumerate(answers, start=1):
        p = a["parsed"]
        rows.append([
            i, a["question"], p["intent"], p["sku"], p["location"],
            "срок поставки" if p["period_mode"] == "lead_time" else p["period_days"],
            p["budget_limit"], p["load_factor"], a["confidence"],
            "да" if p["needs_clarification"] else "нет", p["clarification_reason"] or "—",
            "да" if p["low_forecast_confidence"] else "нет", p["source"], a["text"],
        ])
    n = len(answers)
    flagged = sum(a["parsed"]["needs_clarification"] for a in answers)
    unknown = sum(a["parsed"]["intent"] == "unknown" for a in answers)
    location = sum(a["parsed"]["clarification_reason"] == "location_not_in_data" for a in answers)
    low = sum(a["parsed"]["low_forecast_confidence"] for a in answers)
    summary = (
        f"Доля ответов «требуется уточнение вопроса»: {flagged}/{n} ({flagged / n:.0%}); "
        f"из них intent=unknown (вопрос не понят): {unknown}, вопрос про филиал без данных по филиалу: {location}. "
        f"Отдельно, «низкая уверенность прогноза» (вопрос понят, данных мало): {low}/{n} ({low / n:.0%}); "
        "в долю уточнений этот флаг не входит, флаги независимы."
    )
    scores = _table(
        ["#", "Баллы интентов", "Доля лучшего", "Разрыв 1-го и 2-го"],
        [[i, a["parsed"]["intent_scores"] or "—", a["parsed"]["intent_share"], a["parsed"]["intent_margin"]]
         for i, a in enumerate(answers, start=1)],
    )
    return _table(header, rows) + "\n\n" + summary + "\n\n" + scores


def build_report(
    part1: list[dict[str, Any]],
    expected: dict[str, dict[str, Any]],
    part2: list[dict[str, Any]],
    part3: list[dict[str, Any]],
    meta: dict[str, Any],
    sensitivity: list[dict[str, Any]] | None = None,
) -> str:
    """Собрать полный Markdown-отчёт по трём частям.

    Args:
        part1: Результаты нормализации.
        expected: Ручная разметка для Части 1.
        part2: Результаты прогноза.
        part3: Результаты ответов на вопросы.
        meta: Сведения о запуске (дата данных, режим LLM).
        sensitivity: Сравнение трактовок сдвига уровня спроса.

    Returns:
        Текст отчёта в Markdown.
    """
    return "\n\n".join([
        "# Отчёт пайплайна",
        f"Данные на {meta['as_of']}. Режим разбора вопросов: {meta['mode']}. "
        f"Порог confidence: {cfg.LOW_CONFIDENCE_THRESHOLD}; порог разрыва интентов: "
        f"{cfg.INTENT_MARGIN_THRESHOLD}.",
        "## Часть 1. normalize_movement (M1–M8)",
        part1_markdown(part1, expected),
        "## Часть 2. forecast_demand (горизонты 30 и 90 дней)",
        part2_markdown(part2),
        "### Чувствительность: сдвиг уровня как новый спрос или как выброс",
        sensitivity_markdown(sensitivity or []),
        "## Часть 3. answer_question (15 вопросов)",
        part3_markdown(part3),
    ]) + "\n"
