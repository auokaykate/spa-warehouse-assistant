"""Тесты Части 3: интенты, сущности, unknown и отсутствие выдуманных чисел."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import pytest

import spa_assistant.llm as llm
import spa_assistant.qa as qa
from spa_assistant.forecast import forecast_demand
from spa_assistant.llm import numbers_are_grounded
from spa_assistant.qa import answer_question, extract_budget_limit, extract_load_factor, extract_period

AS_OF = date(2026, 9, 15)


@pytest.mark.parametrize(
    ("question", "days"),
    [
        ("на три месяца", 90),
        ("в ближайшие 14 дней", 14),
        ("на квартал", 90),
        ("на полгода", 180),
        ("за месяц", 30),
        ("в этом месяце", 15),
        ("без периода", None),
    ],
)
def test_extract_period(question: str, days: int | None) -> None:
    """Период извлекается из разных формулировок."""
    assert extract_period(question, AS_OF)[0] == days


def test_extract_budget_and_load() -> None:
    """Бюджетный лимит и сценарий загрузки извлекаются числами."""
    assert extract_budget_limit("при лимите 200 тысяч") == 200_000
    assert extract_budget_limit("до 1,5 млн руб") == 1_500_000
    assert extract_load_factor("если загрузка вырастет на 20%") == pytest.approx(1.2)


@pytest.mark.parametrize(
    ("question", "intent"),
    [
        ("Сколько масла закупить на три месяца и сколько это будет стоить?", "unknown"),
        ("Сколько базового масла закупить на три месяца?", "forecast_purchase"),
        ("Что нужно заказать в ближайшие 14 дней?", "reorder_list"),
        ("Какой бюджет закупок на квартал?", "budget"),
        ("Что закончится до следующей поставки в Сочи?", "deficit_risk"),
        ("Какие партии сгорят в этом месяце?", "expiry_risk"),
        ("Насколько подорожало ароматическое масло у поставщика?", "price_dynamics"),
        ("Заказать масло", "unknown"),
        ("Сколько это будет стоить?", "unknown"),
        ("Какая погода в Сочи на выходных?", "unknown"),
    ],
)
def test_intents(question: str, intent: str, context: dict[str, Any]) -> None:
    """Интенты и edge-кейсы (без периода, без позиции, вне домена)."""
    parsed, confidence, text = answer_question(question, context)
    assert parsed["intent"] == intent
    assert 0.0 <= confidence <= 1.0
    if intent == "unknown":
        assert "уточн" in text.lower() or "вне области" in text.lower()


def test_missing_period_asks_clarification(context: dict[str, Any]) -> None:
    """Без периода и с неоднозначной позицией — unknown и вопрос про оба параметра."""
    parsed, _, text = answer_question("Заказать масло", context)
    assert parsed["missing"] == ["sku", "period_days"]
    assert "какое именно масло" in text and "период" in text


@pytest.mark.parametrize(
    "question",
    [
        "Сколько масла закупить на три месяца и сколько это будет стоить?",
        "Сколько масла уйдёт за месяц, если загрузка вырастет на 20%?",
    ],
)
def test_ambiguous_name_returns_unknown(question: str, context: dict[str, Any]) -> None:
    """«Масло» подходит к OIL-001 и OIL-002: unknown и вопрос «какое именно масло»."""
    parsed, confidence, text = answer_question(question, context)
    assert parsed["intent"] == "unknown" and parsed["reason"] == "ambiguous_sku"
    assert parsed["sku"] is None and parsed["sku_candidates"] == ["OIL-001", "OIL-002"]
    assert parsed["needs_clarification"] and not parsed["low_forecast_confidence"]
    assert text.startswith("Требуется уточнение: какое именно масло")
    assert "OIL-001" in text and "OIL-002" in text


def test_location_without_data_is_flagged(context: dict[str, Any]) -> None:
    """Вопрос про филиал: ответ по сети начинается с оговорки и помечен как уточнение."""
    parsed, _, text = answer_question("Сколько альгинатной маски осталось в Красной Поляне?", context)
    assert parsed["intent"] == "deficit_risk" and parsed["sku"] == "WRAP-030"
    assert parsed["needs_clarification"] and parsed["clarification_reason"] == "location_not_in_data"
    assert text.startswith("По филиалу «Красная Поляна» данных нет")


def test_location_with_data_is_not_flagged(context: dict[str, Any]) -> None:
    """Если данные по филиалу есть, оговорки и пометки об уточнении нет."""
    history = {**context["history"], "by_location": {"Красная Поляна": {}}}
    parsed, _, text = answer_question(
        "Сколько альгинатной маски осталось в Красной Поляне?", {**context, "history": history},
    )
    assert not parsed["needs_clarification"]
    assert not text.startswith("По филиалу")


def test_flags_are_independent(context: dict[str, Any]) -> None:
    """Вопрос понят, но прогноз неуверенный: уточнения вопроса нет, флаг низкой уверенности есть."""
    parsed, _, _ = answer_question("Какой бюджет закупок на квартал?", context)
    assert parsed["intent"] == "budget"
    assert not parsed["needs_clarification"]
    assert parsed["low_forecast_confidence"]


def test_answer_numbers_come_from_forecast(context: dict[str, Any]) -> None:
    """Числа ответа совпадают с результатом forecast_demand."""
    parsed, _, text = answer_question("Посчитай закупку скраба на полгода при лимите 200 тысяч", context)
    expected = forecast_demand(context["history"], "SCRB-020", 180, {"catalog": context["catalog"]})
    assert parsed["sku"] == "SCRB-020" and parsed["period_days"] == 180
    assert f"{expected['recommended_qty']:g} кг" in text
    assert f"{expected['estimated_cost']:,.2f}".replace(",", " ") in text


def test_budget_total_equals_sum_of_forecasts(context: dict[str, Any]) -> None:
    """Бюджет квартала равен сумме estimated_cost из forecast_demand по позициям."""
    parsed, _, text = answer_question("Какой бюджет закупок на квартал?", context)
    total = sum(
        forecast_demand(context["history"], sku, 90, {"catalog": context["catalog"]})["estimated_cost"]
        for sku in context["history"]["weekly_consumption"]
    )
    assert parsed["period_days"] == 90
    assert f"{total:,.2f}".replace(",", " ") in text


def test_single_sku_answer_is_grounded(context: dict[str, Any]) -> None:
    """Все числа ответа по одной позиции берутся из forecast_demand, справочника и вопроса."""
    question = "Сколько базового масла закупить на три месяца и сколько это будет стоить?"
    parsed, _, text = answer_question(question, context)
    assert parsed["sku"] == "OIL-001"
    result = forecast_demand(context["history"], "OIL-001", 90, {"catalog": context["catalog"]})
    source = question + str(result) + str(context["catalog"]["OIL-001"])
    source += f" {result['estimated_cost']:,.2f} {result['price']:,.2f}".replace(",", " ")
    assert numbers_are_grounded(text, source)


def test_llm_number_guard() -> None:
    """Проверка чисел отклоняет текст с числом, которого нет в черновике."""
    assert numbers_are_grounded("Закупка 150 л на 188 857.50 руб.", "150 л, 188 857.50 руб.")
    assert not numbers_are_grounded("Закупка 160 л", "150 л")


def test_llm_path_rejects_invented_numbers(monkeypatch: pytest.MonkeyPatch, context: dict[str, Any]) -> None:
    """LLM разбирает вопрос, но ответ с выдуманным числом заменяется шаблоном."""
    parse_reply = json.dumps({
        "intent_scores": {"forecast_purchase": 0.1, "reorder_list": 0.0, "budget": 0.9, "deficit_risk": 0.0,
                          "expiry_risk": 0.0, "price_dynamics": 0.0, "out_of_scope": 0.0},
        "sku": None, "location": None,
    })

    def fake_call(system: str, user: str, schema: dict[str, Any] | None) -> str:
        """Подменный вызов модели: JSON для разбора и текст с чужим числом для ответа."""
        return parse_reply if schema is not None else "Бюджет квартала 999 999 руб."

    monkeypatch.setattr(llm, "_call", fake_call)
    monkeypatch.setattr(qa, "llm_enabled", lambda ctx: True)
    parsed, _, text = qa.answer_question("Какой бюджет закупок на квартал?", context)
    assert parsed["source"] == "llm" and parsed["intent"] == "budget"
    assert parsed["answer_source"] == "template"
    assert "999 999" not in text
