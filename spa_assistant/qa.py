"""Часть 3. Разбор вопроса пользователя и ответ с объяснением расчёта.

Интент определяется LLM (если доступен) или правилами по ключевым словам.
Все числа в ответе берутся из ``forecast_demand`` и справочника: ни LLM,
ни шаблоны не подставляют остатки, цены и объёмы от себя.
"""

from __future__ import annotations

import calendar
import math
import re
from datetime import date
from typing import Any, Callable

from spa_assistant import config as cfg
from spa_assistant.forecast import forecast_demand
from spa_assistant.llm import llm_enabled, llm_parse_question, llm_render_answer
from spa_assistant.normalize import match_sku_by_name, parse_location, parse_sku, shared_name_word

UNKNOWN = "unknown"
OUT_OF_SCOPE = "out_of_scope"

_NUM_WORD_RE = "|".join(sorted(cfg.NUMBER_WORDS, key=len, reverse=True))
_PERIOD_RE = re.compile(
    r"(\d+|" + _NUM_WORD_RE + r")\s*(" + "|".join(cfg.PERIOD_UNIT_DAYS) + r")[а-яё]*",
    re.IGNORECASE,
)
_MONEY_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*("
    + "|".join(sorted(map(re.escape, cfg.MONEY_MULTIPLIERS), key=len, reverse=True))
    + r")(?![а-яё])",
    re.IGNORECASE,
)
_DOUBLE_DOT = re.compile(r"\.\.(?!\.)")
_LIMIT_RE = re.compile(r"(?:лимит\w*|не более|не больше)\s*(?:в|до)?\s*(\d+(?:[.,]\d+)?)", re.IGNORECASE)


def fmt_qty(value: float) -> str:
    """Отформатировать количество без лишних нулей."""
    return f"{round(value, cfg.QTY_DECIMALS):g}"


def fmt_money(value: float) -> str:
    """Отформатировать сумму в рублях с разделителем тысяч."""
    return f"{value:,.{cfg.MONEY_DECIMALS}f}".replace(",", " ") + " руб."


def extract_period(question: str, as_of: date) -> tuple[int | None, str | None]:
    """Извлечь период планирования из вопроса.

    Args:
        question: Вопрос пользователя.
        as_of: Дата актуальности данных (для «в этом месяце»).

    Returns:
        Пара (число дней или None, режим периода). Режим ``lead_time``
        означает «до следующей поставки» — срок поставки каждой позиции.
    """
    text = question.lower()
    if re.search(cfg.NEXT_DELIVERY_PATTERN, text):
        return None, "lead_time"
    if re.search(cfg.CURRENT_MONTH_PATTERN, text):
        last_day = calendar.monthrange(as_of.year, as_of.month)[1]
        return last_day - as_of.day, "month_end"
    match = _PERIOD_RE.search(text)
    if match:
        raw, unit = match.groups()
        count = int(raw) if raw.isdigit() else cfg.NUMBER_WORDS[raw.lower()]
        unit_days = next(v for k, v in cfg.PERIOD_UNIT_DAYS.items() if unit.lower().startswith(k))
        return count * unit_days, "explicit"
    for pattern, days in cfg.PERIOD_WORDS.items():
        if re.search(pattern, text):
            return days, "explicit"
    return None, None


def extract_budget_limit(question: str) -> float | None:
    """Извлечь бюджетный лимит («при лимите 200 тысяч», «до 1,5 млн руб»).

    Args:
        question: Вопрос пользователя.

    Returns:
        Лимит в рублях или None.
    """
    match = _MONEY_RE.search(question)
    if match:
        value, unit = match.groups()
        return float(value.replace(",", ".")) * cfg.MONEY_MULTIPLIERS[unit.lower()]
    match = _LIMIT_RE.search(question)
    if match:
        return float(match.group(1).replace(",", "."))
    return None


def extract_load_factor(question: str) -> float | None:
    """Извлечь сценарное изменение загрузки («загрузка вырастет на 20%»).

    Args:
        question: Вопрос пользователя.

    Returns:
        Множитель расхода (например 1.2) или None.
    """
    text = question.lower()
    match = re.search(cfg.PERCENT_PATTERN, text)
    if not match or not any(h in text for h in cfg.LOAD_GROWTH_HINTS):
        return None
    pct = float(match.group(1).replace(",", ".")) / cfg.PERCENT_BASE
    sign = -1 if any(h in text for h in cfg.LOAD_DECLINE_HINTS) else 1
    return round(1 + sign * pct, cfg.QTY_DECIMALS)


def extract_sku(question: str, catalog: dict[str, dict[str, Any]]) -> tuple[str | None, list[str]]:
    """Определить позицию по коду или по названию.

    Если название подходит нескольким позициям (например, «масло» —
    OIL-001 и OIL-002), SKU не выбирается: возвращаются все кандидаты,
    и вопрос уходит в уточнение.

    Args:
        question: Вопрос пользователя.
        catalog: Справочник ``{sku: карточка}``.

    Returns:
        Пара (SKU или None, все кандидаты).
    """
    sku, _ = parse_sku(question, catalog)
    if sku:
        return sku, [sku]
    candidates = match_sku_by_name(question, catalog)
    return (candidates[0] if len(candidates) == 1 else None), candidates


def score_intents_rules(question: str) -> dict[str, float]:
    """Посчитать баллы интентов по словарю ключевых слов.

    Args:
        question: Вопрос пользователя.

    Returns:
        Словарь ``{интент: балл}``, включая псевдо-интент ``out_of_scope``.
    """
    text = question.lower()
    scores = {
        intent: sum(w for kw, w in words.items() if kw in text)
        for intent, words in cfg.INTENT_KEYWORDS.items()
    }
    scores[OUT_OF_SCOPE] = sum(w for kw, w in cfg.OUT_OF_SCOPE_KEYWORDS.items() if kw in text)
    return scores


def rank_intents(scores: dict[str, float]) -> tuple[str, float, float]:
    """Выбрать лучший интент и посчитать разрыв с вторым.

    Args:
        scores: Баллы интентов (включая ``out_of_scope``).

    Returns:
        Тройка (лучший интент, его доля в сумме баллов, разрыв долей 1-го и 2-го).
    """
    total = sum(scores.values())
    if total <= 0:
        return UNKNOWN, 0.0, 0.0
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top, top_score = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    return top, top_score / total, (top_score - second) / total


def parse_question(question: str, context: dict[str, Any]) -> dict[str, Any]:
    """Разобрать вопрос: интент, сущности, недостающие параметры.

    Args:
        question: Вопрос пользователя.
        context: ``history``, ``catalog`` и необязательный ``use_llm``.

    Returns:
        Словарь разобранных параметров.
    """
    history, catalog = context["history"], context["catalog"]
    as_of = date.fromisoformat(history["as_of"])
    sku, candidates = extract_sku(question, catalog)
    location, _ = parse_location(question)
    period_days, period_mode = extract_period(question, as_of)

    source = "rules"
    scores = score_intents_rules(question)
    if llm_enabled(context):
        llm = llm_parse_question(question, catalog)
        if llm is not None:
            source = "llm"
            scores = {k: float(v) for k, v in llm["intent_scores"].items()}
            if not candidates and llm.get("sku"):
                sku, candidates = llm["sku"], [llm["sku"]]
            location = location or llm.get("location")

    top, share, margin = rank_intents(scores)
    parsed: dict[str, Any] = {
        "intent": top,
        "sku": sku,
        "sku_candidates": candidates,
        "ambiguous_sku": sku is None and len(candidates) > 1,
        "location": location,
        "period_days": period_days,
        "period_mode": period_mode,
        "budget_limit": extract_budget_limit(question),
        "load_factor": extract_load_factor(question),
        "intent_scores": {k: round(v, cfg.CONFIDENCE_DECIMALS) for k, v in scores.items() if v},
        "intent_share": round(share, cfg.CONFIDENCE_DECIMALS),
        "intent_margin": round(margin, cfg.CONFIDENCE_DECIMALS),
        "source": source,
        "missing": [],
        "reason": None,
    }
    best_score = max(scores.values(), default=0.0)
    if top == OUT_OF_SCOPE:
        parsed.update(intent=UNKNOWN, reason="out_of_scope")
    elif top == UNKNOWN or (source == "rules" and best_score < cfg.INTENT_MIN_SCORE):
        parsed.update(intent=UNKNOWN, reason="no_intent")
    elif margin < cfg.INTENT_MARGIN_THRESHOLD:
        parsed.update(intent=UNKNOWN, reason="ambiguous_intent", candidate_intent=top)
    else:
        missing = [
            name for name in cfg.INTENT_REQUIRED_ENTITIES[top]
            if parsed[name] is None
        ]
        if parsed["ambiguous_sku"] and "sku" not in missing:
            missing.insert(0, "sku")
        if missing:
            reason = "ambiguous_sku" if parsed["ambiguous_sku"] else "missing_params"
            parsed.update(intent=UNKNOWN, reason=reason, candidate_intent=top, missing=missing)
    return parsed


def _clarification(parsed: dict[str, Any], question: str, context: dict[str, Any]) -> str:
    """Сформировать уточняющий вопрос вместо догадки."""
    catalog = context["catalog"]
    reason = parsed["reason"]
    if reason == "out_of_scope":
        text = question.lower()
        if "привез" in text or "придёт" in text or "придет" in text:
            incoming = context["history"].get("incoming_qty", {})
            in_transit = ", ".join(
                f"{sku}: {fmt_qty(qty)} {catalog[sku]['unit']}" for sku, qty in incoming.items() if qty
            )
            return (
                "Требуется уточнение. В данных нет дат поставок по заказам, есть только объём "
                f"в пути ({in_transit}). Уточните номер заказа или поставщика — дату "
                "доставки помощник не придумывает."
            )
        if "почему" in text or "сравнению" in text:
            return (
                "Требуется уточнение. В данных нет сохранённых планов прошлых периодов, "
                "сравнить с ними нельзя. Могу показать, из чего складывается текущий план "
                "по конкретной позиции (уровень спроса, тренд, страховой запас) — укажите позицию и период."
            )
        return (
            "Вопрос вне области складского помощника. Я отвечаю на вопросы о закупках, "
            "остатках, рисках дефицита и списания, бюджете и ценах."
        )
    if reason == "no_intent":
        return (
            "Требуется уточнение: не удалось понять, что нужно посчитать. Например: "
            "«Сколько масла OIL-001 закупить на квартал?» или «Что нужно заказать в ближайшие 14 дней?»"
        )
    if reason == "ambiguous_intent":
        top_two = sorted(parsed["intent_scores"].items(), key=lambda kv: kv[1], reverse=True)[:2]
        names = " или ".join(cfg.INTENT_TITLES.get(k, k) for k, _ in top_two)
        details = []
        if parsed["sku"] is None:
            details.append("позицию")
        if parsed["period_days"] is None:
            details.append("период")
        tail = f" Укажите также {' и '.join(details)}." if details else ""
        return f"Требуется уточнение: вопрос можно понять по-разному — {names}. Что именно посчитать?{tail}"
    asks: list[str] = []
    if "sku" in parsed["missing"]:
        candidates = parsed["sku_candidates"]
        if len(candidates) > 1:
            options = " или ".join(f"{s} «{catalog[s]['name']}»" for s in candidates)
            word = shared_name_word(question, candidates, catalog)
            if word:
                pronoun = cfg.QUESTION_WORD_BY_ENDING.get(word[-1], cfg.QUESTION_WORD_DEFAULT)
                asks.append(f"{pronoun} именно {word} — {options}?")
            else:
                asks.append(f"какую именно позицию — {options}?")
        else:
            asks.append("по какой позиции (SKU или название)?")
    if "period_days" in parsed["missing"]:
        asks.append("на какой период (например, месяц, квартал, 90 дней)?")
    return "Требуется уточнение: " + " ".join(a if i == 0 else a[:1].upper() + a[1:] for i, a in enumerate(asks))


def _run_forecast(context: dict[str, Any], sku: str, days: int, load: float | None = None) -> dict[str, Any]:
    """Вызвать прогноз Части 2 с параметрами из контекста."""
    params: dict[str, Any] = {"catalog": context["catalog"]}
    if load is not None:
        params["load_factor"] = load
    return forecast_demand(context["history"], sku, days, params)


def _forecast_skus(context: dict[str, Any]) -> list[str]:
    """Позиции, по которым есть история расхода."""
    return [s for s in context["history"].get("weekly_consumption", {}) if s in context["catalog"]]


def _no_history_note(context: dict[str, Any]) -> str:
    """Примечание о позициях справочника без истории расхода."""
    skipped = [s for s in context["catalog"] if s not in _forecast_skus(context)]
    return f" Позиции без истории расхода ({', '.join(skipped)}) в расчёт не вошли." if skipped else ""


def _flag(result: dict[str, Any]) -> str:
    """Пометка о низкой уверенности прогноза."""
    if result["low_confidence"]:
        return (
            f" Низкая уверенность прогноза: {result['confidence']} ниже порога "
            f"{cfg.LOW_CONFIDENCE_THRESHOLD} (подтвердите, что новый уровень спроса сохранится)."
        )
    return f" Уверенность прогноза {result['confidence']}."


def _answer_forecast(parsed: dict[str, Any], context: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Ответ на forecast_purchase: объём, стоимость, расход с объяснением."""
    sku, days = parsed["sku"], parsed["period_days"]
    result = _run_forecast(context, sku, days, parsed["load_factor"])
    if result["forecast_demand"] is None:
        return f"По позиции {sku} прогноз построить нельзя: {result['explanation']}", [result]
    unit = result["unit"]
    parts = []
    if parsed["load_factor"] is not None:
        base = _run_forecast(context, sku, days)
        parts.append(
            f"При росте загрузки (коэффициент {fmt_qty(parsed['load_factor'])}) за {days} дн. уйдёт "
            f"{fmt_qty(result['forecast_demand'])} {unit} {sku} против {fmt_qty(base['forecast_demand'])} {unit} "
            f"при текущей загрузке."
        )
    parts.append(
        f"{sku}: на {days} дн. прогноз расхода {fmt_qty(result['forecast_demand'])} {unit}, "
        f"остаток {fmt_qty(result['current_stock'])} {unit}, в пути {fmt_qty(result['incoming_qty'])} {unit}, "
        f"страховой запас {fmt_qty(result['safety_stock'])} {unit}. Рекомендуемая закупка "
        f"{fmt_qty(result['recommended_qty'])} {unit} на сумму {fmt_money(result['estimated_cost'])}"
        f" (цена {fmt_money(result['price'])} за {unit})."
    )
    limit = parsed["budget_limit"]
    if limit is not None:
        item = context["catalog"][sku]
        if result["estimated_cost"] <= limit:
            parts.append(
                f"Укладывается в лимит {fmt_money(limit)}, "
                f"остаток лимита {fmt_money(limit - result['estimated_cost'])}."
            )
        else:
            fit = math.floor(limit / item["price"] / item["pack_size"]) * item["pack_size"]
            fit_text = (
                f"в лимит помещается {fmt_qty(fit)} {unit} на {fmt_money(fit * item['price'])}"
                if fit >= item["min_order_qty"] else "лимита не хватает даже на минимальную партию"
            )
            parts.append(
                f"Превышает лимит {fmt_money(limit)} на {fmt_money(result['estimated_cost'] - limit)}; {fit_text}."
            )
    parts.append(f"Расчёт: {result['explanation']}")
    if result["low_confidence"]:
        parts.append(_flag(result).strip())
    return " ".join(parts), [result]


def _answer_reorder(parsed: dict[str, Any], context: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Ответ на reorder_list: позиции, которые нужно заказать в окне."""
    window = parsed["period_days"] or cfg.DEFAULT_REORDER_WINDOW_DAYS
    as_of = date.fromisoformat(context["history"]["as_of"])
    results = [_run_forecast(context, s, cfg.DEFAULT_ORDER_COVER_DAYS) for s in _forecast_skus(context)]
    due = [r for r in results if r["order_by_date"] and (date.fromisoformat(r["order_by_date"]) - as_of).days <= window]
    if not due:
        text = f"В ближайшие {window} дн. ни одна позиция не опускается до точки заказа."
    else:
        lines = [
            f"{r['sku']}: заказать до {r['order_by_date']} (точка заказа {fmt_qty(r['reorder_point'])} {r['unit']}, "
            f"остаток {fmt_qty(r['current_stock'])} + в пути {fmt_qty(r['incoming_qty'])}), рекомендуемый объём "
            f"на {cfg.DEFAULT_ORDER_COVER_DAYS} дн. {fmt_qty(r['recommended_qty'])} {r['unit']} на "
            f"{fmt_money(r['estimated_cost'])}.{_flag(r)}"
            for r in due
        ]
        text = f"В ближайшие {window} дн. нужно заказать: " + " ".join(lines)
    rest = [f"{r['sku']} (точка заказа {r['order_by_date']})" for r in results if r not in due]
    if rest:
        text += " Позже: " + ", ".join(rest) + "."
    return text + _no_history_note(context), results


def _answer_budget(parsed: dict[str, Any], context: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Ответ на budget: сумма рекомендуемых закупок за период."""
    days = parsed["period_days"]
    results = [_run_forecast(context, s, days) for s in _forecast_skus(context)]
    total = round(sum(r["estimated_cost"] for r in results), cfg.MONEY_DECIMALS)
    lines = "; ".join(
        f"{r['sku']}: {fmt_qty(r['recommended_qty'])} {r['unit']} × {fmt_money(r['price'])} "
        f"= {fmt_money(r['estimated_cost'])}"
        f" (уверенность {r['confidence']})"
        for r in results
    )
    text = f"Бюджет закупок на {days} дн.: {fmt_money(total)}. Состав: {lines}."
    low = [r["sku"] for r in results if r["low_confidence"]]
    if low:
        text += f" По {', '.join(low)} низкая уверенность прогноза (ниже порога {cfg.LOW_CONFIDENCE_THRESHOLD})."
    limit = parsed["budget_limit"]
    if limit is not None:
        verdict = "укладываемся" if total <= limit else "превышение " + fmt_money(total - limit)
        text += f" Лимит {fmt_money(limit)}: {verdict}."
    return text + _no_history_note(context), results


def _within(iso_date: str | None, as_of: date, days: int) -> bool:
    """Проверить, что дата наступает не позже чем через ``days`` дней."""
    return iso_date is not None and (date.fromisoformat(iso_date) - as_of).days <= days


def _answer_deficit(parsed: dict[str, Any], context: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Ответ на deficit_risk: остатки, даты исчерпания, стоимость закрытия риска."""
    as_of = date.fromisoformat(context["history"]["as_of"])
    catalog = context["catalog"]
    skus = [parsed["sku"]] if parsed["sku"] else _forecast_skus(context)
    if parsed["sku"]:
        r = _run_forecast(context, parsed["sku"], cfg.DEFAULT_RISK_HORIZON_DAYS)
        if r["current_stock"] is None:
            return f"По позиции {parsed['sku']} нет данных об остатке.", [r]
        text = (
            f"{r['sku']} «{catalog[r['sku']]['name']}»: остаток {fmt_qty(r['current_stock'])} {r['unit']}, "
            f"в пути {fmt_qty(r['incoming_qty'])} {r['unit']}."
        )
        if r["forecast_demand"] is not None:
            text += (
                f" При текущем расходе {fmt_qty(r['avg_daily_consumption'])} {r['unit']}/день остатка хватит до "
                f"{r['stockout_date_on_hand']}, с учётом поставки в пути — до {r['stockout_date']}.{_flag(r)}"
            )
        return text, [r]

    results: list[dict[str, Any]] = []
    critical: list[dict[str, Any]] = []
    warning: list[dict[str, Any]] = []
    for sku in skus:
        horizon = catalog[sku]["lead_time_days"] if parsed["period_mode"] == "lead_time" else (
            parsed["period_days"] or cfg.DEFAULT_RISK_HORIZON_DAYS
        )
        r = _run_forecast(context, sku, cfg.DEFAULT_ORDER_COVER_DAYS)
        r["risk_horizon_days"] = horizon
        results.append(r)
        if _within(r["stockout_date"], as_of, horizon):
            critical.append(r)
        elif _within(r["stockout_date_on_hand"], as_of, horizon):
            warning.append(r)
    horizon_text = "до следующей поставки (срок поставки каждой позиции)" if parsed["period_mode"] == "lead_time" \
        else f"в ближайшие {results[0]['risk_horizon_days'] if results else cfg.DEFAULT_RISK_HORIZON_DAYS} дн."
    parts = [f"Риск дефицита {horizon_text}."]
    if critical:
        parts.append("Закончится даже с учётом товара в пути: " + "; ".join(
            f"{r['sku']} (остаток {fmt_qty(r['current_stock'])} {r['unit']}, в пути {fmt_qty(r['incoming_qty'])}, "
            f"расход {fmt_qty(r['avg_daily_consumption'])} {r['unit']}/день, закончится {r['stockout_date']}, "
            f"срок поставки {catalog[r['sku']]['lead_time_days']} дн.)" for r in critical) + ".")
    if warning:
        parts.append("Закончится текущий остаток, если товар в пути не придёт вовремя: " + "; ".join(
            f"{r['sku']} (остаток {fmt_qty(r['current_stock'])} {r['unit']} до {r['stockout_date_on_hand']}, "
            f"в пути {fmt_qty(r['incoming_qty'])} {r['unit']})" for r in warning) + ".")
    if not critical and not warning:
        parts.append("Позиций с риском нет.")
    at_risk = critical + warning
    if at_risk and ("стоит" in parsed.get("question", "") or "стоим" in parsed.get("question", "")):
        total = round(sum(r["estimated_cost"] for r in at_risk), cfg.MONEY_DECIMALS)
        parts.append(
            f"Закупка позиций в риске на {cfg.DEFAULT_ORDER_COVER_DAYS} дн.: " + "; ".join(
                f"{r['sku']} {fmt_qty(r['recommended_qty'])} {r['unit']} = {fmt_money(r['estimated_cost'])}"
                for r in at_risk) + f"; итого {fmt_money(total)}."
        )
    low = [r["sku"] for r in at_risk if r["low_confidence"]]
    if low:
        parts.append(
            f"По {', '.join(low)} низкая уверенность прогноза (ниже порога {cfg.LOW_CONFIDENCE_THRESHOLD})."
        )
    return " ".join(parts) + _no_history_note(context), results


def _answer_expiry(parsed: dict[str, Any], context: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Ответ на expiry_risk: в данных нет сроков годности партий."""
    period = ""
    if parsed["period_mode"] == "month_end":
        period = f" в ближайшие {parsed['period_days']} дн. (до конца месяца)"
    return (
        f"Не могу определить, какие партии сгорят{period}: в переданных данных нет сроков годности "
        "и остатков по партиям, только общий остаток по позициям. Чтобы посчитать риск списания, "
        "нужна выгрузка партий с датами истечения срока.",
        [],
    )


def _answer_price(parsed: dict[str, Any], context: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Ответ на price_dynamics: доступна только текущая цена справочника."""
    item = context["catalog"][parsed["sku"]]
    return (
        f"По {item['sku']} «{item['name']}» в справочнике есть только текущая цена "
        f"{fmt_money(item['price'])} за {item['unit']}; истории цен поставщика в данных нет, "
        "поэтому изменение цены посчитать нельзя. Нужна история закупочных цен.",
        [],
    )


HANDLERS: dict[str, Callable[[dict[str, Any], dict[str, Any]], tuple[str, list[dict[str, Any]]]]] = {
    "forecast_purchase": _answer_forecast,
    "reorder_list": _answer_reorder,
    "budget": _answer_budget,
    "deficit_risk": _answer_deficit,
    "expiry_risk": _answer_expiry,
    "price_dynamics": _answer_price,
}


def _location_in_data(location: str, context: dict[str, Any]) -> bool:
    """Есть ли в истории данные по конкретному филиалу."""
    return location in context["history"].get(cfg.LOCATION_HISTORY_KEY, {})


def _location_notice(location: str) -> str:
    """Начало ответа, когда спрошен филиал, а данные есть только по сети."""
    return (
        f"По филиалу «{location}» данных нет: остатки и расход есть только по сети в целом. "
        "Требуется уточнение: подходят ли показатели по всей сети или нужны данные филиала? "
        "Ниже — показатели по всей сети. "
    )


def answer_question(question: str, context: dict[str, Any]) -> tuple[dict[str, Any], float, str]:
    """Разобрать вопрос и ответить с объяснением расчёта.

    Args:
        question: Вопрос пользователя на русском языке.
        context: ``history`` (блок ``part2_history``), ``catalog``
            (справочник ``{sku: карточка}``) и необязательный ``use_llm``.

    Returns:
        Тройка (разобранные параметры, уверенность от 0 до 1, текст ответа).
        Если интент неоднозначен, позиция названа неоднозначно или не хватает
        параметров, интент равен ``unknown``, а текст содержит уточняющий вопрос.

        В параметрах два независимых флага:
        ``needs_clarification`` — требуется уточнение вопроса (``unknown``
        либо спрошен филиал, по которому нет данных);
        ``low_forecast_confidence`` — вопрос понят, но уверенность расчёта
        ниже ``LOW_CONFIDENCE_THRESHOLD`` (мало данных).
    """
    parsed = parse_question(question, context)
    parsed["question"] = question
    if parsed["intent"] == UNKNOWN:
        confidence = round(min(parsed["intent_share"], cfg.UNKNOWN_CONFIDENCE_CAP), cfg.CONFIDENCE_DECIMALS)
        parsed.update(needs_clarification=True, clarification_reason=parsed["reason"], low_forecast_confidence=False)
        return parsed, confidence, _clarification(parsed, question, context)

    draft, results = HANDLERS[parsed["intent"]](parsed, context)
    forecasts = [r for r in results if r.get("forecast_demand") is not None]
    if parsed["intent"] in ("expiry_risk", "price_dynamics"):
        data_conf = cfg.NO_DATA_CONFIDENCE
    elif parsed["sku"] and forecasts:
        data_conf = forecasts[0]["confidence"]
    elif forecasts:
        data_conf = min(r["confidence"] for r in forecasts)
    else:
        data_conf = cfg.NO_DATA_CONFIDENCE
    confidence = min(parsed["intent_share"], data_conf)
    location_missing = bool(parsed["location"]) and not _location_in_data(parsed["location"], context)
    if location_missing:
        confidence *= cfg.LOCATION_NOT_IN_DATA_PENALTY
        draft = _location_notice(parsed["location"]) + draft
    confidence = round(min(1.0, max(0.0, confidence)), cfg.CONFIDENCE_DECIMALS)
    parsed["needs_clarification"] = location_missing
    parsed["clarification_reason"] = "location_not_in_data" if location_missing else None
    parsed["low_forecast_confidence"] = confidence < cfg.LOW_CONFIDENCE_THRESHOLD or any(
        r["low_confidence"] for r in forecasts if parsed["sku"] in (None, r["sku"])
    )
    text = _DOUBLE_DOT.sub(".", draft)
    draft = text
    if llm_enabled(context):
        polished = llm_render_answer(draft)
        if polished is not None:
            text = polished
            parsed["answer_source"] = "llm"
    parsed.setdefault("answer_source", "template")
    return parsed, confidence, text
