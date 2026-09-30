"""Часть 2. Детерминированный прогноз потребности и рекомендация закупки.

Модуль не обращается к LLM. Конвейер: заполнение пропусков, поиск сдвига
уровня спроса, фильтр выбросов Хампеля, модель Холта с затухающим трендом,
страховой запас, точка заказа, округление до упаковки и минимальной партии.
"""

from __future__ import annotations

import math
import statistics
from datetime import date, timedelta
from typing import Any

from spa_assistant import config as cfg


def round_order_qty(need: float, pack_size: float, min_order_qty: float) -> float:
    """Округлить потребность вверх до кратности упаковки с учётом минимальной партии.

    Args:
        need: Нетто-потребность в базовых единицах.
        pack_size: Кратность упаковки поставщика.
        min_order_qty: Минимальная партия поставщика.

    Returns:
        0, если потребности нет; иначе наименьшее кратное ``pack_size``,
        не меньшее ``max(need, min_order_qty)``.
    """
    if need <= cfg.ROUNDING_EPS:
        return 0.0
    target = max(need, min_order_qty)
    packs = math.ceil(target / pack_size - cfg.ROUNDING_EPS)
    return float(round(packs * pack_size, cfg.QTY_DECIMALS))


def fill_gaps(series: list[float | None]) -> tuple[list[float], list[int]]:
    """Заполнить пропуски линейной интерполяцией между соседними неделями.

    Пропуск не считается нулевым расходом: отсутствие записи означает
    отсутствие данных, а не отсутствие потребления. Пропуски на краях ряда
    заполняются ближайшим известным значением.

    Args:
        series: Недельный расход, пропуски обозначены None.

    Returns:
        Пара (заполненный ряд, индексы пропущенных недель).
    """
    known = [i for i, v in enumerate(series) if v is not None]
    missing = [i for i, v in enumerate(series) if v is None]
    if not known:
        return [], missing
    filled: list[float] = []
    for i, value in enumerate(series):
        if value is not None:
            filled.append(float(value))
            continue
        left = max((k for k in known if k < i), default=None)
        right = min((k for k in known if k > i), default=None)
        if left is None:
            filled.append(float(series[right]))
        elif right is None:
            filled.append(float(series[left]))
        else:
            share = (i - left) / (right - left)
            filled.append(series[left] + share * (series[right] - series[left]))
    return filled, missing


def detect_level_shift(series: list[float], params: dict[str, float]) -> int | None:
    """Найти устойчивый сдвиг уровня спроса в конце ряда.

    Хвост длиной не меньше ``level_shift_min_tail`` недель считается новым
    уровнем, если все его точки отличаются от медианы предшествующего
    участка не менее чем в ``level_shift_ratio`` раз (вверх или вниз).
    Одиночный всплеск этому условию не удовлетворяет и обрабатывается
    фильтром выбросов.

    Args:
        series: Недельный расход без пропусков.
        params: Параметры прогноза.

    Returns:
        Индекс первой недели нового уровня или None.
    """
    ratio = params["level_shift_ratio"]
    min_tail = int(params["level_shift_min_tail"])
    min_prior = int(params["level_shift_min_prior"])
    best: int | None = None
    for tail in range(min_tail, len(series) - min_prior + 1):
        start = len(series) - tail
        prior_median = statistics.median(series[:start])
        if prior_median <= 0:
            continue
        head = series[start:]
        up = min(head) >= ratio * prior_median
        down = max(head) * ratio <= prior_median
        if up or down:
            best = start
    return best


def theil_sen(values: list[float]) -> tuple[float, float]:
    """Устойчивая линейная регрессия Тейла-Сена.

    Args:
        values: Ряд значений с равным шагом.

    Returns:
        Пара (наклон, свободный член) прямой ``y = intercept + slope * t``.
    """
    n = len(values)
    if n < 2:
        return 0.0, values[0] if values else 0.0
    slopes = [
        (values[j] - values[i]) / (j - i) for i in range(n) for j in range(i + 1, n)
    ]
    slope = statistics.median(slopes)
    intercept = statistics.median(v - slope * t for t, v in enumerate(values))
    return slope, intercept


def hampel_clean(values: list[float], params: dict[str, float]) -> tuple[list[float], list[int]]:
    """Заменить одиночные выбросы значением тренда (фильтр Хампеля).

    Остатки считаются от прямой Тейла-Сена, поэтому плавный рост спроса
    не принимается за выброс. Точка считается выбросом, если её остаток
    отклоняется от медианы остатков больше чем на ``hampel_k`` робастных сигм
    и одновременно больше чем на ``outlier_min_rel_dev`` от значения тренда:
    на гладком ряду MAD очень мал, и без второго условия выбросом
    объявлялось бы обычное операционное колебание в несколько процентов.

    Args:
        values: Ряд без пропусков.
        params: Параметры прогноза.

    Returns:
        Пара (очищенный ряд, индексы заменённых точек).
    """
    if len(values) < params["min_points_for_outliers"]:
        return list(values), []
    slope, intercept = theil_sen(values)
    fitted = [intercept + slope * t for t in range(len(values))]
    residuals = [v - f for v, f in zip(values, fitted)]
    center = statistics.median(residuals)
    sigma = params["mad_to_sigma"] * statistics.median(abs(r - center) for r in residuals)
    if sigma <= cfg.ROUNDING_EPS:
        return list(values), []
    cleaned = list(values)
    outliers: list[int] = []
    for i, r in enumerate(residuals):
        statistically_extreme = abs(r - center) > params["hampel_k"] * sigma
        materially_large = abs(r) > params["outlier_min_rel_dev"] * abs(fitted[i])
        if statistically_extreme and materially_large:
            cleaned[i] = fitted[i]
            outliers.append(i)
    return cleaned, outliers


def holt_damped(values: list[float], params: dict[str, float]) -> tuple[float, float]:
    """Сглаживание Холта с затухающим трендом.

    Args:
        values: Очищенный недельный ряд.
        params: Параметры ``holt_alpha``, ``holt_beta``, ``holt_phi``,
            ``trend_init_points``, ``min_points_for_trend``.

    Returns:
        Пара (уровень на последней неделе, недельный тренд). Если точек
        меньше ``min_points_for_trend``, тренд принимается нулевым, а
        уровень равен среднему ряда.
    """
    if len(values) < params["min_points_for_trend"]:
        return statistics.fmean(values), 0.0
    alpha, beta, phi = params["holt_alpha"], params["holt_beta"], params["holt_phi"]
    init = min(int(params["trend_init_points"]), len(values)) - 1
    level = values[0]
    trend = (values[init] - values[0]) / init if init > 0 else 0.0
    for value in values[1:]:
        prev_level = level
        level = alpha * value + (1 - alpha) * (prev_level + phi * trend)
        trend = beta * (level - prev_level) + (1 - beta) * phi * trend
    return level, trend


def damped_weekly(level: float, trend: float, phi: float, weeks_ahead: float) -> float:
    """Недельный расход через ``weeks_ahead`` недель по модели Холта."""
    if abs(1 - phi) <= cfg.ROUNDING_EPS:
        damp = weeks_ahead
    else:
        damp = phi * (1 - phi ** weeks_ahead) / (1 - phi)
    return max(level + trend * damp, 0.0)


def daily_path(level: float, trend: float, phi: float, days: int, load: float) -> list[float]:
    """Прогноз дневного расхода на ``days`` дней вперёд."""
    return [
        damped_weekly(level, trend, phi, d / cfg.DAYS_PER_WEEK) / cfg.DAYS_PER_WEEK * load
        for d in range(1, days + 1)
    ]


def _first_day_below(start: float, path: list[float], threshold: float) -> int | None:
    """Номер первого дня, когда остаток опускается ниже порога."""
    remaining = start
    if remaining < threshold:
        return 0
    for day, usage in enumerate(path, start=1):
        remaining -= usage
        if remaining < threshold:
            return day
    return None


def compute_confidence(
    n_points: int,
    cv: float,
    missing_share: float,
    horizon_days: int,
    level_shift: bool,
) -> tuple[float, dict[str, float]]:
    """Рассчитать уверенность прогноза как произведение факторов.

    Args:
        n_points: Число фактически наблюдённых недель в используемом окне.
        cv: Коэффициент вариации остатков относительно тренда.
        missing_share: Доля пропущенных недель в используемом окне.
        horizon_days: Горизонт прогноза.
        level_shift: Обнаружен ли сдвиг уровня спроса.

    Returns:
        Пара (итоговая уверенность, значения отдельных факторов).
    """
    w = cfg.CONFIDENCE_WEIGHTS
    factors = {
        "history": min(1.0, n_points / w["full_history_weeks"]),
        "volatility": max(0.0, 1 - cv / w["cv_zero_confidence"]),
        "completeness": max(0.0, 1 - w["missing_penalty"] * missing_share),
        "horizon": 1 / (1 + horizon_days / w["horizon_scale_days"]),
        "regime": w["level_shift_factor"] if level_shift else 1.0,
    }
    value = math.prod(factors.values())
    value = min(w["max_confidence"], max(w["min_confidence"], value))
    return round(value, cfg.CONFIDENCE_DECIMALS), {
        k: round(v, cfg.CONFIDENCE_DECIMALS) for k, v in factors.items()
    }


def _sentence(text: str) -> str:
    """Сделать из фрагмента предложение: заглавная буква и одна точка в конце."""
    return text[:1].upper() + text[1:].rstrip(".") + ". "


def _offset_date(as_of: date, days: int | None) -> str | None:
    """Дата через ``days`` дней от ``as_of`` в ISO-формате или None."""
    return (as_of + timedelta(days=days)).isoformat() if days is not None else None


def _empty_result(sku: str, reason: str, stock: float | None, incoming: float | None) -> dict[str, Any]:
    """Результат для позиции, по которой прогноз построить нельзя."""
    return {
        "sku": sku, "avg_daily_consumption": None, "forecast_demand": None,
        "current_stock": stock, "incoming_qty": incoming, "safety_stock": None,
        "reorder_point": None, "recommended_qty": None, "estimated_cost": None,
        "stockout_date": None, "confidence": 0.0, "low_confidence": True,
        "explanation": reason,
    }


def forecast_demand(
    history: dict[str, Any], sku: str, horizon_days: int, params: dict[str, Any]
) -> dict[str, Any]:
    """Спрогнозировать потребность и рассчитать рекомендуемую закупку.

    Args:
        history: Блок ``part2_history``: ``as_of``, ``weekly_consumption``,
            ``current_stock``, ``incoming_qty``.
        sku: Код позиции.
        horizon_days: Горизонт планирования в днях.
        params: ``catalog`` (справочник ``{sku: карточка}``) и необязательные
            переопределения ``FORECAST_DEFAULTS``, например ``load_factor``
            для сценария роста загрузки.

    Returns:
        Словарь с полями avg_daily_consumption, forecast_demand,
        current_stock, incoming_qty, safety_stock, reorder_point,
        recommended_qty, estimated_cost, stockout_date, confidence,
        explanation и вспомогательными полями расчёта.
    """
    p: dict[str, Any] = {**cfg.FORECAST_DEFAULTS, **params}
    catalog: dict[str, dict[str, Any]] = p["catalog"]
    stock = history.get("current_stock", {}).get(sku)
    incoming = history.get("incoming_qty", {}).get(sku, 0.0)
    if sku not in catalog:
        return _empty_result(sku, f"Позиция {sku} отсутствует в справочнике.", stock, incoming)
    raw = history.get("weekly_consumption", {}).get(sku)
    if not raw or all(v is None for v in raw) or stock is None:
        return _empty_result(
            sku, f"По позиции {sku} нет истории расхода или остатка; прогноз не строится.",
            stock, incoming,
        )

    item = catalog[sku]
    unit = item["unit"]
    as_of = date.fromisoformat(history["as_of"])
    filled, missing = fill_gaps(raw)
    policy = p["level_shift_policy"]
    shift_start = detect_level_shift(filled, p) if policy != "none" else None
    series, window_start, replaced = filled, 0, []
    if shift_start is not None and policy == "new_level":
        window_start = shift_start
    elif shift_start is not None and policy == "outlier":
        prior_level = statistics.median(filled[:shift_start])
        replaced = list(range(shift_start, len(filled)))
        series = filled[:shift_start] + [prior_level] * len(replaced)
    window = series[window_start:]
    missing_in_window = [i for i in missing if i >= window_start]
    cleaned, outliers = hampel_clean(window, p)
    level, trend = holt_damped(cleaned, p)
    load = float(p["load_factor"])

    path = daily_path(level, trend, p["holt_phi"], max(horizon_days, int(p["stockout_search_days"])), load)
    horizon_path = path[:horizon_days]
    demand = sum(horizon_path)
    avg_daily = demand / horizon_days
    current_daily = level / cfg.DAYS_PER_WEEK * load
    safety = item["safety_stock_days"] * current_daily
    lead_demand = sum(path[: item["lead_time_days"]])
    rop = lead_demand + safety
    available = stock + incoming
    need = demand + safety - available
    qty = round_order_qty(need, item["pack_size"], item["min_order_qty"])
    cost = round(qty * item["price"], cfg.MONEY_DECIMALS)

    stockout_day = _first_day_below(available, path, 0.0)
    stockout_day_on_hand = _first_day_below(stock, path, 0.0)
    order_day = _first_day_below(available, path, rop)

    slope, intercept = theil_sen(cleaned)
    residuals = [v - (intercept + slope * t) for t, v in enumerate(cleaned)]
    mean_value = statistics.fmean(cleaned)
    cv = statistics.pstdev(residuals) / mean_value if mean_value > 0 else 1.0
    observed = len(window) - len(missing_in_window)
    confidence, factors = compute_confidence(
        observed, cv, len(missing_in_window) / len(window), horizon_days, shift_start is not None,
    )
    low_confidence = confidence < cfg.LOW_CONFIDENCE_THRESHOLD

    q = cfg.QTY_DECIMALS
    notes: list[str] = []
    if missing:
        weeks = ", ".join(str(i + 1) for i in missing)
        notes.append(f"пропуск данных за неделю {weeks} заполнен линейной интерполяцией (не нулём)")
    if shift_start is not None and policy == "new_level":
        notes.append(
            f"с недели {shift_start + 1} расход устойчиво изменился более чем в "
            f"{p['level_shift_ratio']} раза — принят как новый уровень спроса, "
            f"в расчёт взяты только последние {len(window)} нед."
        )
    if replaced:
        notes.append(
            f"сдвиг уровня с недели {shift_start + 1} трактован как выброс: недели "
            f"{shift_start + 1}–{len(filled)} заменены медианой прежнего уровня"
        )
    if outliers:
        notes.append("выбросы заменены значением тренда в неделях " + ", ".join(
            str(i + window_start + 1) for i in outliers))
    if load != 1.0:
        notes.append(f"применён сценарный коэффициент загрузки ×{load:g}")
    explanation = (
        f"{sku} «{item['name']}», данные на {as_of.isoformat()}, горизонт {horizon_days} дн. "
        f"Метод: сглаживание Холта с затухающим трендом (alpha={p['holt_alpha']}, "
        f"beta={p['holt_beta']}, phi={p['holt_phi']}) по {len(window)} нед. истории. "
        + (_sentence("; ".join(notes)) if notes else "")
        + f"Текущий уровень {round(level, q)} {unit}/нед, тренд {round(trend, q):+} {unit}/нед "
        f"(затухающий). Прогноз расхода за горизонт {round(demand, q)} {unit} "
        f"(в среднем {round(avg_daily, q)} {unit}/день). "
        f"Страховой запас = {item['safety_stock_days']} дн × {round(current_daily, q)} {unit}/день "
        f"= {round(safety, q)} {unit}. Точка заказа = расход за срок поставки "
        f"{item['lead_time_days']} дн ({round(lead_demand, q)}) + страховой запас = {round(rop, q)} {unit}. "
        f"Потребность = прогноз {round(demand, q)} + страховой запас {round(safety, q)} − остаток "
        f"{stock} − в пути {incoming} = {round(need, q)} {unit}. "
        f"Округление вверх до упаковки {item['pack_size']} и минимальной партии "
        f"{item['min_order_qty']}: {qty} {unit}. Стоимость {qty} × {item['price']} = {cost} руб. "
        f"Уверенность {confidence} (история {factors['history']}, волатильность "
        f"{factors['volatility']}, полнота {factors['completeness']}, горизонт "
        f"{factors['horizon']}, режим {factors['regime']})"
        + ("; ниже порога — низкая уверенность прогноза." if low_confidence else ".")
    )
    return {
        "sku": sku,
        "avg_daily_consumption": round(avg_daily, q),
        "forecast_demand": round(demand, q),
        "current_stock": stock,
        "incoming_qty": incoming,
        "safety_stock": round(safety, q),
        "reorder_point": round(rop, q),
        "recommended_qty": qty,
        "estimated_cost": cost,
        "stockout_date": _offset_date(as_of, stockout_day),
        "confidence": confidence,
        "explanation": explanation,
        "unit": unit,
        "price": item["price"],
        "horizon_days": horizon_days,
        "net_need": round(need, q),
        "stockout_date_on_hand": _offset_date(as_of, stockout_day_on_hand),
        "order_by_date": _offset_date(as_of, order_day),
        "low_confidence": low_confidence,
        "level_weekly": round(level, q),
        "trend_weekly": round(trend, q),
        "level_shift_week": shift_start + 1 if shift_start is not None else None,
        "missing_weeks": [i + 1 for i in missing],
        "outlier_weeks": [i + 1 for i in replaced] + [i + window_start + 1 for i in outliers],
        "level_shift_policy": policy,
        "confidence_factors": factors,
        "load_factor": load,
    }
