"""Тесты Части 2: округление закупки, пропуски, сдвиг уровня, confidence."""

from __future__ import annotations

from typing import Any

import pytest

from spa_assistant import config as cfg
from spa_assistant.forecast import (
    detect_level_shift,
    fill_gaps,
    forecast_demand,
    hampel_clean,
    round_order_qty,
)

REQUIRED_KEYS = (
    "avg_daily_consumption", "forecast_demand", "current_stock", "incoming_qty", "safety_stock",
    "reorder_point", "recommended_qty", "estimated_cost", "stockout_date", "confidence", "explanation",
)


@pytest.mark.parametrize(
    ("need", "pack", "moq", "expected"),
    [
        (15.52, 5, 10, 20.0),
        (3.0, 5, 10, 10.0),
        (10.0, 5, 10, 10.0),
        (10.01, 5, 10, 15.0),
        (36.014, 1, 5, 37.0),
        (1.0, 1, 5, 5.0),
        (120.0, 50, 100, 150.0),
        (230.0, 100, 200, 300.0),
        (150.0, 100, 150, 200.0),
        (0.0, 5, 10, 0.0),
        (-4.0, 1, 5, 0.0),
    ],
)
def test_round_order_qty(need: float, pack: float, moq: float, expected: float) -> None:
    """Округление вверх до кратности упаковки и не ниже минимальной партии."""
    assert round_order_qty(need, pack, moq) == expected


def test_round_order_qty_float_noise() -> None:
    """Погрешность float не добавляет лишнюю упаковку."""
    assert round_order_qty(0.1 + 0.2, 0.3, 0.3) == pytest.approx(0.3)


def test_fill_gaps_interpolates() -> None:
    """Пропуск заполняется интерполяцией, а не нулём."""
    filled, missing = fill_gaps([6.0, None, 6.4])
    assert missing == [1]
    assert filled[1] == pytest.approx(6.2)


def test_level_shift_detected_for_scrub(history: dict[str, Any]) -> None:
    """Устойчивый рост расхода скраба вдвое распознаётся как новый уровень."""
    series, _ = fill_gaps(history["weekly_consumption"]["SCRB-020"])
    assert detect_level_shift(series, cfg.FORECAST_DEFAULTS) == 8


def test_gradual_trend_is_not_level_shift(history: dict[str, Any]) -> None:
    """Плавный рост масла не считается сдвигом уровня."""
    series, _ = fill_gaps(history["weekly_consumption"]["OIL-001"])
    assert detect_level_shift(series, cfg.FORECAST_DEFAULTS) is None


def test_single_spike_is_outlier() -> None:
    """Одиночный всплеск заменяется значением тренда."""
    params = {**cfg.FORECAST_DEFAULTS}
    cleaned, outliers = hampel_clean([6.0, 6.2, 6.1, 18.0, 6.3, 6.2, 6.4, 6.1], params)
    assert outliers == [3]
    assert cleaned[3] < 7.0


def test_forecast_contract_and_rounding(history: dict[str, Any], catalog: dict[str, Any]) -> None:
    """Результат содержит все поля, объём кратен упаковке, стоимость по цене справочника."""
    for sku in history["weekly_consumption"]:
        for horizon in cfg.REPORT_HORIZONS_DAYS:
            result = forecast_demand(history, sku, horizon, {"catalog": catalog})
            assert all(k in result for k in REQUIRED_KEYS)
            item = catalog[sku]
            qty = result["recommended_qty"]
            assert qty == 0 or qty >= item["min_order_qty"]
            assert (qty / item["pack_size"]) == pytest.approx(round(qty / item["pack_size"]))
            assert result["estimated_cost"] == pytest.approx(qty * item["price"])
            assert 0.0 <= result["confidence"] <= 1.0


def test_forecast_is_deterministic(history: dict[str, Any], catalog: dict[str, Any]) -> None:
    """Повторный расчёт даёт идентичный результат."""
    first = forecast_demand(history, "OIL-001", 90, {"catalog": catalog})
    assert first == forecast_demand(history, "OIL-001", 90, {"catalog": catalog})


def test_confidence_lower_for_short_history(history: dict[str, Any], catalog: dict[str, Any]) -> None:
    """После сдвига уровня история короче, confidence ниже порога уточнения."""
    scrub = forecast_demand(history, "SCRB-020", 30, {"catalog": catalog})
    oil = forecast_demand(history, "OIL-001", 30, {"catalog": catalog})
    assert scrub["confidence"] < cfg.LOW_CONFIDENCE_THRESHOLD <= oil["confidence"]


def test_confidence_lower_for_volatile_series(catalog: dict[str, Any]) -> None:
    """Высокая волатильность снижает confidence."""
    base = {"as_of": "2026-09-15", "current_stock": {"OIL-001": 10.0}, "incoming_qty": {"OIL-001": 0.0}}
    stable = {**base, "weekly_consumption": {"OIL-001": [10.0, 10.2, 9.9, 10.1, 10.0, 9.8, 10.1, 10.0]}}
    noisy = {**base, "weekly_consumption": {"OIL-001": [10.0, 5.0, 14.0, 6.0, 13.0, 7.0, 12.0, 8.0]}}
    c_stable = forecast_demand(stable, "OIL-001", 30, {"catalog": catalog})["confidence"]
    c_noisy = forecast_demand(noisy, "OIL-001", 30, {"catalog": catalog})["confidence"]
    assert c_noisy < c_stable


def test_missing_week_lowers_confidence(history: dict[str, Any], catalog: dict[str, Any]) -> None:
    """Пропуск недели отражён в объяснении и в факторе полноты."""
    result = forecast_demand(history, "WRAP-030", 30, {"catalog": catalog})
    assert result["missing_weeks"] == [5]
    assert result["confidence_factors"]["completeness"] < 1.0
    assert "интерполяцией" in result["explanation"]


def test_load_factor_scales_demand(history: dict[str, Any], catalog: dict[str, Any]) -> None:
    """Сценарий роста загрузки на 20% увеличивает прогноз ровно в 1.2 раза."""
    base = forecast_demand(history, "OIL-001", 30, {"catalog": catalog})
    boosted = forecast_demand(history, "OIL-001", 30, {"catalog": catalog, "load_factor": 1.2})
    assert boosted["forecast_demand"] == pytest.approx(base["forecast_demand"] * 1.2, rel=1e-3)


def test_no_history_returns_empty(history: dict[str, Any], catalog: dict[str, Any]) -> None:
    """Без истории числа не придумываются."""
    result = forecast_demand(history, "OIL-002", 30, {"catalog": catalog})
    assert result["forecast_demand"] is None and result["recommended_qty"] is None
    assert result["confidence"] == 0.0


def test_level_shift_policies_differ(history: dict[str, Any], catalog: dict[str, Any]) -> None:
    """Трактовка сдвига как выброса даёт меньший прогноз, чем новый уровень."""
    new_level = forecast_demand(history, "SCRB-020", 30, {"catalog": catalog})
    as_outlier = forecast_demand(history, "SCRB-020", 30, {"catalog": catalog, "level_shift_policy": "outlier"})
    assert new_level["level_shift_week"] == 9
    assert as_outlier["forecast_demand"] < new_level["forecast_demand"]
    assert as_outlier["outlier_weeks"] == [9, 10, 11, 12]
