"""Тесты Части 1: разбор дат, единиц, операций и SKU."""

from __future__ import annotations

from typing import Any

import pytest

from spa_assistant.normalize import (
    RESULT_KEYS,
    convert_to_base,
    normalize_movement,
    parse_date,
    parse_operation,
    parse_quantity,
    parse_sku,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("05.03.2026 приход", "2026-03-05"),
        ("1 марта 2026 расход", "2026-03-01"),
        ("Возврат 12 марта 2026 г.: товар", "2026-03-12"),
        ("03/06/26 Сочи", "2026-03-06"),
        ("2026-03-08; MS-01", "2026-03-08"),
        ("07.03.26 ms-02", "2026-03-07"),
        ("31.02.2026 некорректная дата", None),
        ("без даты", None),
    ],
)
def test_parse_date_formats(text: str, expected: str | None) -> None:
    """Все форматы дат из задания приводятся к ISO."""
    assert parse_date(text)[0] == expected


@pytest.mark.parametrize(
    ("value", "unit", "base", "expected"),
    [
        (450, "мл", "л", 0.45),
        (3.5, "кг", "кг", 3.5),
        (1200, "г", "кг", 1.2),
        (2, "литра", "л", 2.0),
        (48, "пар", "пар", 48.0),
        (5, "кг", "л", None),
    ],
)
def test_convert_to_base(value: float, unit: str, base: str, expected: float | None) -> None:
    """Перевод в базовую единицу и отказ при несовместимых единицах."""
    assert convert_to_base(value, unit, base)[0] == expected


@pytest.mark.parametrize(
    ("text", "sku", "expected_qty", "expected_unit"),
    [
        ("450 мл", "OIL-001", 0.45, "л"),
        ("3,5кг", "WRAP-030", 3.5, "кг"),
        ("1,2 кг", "SCRB-020", 1.2, "кг"),
        ("2 канистры по 5 л", "OIL-001", 10.0, "л"),
        ("4 уп. по 50 пар", "CONS-051", 200.0, "пар"),
        ("3 уп.", "CONS-052", 300.0, "шт"),
        (": −120 шт", "CONS-052", -120.0, "шт"),
    ],
)
def test_parse_quantity(
    text: str, sku: str, expected_qty: float, expected_unit: str, catalog: dict[str, Any]
) -> None:
    """Количества в разных записях приводятся к базовой единице позиции."""
    assert parse_quantity(text, sku, catalog) == (expected_qty, expected_unit)


def test_dash_is_not_minus(catalog: dict[str, Any]) -> None:
    """Тире-разделитель перед количеством не делает его отрицательным."""
    assert parse_quantity("расход — 450 мл", "OIL-001", catalog)[0] == 0.45


@pytest.mark.parametrize("raw", ["OIL-001", "oil 001", "«oil 001»", "Oil_001"])
def test_parse_sku_spellings(raw: str, catalog: dict[str, Any]) -> None:
    """SKU распознаётся в разном написании."""
    assert parse_sku(f"расход {raw} 1 л", catalog)[0] == "OIL-001"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("приход товара", "receipt"),
        ("расход", "consume"),
        ("списание, истёк срок", "writeoff"),
        ("Возврат, брак упаковки", "return"),
        ("Корректировка остатков", "correction"),
        ("просто текст", None),
    ],
)
def test_parse_operation(text: str, expected: str | None) -> None:
    """Операция приводится к одному из пяти значений."""
    assert parse_operation(text) == expected


def test_all_movements_match_expected(
    dataset: dict[str, Any], catalog: dict[str, Any], expected_movements: dict[str, dict[str, Any]]
) -> None:
    """Записи M1–M8 разбираются в соответствии с ручной разметкой."""
    for record in dataset["part1_movements"]:
        result = normalize_movement(record["text"], catalog)
        assert tuple(result) == RESULT_KEYS
        assert result == expected_movements[record["id"]], record["id"]


def test_missing_fields_are_none(catalog: dict[str, Any]) -> None:
    """Ненайденные поля возвращаются как None."""
    result = normalize_movement("что-то непонятное", catalog)
    assert all(result[k] is None for k in RESULT_KEYS)


def test_sku_by_name_ambiguous_returns_none(catalog: dict[str, Any]) -> None:
    """Название, подходящее двум позициям, не угадывается."""
    assert normalize_movement("Приход масла 2 л", catalog)["sku"] is None
