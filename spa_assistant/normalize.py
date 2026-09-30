"""Часть 1. Нормализация текстовых записей движения товара.

Разбор идёт последовательно: найденный фрагмент (партия, дата, объект,
SKU, документ) маскируется пробелами, чтобы его цифры не попали в разбор
количества на следующих шагах.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from spa_assistant import config as cfg
from spa_assistant.data import load_catalog

RESULT_KEYS: tuple[str, ...] = (
    "date", "sku", "location", "operation", "qty", "unit", "batch", "doc_no",
)

_ISO_DATE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_DOT_DATE = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4}|\d{2})\b")
_SLASH_DATE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})\b")
_TEXT_DATE = re.compile(
    r"\b(\d{1,2})\s+(" + "|".join(cfg.MONTHS_GENITIVE) + r")\s+(\d{4})(?:\s*г\.?)?",
    re.IGNORECASE,
)
_UNIT_RE = "|".join(sorted(map(re.escape, cfg.UNIT_ALIASES), key=len, reverse=True))
_NUM = r"(\d+(?:[.,]\d+)?)"
_SIGNED_NUM = r"(?:(?<=[\s:;,(])|^)([-−]?\d+(?:[.,]\d+)?)"
_CONTAINER_WITH_INNER = re.compile(
    _NUM + r"\s*" + cfg.CONTAINER_PATTERN + r"\s*(?:по\s*)" + _NUM
    + r"\s*(" + _UNIT_RE + r")(?![а-яёa-z])",
    re.IGNORECASE,
)
_CONTAINER_ONLY = re.compile(_NUM + r"\s*" + cfg.CONTAINER_PATTERN + r"(?![а-яёa-z])", re.IGNORECASE)
_QTY_WITH_UNIT = re.compile(
    _SIGNED_NUM + r"\s*(" + _UNIT_RE + r")\.?(?![а-яёa-z])", re.IGNORECASE,
)
_WORD = re.compile(r"[а-яёa-z]+", re.IGNORECASE)


def _mask(text: str, start: int, end: int) -> str:
    """Заменить фрагмент строки пробелами, сохранив длину строки."""
    return text[:start] + " " * (end - start) + text[end:]


def _to_float(raw: str) -> float:
    """Перевести строку с запятой или «−» в число с плавающей точкой."""
    value = raw.replace(" ", "").replace(",", ".")
    for minus in cfg.MINUS_CHARS:
        value = value.replace(minus, "-")
    return float(value)


def _expand_year(raw: str) -> int:
    """Превратить двузначный год в четырёхзначный."""
    year = int(raw)
    return year + cfg.TWO_DIGIT_YEAR_BASE if len(raw) == 2 else year


def _safe_date(year: int, month: int, day: int) -> date | None:
    """Собрать дату или вернуть None для невалидной комбинации."""
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_date(text: str) -> tuple[str | None, tuple[int, int] | None]:
    """Найти дату в строке и привести её к ISO-формату.

    Поддерживаются форматы ``2026-03-08``, ``05.03.2026``, ``07.03.26``,
    ``03/06/26`` (порядок частей задаётся ``SLASH_DATE_ORDER``) и
    ``1 марта 2026 г.``.

    Args:
        text: Исходная строка.

    Returns:
        Пара (ISO-дата или None, границы найденного фрагмента или None).
    """
    candidates: list[tuple[re.Match[str], date | None]] = []
    for match in _ISO_DATE.finditer(text):
        y, m, d = match.groups()
        candidates.append((match, _safe_date(int(y), int(m), int(d))))
    for match in _DOT_DATE.finditer(text):
        d, m, y = match.groups()
        candidates.append((match, _safe_date(_expand_year(y), int(m), int(d))))
    for match in _SLASH_DATE.finditer(text):
        first, second, y = match.groups()
        month, day = (first, second) if cfg.SLASH_DATE_ORDER == "MDY" else (second, first)
        candidates.append((match, _safe_date(_expand_year(y), int(month), int(day))))
    for match in _TEXT_DATE.finditer(text):
        d, month_word, y = match.groups()
        month = cfg.MONTHS_GENITIVE[month_word.lower()]
        candidates.append((match, _safe_date(int(y), month, int(d))))
    valid = [(m, d) for m, d in candidates if d is not None]
    if not valid:
        return None, None
    match, found = min(valid, key=lambda item: item[0].start())
    return found.isoformat(), match.span()


def _extract_first(patterns: tuple[str, ...], text: str, flags: int = 0) -> tuple[str | None, str]:
    """Найти первое совпадение из списка шаблонов и замаскировать его."""
    for pattern in patterns:
        match = re.search(pattern, text, flags)
        if match:
            return match.group(1), _mask(text, *match.span())
    return None, text


def parse_location(text: str) -> tuple[str | None, str]:
    """Извлечь объект (код MS-xx или известное название филиала).

    Args:
        text: Строка записи.

    Returns:
        Пара (нормализованный объект или None, строка с замаскированным объектом).
    """
    match = re.search(cfg.LOCATION_CODE_PATTERN, text, re.IGNORECASE)
    if match:
        return cfg.LOCATION_CODE_FORMAT.format(num=match.group(1)), _mask(text, *match.span())
    for pattern, name in cfg.NAMED_LOCATIONS.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return name, _mask(text, *match.span())
    return None, text


def _stem(token: str) -> str:
    """Грубая основа слова: первые ``NAME_STEM_LEN`` букв, ё заменена на е."""
    return token.lower().replace("ё", "е")[: cfg.NAME_STEM_LEN]


def _stems(text: str) -> set[str]:
    """Множество основ значимых слов строки."""
    return {
        _stem(tok) for tok in _WORD.findall(text)
        if len(tok) >= cfg.NAME_MIN_TOKEN_LEN and _stem(tok) not in cfg.NAME_STOP_STEMS
    }


def match_sku_by_name(text: str, catalog: dict[str, dict[str, Any]]) -> list[str]:
    """Сопоставить словесное описание позиции со справочником.

    Для каждой позиции считается число совпавших основ слов её названия.
    Возвращаются все позиции с максимальным счётом, если он не ниже
    ``NAME_MIN_MATCHED_STEMS``. Несколько кандидатов означают неоднозначность.

    Args:
        text: Текст записи или вопроса.
        catalog: Справочник ``{sku: карточка}``.

    Returns:
        Список SKU-кандидатов (пустой, если совпадений нет).
    """
    query = _stems(text)
    scores = {sku: len(query & _stems(item["name"])) for sku, item in catalog.items()}
    best = max(scores.values(), default=0)
    if best < cfg.NAME_MIN_MATCHED_STEMS:
        return []
    return [sku for sku, score in scores.items() if score == best]


def shared_name_word(text: str, candidates: list[str], catalog: dict[str, dict[str, Any]]) -> str | None:
    """Найти слово из вопроса, которое подходит ко всем позициям-кандидатам.

    Нужен для уточняющего вопроса: для «масла» при кандидатах OIL-001 и
    OIL-002 возвращается «масло» в той форме, в какой оно стоит в справочнике.

    Args:
        text: Текст вопроса.
        candidates: SKU, между которыми выбор неоднозначен.
        catalog: Справочник ``{sku: карточка}``.

    Returns:
        Слово из названия первой позиции-кандидата или None.
    """
    shared = _stems(text).intersection(*(_stems(catalog[s]["name"]) for s in candidates))
    for token in _WORD.findall(catalog[candidates[0]]["name"]):
        if _stem(token) in shared:
            return token.lower()
    return None


def parse_sku(text: str, catalog: dict[str, dict[str, Any]]) -> tuple[str | None, str]:
    """Найти SKU по коду в любом написании (``OIL-001``, ``oil 001``, ``WRAP 030``).

    Args:
        text: Строка записи.
        catalog: Справочник ``{sku: карточка}``.

    Returns:
        Пара (SKU из справочника или None, строка с замаскированным кодом).
    """
    prefixes = sorted({sku.split("-")[0] for sku in catalog}, key=len, reverse=True)
    pattern = cfg.SKU_CODE_TEMPLATE.format(prefixes="|".join(map(re.escape, prefixes)))
    for match in re.finditer(pattern, text, re.IGNORECASE):
        sku = f"{match.group(1).upper()}-{match.group(2)}"
        if sku in catalog:
            return sku, _mask(text, *match.span())
    return None, text


def parse_operation(text: str) -> str | None:
    """Определить тип операции по ключевым словам.

    Берётся операция, ключевое слово которой встречается раньше всех;
    при равной позиции используется порядок ``OPERATION_PRIORITY``.

    Args:
        text: Строка записи.

    Returns:
        Одно из receipt, consume, writeoff, return, correction или None.
    """
    lowered = text.lower()
    found: list[tuple[int, int, str]] = []
    for op, stems in cfg.OPERATION_KEYWORDS.items():
        for stem in stems:
            match = re.search(r"(?<![а-яё])" + re.escape(stem), lowered)
            if match:
                found.append((match.start(), cfg.OPERATION_PRIORITY.index(op), op))
    return min(found)[2] if found else None


def convert_to_base(value: float, unit: str, base_unit: str | None) -> tuple[float | None, str | None]:
    """Привести количество к базовой единице позиции.

    Args:
        value: Количество в исходной единице.
        unit: Исходная единица (любой синоним из ``UNIT_ALIASES``).
        base_unit: Базовая единица позиции или None, если позиция неизвестна.

    Returns:
        Пара (количество, единица). Если единицы несовместимы
        (например, кг для позиции в литрах), количество равно None.
    """
    canonical = cfg.UNIT_ALIASES.get(unit.lower().rstrip("."))
    if canonical is None:
        return None, base_unit
    target, factor = cfg.UNIT_TO_BASE[canonical]
    if base_unit is not None and target != base_unit:
        return None, base_unit
    return round(value * factor, cfg.QTY_DECIMALS), target


def parse_quantity(
    text: str, sku: str | None, catalog: dict[str, dict[str, Any]]
) -> tuple[float | None, str | None]:
    """Извлечь количество и привести его к базовой единице позиции.

    Поддерживаются варианты ``450 мл``, ``3,5кг``, ``−120 шт``,
    ``2 канистры по 5 л``, ``4 уп. по 50 пар`` и ``4 уп.`` (через
    ``pack_size`` позиции из справочника).

    Args:
        text: Строка записи с уже замаскированными датой, SKU и т. п.
        sku: Найденный SKU или None.
        catalog: Справочник ``{sku: карточка}``.

    Returns:
        Пара (количество в базовой единице или None, базовая единица или None).
    """
    base_unit = catalog[sku]["unit"] if sku else None
    match = _CONTAINER_WITH_INNER.search(text)
    if match:
        count, inner, unit = match.groups()
        return convert_to_base(_to_float(count) * _to_float(inner), unit, base_unit)
    match = _QTY_WITH_UNIT.search(text)
    if match:
        raw, unit = match.groups()
        return convert_to_base(_to_float(raw), unit, base_unit)
    match = _CONTAINER_ONLY.search(text)
    if match and sku:
        item = catalog[sku]
        return round(_to_float(match.group(1)) * item["pack_size"], cfg.QTY_DECIMALS), item["unit"]
    return None, base_unit


def normalize_movement(text: str, catalog: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Нормализовать текстовую запись движения товара.

    Args:
        text: Запись в свободной форме, например
            ``"05.03.2026 MS-01 приход OIL-001, 2 канистры по 5 л, НК-345"``.
        catalog: Справочник ``{sku: карточка}``; по умолчанию читается catalog.json.

    Returns:
        Словарь с ключами date (ISO), sku, location, operation,
        qty (в базовой единице позиции), unit, batch, doc_no.
        Ненайденные поля равны None.
    """
    catalog = catalog if catalog is not None else load_catalog()
    work = text
    batch, work = _extract_first(cfg.BATCH_PATTERNS, work, re.IGNORECASE)
    iso_date, span = parse_date(work)
    if span:
        work = _mask(work, *span)
    location, work = parse_location(work)
    sku, work = parse_sku(work, catalog)
    doc_no, work = _extract_first(cfg.DOC_NO_PATTERNS, work)
    operation = parse_operation(text)
    if sku is None:
        candidates = match_sku_by_name(work, catalog)
        sku = candidates[0] if len(candidates) == 1 else None
    qty, unit = parse_quantity(work, sku, catalog)
    return {
        "date": iso_date,
        "sku": sku,
        "location": location,
        "operation": operation,
        "qty": qty,
        "unit": unit,
        "batch": batch.upper() if batch else None,
        "doc_no": doc_no,
    }
