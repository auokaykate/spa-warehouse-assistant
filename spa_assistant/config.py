"""Конфигурация: все пороги, коэффициенты, словари и шаблоны разбора.

Модуль не содержит логики. Любое число, влияющее на результат расчёта,
объявлено здесь, чтобы его можно было менять без правки алгоритмов.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATASET_PATH: Final[Path] = PROJECT_ROOT / "dataset.json"
CATALOG_PATH: Final[Path] = PROJECT_ROOT / "catalog.json"
OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "output"

DAYS_PER_WEEK: Final[int] = 7
DAYS_PER_MONTH: Final[int] = 30
DAYS_PER_QUARTER: Final[int] = 90
DAYS_PER_HALF_YEAR: Final[int] = 180
DAYS_PER_YEAR: Final[int] = 365
TWO_DIGIT_YEAR_BASE: Final[int] = 2000

MONEY_DECIMALS: Final[int] = 2
QTY_DECIMALS: Final[int] = 3
CONFIDENCE_DECIMALS: Final[int] = 2
ROUNDING_EPS: Final[float] = 1e-9

SLASH_DATE_ORDER: Final[str] = "MDY"

MONTHS_GENITIVE: Final[dict[str, int]] = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4,
    "мая": 5, "июня": 6, "июля": 7, "августа": 8,
    "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}

UNIT_ALIASES: Final[dict[str, str]] = {
    "мл": "мл", "миллилитр": "мл", "миллилитров": "мл", "миллилитра": "мл",
    "л": "л", "литр": "л", "литра": "л", "литров": "л",
    "г": "г", "гр": "г", "грамм": "г", "грамма": "г", "граммов": "г",
    "кг": "кг", "килограмм": "кг", "килограмма": "кг", "килограммов": "кг",
    "шт": "шт", "штук": "шт", "штуки": "шт", "штука": "шт",
    "пар": "пар", "пара": "пар", "пары": "пар",
}

UNIT_TO_BASE: Final[dict[str, tuple[str, float]]] = {
    "мл": ("л", 0.001),
    "л": ("л", 1.0),
    "г": ("кг", 0.001),
    "кг": ("кг", 1.0),
    "шт": ("шт", 1.0),
    "пар": ("пар", 1.0),
}

CONTAINER_PATTERN: Final[str] = (
    r"(?:канистр\w*|уп\.?|упак\w*|флакон\w*|бутыл\w*|коробк\w*|короб\w*|"
    r"пачк\w*|мешк\w*|мешок|ведр\w*|ведро|банк\w*)"
)

OPERATION_KEYWORDS: Final[dict[str, tuple[str, ...]]] = {
    "correction": ("корректир", "инвентаризац", "пересорт"),
    "writeoff": ("списан", "утилизац", "брак партии", "бой"),
    "return": ("возврат", "вернул", "вернули"),
    "receipt": ("приход", "поступлен", "оприходов", "получен", "приём", "прием"),
    "consume": ("расход", "израсход", "выдан", "использован"),
}

OPERATION_PRIORITY: Final[tuple[str, ...]] = (
    "correction", "return", "writeoff", "receipt", "consume",
)

LOCATION_CODE_PATTERN: Final[str] = r"\bms[-\s]?(\d{2})\b"
LOCATION_CODE_FORMAT: Final[str] = "MS-{num}"
NAMED_LOCATIONS: Final[dict[str, str]] = {
    r"\bсоч\w*": "Сочи",
    r"\bкрасн\w*\s+полян\w*": "Красная Поляна",
}

BATCH_PATTERNS: Final[tuple[str, ...]] = (
    r"парт(?:ия|\.)?\s*[:№]?\s*([A-ZА-Я]{1,3}-[A-ZА-Я]+-\d{3}-\d+)",
    r"\b([BВ]-[A-ZА-Я]+-\d{3}-\d+)\b",
)

DOC_NO_PATTERNS: Final[tuple[str, ...]] = (
    r"(?:накл\w*\.?|док\w*\.?|№)\s*([A-Za-zА-Яа-я]{0,4}-?\d+)",
    r"\b([А-ЯA-Z]{2,4}-\d{2,})\b",
)

SKU_CODE_TEMPLATE: Final[str] = r"(?<![A-Za-z0-9-])({prefixes})[\s\-_]*(\d{{3}})(?![\d-])"

NAME_STEM_LEN: Final[int] = 4
NAME_MIN_TOKEN_LEN: Final[int] = 4
NAME_MIN_MATCHED_STEMS: Final[int] = 1
NAME_STOP_STEMS: Final[frozenset[str]] = frozenset({"сколь", "для", "тела"})

QTY_NUMBER_PATTERN: Final[str] = r"(?<![\w.,/-])([-−–]?\s?\d+(?:[.,]\d+)?)"
MINUS_CHARS: Final[tuple[str, ...]] = ("−", "–")

LEVEL_SHIFT_POLICIES: Final[tuple[str, ...]] = ("new_level", "outlier", "none")

FORECAST_DEFAULTS: Final[dict[str, float | str]] = {
    "level_shift_policy": "new_level",
    "holt_alpha": 0.5,
    "holt_beta": 0.3,
    "holt_phi": 0.9,
    "trend_init_points": 4,
    "min_points_for_trend": 6,
    "level_shift_ratio": 1.5,
    "level_shift_min_tail": 3,
    "level_shift_min_prior": 4,
    "hampel_k": 3.5,
    "outlier_min_rel_dev": 0.25,
    "mad_to_sigma": 1.4826,
    "min_points_for_outliers": 5,
    "load_factor": 1.0,
    "stockout_search_days": 365,
}

CONFIDENCE_WEIGHTS: Final[dict[str, float]] = {
    "full_history_weeks": 8.0,
    "cv_zero_confidence": 0.5,
    "missing_penalty": 1.0,
    "horizon_scale_days": 365.0,
    "level_shift_factor": 0.8,
    "min_confidence": 0.0,
    "max_confidence": 1.0,
}

LOW_CONFIDENCE_THRESHOLD: Final[float] = 0.6

INTENTS: Final[tuple[str, ...]] = (
    "forecast_purchase", "reorder_list", "budget",
    "deficit_risk", "expiry_risk", "price_dynamics",
)

INTENT_KEYWORDS: Final[dict[str, dict[str, float]]] = {
    "forecast_purchase": {
        "закуп": 1.5, "заказать": 1.0, "сколько": 0.5, "посчитай": 0.5,
        "стоить": 0.5, "стоит": 0.5, "уйдёт": 2.0, "уйдет": 2.0,
        "потребност": 1.5, "прогноз": 1.5, "израсход": 1.5,
        "загрузк": 1.0, "расход": 1.0,
    },
    "reorder_list": {
        "что нужно заказать": 3.0, "что заказать": 3.0, "список": 1.0,
        "дозаказ": 2.0, "ближайшие": 1.0, "точк заказ": 2.0,
    },
    "budget": {
        "бюджет": 3.0, "стоить": 0.5, "стоит": 0.5, "стоимост": 0.5,
    },
    "deficit_risk": {
        "дефицит": 3.0, "риск": 2.0, "закончится": 3.0, "кончится": 3.0,
        "хватит": 2.5, "осталось": 2.5, "остат": 2.5,
        "до следующей поставки": 1.5,
    },
    "expiry_risk": {
        "сгорят": 3.0, "сгорит": 3.0, "срок годност": 3.0, "просроч": 3.0,
        "истека": 3.0, "истёк": 3.0, "годност": 2.0,
    },
    "price_dynamics": {
        "подорожал": 3.0, "подешевел": 3.0, "цена": 2.0, "цены": 2.0,
        "динамик": 1.5, "рост цен": 2.0,
    },
}

OUT_OF_SCOPE_KEYWORDS: Final[dict[str, float]] = {
    "погод": 3.0, "курс валют": 3.0, "новост": 3.0, "анекдот": 3.0,
    "привезут": 3.0, "когда придёт": 3.0, "когда придет": 3.0,
    "почему": 1.5, "по сравнению": 1.5,
}

INTENT_TITLES: Final[dict[str, str]] = {
    "forecast_purchase": "закупка конкретной позиции",
    "reorder_list": "список к заказу",
    "budget": "общий бюджет закупок",
    "deficit_risk": "риск дефицита",
    "expiry_risk": "риск списания по сроку годности",
    "price_dynamics": "динамика цен",
    "out_of_scope": "вне области склада",
}

LOCATION_HISTORY_KEY: Final[str] = "by_location"

QUESTION_WORD_BY_ENDING: Final[dict[str, str]] = {
    "о": "какое", "е": "какое", "а": "какая", "я": "какая", "и": "какие", "ы": "какие",
}
QUESTION_WORD_DEFAULT: Final[str] = "какой"

INTENT_MIN_SCORE: Final[float] = 1.0
INTENT_MARGIN_THRESHOLD: Final[float] = 0.25

INTENT_REQUIRED_ENTITIES: Final[dict[str, tuple[str, ...]]] = {
    "forecast_purchase": ("sku", "period_days"),
    "reorder_list": (),
    "budget": ("period_days",),
    "deficit_risk": (),
    "expiry_risk": (),
    "price_dynamics": ("sku",),
}

REPORT_HORIZONS_DAYS: Final[tuple[int, ...]] = (30, 90)
DEFAULT_REORDER_WINDOW_DAYS: Final[int] = 14
DEFAULT_RISK_HORIZON_DAYS: Final[int] = 30
DEFAULT_ORDER_COVER_DAYS: Final[int] = 30

CURRENT_MONTH_PATTERN: Final[str] = r"\b(?:этом|текущем|этот|текущий)\s+месяц\w*"
NEXT_DELIVERY_PATTERN: Final[str] = r"до\s+следующей\s+поставк\w*"

PERIOD_WORDS: Final[dict[str, int]] = {
    r"\bполгода\b|\bполугоди\w*": DAYS_PER_HALF_YEAR,
    r"\bквартал\w*": DAYS_PER_QUARTER,
    r"\bгод\b|\bгода\b": DAYS_PER_YEAR,
    r"\bмесяц\w*": DAYS_PER_MONTH,
    r"\bнедел[юяи]\b": DAYS_PER_WEEK,
}

PERIOD_UNIT_DAYS: Final[dict[str, int]] = {
    "дн": 1, "ден": 1, "сут": 1,
    "недел": DAYS_PER_WEEK,
    "месяц": DAYS_PER_MONTH,
    "квартал": DAYS_PER_QUARTER,
    "год": DAYS_PER_YEAR, "лет": DAYS_PER_YEAR,
}

NUMBER_WORDS: Final[dict[str, int]] = {
    "один": 1, "одну": 1, "одного": 1, "два": 2, "две": 2, "двух": 2,
    "три": 3, "трёх": 3, "трех": 3, "четыре": 4, "четырёх": 4,
    "пять": 5, "шесть": 6, "семь": 7, "восемь": 8, "девять": 9,
    "десять": 10, "одиннадцать": 11, "двенадцать": 12,
}

MONEY_MULTIPLIERS: Final[dict[str, float]] = {
    "тыс": 1_000.0, "тысяч": 1_000.0, "тысячи": 1_000.0, "k": 1_000.0,
    "млн": 1_000_000.0, "миллион": 1_000_000.0, "миллиона": 1_000_000.0,
    "руб": 1.0, "рублей": 1.0, "₽": 1.0,
}

PERCENT_PATTERN: Final[str] = r"(\d+(?:[.,]\d+)?)\s*%"
LOAD_GROWTH_HINTS: Final[tuple[str, ...]] = ("загрузк", "спрос", "поток", "клиент")
LOAD_DECLINE_HINTS: Final[tuple[str, ...]] = ("снизит", "упадёт", "упадет", "уменьш")
PERCENT_BASE: Final[float] = 100.0

LOCATION_NOT_IN_DATA_PENALTY: Final[float] = 0.9
NO_DATA_CONFIDENCE: Final[float] = 0.5
UNKNOWN_CONFIDENCE_CAP: Final[float] = 0.3

LLM_MODEL: Final[str] = "claude-opus-5-5"
LLM_MAX_TOKENS: Final[int] = 2048
LLM_EFFORT: Final[str] = "low"
LLM_TIMEOUT_SECONDS: Final[float] = 60.0
LLM_ENABLE_ENV: Final[str] = "SPA_USE_LLM"
LLM_API_KEY_ENV: Final[str] = "ANTHROPIC_API_KEY"
LLM_FALLBACK_BETA: Final[str] = "server-side-fallback-2026-07-01"
LLM_NUMBER_TOLERANCE: Final[float] = 0.01
