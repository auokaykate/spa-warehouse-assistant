"""Загрузка тестовых данных и справочника товаров."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from spa_assistant.config import CATALOG_PATH, DATASET_PATH


def load_json(path: Path) -> Any:
    """Прочитать JSON-файл в кодировке UTF-8.

    Args:
        path: Путь к файлу.

    Returns:
        Разобранное содержимое файла.
    """
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_catalog(path: Path = CATALOG_PATH) -> dict[str, dict[str, Any]]:
    """Загрузить справочник товаров и проиндексировать его по SKU.

    Args:
        path: Путь к catalog.json.

    Returns:
        Словарь ``{sku: карточка позиции}``.
    """
    return {item["sku"]: item for item in load_json(path)}


def load_dataset(path: Path = DATASET_PATH) -> dict[str, Any]:
    """Загрузить тестовый набор (записи движения, история, вопросы).

    Args:
        path: Путь к dataset.json.

    Returns:
        Словарь с ключами ``part1_movements``, ``part2_history``,
        ``part3_questions``.
    """
    return load_json(path)
