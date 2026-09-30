"""Общие фикстуры тестов."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from spa_assistant.data import load_catalog, load_dataset, load_json


@pytest.fixture(scope="session")
def catalog() -> dict[str, dict[str, Any]]:
    """Справочник товаров из catalog.json."""
    return load_catalog()


@pytest.fixture(scope="session")
def dataset() -> dict[str, Any]:
    """Тестовый набор из dataset.json."""
    return load_dataset()


@pytest.fixture(scope="session")
def history(dataset: dict[str, Any]) -> dict[str, Any]:
    """История расхода и остатков (Часть 2)."""
    return dataset["part2_history"]


@pytest.fixture(scope="session")
def expected_movements() -> dict[str, dict[str, Any]]:
    """Ручная разметка записей M1–M8."""
    return load_json(Path(__file__).parent / "fixtures" / "expected_movements.json")


@pytest.fixture()
def context(history: dict[str, Any], catalog: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Контекст answer_question без обращения к LLM."""
    return {"history": history, "catalog": catalog, "use_llm": False}
