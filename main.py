"""Запуск всего пайплайна на тестовых данных одной командой: ``python main.py``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from spa_assistant import config as cfg
from spa_assistant.data import load_catalog, load_dataset, load_json
from spa_assistant.forecast import forecast_demand
from spa_assistant.llm import llm_enabled
from spa_assistant.normalize import normalize_movement
from spa_assistant.qa import answer_question
from spa_assistant.report import build_report

EXPECTED_PATH = cfg.PROJECT_ROOT / "tests" / "fixtures" / "expected_movements.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(description="Складской ИИ-помощник спа-оператора: прогон пайплайна.")
    parser.add_argument("--no-llm", action="store_true", help="не обращаться к LLM, только правила")
    parser.add_argument("--output", type=Path, default=cfg.OUTPUT_DIR, help="каталог для отчёта")
    parser.add_argument("--quiet", action="store_true", help="не печатать отчёт в консоль")
    return parser.parse_args(argv)


def run_pipeline(use_llm: bool) -> dict[str, Any]:
    """Прогнать три части на тестовых данных.

    Args:
        use_llm: Разрешить обращение к LLM в Части 3.

    Returns:
        Словарь с результатами всех частей и метаданными запуска.
    """
    dataset = load_dataset()
    catalog = load_catalog()
    history = dataset["part2_history"]

    part1 = [
        {"id": rec["id"], "text": rec["text"], "result": normalize_movement(rec["text"], catalog)}
        for rec in dataset["part1_movements"]
    ]
    part2 = [
        forecast_demand(history, sku, horizon, {"catalog": catalog})
        for sku in history["weekly_consumption"]
        for horizon in cfg.REPORT_HORIZONS_DAYS
    ]
    sensitivity = [
        {
            "sku": base["sku"],
            "horizon_days": base["horizon_days"],
            "variants": {
                policy: forecast_demand(
                    history, base["sku"], base["horizon_days"], {"catalog": catalog, "level_shift_policy": policy},
                )
                for policy in cfg.LEVEL_SHIFT_POLICIES
            },
        }
        for base in part2
        if base["level_shift_week"] is not None
    ]
    context = {"history": history, "catalog": catalog, "use_llm": use_llm}
    part3 = []
    for question in dataset["part3_questions"]:
        parsed, confidence, text = answer_question(question, context)
        part3.append({"question": question, "parsed": parsed, "confidence": confidence, "text": text})
    mode = "LLM (Claude) + проверка чисел" if llm_enabled(context) else "fallback на правилах (без LLM)"
    return {
        "meta": {"as_of": history["as_of"], "mode": mode},
        "part1": part1, "part2": part2, "sensitivity": sensitivity, "part3": part3,
    }


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI: считает, сохраняет JSON и Markdown-отчёт.

    Args:
        argv: Аргументы командной строки (по умолчанию ``sys.argv``).

    Returns:
        Код возврата процесса.
    """
    args = parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    results = run_pipeline(use_llm=not args.no_llm)
    report = build_report(
        results["part1"], load_json(EXPECTED_PATH), results["part2"], results["part3"], results["meta"],
        results["sensitivity"],
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (args.output / "report.md").write_text(report, encoding="utf-8")
    if not args.quiet:
        print(report)
    print(f"Отчёт: {args.output / 'report.md'}; данные: {args.output / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
