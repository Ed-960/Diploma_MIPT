"""
Проверки search_menu: семантика, аллергены, калории.

Перед запуском: python load_menu_to_chroma.py

Запуск:
  python test_menu_search.py           — демо + автопроверки корректности
  python test_menu_search.py --demo    — только печать сценариев
  python test_menu_search.py --check   — только assert-проверки (тихо при успехе)
"""

from __future__ import annotations

import argparse
import sys

from mcd_voice.menu.search import search_menu
from mcd_voice.menu.search_checks import run_correctness_checks


def _fmt_allergens(allergens: list[str]) -> str:
    return ", ".join(allergens) if allergens else "нет"


def _print_block(title: str, query: str, rows: list[dict], extra: str = "") -> None:
    sep = "=" * 72
    print(f"\n{sep}\n{title}\nЗапрос: {query!r}{extra}\n{sep}")
    if not rows:
        print("(пусто)")
        return
    for i, row in enumerate(rows, start=1):
        name = row["name"]
        energy = row["energy"]
        ag = _fmt_allergens(row["allergens"])
        print(
            f"{i}. {name}\n"
            f"   Калории: {energy} ккал\n"
            f"   Аллергены: {ag}\n"
            f"   distance: {row['distance']:.4f}"
        )


def run_demos() -> None:
    queries = [
        "something spicy",
        "vegetarian burger",
        "dessert with low sugar",
    ]
    for q in queries:
        rows = search_menu(q, allergens_blacklist=None, top_k=3)
        _print_block("Семантический поиск (топ-3)", q, rows)

    q_chicken = "chicken"
    blacklist = ["Milk", "Gluten"]
    rows_f = search_menu(q_chicken, allergens_blacklist=blacklist, top_k=3)
    _print_block(
        f"Фильтр аллергенов {blacklist}",
        q_chicken,
        rows_f,
    )

    # Не больше N ккал (метаданные energy)
    cap = 250.0
    rows_cal = search_menu(
        "drink or light snack",
        allergens_blacklist=None,
        top_k=5,
        max_energy=cap,
    )
    _print_block(
        f"Калории ≤ {cap} ккал",
        "drink or light snack",
        rows_cal,
        extra=f"\nФильтр: max_energy={cap}",
    )

    # Комбинация: без молока и не выше порога калорий
    cap2 = 400.0
    rows_combo = search_menu(
        "burger",
        allergens_blacklist=["Milk"],
        top_k=5,
        max_energy=cap2,
    )
    _print_block(
        f"Без Milk и ≤ {cap2} ккал",
        "burger",
        rows_combo,
        extra=f"\nФильтр: blacklist Milk, max_energy={cap2}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Демо и проверки search_menu")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="только сценарии с печатью",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="только assert-проверки",
    )
    args = parser.parse_args()

    if args.check:
        run_correctness_checks()
        print("OK: все проверки корректности прошли.", file=sys.stderr)
        return

    if args.demo:
        run_demos()
        return

    run_demos()
    print("\n" + "=" * 72 + "\nАвтопроверки корректности фильтров…\n" + "=" * 72)
    run_correctness_checks()
    print("OK: все проверки корректности прошли.", file=sys.stderr)


if __name__ == "__main__":
    main()
