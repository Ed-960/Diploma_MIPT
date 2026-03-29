"""Проверки инвариантов выдачи search_menu (для скриптов и pytest)."""

from __future__ import annotations

from typing import Any

from mcd_voice.menu.search import search_menu


def assert_energy_bounds(
    rows: list[dict[str, Any]],
    *,
    max_energy: float | None = None,
    min_energy: float | None = None,
) -> None:
    """Проверяет, что у всех строк energy в заданных границах."""
    for r in rows:
        e = float(r["energy"])
        if max_energy is not None and e > float(max_energy) + 1e-6:
            raise AssertionError(f"energy {e} > max_energy {max_energy}: {r}")
        if min_energy is not None and e < float(min_energy) - 1e-6:
            raise AssertionError(f"energy {e} < min_energy {min_energy}: {r}")


def assert_forbidden_allergens_absent(
    rows: list[dict[str, Any]],
    forbidden: list[str],
) -> None:
    """Запрещённые токены не должны встречаться в списке allergens."""
    for r in rows:
        for token in forbidden:
            t = token.strip()
            if not t:
                continue
            if t in r["allergens"]:
                raise AssertionError(
                    f"forbidden {t!r} in {r['name']!r}: {r['allergens']}"
                )


def run_correctness_checks() -> None:
    """Полный набор проверок: фильтры Chroma не должны «протекать» в выдачу."""
    cap = 150.0
    rows = search_menu("anything", top_k=20, max_energy=cap)
    assert_energy_bounds(rows, max_energy=cap)

    lo = 500.0
    rows2 = search_menu("meal", top_k=20, min_energy=lo)
    assert_energy_bounds(rows2, min_energy=lo)

    rows3 = search_menu("chicken", allergens_blacklist=["Milk"], top_k=10)
    assert_forbidden_allergens_absent(rows3, ["Milk"])

    rows4 = search_menu(
        "food",
        allergens_blacklist=["Milk"],
        top_k=15,
        max_energy=300.0,
    )
    assert_energy_bounds(rows4, max_energy=300.0)
    assert_forbidden_allergens_absent(rows4, ["Milk"])

    rows5 = search_menu("coffee", top_k=2)
    if len(rows5) < 1:
        raise AssertionError("ожидалась хотя бы одна позиция по запросу 'coffee'")
