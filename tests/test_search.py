"""Pytest: семантический поиск и фильтры по метаданным (нужна загруженная Chroma)."""

from __future__ import annotations

import pytest

from mcd_voice.menu.search import search_menu
from mcd_voice.menu.search_checks import (
    assert_energy_bounds,
    assert_forbidden_allergens_absent,
    run_correctness_checks,
)


def test_search_returns_results_for_coffee() -> None:
    rows = search_menu("coffee", top_k=2)
    assert len(rows) >= 1
    assert "name" in rows[0] and "distance" in rows[0]


def test_max_energy_filter_respected() -> None:
    cap = 150.0
    rows = search_menu("anything", top_k=20, max_energy=cap)
    assert_energy_bounds(rows, max_energy=cap)


def test_min_energy_filter_respected() -> None:
    lo = 500.0
    rows = search_menu("meal", top_k=20, min_energy=lo)
    assert_energy_bounds(rows, min_energy=lo)


def test_milk_blacklist_respected() -> None:
    rows = search_menu("chicken", allergens_blacklist=["Milk"], top_k=10)
    assert_forbidden_allergens_absent(rows, ["Milk"])


def test_combo_milk_and_max_energy() -> None:
    rows = search_menu(
        "food",
        allergens_blacklist=["Milk"],
        top_k=15,
        max_energy=300.0,
    )
    assert_energy_bounds(rows, max_energy=300.0)
    assert_forbidden_allergens_absent(rows, ["Milk"])


def test_full_correctness_suite() -> None:
    """Тот же набор, что `python test_menu_search.py --check`."""
    run_correctness_checks()


@pytest.mark.parametrize(
    "query,top_k",
    [
        ("spicy", 3),
        ("vegetarian", 3),
    ],
)
def test_semantic_search_nonempty(query: str, top_k: int) -> None:
    rows = search_menu(query, top_k=top_k)
    assert len(rows) >= 1
