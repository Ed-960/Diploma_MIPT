"""Лексические исключения RAG (не аллергены Chroma)."""

from __future__ import annotations

from mcd_voice.menu.rag_lexical import (
    filter_rows_by_excluded_lexical,
    normalize_excluded_lexical_terms,
    row_violates_excluded_lexical,
)


def test_normalize_dedupes_and_skips_garbage() -> None:
    assert normalize_excluded_lexical_terms(["Beef", " beef ", "no", "x" * 99]) == ["beef"]


def test_row_violates_single_word_boundary() -> None:
    row = {
        "name": "Test Patty",
        "ingredients": "100% beef patty, bun",
        "description": "",
        "tag": "",
    }
    assert row_violates_excluded_lexical(row, ["beef"])
    assert not row_violates_excluded_lexical(row, ["chicken"])


def test_filter_keeps_order() -> None:
    rows = [
        {"name": "A", "ingredients": "chicken", "description": "", "tag": ""},
        {"name": "B", "ingredients": "beef", "description": "", "tag": ""},
        {"name": "C", "ingredients": "fish", "description": "", "tag": ""},
    ]
    out = filter_rows_by_excluded_lexical(rows, ["beef"])
    assert [r["name"] for r in out] == ["A", "C"]


def test_phrase_match() -> None:
    row = {
        "name": "Special",
        "ingredients": "big mac sauce",
        "description": "",
        "tag": "",
    }
    assert row_violates_excluded_lexical(row, ["big mac"])
    assert not row_violates_excluded_lexical(row, ["mac big"])
