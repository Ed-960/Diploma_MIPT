"""Тесты универсальных helper-функций RAG/intent в CashierAgent."""

from __future__ import annotations

import pytest

from mcd_voice.llm.agent import (
    _extract_names_from_rag_context,
    _is_non_food_client_utterance,
    _normalize_rewrite_output,
    _rag_intent,
    _should_skip_rag,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("thanks", True),
        ("Okay, that's all.", True),
        ("A burger and fries, please", False),
        ("Can I get salad?", False),
    ],
)
def test_non_food_utterance_detection(text: str, expected: bool) -> None:
    assert _is_non_food_client_utterance(text) is expected


@pytest.mark.parametrize(
    "spec,expected",
    [
        ({"intent": "lookup"}, "lookup"),
        ({"intent": "alternatives"}, "alternatives"),
        ({"intent": "details"}, "details"),
        ({"intent": "calorie_tune"}, "calorie_tune"),
        ({"intent": "compare"}, "compare"),
        ({"intent": "unknown"}, "lookup"),
        ({}, "lookup"),
        (None, "lookup"),
    ],
)
def test_rag_intent_normalization(spec, expected: str) -> None:
    assert _rag_intent(spec) == expected


def test_extract_names_from_rag_context_unique_and_ordered() -> None:
    rag_context = (
        "- Side Salad (~15 kcal)\n"
        "- Big Mac (~550 kcal)\n"
        "- Side Salad (~15 kcal)\n"
        "Some other line\n"
    )
    assert _extract_names_from_rag_context(rag_context) == ["Side Salad", "Big Mac"]


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("food type: burgers", "food type burgers"),
        ("  all  ", "general menu items"),
        ("okay thanks", "general menu items"),
        ("grilled chicken, no beef", "grilled chicken no beef"),
    ],
)
def test_normalize_rewrite_output(raw: str, expected: str) -> None:
    assert _normalize_rewrite_output(raw) == expected


@pytest.mark.parametrize(
    "client_text,search_query,expected",
    [
        ("thanks", "thanks", True),
        ("I want a burger", "big mac burger", False),
        ("", "", True),
        ("ok", "general menu items", True),
    ],
)
def test_should_skip_rag(client_text: str, search_query: str, expected: bool) -> None:
    assert _should_skip_rag(client_text, search_query) is expected
