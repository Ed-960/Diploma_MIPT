"""Тесты универсальных helper-функций RAG/intent в CashierAgent."""

from __future__ import annotations

import pytest

from mcd_voice.llm.agent import (
    _detect_restriction_override,
    _extract_names_from_rag_context,
    _is_non_food_client_utterance,
    _normalize_rewrite_output,
    _rag_intent,
    _sanitize_cashier_response,
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


def test_restriction_override_requires_prior_item_warning() -> None:
    history = [
        {
            "speaker": "cashier",
            "text": "Just so you know, Big Mac contains milk. Would nuggets work?",
        },
        {"speaker": "client", "text": "No, I'll take the Big Mac anyway."},
    ]
    assert _detect_restriction_override(
        "No, I'll take the Big Mac anyway.",
        history,
        ["Big Mac"],
    ) is True


def test_restriction_override_ignores_unwarned_item() -> None:
    assert _detect_restriction_override(
        "I'll take it anyway.",
        [{"speaker": "cashier", "text": "Sure, anything else?"}],
        ["Big Mac"],
    ) is False


def test_sanitize_cashier_response_removes_catalog_dump_sentence() -> None:
    raw = (
        "Chicken McNuggets: Bite-sized pieces of breaded chicken. "
        "Would you like to add it to your order? "
        "Got it, one nuggets and a Coke."
    )
    cleaned = _sanitize_cashier_response(raw)
    assert "Bite-sized pieces" not in cleaned
    assert "Would you like to add it to your order" not in cleaned
    assert "Got it" in cleaned
