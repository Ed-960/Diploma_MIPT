"""Детектор «другие варианты кофе» для RAG rewrite."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mcd_voice.llm.agent import (
    _COFFEE_BROAD_RAG_QUERY,
    _client_asks_coffee_variant_count,
    _client_wants_coffee_variety_scan,
    _coffee_count_response_from_rag_context,
    _enrich_client_text_for_menu_rag,
    _rewrite_query,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "dont you have any other kind of coffee besides all these kinds you called ?",
            True,
        ),
        ("yes, I mean another coffee", True),
        ("What other kinds of coffee do you have?", True),
        ("I'd like a black coffee", False),
        ("something else", False),
        ("другой кофе есть?", True),
        ("coffee other variants", True),
        ("other variants of coffee please", True),
    ],
)
def test_coffee_variety_scan(text: str, expected: bool) -> None:
    assert _client_wants_coffee_variety_scan(text) is expected


def test_enrich_short_followup_carries_coffee_topic_from_history() -> None:
    history = [
        {"speaker": "client", "text": "Which kind of coffee do you have"},
        {"speaker": "cashier", "text": "We have black coffee and cold coffee."},
        {"speaker": "client", "text": "what else"},
    ]
    out = _enrich_client_text_for_menu_rag(history, "what else")
    assert "what else" in out
    assert "coffee" in out.lower()


def test_enrich_other_variants_after_coffee_thread() -> None:
    history = [
        {"speaker": "client", "text": "coffee other variants"},
        {"speaker": "cashier", "text": "Cold Coffee McFloat"},
        {"speaker": "client", "text": "other variants"},
    ]
    out = _enrich_client_text_for_menu_rag(history, "other variants")
    assert "other variants" in out
    assert "premium roast" in out.lower() or "latte" in out.lower()


def test_rewrite_query_overrides_narrow_coffee_with_broad_keywords(monkeypatch) -> None:
    """Узкий ответ mini-LLM («coffee») заменяется на разнообразный запрос к Chroma."""
    monkeypatch.setattr("mcd_voice.llm.agent._call_llm", lambda *a, **k: "coffee")
    out = _rewrite_query(
        "What other kinds of coffee do you have?",
        MagicMock(),
        "rewrite-model",
        llm_trace=None,
    )
    assert out == _COFFEE_BROAD_RAG_QUERY


@pytest.mark.parametrize(
    "text,expected",
    [
        ("I need the count of all the coffee variants in menu", True),
        ("How many coffee variants do you have?", True),
        ("сколько видов кофе у вас в меню", True),
        ("I need a coffee", False),
    ],
)
def test_detects_coffee_count_intent(text: str, expected: bool) -> None:
    assert _client_asks_coffee_variant_count(text) is expected


def test_coffee_count_response_uses_only_rag_context_names() -> None:
    rag_context = (
        "- Premium Roast Coffee (120 kcal, allergens: none listed)\n"
        "- Caramel Macchiato (260 kcal, allergens: Milk)\n"
        "- Caramel Frappe (380 kcal, allergens: Milk)\n"
        "- Big Mac (563 kcal, allergens: gluten)\n"
    )
    out = _coffee_count_response_from_rag_context(rag_context)
    assert out is not None
    assert "3 coffee variants" in out
    assert "Premium Roast Coffee" in out
    assert "Caramel Macchiato" in out
    assert "Caramel Frappe" in out
    assert "Big Mac" not in out
