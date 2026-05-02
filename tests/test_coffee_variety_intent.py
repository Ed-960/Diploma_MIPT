"""Тесты универсальных helper-функций RAG/intent в CashierAgent."""

from __future__ import annotations

import json

import pytest

from mcd_voice.config import MCD_JSON_PATH
from mcd_voice.llm.agent import (
    CashierAgent,
    RAG_CATALOG_TOP_K,
    _detect_restriction_override,
    _deterministic_full_catalog_nutrition_reply,
    _extract_names_from_rag_context,
    _derive_secondary_search_queries,
    _grounded_rows_for_names,
    _grounding_target_names,
    _is_confirm_plus_browse_utterance,
    _is_menu_browse_request,
    _is_non_food_client_utterance,
    _is_service_meta_question,
    _normalize_rewrite_output,
    _rag_intent,
    _recommendation_search_query,
    _render_grounded_rows,
    _sanitize_cashier_response,
    _split_intent_clauses,
    _effective_rag_top_k,
    _should_skip_rag,
    _use_turn_orchestrator,
    _wants_menu_item_details,
    _wants_full_nutrition_context,
)


def test_rag_catalog_top_k_tracks_menu_size() -> None:
    rows = json.loads(MCD_JSON_PATH.read_text(encoding="utf-8"))

    assert RAG_CATALOG_TOP_K >= len(rows)


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
        ("Why did you ask me that?", "khachapuri", True),
    ],
)
def test_should_skip_rag(client_text: str, search_query: str, expected: bool) -> None:
    assert _should_skip_rag(client_text, search_query) is expected


def test_recommendation_query_targets_food_after_coffee_context() -> None:
    history = [
        {"speaker": "client", "text": "what kind of coffee you have"},
        {"speaker": "cashier", "text": "We have black coffee and iced coffee."},
        {"speaker": "client", "text": "yes I would like and what do you suggest to eat"},
    ]

    assert (
        _recommendation_search_query("I've just asked you to suggest sth to it", history)
        == "main food items burgers chicken sandwiches wraps fries sides"
    )


def test_recommendation_query_targets_second_intent_in_mixed_order() -> None:
    assert (
        _recommendation_search_query(
            "yes I would like black coffee and what do you suggest to eat",
            [],
        )
        == "main food items burgers chicken sandwiches wraps fries sides"
    )


def test_detect_confirm_plus_browse_utterance() -> None:
    assert _is_confirm_plus_browse_utterance(
        "yes I would like black coffee and what do you suggest to eat",
        {
            "intent": "lookup",
            "requested_items": ["black coffee"],
        },
    ) is True
    assert _is_confirm_plus_browse_utterance(
        "what do you suggest to eat",
        {
            "intent": "lookup",
            "requested_items": [],
        },
    ) is False


def test_split_intent_clauses_for_compound_utterance() -> None:
    assert _split_intent_clauses(
        "I would like black coffee and what do you suggest to eat?"
    ) == [
        "I would like black coffee",
        "what do you suggest to eat",
    ]


def test_derive_secondary_queries_include_each_meaningful_clause() -> None:
    queries = _derive_secondary_search_queries(
        "yes I would like black coffee and what do you suggest to eat",
        [],
        "main food items burgers chicken sandwiches wraps fries sides",
        {"requested_items": ["black coffee"]},
    )

    assert len(queries) == 1
    assert "black coffee" in queries[0].lower()


def test_recommendation_query_targets_drinks_for_drink_request() -> None:
    assert (
        _recommendation_search_query("can I have anything to drink", [])
        == "drinks beverages coffee tea soda water juice"
    )


def test_menu_browse_followup_uses_history_category() -> None:
    history = [
        {"speaker": "client", "text": "what kind of coffee you have"},
        {"speaker": "cashier", "text": "We have Iced Coffee and Cold Coffee."},
    ]

    assert (
        _recommendation_search_query("that's all you have?", history)
        == "coffee drinks hot cold espresso frappe"
    )
    assert _is_menu_browse_request("that's all you have?", history, None) is True


def test_menu_browse_request_not_inherited_from_history_on_confirmation_turn() -> None:
    history = [
        {"speaker": "client", "text": "can I have more options with 5-6 gram protein"},
        {"speaker": "cashier", "text": "Sure — options around 5-6 g protein are Dosa and Veg Surprise."},
    ]
    assert (
        _is_menu_browse_request(
            "I'd like both of them and of course the coffee",
            history,
            {"intent": "lookup", "requested_items": ["coffee"]},
        )
        is False
    )


def test_deterministic_menu_browse_reply_uses_only_rag_context_names() -> None:
    rag_context = "\n".join(
        [
            "- Iced Coffee (allergens: Milk)",
            "- Premium Roast Coffee (allergens: none listed)",
            "- Cold Coffee (allergens: Milk)",
        ]
    )

    reply = CashierAgent._deterministic_menu_browse_reply(
        object(),
        "that's all you have?",
        [{"speaker": "client", "text": "what kind of coffee you have"}],
        rag_context,
        {"intent": "lookup", "requested_items": ["black coffee"]},
    )

    assert reply == (
        "Got it, black coffee. We have Iced Coffee, Premium Roast Coffee, and Cold Coffee. "
        "Which one would you like?"
    )


def test_deterministic_menu_browse_reply_honors_lexical_exclusions_for_category() -> None:
    rag_context = "\n".join(
        [
            "- Hamburger (allergens: none listed)",
            "- Chicken McNuggets (allergens: Cereal containing gluten)",
            "- Happy Meal (Hamburger) (allergens: Cereal containing gluten)",
            "- Our World Famous Fries (allergens: none listed)",
        ]
    )

    reply = CashierAgent._deterministic_menu_browse_reply(
        object(),
        "what would you suggest from burgers? not chicken",
        [],
        rag_context,
        {
            "intent": "lookup",
            "search_query": "burgers without chicken",
            "excluded_lexical": ["chicken"],
            "requested_items": [],
        },
    )

    assert reply == "I'd suggest Hamburger. Would you like that?"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Why did you say that?", True),
        ("So why you asked me about the menu?", True),
        ("I want a Big Mac", False),
        ("What's in the nuggets?", False),
    ],
)
def test_service_meta_question(text: str, expected: bool) -> None:
    assert _is_service_meta_question(text) is expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("What's in a Big Mac?", True),
        ("Any allergens in the fries?", True),
        ("Why did you ask that?", False),
        ("I want nuggets", False),
    ],
)
def test_wants_menu_item_details(text: str, expected: bool) -> None:
    assert _wants_menu_item_details(text) is expected


def test_wants_full_nutrition_context_for_nutrient_request_even_lookup_intent() -> None:
    assert _wants_full_nutrition_context(
        "can I have a burger around 5-6 grams protein and no sugar coffee",
        {"intent": "lookup", "compare_metrics": [], "restrictions": ["sugar"]},
    ) is True


def test_effective_rag_top_k_always_uses_full_catalog() -> None:
    assert _effective_rag_top_k(12, include_full_nutrition=False) >= RAG_CATALOG_TOP_K
    assert _effective_rag_top_k(12, include_full_nutrition=True) >= RAG_CATALOG_TOP_K


def test_cashier_handles_malformed_provider_payload_without_crash(monkeypatch) -> None:
    class _DummyResp:
        choices = None

    class _DummyCompletions:
        @staticmethod
        def create(**_kwargs):
            return _DummyResp()

    class _DummyChat:
        completions = _DummyCompletions()

    class _DummyClient:
        chat = _DummyChat()

    monkeypatch.setattr(
        "mcd_voice.llm.agent._build_openai_client",
        lambda *_a, **_k: _DummyClient(),
    )
    monkeypatch.setattr("mcd_voice.llm.agent._use_rag_json_rewrite", lambda: False)

    agent = CashierAgent(rag_top_k=0)
    text = agent.generate_response(
        {"language": "EN", "psycho": "regular"},
        [{"speaker": "client", "text": "but Veg Surprise Burger has 5.71, don't ?"}],
        {"persons": []},
    )

    assert "Could you repeat your order?" in text


def test_full_catalog_nutrition_reply_avoids_false_no_match_and_handles_no_sugar_coffee() -> None:
    reply = _deterministic_full_catalog_nutrition_reply(
        "Can I have a burger which have around 5-6 gram protein? and a coffee which is with no sugar"
    )
    assert reply is not None
    assert "don't see a burger in the 5-6 g range" not in reply
    assert "about" in reply and "protein" in reply
    assert "no-sugar coffee" in reply


def test_full_catalog_nutrition_reply_more_options_uses_range_and_history_context() -> None:
    reply = _deterministic_full_catalog_nutrition_reply(
        "can I have please more options with 5-6 gram protein",
        history=[
            {"speaker": "client", "text": "Can I have a burger which have around 5-6 gram protein?"},
            {"speaker": "cashier", "text": "Yes, Dosa Masala Burger is about 5.66 g protein."},
        ],
    )
    assert reply is not None
    assert "options around 5-6 g protein" in reply
    assert "Dosa Masala Burger" in reply
    assert "Which one would you like?" in reply


def test_full_catalog_nutrition_reply_more_options_like_this_inherits_last_range() -> None:
    reply = _deterministic_full_catalog_nutrition_reply(
        "good, can I have please more options like this ?",
        history=[
            {
                "speaker": "client",
                "text": "Can I have a burger which have around 11-12 gram protein? and a coffee which is with no sugar ?",
            },
            {
                "speaker": "cashier",
                "text": "Yes, Hamburger is about 12.00 g protein. For no-sugar coffee, Black Coffee and Premium Roast Coffee fit (about 0 g sugar).",
            },
            {"speaker": "client", "text": "good, can I have please more options like this ?"},
        ],
    )
    assert reply is not None
    assert "options around 11-12 g protein" in reply
    assert "Hamburger" in reply
    assert "Which one would you like?" in reply


def test_full_catalog_nutrition_reply_handles_non_burger_scope(monkeypatch) -> None:
    monkeypatch.setattr(
        "mcd_voice.llm.agent._load_menu_rows_for_nutrition",
        lambda: (
            {"name": "Chicken Sandwich", "protein": 10.4},
            {"name": "Veg Sandwich", "protein": 10.9},
            {"name": "Hamburger", "protein": 12.0},
        ),
    )
    reply = _deterministic_full_catalog_nutrition_reply(
        "good, more options like this please",
        history=[
            {"speaker": "client", "text": "Can I have a sandwich around 10-11 gram protein?"},
            {"speaker": "cashier", "text": "Yes, Chicken Sandwich is about 10.4 g protein."},
        ],
    )
    assert reply is not None
    assert "for sandwiches" in reply
    assert "Chicken Sandwich" in reply
    assert "Veg Sandwich" in reply


def test_grounding_targets_use_recent_named_item_for_followup() -> None:
    rows = [
        {"name": "Veg Surprise Burger", "distance": 0.59},
        {"name": "McAloo Tikki Burger", "distance": 0.65},
    ]
    history = [
        {
            "speaker": "cashier",
            "text": "Just so you know, the McAloo Tikki Burger contains gluten, milk and soya.",
        },
    ]

    assert _grounding_target_names(
        "yes, but only if it contains potato menu items containing potato",
        history,
        rows,
        [],
    ) == ["McAloo Tikki Burger"]


def test_grounded_rows_render_exact_ingredients_even_above_threshold() -> None:
    rows = [
        {
            "name": "McAloo Tikki Burger",
            "distance": 0.65,
            "ingredients": "Regular bun crown, Tom-Mayo sauce, Aloo tikki patty",
            "description": "A golden fried vegetarian patty prepared with peas and potato.",
            "allergens": ["Cereal containing gluten", "Milk", "Soya"],
        },
    ]

    grounded = _grounded_rows_for_names(rows, ["McAloo Tikki Burger"])
    rendered = _render_grounded_rows(grounded)

    assert "McAloo Tikki Burger" in rendered
    assert "Aloo tikki patty" in rendered
    assert "do not guess" in rendered


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


def test_turn_orchestrator_flag_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("LLM_TURN_ORCHESTRATOR", "0")
    assert _use_turn_orchestrator() is False


def test_cashier_generate_response_uses_turn_orchestrator(monkeypatch) -> None:
    monkeypatch.setenv("LLM_TURN_ORCHESTRATOR", "1")
    captured: dict[str, object] = {}

    def _fake_run_cashier_turn(**kwargs):
        captured.update(kwargs)
        return "orchestrated-path"

    monkeypatch.setattr(
        "mcd_voice.llm.turn_orchestrator.run_cashier_turn",
        _fake_run_cashier_turn,
    )
    monkeypatch.setattr(
        "mcd_voice.llm.agent._build_openai_client",
        lambda *_a, **_k: object(),
    )
    agent = CashierAgent(rag_top_k=0)
    out = agent.generate_response(
        {"language": "EN", "psycho": "regular"},
        [{"speaker": "client", "text": "hello"}],
        {"persons": []},
    )

    assert out == "orchestrated-path"
    assert captured["agent"] is agent
