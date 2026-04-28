"""JSON RAG-спек: нормализация аллергенов, парсинг, энергия."""

import json

import pytest

from mcd_voice.menu.rag_structured import (
    get_rag_json_system_prompt,
    parse_rag_json_response,
    normalize_excluded_allergen_list,
    chroma_excludable_allergen_vocabulary,
)


def test_rag_json_system_is_non_empty() -> None:
    p = get_rag_json_system_prompt()
    assert "search_query" in p
    assert "excluded_allergens" in p
    assert "Milk" in p or "milk" in p.lower()
    assert len(chroma_excludable_allergen_vocabulary()) >= 1


def test_parse_minimal() -> None:
    raw = json.dumps(
        {
            "search_query": "grilled chicken and sides",
            "excluded_allergens": ["Milk", "Nuts"],
            "max_kcal": 420.0,
            "min_kcal": None,
        }
    )
    s = parse_rag_json_response(raw)
    assert s["search_query"] == "grilled chicken and sides"
    assert s["excluded_allergens"] == ["Milk", "Nuts"]
    assert s.get("excluded_lexical") == []
    assert s["max_kcal"] == 420.0
    assert s["min_kcal"] is None


def test_parse_excluded_lexical() -> None:
    raw = json.dumps(
        {
            "search_query": "savory protein sandwich",
            "excluded_allergens": [],
            "excluded_lexical": ["beef", "bacon"],
            "max_kcal": None,
            "min_kcal": None,
        }
    )
    s = parse_rag_json_response(raw)
    assert s["excluded_lexical"] == ["beef", "bacon"]


def test_parse_excluded_menu_terms_alias() -> None:
    raw = json.dumps(
        {
            "search_query": "burger options",
            "excluded_allergens": [],
            "excluded_menu_terms": ["pickle"],
            "max_kcal": None,
            "min_kcal": None,
        }
    )
    s = parse_rag_json_response(raw)
    assert s["excluded_lexical"] == ["pickle"]


def test_parse_allergies_alias() -> None:
    raw = json.dumps(
        {
            "search_query": "light options",
            "allergies": ["milk"],
            "max_kcal": None,
            "min_kcal": None,
        }
    )
    s = parse_rag_json_response(raw)
    assert s["excluded_allergens"] == ["Milk"]


def test_parse_code_fence() -> None:
    raw = '```json\n{"search_query": "fries", "excluded_allergens": [], "max_kcal": null, "min_kcal": null}\n```\n'
    s = parse_rag_json_response(raw)
    assert s["search_query"] == "fries"
    assert s["excluded_allergens"] == []


def test_alias_gluten() -> None:
    assert normalize_excluded_allergen_list(["gluten", "GLUTEN", "Cereal containing gluten"]) == [
        "Cereal containing gluten",
    ]


def test_energy_single_key_maps_to_max() -> None:
    s = parse_rag_json_response(
        '{"search_query": "light lunch", "excluded_allergens": [], "energy": 500}'
    )
    assert s["max_kcal"] == 500.0


def test_compare_metrics_normalization() -> None:
    s = parse_rag_json_response(
        json.dumps(
            {
                "intent": "compare",
                "search_query": "salad options",
                "compare_metrics": [
                    {"field": "protein", "goal": "max"},
                    {"field": "fat", "goal": "min"},
                    {"field": "kcal", "goal": "min"},
                ],
                "excluded_allergens": [],
            }
        )
    )
    assert s["intent"] == "compare"
    assert s["compare_metrics"] == [
        {"field": "protein", "goal": "max"},
        {"field": "total_fat", "goal": "min"},
        {"field": "energy", "goal": "min"},
    ]


def test_compare_metrics_extended_aliases() -> None:
    s = parse_rag_json_response(
        json.dumps(
            {
                "intent": "compare",
                "search_query": "chicken options",
                "compare_metrics": [
                    {"field": "cholesterol", "goal": "min"},
                    {"field": "saturated_fat", "goal": "min"},
                ],
                "excluded_allergens": [],
            }
        )
    )
    assert s["compare_metrics"] == [
        {"field": "chol", "goal": "min"},
        {"field": "sat_fat", "goal": "min"},
    ]


def test_parse_rejects_empty_query() -> None:
    with pytest.raises(ValueError):
        parse_rag_json_response('{"search_query": "", "excluded_allergens": []}')
