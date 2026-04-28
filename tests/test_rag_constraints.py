"""Правила Chroma-фильтра из устного текста (без профиля / вместе с ним)."""

from mcd_voice.menu.rag_constraints import (
    extract_utterance_chroma_allergen_exclusions,
    merge_rag_allergen_blacklist,
)


def test_no_milk_finds_milk():
    t = "I need something with no milk, please"
    assert "Milk" in extract_utterance_chroma_allergen_exclusions(t)


def test_lactose_intolerant_finds_milk():
    t = "I'm lactose intolerant"
    assert "Milk" in extract_utterance_chroma_allergen_exclusions(t)


def test_without_nuts():
    t = "A burger but without any nuts in it"
    assert "Nuts" in extract_utterance_chroma_allergen_exclusions(t)


def test_allergy_to_milk_without_no():
    t = "I have an allergy to milk"
    assert "Milk" in extract_utterance_chroma_allergen_exclusions(t)


def test_gluten_free_compound():
    t = "gluten-free options at the window"
    assert "Cereal containing gluten" in extract_utterance_chroma_allergen_exclusions(t)


def test_vegan_adds_block():
    t = "we are completely vegan"
    s = set(extract_utterance_chroma_allergen_exclusions(t))
    assert s >= {"Milk", "Egg", "Fish"}


def test_love_milk_excluded():
    t = "I love milkshakes, give me one"
    assert "Milk" not in extract_utterance_chroma_allergen_exclusions(t)


def test_merge_unions():
    u, m = merge_rag_allergen_blacklist(
        ["Fish"],
        ["I can't have any milk today"],
    )
    assert "Milk" in u
    assert "Fish" in u
    assert m["utterance_allergen_exclusions"] == ["Milk"]


def test_empty():
    assert extract_utterance_chroma_allergen_exclusions("") == []
    u, _ = merge_rag_allergen_blacklist(["Egg"], [])
    assert u == ["Egg"]
