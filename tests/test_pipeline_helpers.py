"""
Юнит-тесты для новых функций pipeline:
  - parse_order_from_text (парсинг количеств)
  - _detect_target_person (детекция персоны)
  - _resolve_person_index (маппинг на индекс)
  - build_initial_order_state (построение order_state)
  - validate_dialog (per-person валидация)
"""

from __future__ import annotations

import pytest

from mcd_voice.dialog.pipeline import (
    _detect_target_person,
    _resolve_person_index,
    build_initial_order_state,
    parse_order_from_text,
    validate_dialog,
)

# ── Фикстуры ─────────────────────────────────────────────────────────

MENU_NAMES = [
    "Big Mac",
    "McChicken",
    "Large French Fries",
    "Cola",
    "McFlurry with Oreo Cookies",
    "Kids Ice Cream Cone",
    "Chicken McNuggets",
    "Oatmeal Raisin Cookie",
    "Apple Juice",
]


@pytest.fixture()
def family_profile() -> dict:
    return {
        "sex": "male",
        "age": 35,
        "psycho": "friendly",
        "language": "EN",
        "calApprValue": 2200,
        "noMilk": False,
        "noFish": False,
        "noNuts": False,
        "noEggs": False,
        "noGluten": False,
        "noBeef": False,
        "isVegan": False,
        "noSugar": False,
        "childQuant": 2,
        "friendsQuant": 0,
        "companions": [
            {
                "role": "child",
                "label": "child_1",
                "age": 4,
                "restrictions": {"noMilk": True, "noEggs": False, "noNuts": False, "noGluten": False},
            },
            {
                "role": "child",
                "label": "child_2",
                "age": 7,
                "restrictions": {"noMilk": False, "noEggs": False, "noNuts": True, "noGluten": False},
            },
        ],
    }


# ── parse_order_from_text ─────────────────────────────────────────────

class TestParseOrderFromText:
    def test_single_item_no_quantity(self):
        result = parse_order_from_text("I want a Big Mac", MENU_NAMES)
        assert result == [("Big Mac", 1)]

    def test_single_item_with_quantity(self):
        result = parse_order_from_text("3 Big Mac please", MENU_NAMES)
        assert result == [("Big Mac", 3)]

    def test_multiple_items_with_quantities(self):
        result = parse_order_from_text(
            "I'll have 2 Big Mac and 3 Large French Fries",
            MENU_NAMES,
        )
        assert ("Big Mac", 2) in result
        assert ("Large French Fries", 3) in result

    def test_mixed_quantity_and_no_quantity(self):
        result = parse_order_from_text(
            "2 McChicken and a Cola",
            MENU_NAMES,
        )
        assert ("McChicken", 2) in result
        assert ("Cola", 1) in result

    def test_long_name_match(self):
        result = parse_order_from_text(
            "One McFlurry with Oreo Cookies for dessert",
            MENU_NAMES,
        )
        assert result == [("McFlurry with Oreo Cookies", 1)]

    def test_no_match(self):
        result = parse_order_from_text("Nothing from the menu", MENU_NAMES)
        assert result == []

    def test_short_names_skipped(self):
        result = parse_order_from_text("I want tea", ["Tea"])
        assert result == []


# ── _detect_target_person ─────────────────────────────────────────────

class TestDetectTargetPerson:
    def test_self_explicit(self):
        assert _detect_target_person("For me, a Big Mac") == "self"

    def test_self_ill_have(self):
        assert _detect_target_person("I'll have a cola") == "self"

    def test_wife(self):
        assert _detect_target_person("For my wife, a McChicken") == "spouse"

    def test_husband(self):
        assert _detect_target_person("For my husband, a burger") == "spouse"

    def test_youngest_child(self):
        assert _detect_target_person("For the youngest, ice cream") == "child_youngest"

    def test_oldest_child(self):
        assert _detect_target_person("For the oldest son, a burger") == "child_oldest"

    def test_generic_child(self):
        assert _detect_target_person("For the kids, some fries") == "child_generic"

    def test_friend(self):
        assert _detect_target_person("For my friend, a Cola") == "friend_generic"

    def test_default_self(self):
        assert _detect_target_person("A burger and fries") == "self"


# ── _resolve_person_index ─────────────────────────────────────────────

class TestResolvePersonIndex:
    def test_self_returns_zero(self, family_profile):
        os = build_initial_order_state(family_profile)
        assert _resolve_person_index("self", os["persons"]) == 0

    def test_child_youngest(self, family_profile):
        os = build_initial_order_state(family_profile)
        idx = _resolve_person_index("child_youngest", os["persons"])
        assert os["persons"][idx]["label"] == "child_1"
        assert os["persons"][idx]["age"] == 4

    def test_child_oldest(self, family_profile):
        os = build_initial_order_state(family_profile)
        idx = _resolve_person_index("child_oldest", os["persons"])
        assert os["persons"][idx]["label"] == "child_2"
        assert os["persons"][idx]["age"] == 7

    def test_child_generic_prefers_empty(self, family_profile):
        os = build_initial_order_state(family_profile)
        os["persons"][1]["items"] = [{"name": "Big Mac", "quantity": 1}]
        idx = _resolve_person_index("child_generic", os["persons"])
        assert os["persons"][idx]["label"] == "child_2"


# ── build_initial_order_state ─────────────────────────────────────────

class TestBuildInitialOrderState:
    def test_solo_profile(self):
        profile = {"companions": []}
        os = build_initial_order_state(profile)
        assert len(os["persons"]) == 1
        assert os["persons"][0]["role"] == "self"
        assert os["order_complete"] is False

    def test_family_profile(self, family_profile):
        os = build_initial_order_state(family_profile)
        assert len(os["persons"]) == 3
        labels = [p["label"] for p in os["persons"]]
        assert labels == ["customer", "child_1", "child_2"]

    def test_persons_have_empty_items(self, family_profile):
        os = build_initial_order_state(family_profile)
        for p in os["persons"]:
            assert p["items"] == []
            assert p["total_energy"] == 0.0


# ── validate_dialog ──────────────────────────────────────────────────

class TestValidateDialog:
    def test_empty_order(self, family_profile):
        os = build_initial_order_state(family_profile)
        flags = validate_dialog(family_profile, os, [])
        assert flags["empty_order"] is True
        assert flags["total_items"] == 0

    def test_child_allergen_violation(self, family_profile):
        os = build_initial_order_state(family_profile)
        os["persons"][1]["items"] = [{"name": "McFlurry", "quantity": 1}]
        os["persons"][1]["allergens"] = ["Milk"]
        flags = validate_dialog(family_profile, os, [])
        assert flags["per_person"][1]["allergen_violation"] == ["Milk"]
        assert "Milk" in flags["allergen_violation"]

    def test_no_violation_for_self(self, family_profile):
        os = build_initial_order_state(family_profile)
        os["persons"][0]["items"] = [{"name": "Big Mac", "quantity": 1}]
        os["persons"][0]["allergens"] = ["Milk"]
        flags = validate_dialog(family_profile, os, [])
        assert flags["per_person"][0]["allergen_violation"] == []

    def test_calorie_warning(self, family_profile):
        os = build_initial_order_state(family_profile)
        os["persons"][0]["total_energy"] = 5000.0
        flags = validate_dialog(family_profile, os, [])
        assert flags["calorie_warning"] is True

    def test_turns_count(self, family_profile):
        os = build_initial_order_state(family_profile)
        history = [
            {"speaker": "cashier", "text": "Hi"},
            {"speaker": "client", "text": "Hi"},
            {"speaker": "cashier", "text": "What?"},
        ]
        flags = validate_dialog(family_profile, os, history)
        assert flags["turns"] == 3
