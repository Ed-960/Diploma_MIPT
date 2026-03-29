"""
Юнит-тесты для ProfileGenerator и связанных функций.
"""

from __future__ import annotations

import random

import pytest

from mcd_voice.profile.generator import (
    ProfileGenerator,
    generate_text_description,
    get_allergen_blacklist,
    get_group_allergen_blacklist,
)


@pytest.fixture()
def seeded_gen() -> ProfileGenerator:
    return ProfileGenerator(rng=random.Random(42))


class TestProfileGenerator:
    def test_generate_returns_required_keys(self, seeded_gen):
        p = seeded_gen.generate()
        required = {"sex", "age", "psycho", "language", "calApprValue",
                     "childQuant", "friendsQuant", "companions"}
        assert required.issubset(p.keys())

    def test_companions_match_quant(self, seeded_gen):
        for _ in range(50):
            p = seeded_gen.generate()
            children = [c for c in p["companions"] if c["role"] == "child"]
            friends = [c for c in p["companions"] if c["role"] == "friend"]
            assert len(children) == p["childQuant"]
            assert len(friends) == p["friendsQuant"]

    def test_child_has_age_and_restrictions(self, seeded_gen):
        for _ in range(100):
            p = seeded_gen.generate()
            if p["childQuant"] > 0:
                child = p["companions"][0]
                assert 3 <= child["age"] <= 14
                assert "restrictions" in child
                assert isinstance(child["restrictions"], dict)
                return
        pytest.skip("No children generated in 100 attempts")

    def test_vegan_implies_flags(self, seeded_gen):
        for _ in range(500):
            p = seeded_gen.generate()
            if p.get("isVegan"):
                assert p["noMilk"] is True
                assert p["noFish"] is True
                assert p["noBeef"] is True
                assert p["noEggs"] is True
                return
        pytest.skip("No vegan generated in 500 attempts")

    def test_calories_in_range(self, seeded_gen):
        for _ in range(100):
            p = seeded_gen.generate()
            assert 800 <= p["calApprValue"] <= 3500


class TestAllergenBlacklist:
    def test_no_restrictions(self):
        assert get_allergen_blacklist({"noMilk": False}) == []

    def test_milk(self):
        bl = get_allergen_blacklist({"noMilk": True})
        assert "Milk" in bl

    def test_vegan_adds_extra(self):
        bl = get_allergen_blacklist({"isVegan": True, "noMilk": True, "noEggs": True, "noFish": True})
        assert "Milk" in bl
        assert "Egg" in bl
        assert "Fish" in bl


class TestGroupBlacklist:
    def test_combines_self_and_children(self):
        profile = {
            "noMilk": False,
            "companions": [
                {"role": "child", "label": "c1", "age": 5,
                 "restrictions": {"noMilk": True}},
            ],
        }
        bl = get_group_allergen_blacklist(profile)
        assert "Milk" in bl

    def test_empty_companions(self):
        profile = {"noMilk": True, "companions": []}
        bl = get_group_allergen_blacklist(profile)
        assert "Milk" in bl


class TestTextDescription:
    def test_solo_ordering(self):
        p = {"age": 30, "sex": "male", "psycho": "regular",
             "calApprValue": 2000, "companions": []}
        desc = generate_text_description(p)
        assert "Ordering alone" in desc
        assert "30-year-old male" in desc

    def test_group_ordering(self):
        p = {
            "age": 35, "sex": "female", "psycho": "friendly",
            "calApprValue": 1800, "noMilk": True,
            "companions": [
                {"role": "child", "label": "child_1", "age": 5,
                 "restrictions": {"noNuts": True}},
            ],
        }
        desc = generate_text_description(p)
        assert "group of 2" in desc
        assert "child_1" in desc
        assert "nut allergy" in desc
