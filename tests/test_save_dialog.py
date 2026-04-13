"""
Юнит-тесты для save_dialog / load_dialog / aggregate_stats (JSON only).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mcd_voice.dialog.save_dialog import (
    aggregate_stats,
    load_all_dialogs,
    load_dialog,
    save_dialog,
)


@pytest.fixture()
def tmp_dialogs(tmp_path):
    """Создаёт 3 тестовых диалога в tmp_path."""
    for i in range(1, 4):
        save_dialog(
            dialog_id=i,
            profile={"sex": "male", "age": 30 + i, "psycho": "regular",
                      "calApprValue": 2000, "companions": [],
                      "noMilk": i == 2, "childQuant": 0, "friendsQuant": 0},
            history=[{"speaker": "cashier", "text": "Hi"},
                     {"speaker": "client", "text": "Hello"}],
            order_state={"persons": [{"role": "self", "label": "customer",
                                       "items": [], "total_energy": 0, "allergens": []}]},
            flags={"allergen_violation": ["Milk"] if i == 2 else [],
                   "calorie_warning": False,
                   "empty_order": True,
                   "incomplete_order": False,
                   "hallucination": i == 3,
                   "hallucinated_items": ["Alien Burger"] if i == 3 else [],
                   "total_items": 0,
                   "total_energy": 0,
                   "calorie_target": 2000,
                   "turns": 2,
                   "per_person": [],
                   "error_localization": []},
            output_dir=tmp_path,
        )
    return tmp_path


class TestSaveLoadDialog:
    def test_save_creates_file(self, tmp_path):
        p = save_dialog(1, {}, [], {}, {}, output_dir=tmp_path)
        assert p.exists()
        assert p.name == "dialog_0001.json"

    def test_roundtrip(self, tmp_path):
        profile = {"sex": "female", "age": 25}
        save_dialog(1, profile, [{"speaker": "client", "text": "hi"}],
                     {"persons": []}, {"turns": 1}, output_dir=tmp_path)
        loaded = load_dialog(1, input_dir=tmp_path)
        assert loaded["profile"] == profile
        assert loaded["history"][0]["text"] == "hi"

    def test_load_all(self, tmp_dialogs):
        all_d = load_all_dialogs(tmp_dialogs)
        assert len(all_d) == 3


class TestAggregateStats:
    def test_creates_summary_json(self, tmp_dialogs):
        summaries = aggregate_stats(tmp_dialogs)
        assert len(summaries) == 3
        summary_path = tmp_dialogs / "summary.json"
        assert summary_path.exists()

        with open(summary_path, "r") as f:
            data = json.load(f)
        assert len(data) == 3

    def test_summary_fields(self, tmp_dialogs):
        summaries = aggregate_stats(tmp_dialogs)
        s = summaries[0]
        assert "dialog_id" in s
        assert "sex" in s
        assert "turns" in s
        assert "allergen_violation" in s
        assert "group_size" in s
        assert "has_companion_violation" in s
        assert "error_localization" in s
        assert "incomplete_order" in s
        assert "hallucination" in s
        assert "hallucinated_items" in s

    def test_allergen_preserved(self, tmp_dialogs):
        summaries = aggregate_stats(tmp_dialogs)
        dialog_2 = next(s for s in summaries if s["dialog_id"] == 2)
        assert "Milk" in dialog_2["allergen_violation"]

    def test_custom_output_path(self, tmp_dialogs):
        out = tmp_dialogs / "custom" / "report.json"
        aggregate_stats(tmp_dialogs, output_file=out)
        assert out.exists()
