from __future__ import annotations

from mcd_voice.dialog.question_experiment import (
    MenuItem,
    compute_group_completeness,
    detect_constraint_violation,
    detect_hallucination,
    detect_need_to_specify,
    evaluate_single_turn_metrics,
    extract_mentioned_menu_items,
    is_empty_response,
    save_dialogs_by_category,
)


def _fake_item(
    *,
    name: str,
    allergens: set[str] | None = None,
    ingredients: str = "",
    restrictions: dict[str, bool] | None = None,
    nutrients: dict[str, float] | None = None,
) -> MenuItem:
    return MenuItem(
        name=name,
        norm_name=name.lower(),
        ingredients=ingredients,
        allergens=allergens or set(),
        restriction_map=restrictions or {
            "noMilk": False,
            "noFish": False,
            "noNuts": False,
            "noEggs": False,
            "noGluten": False,
            "noBeef": False,
            "isVegan": False,
        },
        nutrients=nutrients or {
            "energy": 100.0,
            "protein": 10.0,
            "carbs": 10.0,
            "total_sugar": 5.0,
            "sodium": 200.0,
        },
    )


def test_extract_mentioned_menu_items_preserves_order() -> None:
    names = ["Big Mac", "Diet Coke", "Side Salad"]
    text = "I can suggest Side Salad first, then Big Mac and maybe Diet Coke."
    found = extract_mentioned_menu_items(text, names)
    assert found == ["Side Salad", "Big Mac", "Diet Coke"]


def test_detect_need_to_specify_when_only_question() -> None:
    assert detect_need_to_specify("Could you clarify if you want beef or chicken?", []) is True
    assert detect_need_to_specify("Could you clarify if you want beef or chicken?", ["Big Mac"]) is False


def test_is_empty_response_heuristics() -> None:
    assert is_empty_response("") is True
    assert is_empty_response("Sorry, I did not understand") is True
    assert is_empty_response("You can order a Big Mac.") is False


def test_detect_hallucination_for_unknown_recommendation() -> None:
    hall = detect_hallucination(
        response_text="I recommend Royal Dragon Burger for your order.",
        mentioned_items=[],
        all_menu_names=["Big Mac", "Side Salad"],
    )
    assert hall is True


def test_detect_constraint_violation_allergen_and_exclude() -> None:
    menu = {
        "Cheesy Fries": _fake_item(
            name="Cheesy Fries",
            allergens={"milk"},
            ingredients="fries, cheese sauce, onion powder",
        )
    }
    violated, reasons = detect_constraint_violation(
        expected_constraints=[
            {"type": "allergen", "value": "Milk"},
            {"type": "exclude", "value": "onion"},
        ],
        mentioned_items=["Cheesy Fries"],
        menu_by_name=menu,
    )
    assert violated is True
    assert len(reasons) == 2


def test_evaluate_single_turn_metrics_success_positions() -> None:
    item = _fake_item(name="Big Mac")
    metrics = evaluate_single_turn_metrics(
        question_row={"expected_item": "Big Mac", "expected_constraints": []},
        response_text="Top options: Side Salad, Big Mac, Diet Coke.",
        mentioned_items=["Side Salad", "Big Mac", "Diet Coke"],
        menu_by_name={
            "Big Mac": item,
            "Side Salad": _fake_item(name="Side Salad"),
            "Diet Coke": _fake_item(name="Diet Coke"),
        },
        all_menu_names=["Big Mac", "Side Salad", "Diet Coke"],
    )
    assert metrics["success_at_1"] is False
    assert metrics["success_at_3"] is True
    assert metrics["success_at_5"] is True


def test_compute_group_completeness_from_group_constraint() -> None:
    completeness = compute_group_completeness(
        category="group",
        expected_constraints=[{"type": "group", "members": ["a", "b", "c"]}],
        mentioned_items=["Big Mac", "Fries"],
    )
    assert completeness == 0.6667


def test_save_dialogs_by_category_writes_category_dirs(tmp_path) -> None:
    rows = [
        {
            "question_id": 1,
            "category": "simple",
            "response_time_ms": 12.0,
            "metrics": {"success_at_1": True, "success_at_3": True, "success_at_5": True},
            "need_to_specify": False,
            "empty_response": False,
        },
        {
            "question_id": 2,
            "category": "diet",
            "response_time_ms": 15.0,
            "metrics": {"success_at_1": False, "success_at_3": True, "success_at_5": True},
            "need_to_specify": True,
            "empty_response": False,
        },
    ]
    out = save_dialogs_by_category(rows, output_dir=str(tmp_path))
    assert (tmp_path / "by_category" / "simple" / "dialogs" / "dialog_0001.json").exists()
    assert (tmp_path / "by_category" / "diet" / "summary.json").exists()
    assert "summary" in out
