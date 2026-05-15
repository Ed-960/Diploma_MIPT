from __future__ import annotations

import json

from mcd_voice.dialog.question_experiment import (
    build_judge_comparison,
    build_metrics_from_judge,
    MenuItem,
    compute_group_completeness,
    detect_constraint_violation,
    detect_hallucination,
    detect_need_to_specify,
    evaluate_single_turn_metrics,
    evaluate_dialog_audit,
    extract_mentioned_menu_items,
    is_empty_response,
    save_incremental_question_row,
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


def test_save_incremental_question_row_writes_json_and_jsonl(tmp_path) -> None:
    save_incremental_question_row(tmp_path, {"question_id": 1, "question": "one"})
    save_incremental_question_row(tmp_path, {"question_id": 2, "question": "two"})
    p1 = tmp_path / "incremental" / "question_0001.json"
    p2 = tmp_path / "incremental" / "question_0002.json"
    jl = tmp_path / "incremental" / "rows_partial.jsonl"
    assert p1.exists() and p2.exists() and jl.exists()
    lines = jl.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["question"] == "one"


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


def test_audit_marks_context_ignorance_when_expected_item_denied() -> None:
    menu = {"Big Mac": _fake_item(name="Big Mac")}
    heur = {
        "success_at_1": False,
        "success_at_3": False,
        "success_at_5": False,
        "hallucination": False,
        "constraint_violation": False,
    }
    audit = evaluate_dialog_audit(
        question_row={"expected_item": "Big Mac"},
        response_text="Sorry, we don't have Big Mac on the menu right now.",
        mentioned_items=[],
        menu_by_name=menu,
        heuristic_metrics=heur,
        need_to_specify=False,
    )
    assert audit["context_ignorance"] is True
    assert audit["critical_error"] is True
    assert audit["expected_item_in_menu"] is True


def test_audit_marks_factual_conflict_for_allergens_and_protein() -> None:
    menu = {
        "Quarter Pounder with Cheese": _fake_item(
            name="Quarter Pounder with Cheese",
            allergens={"cereal containing gluten", "milk", "sesame"},
            nutrients={
                "energy": 520.0,
                "protein": 30.0,
                "carbs": 42.0,
                "total_sugar": 10.0,
                "sodium": 1140.0,
            },
        )
    }
    heur = {
        "success_at_1": True,
        "success_at_3": True,
        "success_at_5": True,
        "hallucination": False,
        "constraint_violation": False,
    }
    text = (
        "Quarter Pounder with Cheese is available. Allergies: gluten, dairy, egg. "
        "Protein: 18g."
    )
    audit = evaluate_dialog_audit(
        question_row={"expected_item": "Quarter Pounder with Cheese"},
        response_text=text,
        mentioned_items=["Quarter Pounder with Cheese"],
        menu_by_name=menu,
        heuristic_metrics=heur,
        need_to_specify=False,
    )
    assert audit["factual_conflicts"]
    assert audit["critical_error"] is True


def test_build_metrics_from_judge_prefers_judge_values() -> None:
    heur = {
        "success_at_1": False,
        "success_at_3": False,
        "success_at_5": False,
        "hallucination": False,
        "constraint_violation": False,
        "need_to_specify": False,
        "empty_response": False,
        "group_completeness": None,
    }
    judge = {
        "success_at_1": True,
        "success_at_3": True,
        "success_at_5": True,
        "hallucination": True,
        "constraint_violation": True,
        "group_completeness": 0.5,
    }
    out, sources = build_metrics_from_judge(judge_parsed=judge, heuristic_metrics=heur)
    assert out["success_at_1"] is True
    assert out["constraint_violation"] is True
    assert out["hallucination"] is True
    assert out["group_completeness"] == 0.5
    assert sources["success_at_1"] == "judge"
    assert sources["group_completeness"] == "judge"


def test_build_metrics_from_judge_falls_back_when_missing() -> None:
    heur = {
        "success_at_1": False,
        "success_at_3": True,
        "success_at_5": True,
        "hallucination": False,
        "constraint_violation": False,
        "need_to_specify": True,
        "empty_response": False,
        "group_completeness": None,
    }
    out, sources = build_metrics_from_judge(judge_parsed={"short_analysis": "ok"}, heuristic_metrics=heur)
    assert out["success_at_3"] is True
    assert out["need_to_specify"] is True
    assert sources["success_at_3"] == "heuristic_fallback"
    assert sources["group_completeness"] == "none"


def test_build_judge_comparison_reports_disagreements() -> None:
    final_metrics = {
        "success_at_1": False,
        "success_at_3": False,
        "success_at_5": False,
        "hallucination": False,
        "constraint_violation": False,
        "need_to_specify": False,
        "empty_response": False,
    }
    heur = {
        "success_at_1": True,
        "success_at_3": True,
        "success_at_5": False,
        "hallucination": True,
        "constraint_violation": False,
        "need_to_specify": False,
        "empty_response": False,
    }
    audit = {
        "critical_error": True,
        "context_ignorance": True,
        "factual_conflicts": ["Allergen mismatch."],
    }
    cmp = build_judge_comparison(final_metrics=final_metrics, heuristic_metrics=heur, audit=audit)
    assert cmp["disagreement_count"] == 3
    assert "success_at_1" in cmp["disagreements"]
    assert cmp["supports_audit_signals"] is False
