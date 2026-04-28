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
    DialogPipeline,
    _detect_target_person,
    _has_cashier_hard_repeat,
    _is_looping_tail,
    _resolve_person_index,
    build_initial_order_state,
    localize_errors,
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

    def test_trademark_symbol_tolerant_match(self):
        result = parse_order_from_text(
            "Yes, confirm large Chicken McNuggets and coffee",
            ["Chicken McNuggets®", "Black Coffee®"],
        )
        assert ("Chicken McNuggets®", 1) in result

    def test_no_match(self):
        result = parse_order_from_text("Nothing from the menu", MENU_NAMES)
        assert result == []

    def test_short_names_skipped(self):
        result = parse_order_from_text("I want tea", ["Tea"])
        assert result == []

    def test_allergy_mention_does_not_add_item(self):
        result = parse_order_from_text(
            "I have allergy on milk, please no milk.",
            ["Milk", "Big Mac"],
        )
        assert ("Milk", 1) not in result

    def test_explicit_order_cue_still_allows_item(self):
        result = parse_order_from_text(
            "Can I have milk please?",
            ["Milk", "Big Mac"],
        )
        assert ("Milk", 1) in result


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

    def test_child_age_phrase(self):
        assert _detect_target_person("For my 7-year-old, a Happy Meal") == "child_generic"

    def test_child_numbered_phrase(self):
        assert _detect_target_person("Child one needs nuggets") == "child_generic"

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


def test_update_order_multi_person_segmented_assignment(family_profile):
    os = build_initial_order_state(family_profile)
    menu_names = ["Big Mac", "Happy Meal (4pc McNuggets)"]
    energy = {"Big Mac": 550.0, "Happy Meal (4pc McNuggets)": 395.0}
    allergen_map = {"Big Mac": [], "Happy Meal (4pc McNuggets)": []}
    DialogPipeline._update_order(
        "I'll take a Big Mac. For my 7-year-old kid, a Happy Meal (4pc McNuggets).",
        menu_names,
        os,
        energy,
        allergen_map,
    )
    assert any(it["name"] == "Big Mac" for it in os["persons"][0]["items"])
    # 7-year-old in fixture is child_2, so assignment should be explicit.
    assert any(it["name"] == "Happy Meal (4pc McNuggets)" for it in os["persons"][2]["items"])


def test_update_order_segmented_assignment_with_label_reference(family_profile):
    os = build_initial_order_state(family_profile)
    menu_names = ["Big Mac", "Happy Meal (4pc McNuggets)"]
    energy = {"Big Mac": 550.0, "Happy Meal (4pc McNuggets)": 395.0}
    allergen_map = {"Big Mac": [], "Happy Meal (4pc McNuggets)": []}
    DialogPipeline._update_order(
        "For me, Big Mac. For child_1, Happy Meal (4pc McNuggets).",
        menu_names,
        os,
        energy,
        allergen_map,
    )
    assert any(it["name"] == "Big Mac" for it in os["persons"][0]["items"])
    assert any(it["name"] == "Happy Meal (4pc McNuggets)" for it in os["persons"][1]["items"])


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
        assert flags["allergen_violation_per_person"] == [
            {"label": "child_1", "role": "child", "allergens": ["Milk"]}
        ]

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

    def test_incomplete_order_when_children_have_no_items(self, family_profile):
        os = build_initial_order_state(family_profile)
        os["persons"][0]["items"] = [{"name": "Big Mac", "quantity": 1}]
        flags = validate_dialog(family_profile, os, [])
        assert flags["incomplete_order"] is True

    def test_hallucination_flag_for_unknown_item(self, family_profile):
        os = build_initial_order_state(family_profile)
        os["persons"][0]["items"] = [{"name": "Alien Burger", "quantity": 1}]
        flags = validate_dialog(family_profile, os, [], menu_names=MENU_NAMES)
        assert flags["hallucination"] is True
        assert flags["hallucinated_items"] == ["Alien Burger"]


def test_localize_errors_detects_cot_and_allergen() -> None:
    history = [
        {"speaker": "cashier", "text": "We have Milk and Egg options for you."},
        {"speaker": "cashier", "text": "Let's reason step-by-step before ordering."},
    ]
    flags = {
        "allergen_violation_per_person": [
            {"label": "child_1", "role": "child", "allergens": ["Milk", "Egg"]}
        ],
    }
    rows = localize_errors(history, flags)
    types = [r["error_type"] for r in rows]
    assert "allergen_suggestion" in types
    assert "cot_leak" in types


def test_localize_errors_includes_hallucinated_items() -> None:
    rows = localize_errors([], {"hallucinated_items": ["Alien Burger"]})
    assert rows == [
        {
            "turn": None,
            "speaker": "system",
            "error_type": "hallucinated_item",
            "item": "Alien Burger",
            "excerpt": "Alien Burger",
        }
    ]


class _StubCatalog:
    def load(self) -> tuple[list[str], dict[str, float]]:
        return ["Big Mac"], {"Big Mac": 550.0}


class _StubClientAgent:
    def generate_response(self, profile: dict, history: list[dict[str, str]]) -> str:
        return "Big Mac please"


class _StubCashierAgent:
    def __init__(self) -> None:
        self._calls = 0

    def generate_response(
        self,
        profile: dict,
        history: list[dict[str, str]],
        order_state: dict[str, object],
        **_: object,
    ) -> str:
        self._calls += 1
        if self._calls == 1:
            return "Hello, what would you like?"
        return "That will be all, thank you."


class _TraceClientAgent:
    def generate_response(
        self,
        profile: dict,
        history: list[dict[str, str]],
        *,
        llm_trace: list[dict] | None = None,
    ) -> str:
        if llm_trace is not None:
            llm_trace.append({"event": "llm_call", "agent": "client", "model": "stub"})
        return "Big Mac please"


class _TraceCashierAgent:
    def __init__(self) -> None:
        self._calls = 0

    def generate_response(
        self,
        profile: dict,
        history: list[dict[str, str]],
        order_state: dict[str, object],
        *,
        llm_trace: list[dict] | None = None,
        **_: object,
    ) -> str:
        self._calls += 1
        if llm_trace is not None:
            llm_trace.append({"event": "llm_call", "agent": "cashier", "model": "stub"})
        if self._calls == 1:
            return "Hello, what would you like?"
        return "That will be all, thank you."


class _ConfirmingClientAgent:
    def __init__(self) -> None:
        self._calls = 0

    def generate_response(self, profile: dict, history: list[dict[str, str]]) -> str:
        self._calls += 1
        if self._calls == 1:
            return "Big Mac please"
        return "That's all, thanks"


class _PassiveClientAgent:
    def __init__(self) -> None:
        self._calls = 0

    def generate_response(self, profile: dict, history: list[dict[str, str]]) -> str:
        self._calls += 1
        if self._calls == 1:
            return "What do you recommend?"
        return "Yes, that's all, thanks"


class _ConfirmingSummaryCashier:
    def __init__(self) -> None:
        self._calls = 0

    def generate_response(
        self,
        profile: dict,
        history: list[dict[str, str]],
        order_state: dict[str, object],
        **_: object,
    ) -> str:
        self._calls += 1
        if self._calls == 1:
            return "Hello, what would you like?"
        return "Your order is confirmed: Big Mac. Enjoy your meal!"


def test_dialog_pipeline_collect_rag_trace_empty_with_stub_cashier(
    monkeypatch, family_profile,
) -> None:
    pipeline = DialogPipeline(
        max_turns=1,
        menu_catalog=_StubCatalog(),
        client_agent=_StubClientAgent(),
        cashier_agent=_StubCashierAgent(),
        collect_rag_trace=True,
    )
    monkeypatch.setattr(pipeline, "_build_allergen_map", lambda: {"Big Mac": []})

    _, _, _, flags = pipeline.run(profile=family_profile)

    assert "rag_trace" in flags
    assert flags["rag_trace"] == []


class _TraceEmitClient:
    def generate_response(
        self,
        profile: dict,
        history: list[dict[str, str]],
        *,
        llm_trace: list[dict] | None = None,
        **_: object,
    ) -> str:
        if llm_trace is not None:
            llm_trace.append({
                "event": "llm_call",
                "agent": "client",
                "model": "stub",
                "response_preview": "client says hi",
            })
        return "client text"


class _TraceEmitCashier:
    def __init__(self) -> None:
        self._n = 0

    def generate_response(
        self,
        profile: dict,
        history: list[dict[str, str]],
        order_state: dict[str, object],
        *,
        rag_trace: list[dict] | None = None,
        llm_trace: list[dict] | None = None,
        **_: object,
    ) -> str:
        self._n += 1
        if rag_trace is not None:
            rag_trace.append({
                "event": "rag",
                "search_query": f"q{self._n}",
                "outcome": "injected",
                "best_distance": 0.1,
                "candidates": [{"name": "Big Mac", "distance": 0.1, "energy": 500}],
            })
        if llm_trace is not None:
            llm_trace.append({
                "event": "llm_call",
                "agent": "cashier",
                "model": "stub",
                "response_preview": f"cashier {self._n}",
            })
        return f"cashier reply {self._n}"


def test_dialog_pipeline_emit_trace_progress_trace_delta(
    monkeypatch, family_profile,
) -> None:
    events: list[dict[str, object]] = []
    pipeline = DialogPipeline(
        max_turns=1,
        menu_catalog=_StubCatalog(),
        client_agent=_TraceEmitClient(),
        cashier_agent=_TraceEmitCashier(),
        progress_callback=events.append,
        collect_rag_trace=True,
        collect_llm_trace=True,
        emit_trace_progress=True,
    )
    monkeypatch.setattr(pipeline, "_build_allergen_map", lambda: {"Big Mac": []})

    pipeline.run(profile=family_profile)

    deltas = [e for e in events if e.get("stage") == "trace_delta"]
    assert len(deltas) >= 3
    assert deltas[0].get("label") == "greeting"
    assert any(d.get("label") == "client" for d in deltas)
    assert any(d.get("label") == "cashier" for d in deltas)


def test_dialog_pipeline_collect_llm_trace_with_trace_agents(
    monkeypatch, family_profile,
) -> None:
    pipeline = DialogPipeline(
        max_turns=1,
        menu_catalog=_StubCatalog(),
        client_agent=_TraceClientAgent(),
        cashier_agent=_TraceCashierAgent(),
        collect_llm_trace=True,
    )
    monkeypatch.setattr(pipeline, "_build_allergen_map", lambda: {"Big Mac": []})

    _, _, _, flags = pipeline.run(profile=family_profile)

    assert "llm_trace" in flags
    llm_calls = [ev for ev in flags["llm_trace"] if ev.get("event") == "llm_call"]
    assert len(llm_calls) == 3
    assert llm_calls[0]["agent"] == "cashier"
    assert llm_calls[1]["agent"] == "client"
    assert llm_calls[2]["agent"] == "cashier"


def test_dialog_pipeline_emits_progress_events(monkeypatch, family_profile) -> None:
    events: list[dict[str, object]] = []
    pipeline = DialogPipeline(
        max_turns=1,
        menu_catalog=_StubCatalog(),
        client_agent=_StubClientAgent(),
        cashier_agent=_StubCashierAgent(),
        progress_callback=events.append,
    )
    monkeypatch.setattr(pipeline, "_build_allergen_map", lambda: {"Big Mac": []})

    history, _, _, flags = pipeline.run(profile=family_profile)

    assert history
    assert flags["turns"] == len(history)
    assert [event["stage"] for event in events] == [
        "prepare",
        "greeting_start",
        "greeting_done",
        "turn_start",
        "client_thinking",
        "client_done",
        "cashier_thinking",
        "cashier_done",
        "finished",
    ]


def test_dialog_pipeline_sets_order_complete_on_confirm(
    monkeypatch, family_profile,
) -> None:
    pipeline = DialogPipeline(
        max_turns=3,
        menu_catalog=_StubCatalog(),
        client_agent=_ConfirmingClientAgent(),
        cashier_agent=_StubCashierAgent(),
    )
    monkeypatch.setattr(pipeline, "_build_allergen_map", lambda: {"Big Mac": []})

    _, _, order_state, _ = pipeline.run(profile=family_profile)

    assert order_state["order_complete"] is True


def test_dialog_pipeline_parses_final_cashier_summary(
    monkeypatch, family_profile,
) -> None:
    pipeline = DialogPipeline(
        max_turns=3,
        menu_catalog=_StubCatalog(),
        client_agent=_PassiveClientAgent(),
        cashier_agent=_ConfirmingSummaryCashier(),
    )
    monkeypatch.setattr(pipeline, "_build_allergen_map", lambda: {"Big Mac": []})

    _, _, order_state, flags = pipeline.run(profile=family_profile)

    assert flags["empty_order"] is False
    assert order_state["persons"][0]["items"] == [{"name": "Big Mac", "quantity": 1}]


def test_looping_tail_detects_repeated_ack_pair() -> None:
    history = [
        {"speaker": "cashier", "text": "Order is on the way."},
        {"speaker": "client", "text": "hurry"},
        {"speaker": "cashier", "text": "almost there"},
        {"speaker": "client", "text": "hurry"},
        {"speaker": "cashier", "text": "almost there"},
        {"speaker": "client", "text": "hurry"},
        {"speaker": "cashier", "text": "almost there"},
    ]
    assert _is_looping_tail(history) is True


def test_cashier_hard_repeat_detects_triplicate() -> None:
    history = [
        {"speaker": "cashier", "text": "Hi!"},
        {"speaker": "client", "text": "yeah"},
        {"speaker": "cashier", "text": "Have a great day!"},
        {"speaker": "client", "text": "you too"},
        {"speaker": "cashier", "text": "Have a great day!"},
        {"speaker": "client", "text": "you too"},
        {"speaker": "cashier", "text": "Have a great day!"},
    ]
    assert _has_cashier_hard_repeat(history) is True
