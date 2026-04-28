import pytest

from mcd_voice.dialog import HumanDriveThroughSession
from mcd_voice.profile import get_group_allergen_blacklist, neutral_drive_through_profile


def test_human_session_step_before_start_raises() -> None:
    s = HumanDriveThroughSession(max_turns=2)
    with pytest.raises(RuntimeError, match="start"):
        s.step("hello")


def test_human_session_double_start_raises() -> None:
    s = HumanDriveThroughSession(max_turns=2)
    s._started = True  # simulate partial init without calling LLM
    with pytest.raises(RuntimeError, match="already started"):
        s.start()


def test_snapshot_for_save_before_start_is_none() -> None:
    s = HumanDriveThroughSession(max_turns=2)
    assert s.snapshot_for_save() is None


def test_neutral_drive_through_profile_is_unrestricted() -> None:
    p = neutral_drive_through_profile()
    assert p["companions"] == []
    assert get_group_allergen_blacklist(p) == []
    assert not any(
        p.get(k)
        for k in (
            "noMilk",
            "noSugar",
            "noBeef",
            "isVegan",
            "noFish",
            "noNuts",
            "noEggs",
            "noGluten",
        )
    )


def test_human_session_step_updates_order_without_live_llm(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ORDER_JSON_REWRITE", "0")
    monkeypatch.setattr(
        "mcd_voice.dialog.catalog.MenuCatalog.load_runtime_index",
        lambda _self: (
            ["Big Mac®"],
            {"Big Mac®": 493.0},
            {"Big Mac®": []},
            {"Big Mac®": {}},
        ),
    )

    class FakeCashierAgent:
        def __init__(self, **_kwargs):
            pass

        def generate_response(self, profile, history, order_state, **_kwargs):
            if not history:
                return "Hi, what can I get for you?"
            return "Got it, a Big Mac."

    monkeypatch.setattr(
        "mcd_voice.dialog.human_voice_session.CashierAgent",
        FakeCashierAgent,
    )
    s = HumanDriveThroughSession(max_turns=2)
    start = s.start()
    assert start["greeting"] == "Hi, what can I get for you?"

    out = s.step("Can I get a Big Mac?")

    assert out["cashier_text"] == "Got it, a Big Mac."
    assert out["validation"]["total_items"] == 1
    snap = s.snapshot_for_save()
    assert snap is not None
    _profile, _history, order_state, flags = snap
    assert flags["total_items"] == 1
    assert order_state["persons"][0]["items"][0]["name"] == "Big Mac®"
