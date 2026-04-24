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
