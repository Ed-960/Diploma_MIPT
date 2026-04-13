import pytest

from mcd_voice.dialog import HumanDriveThroughSession


def test_human_session_step_before_start_raises() -> None:
    s = HumanDriveThroughSession(max_turns=2)
    with pytest.raises(RuntimeError, match="start"):
        s.step("hello")


def test_human_session_double_start_raises() -> None:
    s = HumanDriveThroughSession(max_turns=2)
    s._started = True  # simulate partial init without calling LLM
    with pytest.raises(RuntimeError, match="already started"):
        s.start()
