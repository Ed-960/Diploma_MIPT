"""OpenRouter: provider.ignore из env (обход перегруженных провайдеров)."""

from __future__ import annotations

import pytest

from mcd_voice.llm import agent as ag


def test_no_extra_body_when_not_openrouter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_PROVIDER_IGNORE", raising=False)
    monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com/v1")
    assert ag._openrouter_extra_body() is None


def test_default_ignores_venice_on_openrouter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_PROVIDER_IGNORE", raising=False)
    monkeypatch.setenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    assert ag._openrouter_extra_body() == {"provider": {"ignore": ["Venice", "venice"]}}


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("none", None),
        ("FALSE", None),
        ("venice,deepinfra", {"provider": {"ignore": ["venice", "deepinfra"]}}),
    ],
)
def test_openrouter_ignore_env(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: dict | None,
) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENROUTER_PROVIDER_IGNORE", raw)
    assert ag._openrouter_extra_body() == expected
