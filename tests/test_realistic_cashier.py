"""Режим realistic_cashier: кассир без скрытого профиля и без RAG-фильтра по аллергенам группы."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mcd_voice.llm.agent import CashierAgent
from mcd_voice.llm.prompts import get_cashier_system_prompt, get_client_system_prompt


def _stub_llm_rewrite_then_cashier(client, model, system, messages, temperature=0.8):
    if "short food search query" in (system or "").lower():
        return "burgers chicken combo meal"
    return "Sure."


@pytest.fixture(autouse=True)
def _mock_openai_client(monkeypatch):
    monkeypatch.setattr(
        "mcd_voice.llm.agent._build_openai_client",
        lambda *a, **k: MagicMock(),
    )


def test_realistic_prompt_hides_group_and_psycho() -> None:
    profile = {
        "psycho": "impatient",
        "companions": [
            {
                "role": "child",
                "label": "child_1",
                "age": 5,
                "restrictions": {"noMilk": True},
            },
        ],
    }
    text = get_cashier_system_prompt(profile, realistic=True)
    assert "REALISTIC DRIVE-THROUGH" in text
    assert "Customer personality" not in text
    assert "ordering for a group" not in text.lower()


def test_realistic_cashier_rag_without_allergen_blacklist(
    monkeypatch,
) -> None:
    """RAG не получает blacklist из профиля (кассир «не знает» ограничений заранее)."""
    captured: dict[str, object] = {}

    def fake_search(query: str, allergens_blacklist=None, **_kwargs):
        captured["allergens_blacklist"] = allergens_blacklist
        return []

    monkeypatch.setattr("mcd_voice.llm.agent.search_menu", fake_search)
    agent = CashierAgent(rag_top_k=3, realistic_cashier=True)
    monkeypatch.setattr("mcd_voice.llm.agent._call_llm", _stub_llm_rewrite_then_cashier)
    profile = {
        "language": "EN",
        "noMilk": True,
        "companions": [],
    }
    trace: list[dict] = []
    agent.generate_response(
        profile,
        [{"speaker": "client", "text": "I want a burger"}],
        {"persons": []},
        rag_trace=trace,
        rag_meta={"call": "turn", "turn": 1},
    )
    assert captured.get("allergens_blacklist") is None
    rag_ev = [e for e in trace if e.get("event") == "rag"]
    assert rag_ev[0].get("allergen_blacklist_tokens") == []


def test_realistic_cashier_tracks_utterance_allergen_exclusions(
    monkeypatch,
) -> None:
    """A spoken restriction must not crash excluded-candidate tracing."""
    calls: list[dict[str, object]] = []

    def fake_search(query: str, allergens_blacklist=None, **_kwargs):
        calls.append({"query": query, "allergens_blacklist": allergens_blacklist})
        if allergens_blacklist:
            return []
        return [
            {
                "name": "McFlurry® with OREO® Cookies",
                "distance": 0.12,
                "energy": 510.0,
                "allergens": ["Milk"],
            }
        ]

    monkeypatch.setattr("mcd_voice.llm.agent.search_menu", fake_search)
    monkeypatch.setattr("mcd_voice.llm.agent._call_llm", _stub_llm_rewrite_then_cashier)
    agent = CashierAgent(rag_top_k=3, realistic_cashier=True)
    trace: list[dict] = []

    agent.generate_response(
        {"language": "EN", "companions": []},
        [
            {
                "speaker": "client",
                "text": "I'm lactose intolerant. Can I get a McFlurry with OREO Cookies?",
            }
        ],
        {"persons": []},
        rag_trace=trace,
        rag_meta={"call": "turn", "turn": 1},
    )

    assert any(c["allergens_blacklist"] == ["Milk"] for c in calls)
    rag_ev = [e for e in trace if e.get("event") == "rag"]
    assert rag_ev[0]["allergen_blacklist_tokens"] == ["Milk"]
    assert rag_ev[0]["excluded_by_constraints"] == [
        {
            "name": "McFlurry® with OREO® Cookies",
            "distance": 0.12,
            "allergens": ["Milk"],
            "mentioned_in_query": True,
        }
    ]


def test_client_prompt_language_from_profile() -> None:
    profile = {"language": "RU", "psycho": "regular", "calApprValue": 2000}
    text = get_client_system_prompt(profile)
    assert "Speak in Russian." in text


def test_cashier_prompt_language_from_profile() -> None:
    profile = {"language": "RU"}
    text = get_cashier_system_prompt(profile, realistic=False)
    assert "Speak in Russian." in text
