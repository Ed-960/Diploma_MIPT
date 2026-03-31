"""Трассировка RAG в CashierAgent.

RAG теперь вызывается всегда (нет keyword-фильтра): решение о подстановке контекста
принимается по семантическому расстоянию, а не по ключевым словам.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mcd_voice.llm.agent import CashierAgent

_VALID_OUTCOMES = ("injected", "injected_soft", "above_threshold", "no_chroma_hits")


@pytest.fixture()
def minimal_profile() -> dict:
    return {"language": "EN", "psycho": "regular"}


@pytest.fixture(autouse=True)
def _mock_openai_client(monkeypatch):
    """Не создаём реальный OpenAI-клиент — не нужны API-ключи."""
    monkeypatch.setattr(
        "mcd_voice.llm.agent._build_openai_client",
        lambda *a, **k: MagicMock(),
    )


def test_rag_always_runs_even_for_closing_phrase(
    monkeypatch, minimal_profile: dict,
) -> None:
    """RAG запускается независимо от текста — результат решается расстоянием."""
    agent = CashierAgent(rag_top_k=3)
    monkeypatch.setattr("mcd_voice.llm.agent._call_llm", lambda *a, **k: "ok")
    trace: list[dict] = []
    history = [{"speaker": "client", "text": "That is all, thanks."}]
    agent.generate_response(
        minimal_profile,
        history,
        {"persons": []},
        rag_trace=trace,
        rag_meta={"call": "turn", "turn": 1},
    )
    rag_ev = [e for e in trace if e.get("event") == "rag"]
    assert len(rag_ev) == 1
    assert rag_ev[0]["outcome"] in _VALID_OUTCOMES


def test_rag_runs_for_menu_query(
    monkeypatch, minimal_profile: dict,
) -> None:
    """Запрос про меню даёт event=rag с ожидаемыми полями трассы."""
    agent = CashierAgent(rag_top_k=3)
    monkeypatch.setattr("mcd_voice.llm.agent._call_llm", lambda *a, **k: "ok")
    trace: list[dict] = []
    history = [
        {"speaker": "client", "text": "What would you recommend from the menu?"},
    ]
    agent.generate_response(
        minimal_profile,
        history,
        {"persons": []},
        rag_trace=trace,
        rag_meta={"call": "turn", "turn": 2},
    )
    rag_ev = [e for e in trace if e.get("event") == "rag"]
    assert len(rag_ev) == 1
    event = rag_ev[0]
    assert event["event"] == "rag"
    assert event["call"] == "turn"
    assert event["turn"] == 2
    assert event["metric"] == "cosine_distance"
    assert "candidates" in event
    assert event["outcome"] in _VALID_OUTCOMES
    assert "context_preview" in event


def test_rag_disabled_when_top_k_zero(
    monkeypatch, minimal_profile: dict,
) -> None:
    """При rag_top_k=0 поиск не запускается."""
    agent = CashierAgent(rag_top_k=0)
    monkeypatch.setattr("mcd_voice.llm.agent._call_llm", lambda *a, **k: "ok")
    trace: list[dict] = []
    history = [{"speaker": "client", "text": "What do you have?"}]
    agent.generate_response(
        minimal_profile,
        history,
        {"persons": []},
        rag_trace=trace,
        rag_meta={"call": "turn", "turn": 1},
    )
    assert len(trace) == 1
    assert trace[0]["event"] == "rag_disabled"


def test_rag_uses_fallback_query_when_no_client_text(
    monkeypatch, minimal_profile: dict,
) -> None:
    """Если клиент ещё не говорил (приветствие), используется fallback-запрос."""
    agent = CashierAgent(rag_top_k=3)
    monkeypatch.setattr("mcd_voice.llm.agent._call_llm", lambda *a, **k: "ok")
    trace: list[dict] = []
    agent.generate_response(
        minimal_profile,
        history=[],
        order_state={"persons": []},
        rag_trace=trace,
        rag_meta={"call": "greeting"},
    )
    rag_ev = [e for e in trace if e.get("event") == "rag"]
    assert len(rag_ev) == 1
    assert rag_ev[0].get("fallback") is True
