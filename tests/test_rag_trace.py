"""Трассировка RAG в CashierAgent.

RAG теперь вызывается всегда (нет keyword-фильтра): решение о подстановке контекста
принимается по семантическому расстоянию, а не по ключевым словам.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mcd_voice.llm.agent import CashierAgent, _render_rows

_VALID_OUTCOMES = ("injected", "injected_soft", "above_threshold", "no_chroma_hits")


def _stub_llm_rewrite_then_cashier(client, model, system, messages, temperature=0.8):
    """
    Отличает mini-LLM rewrite от ответа кассира.
    Ответ «ok» в rewrite нормализуется в general menu items и отключает RAG — тесты
    должны возвращать осмысленный поисковый запрос.
    """
    if "short food search query" in (system or "").lower():
        return "burgers chicken sides drinks dessert"
    return "Got it, thanks for stopping by."


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
    monkeypatch.setattr("mcd_voice.llm.agent._call_llm", _stub_llm_rewrite_then_cashier)
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
    monkeypatch.setattr("mcd_voice.llm.agent._call_llm", _stub_llm_rewrite_then_cashier)
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


def test_graph_rag_mode_uses_graph_retrieval(
    monkeypatch, minimal_profile: dict,
) -> None:
    """При rag_mode=graph кассир использует graph retrieval, а не Chroma."""

    def _fake_graph_search(*_args, **_kwargs):
        return (
            [
                {
                    "name": "Graph Burger",
                    "distance": 0.2,
                    "allergens": [],
                    "energy": 410.0,
                    "added_sugar": 1.0,
                    "total_sugar": 4.0,
                },
            ],
            {"retrieval_mode": "graph", "metric": "graph_score_distance"},
        )

    monkeypatch.setattr("mcd_voice.llm.agent.search_menu_graph", _fake_graph_search)
    monkeypatch.setattr(
        "mcd_voice.llm.agent.search_menu",
        lambda *_a, **_k: pytest.fail("vector search should not run in graph mode"),
    )
    monkeypatch.setattr("mcd_voice.llm.agent._call_llm", _stub_llm_rewrite_then_cashier)
    agent = CashierAgent(rag_top_k=3, rag_mode="graph")
    trace: list[dict] = []
    history = [{"speaker": "client", "text": "Any burger options?"}]
    agent.generate_response(
        minimal_profile,
        history,
        {"persons": []},
        rag_trace=trace,
        rag_meta={"call": "turn", "turn": 3},
    )
    rag_ev = [e for e in trace if e.get("event") == "rag"]
    assert len(rag_ev) == 1
    assert rag_ev[0]["retrieval_mode"] == "graph"
    assert rag_ev[0]["metric"] == "graph_score_distance"
    assert rag_ev[0]["outcome"] == "injected_graph"


def test_render_rows_respects_max_lines() -> None:
    """После фильтра по distance остаются не более max_lines строк (порядок релевантности)."""
    rows: list[dict] = []
    for i in range(30):
        rows.append(
            {
                "name": f"Item{i}",
                "distance": 0.1 + i * 0.001,
                "allergens": [],
                "energy": 100.0,
                "added_sugar": 1.0,
                "total_sugar": 2.0,
            },
        )
    lines, used = _render_rows(rows, max_dist=1.0, max_lines=5)
    assert len(lines) == len(used) == 5
    assert lines[0].startswith("- Item0")
    assert "Item4" in lines[4]

    lines_all, _ = _render_rows(rows, max_dist=1.0, max_lines=None)
    assert len(lines_all) == 30


def test_menu_graph_to_mermaid_exports_structure() -> None:
    from mcd_voice.menu.graph_rag import menu_graph_to_mermaid

    text = menu_graph_to_mermaid(max_edges=50, min_weight=0.5)
    assert text.startswith("flowchart LR")
    assert "---|" in text
    assert '["' in text
