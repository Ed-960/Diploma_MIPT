"""Graph RAG по mcd.json + in-memory граф: без Chroma, без LLM, без сети."""

from __future__ import annotations

from mcd_voice.llm.agent import RAG_MODE_GRAPH, CashierAgent, merge_graph_retrieval_query
from mcd_voice.menu.graph_rag import _build_menu_graph, search_menu_graph


def test_merge_graph_retrieval_query_orders_and_dedupes() -> None:
    """Client text first, then rewrite; identical strings appear once."""
    q = merge_graph_retrieval_query(
        "dairy-free burger options",
        "burger no milk please",
        ("burger no milk please", "Big Mac"),
    )
    lines = [ln for ln in q.split("\n") if ln.strip()]
    assert lines[0] == "burger no milk please"
    assert "dairy-free burger options" in lines
    assert "Big Mac" in lines
    assert lines.count("burger no milk please") == 1


def test_search_menu_graph_returns_hits_and_trace_info() -> None:
    rows, info = search_menu_graph("Big Mac", top_k=5)
    assert len(rows) >= 1
    assert info["retrieval_mode"] == "graph"
    assert info["metric"] == "graph_score_distance"
    assert {"dish", "ingredient", "allergen", "category"} <= set(
        info["graph_schema"]["node_kinds"],
    )
    names = [r["name"] for r in rows]
    assert any("Big Mac" in n for n in names)
    assert all("distance" in r for r in rows)
    assert info["retrieval_paths"]


def test_menu_graph_is_heterogeneous() -> None:
    graph = _build_menu_graph()
    kinds = {node["kind"] for node in graph["nodes"].values()}
    assert {"dish", "ingredient", "allergen", "category", "token"} <= kinds
    assert "contains" in graph["schema"]["relations"]
    assert "has_allergen" in graph["schema"]["relations"]
    assert "in_category" in graph["schema"]["relations"]


def test_search_menu_graph_respects_allergen_blacklist_paths() -> None:
    rows, info = search_menu_graph(
        "dairy free burger",
        allergens_blacklist=["Milk"],
        top_k=8,
    )
    assert rows
    assert all("Milk" not in r.get("allergens", []) for r in rows)
    assert any(path["path"][0]["kind"] in {"category", "token", "dish"} for path in info["retrieval_paths"])


def test_cashier_agent_graph_path_calls_search_menu_graph_only(
    monkeypatch,
) -> None:
    """_do_graph_rag использует search_menu_graph; Chroma (search_menu) не вызывается."""
    minimal_profile = {"language": "EN", "psycho": "regular", "companions": []}

    called: dict[str, int] = {"graph": 0, "vector": 0}

    def _track_graph(*args, **kwargs):
        called["graph"] += 1
        return search_menu_graph(*args, **kwargs)

    monkeypatch.setattr(
        "mcd_voice.llm.agent.search_menu_graph",
        _track_graph,
    )
    monkeypatch.setattr(
        "mcd_voice.llm.agent.search_menu",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("vector/Chroma must not run in graph rag_mode"),
        ),
    )

    agent = CashierAgent.__new__(CashierAgent)
    agent.rag_mode = RAG_MODE_GRAPH
    agent._realistic_cashier = True
    agent.rag_top_k = 8
    agent.rag_max_prompt_lines = 15

    ctx, info = CashierAgent._do_graph_rag(
        agent,
        "burger please",
        minimal_profile,
        rag_constraint_texts=(),
        rag_json_spec=None,
    )
    assert info.get("graph_retrieval_query")
    assert called["graph"] == 1
    assert called["vector"] == 0
    assert info.get("outcome") == "injected_graph"
    assert len(ctx) > 0


def test_load_runtime_index_from_json_matches_chroma_shape() -> None:
    from mcd_voice.dialog.catalog import MenuCatalog

    cat = MenuCatalog()
    names_j, energy_j, allergens_j, restr_j = cat.load_runtime_index_from_json()
    assert names_j
    assert isinstance(energy_j, dict)
    assert isinstance(allergens_j, dict)
    assert isinstance(restr_j, dict)
