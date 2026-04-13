from __future__ import annotations

from mcd_voice.profile import PROFILE_DECISION_GRAPH, to_mermaid, walk_graph


def test_walk_graph_returns_rows() -> None:
    rows = walk_graph(PROFILE_DECISION_GRAPH)
    assert rows
    assert rows[0]["name"] == "root"
    assert rows[0]["question"]


def test_mermaid_has_root_node() -> None:
    graph_text = to_mermaid(PROFILE_DECISION_GRAPH)
    assert graph_text.startswith("flowchart TD")
    assert "root" in graph_text
