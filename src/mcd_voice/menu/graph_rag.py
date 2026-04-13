"""Graph-augmented retrieval over menu metadata.

Лёгкая graph-RAG реализация без внешней graph DB:
- узлы: уникальные позиции меню (по name),
- рёбра: общая категория / тег / аллергены / токены ингредиентов,
- поиск: lexical seeds из запроса + расширение по соседям графа.
"""

from __future__ import annotations

from functools import lru_cache
import re
from typing import Any, Sequence

from mcd_voice.menu.dataset import load_menu_from_json
from mcd_voice.menu.parsing import allergens_meta_to_list

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "with", "without", "for", "to", "of",
    "in", "on", "my", "me", "we", "i", "you", "it", "is", "are", "do",
    "have", "want", "would", "like", "please", "menu", "item", "items",
}


def _tokenize(text: str) -> set[str]:
    return {
        t for t in _TOKEN_RE.findall((text or "").lower())
        if len(t) >= 2 and t not in _STOPWORDS
    }


def _safe_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=1)
def _build_menu_graph() -> dict[str, Any]:
    _, _, metas = load_menu_from_json()
    nodes: dict[str, dict[str, Any]] = {}

    for m in metas:
        name = str(m.get("name") or "").strip()
        if not name:
            continue
        node = nodes.setdefault(
            name,
            {
                "name": name,
                "category": str(m.get("category") or "").strip().lower(),
                "tag": str(m.get("tag") or "").strip().lower(),
                "name_tokens": _tokenize(name),
                "text_tokens": set(),
                "allergens": set(),
                "energies": [],
                "added_sugar": [],
                "total_sugar": [],
            },
        )
        node["text_tokens"] |= _tokenize(
            f"{m.get('description') or ''} {m.get('ingredients') or ''}"
        )
        node["allergens"] |= {
            a.strip().lower()
            for a in allergens_meta_to_list(m.get("allergens"))
            if str(a).strip()
        }
        e = _safe_float(m.get("energy"))
        if e is not None:
            node["energies"].append(e)
        a_s = _safe_float(m.get("added_sugar"))
        if a_s is not None:
            node["added_sugar"].append(a_s)
        t_s = _safe_float(m.get("total_sugar"))
        if t_s is not None:
            node["total_sugar"].append(t_s)

    names = sorted(nodes.keys())
    edges: dict[str, list[tuple[str, float]]] = {n: [] for n in names}
    for i, n1 in enumerate(names):
        a = nodes[n1]
        for n2 in names[i + 1:]:
            b = nodes[n2]
            w = 0.0
            if a["category"] and a["category"] == b["category"]:
                w += 2.0
            if a["tag"] and a["tag"] == b["tag"]:
                w += 1.5
            shared_allergens = a["allergens"] & b["allergens"]
            if shared_allergens:
                w += min(2.0, 0.5 * len(shared_allergens))
            shared_text = a["text_tokens"] & b["text_tokens"]
            if shared_text:
                w += min(2.0, 0.25 * len(shared_text))
            if w > 0:
                edges[n1].append((n2, w))
                edges[n2].append((n1, w))

    for name in names:
        edges[name].sort(key=lambda x: x[1], reverse=True)
    return {"nodes": nodes, "edges": edges}


def _seed_score(node: dict[str, Any], query_tokens: set[str], query: str) -> float:
    if not query_tokens:
        return 0.0
    score = 0.0
    for tok in query_tokens:
        if tok in node["name_tokens"]:
            score += 3.0
        elif tok == node["category"] or tok == node["tag"]:
            score += 2.0
        elif tok in node["text_tokens"]:
            score += 1.0
    q = query.strip().lower()
    if q and q in node["name"].lower():
        score += 2.0
    return score


def search_menu_graph(
    query: str,
    *,
    allergens_blacklist: list[str] | None = None,
    top_k: int = 5,
    seed_k: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Возвращает menu rows для prompt и trace-info для graph-RAG."""
    graph = _build_menu_graph()
    nodes: dict[str, dict[str, Any]] = graph["nodes"]
    edges: dict[str, list[tuple[str, float]]] = graph["edges"]
    q_tokens = _tokenize(query)
    black = {str(a).strip().lower() for a in (allergens_blacklist or []) if str(a).strip()}

    scored = [
        (name, _seed_score(node, q_tokens, query))
        for name, node in nodes.items()
    ]
    scored.sort(key=lambda x: (x[1], x[0]), reverse=True)
    seeds = [(n, s) for n, s in scored if s > 0][: max(1, seed_k)]
    if not seeds:
        seeds = scored[: max(1, seed_k)]

    candidate_scores: dict[str, float] = {}
    for name, s in seeds:
        candidate_scores[name] = max(candidate_scores.get(name, 0.0), s + 1.0)
        for nb, w in edges.get(name, [])[: max(4, top_k)]:
            candidate_scores[nb] = max(candidate_scores.get(nb, 0.0), s * 0.5 + w)

    ranked = sorted(candidate_scores.items(), key=lambda x: (x[1], x[0]), reverse=True)
    rows: list[dict[str, Any]] = []
    for name, score in ranked:
        node = nodes[name]
        if black and (node["allergens"] & black):
            continue
        energies = node["energies"] or [0.0]
        added = node["added_sugar"] or [0.0]
        total = node["total_sugar"] or [0.0]
        # "distance" kept for compatibility with existing render/trace pipeline.
        dist = 1.0 / (1.0 + max(0.01, score))
        rows.append(
            {
                "name": name,
                "allergens": sorted(node["allergens"]),
                "energy": sum(energies) / len(energies),
                "added_sugar": sum(added) / len(added),
                "total_sugar": sum(total) / len(total),
                "distance": float(dist),
            }
        )
        if len(rows) >= max(1, top_k):
            break

    info = {
        "retrieval_mode": "graph",
        "metric": "graph_score_distance",
        "seed_nodes": [{"name": n, "seed_score": round(s, 3)} for n, s in seeds],
        "graph_candidate_count": len(candidate_scores),
    }
    return rows, info


def _mermaid_escape_label(text: str) -> str:
    return (
        (text or "")
        .replace("\\", "\\\\")
        .replace('"', "'")
        .replace("\n", " ")
        .replace("\r", " ")
    )


def _menu_graph_draw_edges(
    *,
    max_edges: int | None,
    min_weight: float,
) -> tuple[list[str], dict[str, str], list[tuple[str, str, float]]]:
    """Sorted node names, id n0.., undirected edges; ``max_edges=None`` — без ограничения."""
    graph = _build_menu_graph()
    nodes: dict[str, dict[str, Any]] = graph["nodes"]
    edges_map: dict[str, list[tuple[str, float]]] = graph["edges"]
    names = sorted(nodes.keys())
    id_for: dict[str, str] = {n: f"n{i}" for i, n in enumerate(names)}

    seen: set[tuple[str, str]] = set()
    weighted: list[tuple[str, str, float]] = []
    for a in names:
        for b, w in edges_map.get(a, []):
            if w < min_weight:
                continue
            ia, ib = id_for[a], id_for[b]
            key = (ia, ib) if ia < ib else (ib, ia)
            if key in seen:
                continue
            seen.add(key)
            weighted.append((ia, ib, w))

    weighted.sort(key=lambda t: t[2], reverse=True)
    if max_edges is None:
        trimmed = weighted
    else:
        trimmed = weighted[: max(0, max_edges)]
    return names, id_for, trimmed


def _resolve_one_focus_query(query: str, names: list[str]) -> str:
    """Map user string to exact menu ``name`` (exact, case-insensitive, else unique substring)."""
    q = (query or "").strip()
    if not q:
        raise ValueError("empty focus name")
    names_set = set(names)
    if q in names_set:
        return q
    by_lower = {n.lower(): n for n in names}
    ql = q.lower()
    if ql in by_lower:
        return by_lower[ql]
    hits = [n for n in names if ql in n.lower()]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise ValueError(f"Menu item not found: {query!r}")
    raise ValueError(f"Ambiguous {query!r}; try e.g. {hits[:10]!r}")


def menu_graph_focus_payload(
    focus_queries: Sequence[str],
    *,
    neighbor_hops: int = 1,
    min_weight: float = 0.25,
    max_edges: int | None = None,
    edge_mode: str = "star",
) -> dict[str, Any]:
    """Subgraph around seed menu names.

    ``neighbor_hops``: expand seeds by that many BFS hops (edges with w ≥ min_weight).

    ``edge_mode``:
      - ``star`` — только рёбра, где хотя бы один конец — seed (удобно для слайда);
      - ``induced`` — все рёбра между видимыми узлами (плотнее).
    """
    graph = _build_menu_graph()
    edges_map: dict[str, list[tuple[str, float]]] = graph["edges"]
    all_names = sorted(graph["nodes"].keys())
    seeds: list[str] = []
    for raw in focus_queries:
        r = (raw or "").strip()
        if not r:
            continue
        seeds.append(_resolve_one_focus_query(r, all_names))
    seeds = list(dict.fromkeys(seeds))
    if not seeds:
        raise ValueError("no focus items (use --focus Big Mac or --focus \"McChicken,Big Mac\")")
    seed_set = set(seeds)
    em = (edge_mode or "star").strip().lower()
    if em not in ("star", "induced"):
        raise ValueError("edge_mode must be 'star' or 'induced'")

    visible: set[str] = set(seeds)
    frontier: set[str] = set(seeds)
    for _ in range(max(0, neighbor_hops)):
        nxt: set[str] = set()
        for n in frontier:
            for nb, w in edges_map.get(n, []):
                if w < min_weight or nb in visible:
                    continue
                visible.add(nb)
                nxt.add(nb)
        frontier = nxt

    vis_sorted = sorted(visible)
    seen: set[tuple[str, str]] = set()
    weighted: list[tuple[str, str, float]] = []
    for a in vis_sorted:
        for b, w in edges_map.get(a, []):
            if w < min_weight or b not in visible or b <= a:
                continue
            if em == "star" and a not in seed_set and b not in seed_set:
                continue
            key = (a, b)
            if key in seen:
                continue
            seen.add(key)
            weighted.append((a, b, w))

    weighted.sort(key=lambda t: t[2], reverse=True)
    if max_edges is not None:
        weighted = weighted[: max(0, max_edges)]

    used: set[str] = set(seeds)
    for a, b, _ in weighted:
        used.add(a)
        used.add(b)
    vis_sorted = sorted(used)

    id_for = {n: f"n{i}" for i, n in enumerate(vis_sorted)}
    return {
        "nodes": [{"id": id_for[n], "label": n} for n in vis_sorted],
        "edges": [{"from": id_for[a], "to": id_for[b], "w": w} for a, b, w in weighted],
        "_viz": {
            "mode": "focus",
            "seeds": seeds,
            "neighbor_hops": neighbor_hops,
            "edge_mode": em,
            **({"edge_cap": max_edges} if max_edges is not None else {}),
        },
    }


def graph_payload_to_mermaid(payload: dict[str, Any]) -> str:
    """Mermaid from ``menu_graph_vis_payload`` / ``menu_graph_focus_payload`` shape (nodes + edges only)."""
    nodes: list[dict[str, Any]] = payload["nodes"]
    edges: list[dict[str, Any]] = payload["edges"]
    lines: list[str] = ["flowchart LR", "  %% menu graph-RAG (exported subset)"]
    for n in nodes:
        lid = str(n["id"])
        label = _mermaid_escape_label(str(n.get("label") or ""))
        lines.append(f'  {lid}["{label}"]')
    for e in edges:
        w = float(e.get("w", 0))
        lines.append(f"  {e['from']} ---|{w:.2f}| {e['to']}")
    return "\n".join(lines) + "\n"


def menu_graph_vis_payload(
    *,
    max_edges: int | None = None,
    min_weight: float = 0.25,
) -> dict[str, Any]:
    """JSON-serializable graph (vis HTML и ``visualize_menu_graph.py``)."""
    names, id_for, edges = _menu_graph_draw_edges(
        max_edges=max_edges, min_weight=min_weight
    )
    return {
        "nodes": [{"id": id_for[n], "label": n} for n in names],
        "edges": [{"from": ia, "to": ib, "w": w} for ia, ib, w in edges],
    }


def menu_graph_to_mermaid(
    *,
    max_edges: int | None = None,
    min_weight: float = 0.25,
) -> str:
    """Mermaid flowchart for the menu co-occurrence graph (graph-RAG structure)."""
    names, id_for, edges = _menu_graph_draw_edges(
        max_edges=max_edges, min_weight=min_weight
    )
    lines: list[str] = ["flowchart LR", "  %% menu graph-RAG: category/tag/allergen/token links"]
    for name in names:
        lid = id_for[name]
        label = _mermaid_escape_label(name)
        lines.append(f'  {lid}["{label}"]')
    for ia, ib, w in edges:
        lines.append(f"  {ia} ---|{w:.2f}| {ib}")

    return "\n".join(lines) + "\n"
