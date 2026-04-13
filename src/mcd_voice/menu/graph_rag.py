"""Graph-augmented retrieval over menu metadata.

Лёгкая graph-RAG реализация без внешней graph DB:
- узлы: уникальные позиции меню (по name),
- рёбра: общая категория / тег / аллергены / токены ингредиентов,
- поиск: lexical seeds из запроса + расширение по соседям графа.
"""

from __future__ import annotations

from functools import lru_cache
import re
from typing import Any

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


def menu_graph_to_mermaid(
    *,
    max_edges: int = 220,
    min_weight: float = 0.25,
) -> str:
    """Mermaid flowchart for the menu co-occurrence graph (graph-RAG structure)."""
    graph = _build_menu_graph()
    nodes: dict[str, dict[str, Any]] = graph["nodes"]
    edges_map: dict[str, list[tuple[str, float]]] = graph["edges"]
    names = sorted(nodes.keys())
    id_for: dict[str, str] = {n: f"n{i}" for i, n in enumerate(names)}

    lines: list[str] = ["flowchart LR", "  %% menu graph-RAG: category/tag/allergen/token links"]
    for name in names:
        lid = id_for[name]
        label = _mermaid_escape_label(name)
        lines.append(f'  {lid}["{label}"]')

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
    for ia, ib, w in weighted[: max(0, max_edges)]:
        lines.append(f"  {ia} ---|{w:.2f}| {ib}")

    return "\n".join(lines) + "\n"
