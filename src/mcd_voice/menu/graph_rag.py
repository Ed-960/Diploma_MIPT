"""Graph-augmented retrieval over menu metadata.

Гетерогенный graph-RAG без внешней graph DB:
- узлы: блюда, ингредиенты, аллергены, категории/теги и лексические признаки;
- рёбра: dish -> ingredient/allergen/category/tag/token;
- поиск: lexical seeds из запроса -> соседние узлы -> пути к dish-кандидатам.

Публичный API ``search_menu_graph`` совместим с прежней реализацией: агент получает
menu rows и trace-info, а Chroma/LLM для этого режима не требуются.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Sequence

from mcd_voice.menu.dataset import load_menu_from_json
from mcd_voice.menu.parsing import allergens_meta_to_list
from mcd_voice.menu.rag_lexical import (
    filter_rows_by_excluded_lexical,
    normalize_excluded_lexical_terms,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "with", "without", "for", "to", "of",
    "in", "on", "my", "me", "we", "i", "you", "it", "is", "are", "do",
    "have", "want", "would", "like", "please", "menu", "item", "items",
    "no", "free", "allergic", "allergy", "intolerant", "avoid",
}

_REL_WEIGHTS = {
    "name_token": 3.0,
    "contains": 2.4,
    "has_allergen": 2.1,
    "in_category": 2.0,
    "has_tag": 1.5,
    "mentions": 1.0,
}

_ALLERGEN_QUERY_ALIASES: dict[str, str] = {
    "dairy": "milk",
    "lactose": "milk",
    "milk": "milk",
    "gluten": "cereal containing gluten",
    "wheat": "cereal containing gluten",
    "egg": "egg",
    "eggs": "egg",
    "nut": "nuts",
    "nuts": "nuts",
    "peanut": "nuts",
    "peanuts": "nuts",
    "fish": "fish",
    "soy": "soya",
    "soya": "soya",
    "sulphite": "sulphites",
    "sulphites": "sulphites",
    "sulfite": "sulphites",
    "sulfites": "sulphites",
}

_CATEGORY_HINTS: dict[str, str] = {
    "burger": "burger",
    "burgers": "burger",
    "sandwich": "sandwich",
    "sandwiches": "sandwich",
    "wrap": "wrap",
    "wraps": "wrap",
    "fries": "fries",
    "side": "side",
    "sides": "side",
    "drink": "drink",
    "drinks": "drink",
    "beverage": "drink",
    "beverages": "drink",
    "coffee": "coffee",
    "dessert": "dessert",
    "desserts": "dessert",
    "chicken": "chicken",
    "nuggets": "nuggets",
}


def _tokenize(text: str) -> set[str]:
    return {
        t for t in _TOKEN_RE.findall((text or "").lower())
        if len(t) >= 2 and t not in _STOPWORDS
    }


def _norm_label(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _node_id(kind: str, label: str) -> str:
    return f"{kind}:{_norm_label(label)}"


def _split_ingredients(raw: Any) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    parts = re.split(r",|;|\band\b|\bwith\b", text, flags=re.I)
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        cleaned = re.sub(r"\s+", " ", part.strip(" ."))
        cleaned = re.sub(
            r"^(?:freshly|fresh|premium|mildly|spiced|sliced|shredded|salted)\s+",
            "",
            cleaned,
            flags=re.I,
        ).strip()
        if len(cleaned) < 2:
            continue
        key = _norm_label(cleaned)
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def _add_node(
    nodes: dict[str, dict[str, Any]],
    node_id: str,
    *,
    kind: str,
    label: str,
    tokens: set[str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    node = nodes.setdefault(
        node_id,
        {
            "id": node_id,
            "kind": kind,
            "label": label,
            "tokens": set(),
        },
    )
    node["tokens"].update(tokens or _tokenize(label))
    node.update(extra)
    return node


def _add_edge(
    edges: dict[str, list[tuple[str, float]]],
    edge_meta: dict[tuple[str, str], str],
    a: str,
    b: str,
    *,
    weight: float,
    relation: str,
) -> None:
    def _upsert(src: str, dst: str) -> None:
        rows = edges.setdefault(src, [])
        for i, (old_dst, old_w) in enumerate(rows):
            if old_dst == dst:
                rows[i] = (dst, max(old_w, weight))
                return
        rows.append((dst, weight))

    _upsert(a, b)
    _upsert(b, a)
    edge_meta[(a, b)] = relation
    edge_meta[(b, a)] = relation


def _infer_category_nodes(name: str, tag: str, description: str) -> list[str]:
    tokens = _tokenize(f"{name} {tag} {description}")
    out: list[str] = []
    for tok in sorted(tokens):
        if tok in _CATEGORY_HINTS:
            cat = _CATEGORY_HINTS[tok]
            if cat not in out:
                out.append(cat)
    return out


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
    edges: dict[str, list[tuple[str, float]]] = {}
    edge_meta: dict[tuple[str, str], str] = {}
    dish_ids: list[str] = []
    dish_id_by_name: dict[str, str] = {}

    for m in metas:
        name = str(m.get("name") or "").strip()
        if not name:
            continue
        description = str(m.get("description") or "")
        ingredients_raw = str(m.get("ingredients") or "")
        tag = str(m.get("tag") or "").strip()
        category = str(m.get("category") or "").strip()
        allergens = allergens_meta_to_list(m.get("allergens"))
        dish_id = _node_id("dish", name)
        dish = _add_node(
            nodes,
            dish_id,
            kind="dish",
            label=name,
            tokens=_tokenize(f"{name} {description} {ingredients_raw} {tag} {category}"),
            name=name,
            category=category.lower(),
            tag=tag.lower(),
            ingredients=ingredients_raw,
            description=description,
            allergens=set(),
            allergens_display=[],
            energies=[],
            added_sugar=[],
            total_sugar=[],
        )
        if dish_id not in dish_ids:
            dish_ids.append(dish_id)
            dish_id_by_name[name] = dish_id
        dish["allergens"].update({a.strip().lower() for a in allergens if str(a).strip()})
        for a in allergens:
            token = str(a).strip()
            if token and token not in dish["allergens_display"]:
                dish["allergens_display"].append(token)
        e = _safe_float(m.get("energy"))
        if e is not None:
            dish["energies"].append(e)
        a_s = _safe_float(m.get("added_sugar"))
        if a_s is not None:
            dish["added_sugar"].append(a_s)
        t_s = _safe_float(m.get("total_sugar"))
        if t_s is not None:
            dish["total_sugar"].append(t_s)

        for ingredient in _split_ingredients(ingredients_raw):
            ing_id = _node_id("ingredient", ingredient)
            _add_node(nodes, ing_id, kind="ingredient", label=ingredient)
            _add_edge(
                edges, edge_meta, dish_id, ing_id,
                weight=_REL_WEIGHTS["contains"], relation="contains",
            )

        for allergen in allergens:
            all_id = _node_id("allergen", allergen)
            _add_node(nodes, all_id, kind="allergen", label=allergen)
            _add_edge(
                edges, edge_meta, dish_id, all_id,
                weight=_REL_WEIGHTS["has_allergen"], relation="has_allergen",
            )

        category_labels = []
        if category:
            category_labels.append(category)
        category_labels.extend(_infer_category_nodes(name, tag, description))
        for cat in dict.fromkeys(category_labels):
            cat_id = _node_id("category", cat)
            _add_node(nodes, cat_id, kind="category", label=cat)
            _add_edge(
                edges, edge_meta, dish_id, cat_id,
                weight=_REL_WEIGHTS["in_category"], relation="in_category",
            )

        if tag:
            tag_id = _node_id("tag", tag)
            _add_node(nodes, tag_id, kind="tag", label=tag)
            _add_edge(
                edges, edge_meta, dish_id, tag_id,
                weight=_REL_WEIGHTS["has_tag"], relation="has_tag",
            )

        for tok in _tokenize(f"{name} {description}"):
            tok_id = _node_id("token", tok)
            _add_node(nodes, tok_id, kind="token", label=tok, tokens={tok})
            weight = _REL_WEIGHTS["name_token"] if tok in _tokenize(name) else _REL_WEIGHTS["mentions"]
            _add_edge(
                edges, edge_meta, dish_id, tok_id,
                weight=weight, relation="name_token" if tok in _tokenize(name) else "mentions",
            )

    for node_id in edges:
        edges[node_id].sort(key=lambda x: x[1], reverse=True)
    kind_counts: dict[str, int] = {}
    for node in nodes.values():
        kind_counts[node["kind"]] = kind_counts.get(node["kind"], 0) + 1
    return {
        "nodes": nodes,
        "edges": edges,
        "edge_meta": edge_meta,
        "dish_ids": dish_ids,
        "dish_id_by_name": dish_id_by_name,
        "schema": {
            "node_kinds": kind_counts,
            "relations": sorted(set(edge_meta.values())),
        },
    }


def _seed_score(node: dict[str, Any], query_tokens: set[str], query: str) -> float:
    if not query_tokens:
        return 0.0
    score = 0.0
    node_tokens = set(node.get("tokens") or ())
    label = str(node.get("label") or "")
    kind = str(node.get("kind") or "")
    for tok in query_tokens:
        alias = _ALLERGEN_QUERY_ALIASES.get(tok)
        category_hint = _CATEGORY_HINTS.get(tok)
        if tok in node_tokens:
            score += 3.0 if kind == "dish" else 2.0
        if alias and kind == "allergen" and alias in _norm_label(label):
            score += 3.0
        if category_hint and kind == "category" and category_hint == _norm_label(label):
            score += 2.5
    q = query.strip().lower()
    if q and q in _norm_label(label):
        score += 2.0
    return score


def _dish_row_from_node(node: dict[str, Any], score: float) -> dict[str, Any]:
    energies = node.get("energies") or [0.0]
    added = node.get("added_sugar") or [0.0]
    total = node.get("total_sugar") or [0.0]
    dist = 1.0 / (1.0 + max(0.01, score))
    return {
        "name": node.get("name") or node.get("label") or "",
        "category": node.get("category", ""),
        "tag": node.get("tag", ""),
        "description": node.get("description", ""),
        "ingredients": node.get("ingredients", ""),
        "allergens": list(node.get("allergens_display") or sorted(node.get("allergens") or [])),
        "energy": sum(energies) / len(energies),
        "added_sugar": sum(added) / len(added),
        "total_sugar": sum(total) / len(total),
        "distance": float(dist),
    }


def _candidate_paths(
    seed_id: str,
    seed_score: float,
    graph: dict[str, Any],
    *,
    max_neighbors: int,
) -> list[tuple[str, float, list[str]]]:
    nodes: dict[str, dict[str, Any]] = graph["nodes"]
    edges: dict[str, list[tuple[str, float]]] = graph["edges"]
    out: list[tuple[str, float, list[str]]] = []
    seed_node = nodes[seed_id]
    if seed_node["kind"] == "dish":
        out.append((seed_id, seed_score + 2.0, [seed_id]))

    for nb, w in edges.get(seed_id, [])[:max_neighbors]:
        nb_node = nodes[nb]
        if nb_node["kind"] == "dish":
            out.append((nb, seed_score + w, [seed_id, nb]))
            continue
        for nb2, w2 in edges.get(nb, [])[:max_neighbors]:
            if nodes[nb2]["kind"] != "dish":
                continue
            out.append((nb2, seed_score + 0.75 * w + 0.35 * w2, [seed_id, nb, nb2]))
    return out


def search_menu_graph(
    query: str,
    *,
    allergens_blacklist: list[str] | None = None,
    top_k: int = 5,
    seed_k: int = 4,
    max_energy: float | None = None,
    min_energy: float | None = None,
    excluded_lexical: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Возвращает menu rows для prompt и trace-info для graph-RAG.

    ``max_energy`` / ``min_energy`` — по средней энергии вариантов с тем же name (как в строке ответа).

    ``excluded_lexical`` — те же правила, что у ``search_menu`` (текст карточки по имени из ``mcd.json``).
    """
    graph = _build_menu_graph()
    nodes: dict[str, dict[str, Any]] = graph["nodes"]
    dish_ids: list[str] = graph["dish_ids"]
    q_tokens = _tokenize(query)
    black = {str(a).strip().lower() for a in (allergens_blacklist or []) if str(a).strip()}

    scored = [
        (node_id, _seed_score(node, q_tokens, query))
        for node_id, node in nodes.items()
    ]
    scored.sort(key=lambda x: (x[1], x[0]), reverse=True)
    seeds = [(n, s) for n, s in scored if s > 0][: max(1, seed_k)]
    if not seeds:
        seeds = [(dish_id, 0.1) for dish_id in dish_ids[: max(1, seed_k)]]

    candidate_scores: dict[str, float] = {}
    best_paths: dict[str, list[str]] = {}
    max_neighbors = max(8, top_k * 2)
    for seed_id, s in seeds:
        for dish_id, score, path in _candidate_paths(
            seed_id, s, graph, max_neighbors=max_neighbors,
        ):
            if score <= candidate_scores.get(dish_id, -1.0):
                continue
            candidate_scores[dish_id] = score
            best_paths[dish_id] = path

    ranked = sorted(candidate_scores.items(), key=lambda x: (x[1], x[0]), reverse=True)
    lex = normalize_excluded_lexical_terms(excluded_lexical or ())
    pool_k = max(top_k * 8, top_k + 20) if lex else top_k
    rows: list[dict[str, Any]] = []
    used_path_rows: list[dict[str, Any]] = []
    for dish_id, score in ranked:
        node = nodes[dish_id]
        if black and (node["allergens"] & black):
            continue
        energies = node.get("energies") or [0.0]
        avg_e = sum(energies) / len(energies)
        if max_energy is not None and avg_e > float(max_energy) + 1e-6:
            continue
        if min_energy is not None and avg_e < float(min_energy) - 1e-6:
            continue
        rows.append(_dish_row_from_node(node, score))
        path = best_paths.get(dish_id, [dish_id])
        used_path_rows.append(
            {
                "dish": node.get("name") or node["label"],
                "score": round(score, 3),
                "path": [
                    {
                        "id": pid,
                        "kind": nodes[pid]["kind"],
                        "label": nodes[pid]["label"],
                    }
                    for pid in path
                ],
            }
        )
        if len(rows) >= max(1, pool_k):
            break

    if lex:
        rows = filter_rows_by_excluded_lexical(rows, lex)[: max(1, top_k)]
    else:
        rows = rows[: max(1, top_k)]

    info = {
        "retrieval_mode": "graph",
        "metric": "graph_score_distance",
        "graph_schema": graph["schema"],
        "seed_nodes": [
            {
                "id": n,
                "kind": nodes[n]["kind"],
                "label": nodes[n]["label"],
                "seed_score": round(s, 3),
            }
            for n, s in seeds
        ],
        "retrieval_paths": used_path_rows[: max(1, top_k)],
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
    """Sorted graph node ids, display ids n0.., undirected edges; ``max_edges=None`` — без ограничения."""
    graph = _build_menu_graph()
    nodes: dict[str, dict[str, Any]] = graph["nodes"]
    edges_map: dict[str, list[tuple[str, float]]] = graph["edges"]
    names = sorted(nodes.keys(), key=lambda n: (nodes[n]["kind"], nodes[n]["label"]))
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
    graph = _build_menu_graph()
    dish_names = sorted(graph["dish_id_by_name"].keys())
    names_set = set(dish_names)
    if q in names_set:
        return graph["dish_id_by_name"][q]
    by_lower = {n.lower(): n for n in dish_names}
    ql = q.lower()
    if ql in by_lower:
        return graph["dish_id_by_name"][by_lower[ql]]
    hits = [n for n in dish_names if ql in n.lower()]
    if len(hits) == 1:
        return graph["dish_id_by_name"][hits[0]]
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
    nodes = graph["nodes"]
    return {
        "nodes": [
            {"id": id_for[n], "label": nodes[n]["label"], "kind": nodes[n]["kind"]}
            for n in vis_sorted
        ],
        "edges": [{"from": id_for[a], "to": id_for[b], "w": w} for a, b, w in weighted],
        "_viz": {
            "mode": "focus",
            "seeds": [nodes[s]["label"] for s in seeds],
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
    graph = _build_menu_graph()
    nodes = graph["nodes"]
    return {
        "nodes": [
            {
                "id": id_for[n],
                "label": nodes[n]["label"],
                "kind": nodes[n]["kind"],
            }
            for n in names
        ],
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
    graph = _build_menu_graph()
    nodes = graph["nodes"]
    lines: list[str] = ["flowchart LR", "  %% heterogeneous menu graph-RAG: dish/ingredient/allergen/category/tag links"]
    for name in names:
        lid = id_for[name]
        node = nodes[name]
        label = _mermaid_escape_label(f"{node['kind']}: {node['label']}")
        lines.append(f'  {lid}["{label}"]')
    for ia, ib, w in edges:
        lines.append(f"  {ia} ---|{w:.2f}| {ib}")

    return "\n".join(lines) + "\n"
