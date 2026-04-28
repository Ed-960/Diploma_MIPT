"""
Семантический поиск по меню (косинусная метрика в коллекции).

Косинусное расстояние удобно для эмбеддингов: сравнивается направление векторов смысла,
а не длина; для коротких описаний блюд это стандартный выбор.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

from mcd_voice.menu.chroma import configure_hf_cache, get_menu_collection
from mcd_voice.menu.parsing import allergens_meta_to_list
from mcd_voice.menu.rag_lexical import (
    chroma_fetch_n_for_lexical,
    filter_rows_by_excluded_lexical,
    normalize_excluded_lexical_terms,
)


def _build_where_for_blacklist(allergens_blacklist: list[str] | None) -> dict[str, Any] | None:
    """Фильтр Chroma: для каждого токена — allergens $not_contains; несколько — $and."""
    if not allergens_blacklist:
        return None
    clauses: list[dict[str, Any]] = []
    for a in allergens_blacklist:
        token = str(a).strip()
        if not token:
            continue
        clauses.append({"allergens": {"$not_contains": token}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _where_clauses_flat(where: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Разворачивает where в список атомарных условий (для объединения с калориями)."""
    if where is None:
        return []
    if "$and" in where:
        return list(where["$and"])
    return [where]


def build_where(
    allergens_blacklist: list[str] | None = None,
    *,
    max_energy: float | None = None,
    min_energy: float | None = None,
) -> dict[str, Any] | None:
    """
    Собирает единый фильтр Chroma: аллергены + опционально energy <= / >= (ккал в метаданных).
    """
    clauses: list[dict[str, Any]] = []
    clauses.extend(_where_clauses_flat(_build_where_for_blacklist(allergens_blacklist)))
    if max_energy is not None:
        clauses.append({"energy": {"$lte": float(max_energy)}})
    if min_energy is not None:
        clauses.append({"energy": {"$gte": float(min_energy)}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def search_menu(
    query: str,
    allergens_blacklist: list[str] | None = None,
    top_k: int = 5,
    *,
    max_energy: float | None = None,
    min_energy: float | None = None,
    excluded_lexical: Sequence[str] | None = None,
    chroma_trace: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Семантический поиск с опциональными фильтрами метаданных.

    :param max_energy: только блюда с energy <= этого значения (ккал).
    :param min_energy: только блюда с energy >= этого значения (ккал).
    :param excluded_lexical: слова/короткие фразы; позиции, где они встречаются в тексте
        карточки меню, отбрасываются после векторного поиска (см. ``rag_lexical``).

    Возвращает списки словарей: name, energy, allergens, sugar fields, ingredients, distance.
    """
    configure_hf_cache()
    collection = get_menu_collection()
    where = build_where(
        allergens_blacklist,
        max_energy=max_energy,
        min_energy=min_energy,
    )

    lex = normalize_excluded_lexical_terms(excluded_lexical or ())
    total_docs = max(1, collection.count())
    n = chroma_fetch_n_for_lexical(top_k, total_docs, bool(lex))
    kwargs: dict[str, Any] = {
        "query_texts": [query],
        "n_results": n,
    }
    if where is not None:
        kwargs["where"] = where

    if chroma_trace is not None:
        chroma_trace.append(
            {
                "event": "chroma_request",
                "query_texts": list(kwargs["query_texts"]),
                "n_results": int(kwargs["n_results"]),
                "where": kwargs.get("where"),
            },
        )

    t_query = time.perf_counter()
    try:
        results = collection.query(**kwargs)
    except Exception as exc:
        # WSL/Windows GPU stack can occasionally crash with torch.AcceleratorError.
        # Retry once on CPU to keep generation alive.
        msg = str(exc)
        if "CUDA error" not in msg and "AcceleratorError" not in msg:
            raise
        print(
            "  [!] CUDA error during Chroma query; retrying embeddings on CPU once.",
            flush=True,
        )
        # Do not mutate global EMBEDDING_DEVICE: in parallel mode this would
        # silently force all other workers onto CPU.
        collection = get_menu_collection(device="cpu")
        results = collection.query(**kwargs)
    query_ms = (time.perf_counter() - t_query) * 1000.0

    metas0 = results["metadatas"][0]
    dists0 = results["distances"][0]

    if chroma_trace is not None:
        ids0 = (results.get("ids") or [[]])[0] or []
        docs0 = (results.get("documents") or [[]])[0] or []
        chroma_trace.append(
            {
                "event": "chroma_response",
                "query_duration_ms": round(query_ms, 2),
                "ids": list(ids0),
                "distances": [float(x) for x in dists0],
                "metadatas": list(metas0),
                "documents": list(docs0),
            },
        )

    out: list[dict[str, Any]] = []
    for meta, dist in zip(metas0, dists0):
        out.append(
            {
                # идентификация
                "name":         meta.get("name", ""),
                "category":     meta.get("category", ""),
                "serving_size": meta.get("serving_size", ""),
                "tag":          meta.get("tag", ""),
                "description":  meta.get("description", ""),
                "ingredients":  meta.get("ingredients", ""),
                # аллергены
                "allergens":    allergens_meta_to_list(meta.get("allergens")),
                # нутриенты
                "energy":       meta.get("energy", 0.0),
                "protein":      meta.get("protein", 0.0),
                "total_fat":    meta.get("total_fat", 0.0),
                "sat_fat":      meta.get("sat_fat", 0.0),
                "trans_fat":    meta.get("trans_fat", 0.0),
                "chol":         meta.get("chol", 0.0),
                "carbs":        meta.get("carbs", 0.0),
                "total_sugar":  meta.get("total_sugar", 0.0),
                "added_sugar":  meta.get("added_sugar", 0.0),
                "sodium":       meta.get("sodium", 0.0),
                # расстояние
                "distance":     float(dist),
            }
        )
    if lex:
        out = filter_rows_by_excluded_lexical(out, lex)[: max(1, top_k)]
    return out


def _demo() -> None:
    rows = search_menu(
        "chicken without milk",
        allergens_blacklist=["Milk"],
        top_k=5,
    )
    for i, row in enumerate(rows, start=1):
        print(f"{i}. {row['name']}")
        print(f"   energy: {row['energy']}, distance: {row['distance']:.4f}")
        print(f"   allergens: {row['allergens']}")


if __name__ == "__main__":
    configure_hf_cache()
    _demo()
