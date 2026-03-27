"""
Семантический поиск по меню (косинусная метрика в коллекции).

Косинусное расстояние удобно для эмбеддингов: сравнивается направление векторов смысла,
а не длина; для коротких описаний блюд это стандартный выбор.
"""

from __future__ import annotations

from typing import Any

from mcd_voice.menu.chroma import configure_hf_cache, get_menu_collection
from mcd_voice.menu.parsing import allergens_meta_to_list


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
) -> list[dict[str, Any]]:
    """
    Семантический поиск с опциональными фильтрами метаданных.

    :param max_energy: только блюда с energy <= этого значения (ккал).
    :param min_energy: только блюда с energy >= этого значения (ккал).

    Возвращает списки словарей: name, energy, allergens, distance.
    """
    configure_hf_cache()
    collection = get_menu_collection()
    where = build_where(
        allergens_blacklist,
        max_energy=max_energy,
        min_energy=min_energy,
    )

    n = min(top_k, max(1, collection.count()))
    kwargs: dict[str, Any] = {
        "query_texts": [query],
        "n_results": n,
    }
    if where is not None:
        kwargs["where"] = where

    results = collection.query(**kwargs)
    metas0 = results["metadatas"][0]
    dists0 = results["distances"][0]

    out: list[dict[str, Any]] = []
    for meta, dist in zip(metas0, dists0):
        out.append(
            {
                "name": meta.get("name", ""),
                "energy": meta.get("energy", 0),
                "allergens": allergens_meta_to_list(meta.get("allergens")),
                "distance": float(dist),
            }
        )
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
