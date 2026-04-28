"""Runtime-каталог блюд из Vector DB (Chroma) для диалогового контура."""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Any

from mcd_voice.config import NO_ALLERGEN_SENTINEL
from mcd_voice.menu.chroma import get_menu_collection


class MenuCatalog:
    """Загрузка меню из Chroma; используется конвейером диалога."""

    def load_runtime_index(
        self,
    ) -> tuple[list[str], dict[str, float], dict[str, list[str]], dict[str, dict[str, bool]]]:
        collection = get_menu_collection()
        payload = collection.get(include=["metadatas"])
        metas = payload.get("metadatas") or []

        seen_names: list[str] = []
        energy_acc: dict[str, list[float]] = defaultdict(list)
        allergen_acc: dict[str, set[str]] = defaultdict(set)
        restriction_map: dict[str, dict[str, bool]] = {}

        for meta in metas:
            if not isinstance(meta, dict):
                continue
            name = str(meta.get("name") or "").strip()
            if not name:
                continue
            if name not in energy_acc:
                seen_names.append(name)
            try:
                energy_acc[name].append(float(meta.get("energy") or 0.0))
            except (TypeError, ValueError):
                energy_acc[name].append(0.0)
            allergens = _allergen_set_from_meta(meta.get("allergens"))
            allergen_acc[name].update(allergens)
            blob = " ".join(
                str(meta.get(k, "") or "")
                for k in ("name", "ingredients", "tag", "description")
            ).lower()
            added_sugar = _f(meta.get("added_sugar"))
            cur = restriction_map.setdefault(
                name,
                {
                    "noMilk": False,
                    "noFish": False,
                    "noNuts": False,
                    "noEggs": False,
                    "noGluten": False,
                    "noBeef": False,
                    "isVegan": False,
                    "noSugar": False,
                },
            )
            has_milk = ("milk" in allergens) or bool(
                re.search(r"\b(milk|cheese|paneer|butter|latte|shake|sundae|mcflurry)\b", blob)
            )
            has_fish = ("fish" in allergens) or bool(
                re.search(r"\b(fish|filet[\s-]*o[\s-]*fish|tuna|salmon|shrimp)\b", blob)
            )
            has_eggs = ("egg" in allergens) or bool(re.search(r"\b(egg|mayo|mayonnaise)\b", blob))
            has_nuts = ("nuts" in allergens) or bool(
                re.search(r"\b(nut|nuts|peanut|almond|hazelnut|cashew|walnut)\b", blob)
            )
            has_gluten = ("cereal containing gluten" in allergens) or bool(
                re.search(r"\b(bun|bread|wrap|tortilla|muffin|biscuit|wheat|gluten)\b", blob)
            )
            has_beef = bool(
                re.search(r"\b(beef|hamburger|cheeseburger|big mac|mcdouble|quarter pounder)\b", blob)
            )
            has_animal = has_milk or has_fish or has_eggs or has_beef or bool(
                re.search(r"\b(chicken|beef|fish|egg|bacon|sausage|ham)\b", blob)
            )
            cur["noMilk"] = cur["noMilk"] or has_milk
            cur["noFish"] = cur["noFish"] or has_fish
            cur["noNuts"] = cur["noNuts"] or has_nuts
            cur["noEggs"] = cur["noEggs"] or has_eggs
            cur["noGluten"] = cur["noGluten"] or has_gluten
            cur["noBeef"] = cur["noBeef"] or has_beef
            cur["isVegan"] = cur["isVegan"] or has_animal
            cur["noSugar"] = cur["noSugar"] or (added_sugar > 0.0)

        energy_by_name = {
            name: round(sum(vals) / len(vals), 2)
            for name, vals in energy_acc.items()
        }
        allergen_map = {name: sorted(vals) for name, vals in allergen_acc.items()}
        return seen_names, energy_by_name, allergen_map, restriction_map

    def load(self) -> tuple[list[str], dict[str, float]]:
        """
        Backward-compatible API: names + average calories.
        """
        names, energy, _, _ = self.load_runtime_index()
        return names, energy


def _f(v: Any) -> float:
    try:
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _allergen_set_from_meta(raw: Any) -> set[str]:
    out: set[str] = set()
    if isinstance(raw, list):
        seq = raw
    elif raw is None:
        seq = []
    else:
        seq = [raw]
    for x in seq:
        t = str(x or "").strip()
        if not t or t == NO_ALLERGEN_SENTINEL:
            continue
        out.add(t.lower())
    return out
