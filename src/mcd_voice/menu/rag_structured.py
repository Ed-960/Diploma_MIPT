"""
Структурированный ответ mini-LLM: JSON (search_query + чем отфильтровать Chroma ``where``).

`$not_contains` сопоставляет подстроки в сохранённых `metadata['allergens']`.
Список разрешённых канонов берётся напрямую из Chroma metadata
(`chroma_excludable_allergen_vocabulary`) — без чтения `mcd.json` в runtime.

**Алиасы** (dairy → Milk): разговорное слово **редко** **совпадает** **с** **строкой** **в** **данных**;** **без** **нормализации** **молочное** **блюдо** **могло** **бы** **остаться** **в** **топ-****k** **(****$not_contains** **«**dairy**»** **не** **тронет** **документ** **с** **только** **«**Milk**»****).**
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, TypedDict

from mcd_voice.config import NO_ALLERGEN_SENTINEL
from mcd_voice.menu.chroma import get_menu_collection
from mcd_voice.menu.parsing import allergens_meta_to_list
from mcd_voice.menu.rag_lexical import normalize_excluded_lexical_terms


@lru_cache(maxsize=1)
def chroma_excludable_allergen_vocabulary() -> frozenset[str]:
    """
    Уникальные токены из ``metadata['allergens']`` (как при загрузке в Chroma).
    """
    try:
        collection = get_menu_collection()
        payload = collection.get(include=["metadatas"])
        metas = payload.get("metadatas") or []
    except Exception:
        # Safe fallback when Chroma isn't initialized yet.
        return frozenset(
            [
                "Milk",
                "Egg",
                "Fish",
                "Nuts",
                "Cereal containing gluten",
                "Soya",
                "Sulphites",
            ]
        )
    u: set[str] = set()
    for m in metas:
        if not isinstance(m, dict):
            continue
        for a in allergens_meta_to_list(m.get("allergens")):
            t = str(a).strip()
            if t and t != NO_ALLERGEN_SENTINEL:
                u.add(t)
    return frozenset(u)

# Синонимы / варианты от LLM → каноничный токен
_ALLERGEN_ALIASES: dict[str, str] = {
    "milk": "Milk",
    "dairy": "Milk",
    "lactose": "Milk",
    "egg": "Egg",
    "eggs": "Egg",
    "fish": "Fish",
    "shellfish": "Fish",
    "nuts": "Nuts",
    "nut": "Nuts",
    "peanuts": "Nuts",
    "peanut": "Nuts",
    "gluten": "Cereal containing gluten",
    "wheat": "Cereal containing gluten",
    "soy": "Soya",
    "soya": "Soya",
    "sulphites": "Sulphites",
    "sulfites": "Sulphites",
    "sulphite": "Sulphites",
}

# Шаблон: {allowed} подставляется из меню
_RAG_JSON_TEMPLATE = """You are a menu RAG pre-processor. Read the customer message and output a SINGLE JSON object only (no markdown, no commentary).

Schema:
{{
  "intent": "one of: lookup | alternatives | details | calorie_tune | compare",
  "search_query": "string, 3–12 words, English, positive phrasing for semantic search over menu (food type, category, what they want). If they must avoid an ingredient, still describe what they can have (e.g. grilled chicken, sides, plant-based) — do not only repeat the avoided word.",
  "compare_metrics": [{{"field":"protein","goal":"max"}}],
  "excluded_allergens": [ "Milk" ],
  "excluded_lexical": [ "beef" ],
  "max_kcal": null,
  "min_kcal": null
}}

(You may use the alias key "allergies" with the same meaning as "excluded_allergens" — list of ingredients/allergen types the customer must NOT receive; we exclude matching menu rows in vector search.)

Rules:
- "intent":
  * lookup — default menu retrieval / recommendation
  * alternatives — customer asks for other/new options
  * details — customer asks ingredients/details about selected item
  * calorie_tune — customer asks to match/adjust calories target
  * compare — customer asks to compare menu options by nutrients
- "compare_metrics": optional list for compare intent.
  - field: one of energy, protein, total_fat, sat_fat, trans_fat, chol, carbs, total_sugar, added_sugar, sodium
  - goal: max or min
  - examples: most protein -> {{"field":"protein","goal":"max"}}, least fat -> {{"field":"total_fat","goal":"min"}}
- "excluded_allergens" (or "allergies"): use ONLY these exact strings as they appear in the menu (subset of): {allowed}
- "excluded_lexical": 0–12 short English tokens or 2-word phrases the customer must NOT get, matched against menu item name/description/ingredients/tag (e.g. beef, bacon, pickle, onion, mayo, coffee). Use for avoided foods or ingredients that are NOT covered by excluded_allergens. Empty [] if none. Lowercase words; no sentences.
- If the customer is vegan or fully plant-based, you may list Milk, Egg, Fish as excluded (when they appear in the list above).
- "max_kcal" / "min_kcal": numbers (kcal) or null. Use max_kcal for "under X calories", "light", "not more than X kcal". Use min_kcal for "at least X calories", "filling", "hearty". If unclear, null.
- If there is no food/diet intent, set search_query to: general menu items
- Output valid JSON only, one object.
"""


def get_rag_json_system_prompt() -> str:
    """Системный промпт с актуальным списком токенов из Chroma metadata."""
    allowed = ", ".join(sorted(chroma_excludable_allergen_vocabulary()))
    return _RAG_JSON_TEMPLATE.format(allowed=allowed)


def _strip_code_fence(raw: str) -> str:
    t = (raw or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.I)
        t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def _normalize_excluded_item(x: Any) -> str | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    vocab = chroma_excludable_allergen_vocabulary()
    if s in vocab:
        return s
    k = s.lower()
    if k in _ALLERGEN_ALIASES:
        mapped = _ALLERGEN_ALIASES[k]
        if mapped in vocab:
            return mapped
    for canon in vocab:
        if canon.lower() == k or k in canon.lower() or canon.lower() in k:
            return canon
    return None


def normalize_excluded_allergen_list(raw: list[Any] | None) -> list[str]:
    if not raw:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for x in raw:
        t = _normalize_excluded_item(x)
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _f_or_none(v: Any) -> float | None:
    if v is None or v is False:
        return None
    try:
        f = float(v)
        if f < 0:
            return None
        return f
    except (TypeError, ValueError):
        return None


class RagJsonSpec(TypedDict, total=False):
    compare_metrics: list[dict[str, str]]
    intent: str
    search_query: str
    excluded_allergens: list[str]
    excluded_lexical: list[str]
    max_kcal: float | None
    min_kcal: float | None


_COMPARE_FIELD_ALIASES: dict[str, str] = {
    "calories": "energy",
    "calorie": "energy",
    "kcal": "energy",
    "energy": "energy",
    "protein": "protein",
    "fat": "total_fat",
    "total_fat": "total_fat",
    "saturated_fat": "sat_fat",
    "sat_fat": "sat_fat",
    "trans_fat": "trans_fat",
    "cholesterol": "chol",
    "chol": "chol",
    "carbs": "carbs",
    "carbohydrates": "carbs",
    "sugar": "total_sugar",
    "total_sugar": "total_sugar",
    "added_sugar": "added_sugar",
    "sodium": "sodium",
}
_COMPARE_FIELDS_ALLOWED: frozenset[str] = frozenset(
    {
        "energy",
        "protein",
        "total_fat",
        "sat_fat",
        "trans_fat",
        "chol",
        "carbs",
        "total_sugar",
        "added_sugar",
        "sodium",
    }
)


def _normalize_compare_metrics(raw: Any) -> list[dict[str, str]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raw = [raw]
    out: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        field_raw = str(item.get("field") or item.get("metric") or "").strip().lower()
        goal_raw = str(item.get("goal") or item.get("order") or "").strip().lower()
        field = _COMPARE_FIELD_ALIASES.get(field_raw, field_raw)
        if field not in _COMPARE_FIELDS_ALLOWED:
            continue
        goal = "max" if goal_raw in {"max", "highest", "more", "high"} else "min"
        out.append({"field": field, "goal": goal})
    # Deduplicate keeping order.
    uniq: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for m in out:
        k = (m["field"], m["goal"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(m)
    return uniq[:4]


def parse_rag_json_response(raw: str) -> RagJsonSpec:
    t = _strip_code_fence(raw)
    data = json.loads(t)
    if not isinstance(data, dict):
        raise ValueError("not an object")
    intent = str(data.get("intent") or "lookup").strip().lower()
    if intent not in {"lookup", "alternatives", "details", "calorie_tune", "compare"}:
        intent = "lookup"
    sq = data.get("search_query")
    if sq is not None and not isinstance(sq, str):
        sq = str(sq)
    if not (sq and str(sq).strip()):
        raise ValueError("search_query required")
    ex = data.get("excluded_allergens")
    if ex is None and "allergies" in data:
        ex = data.get("allergies")
    if ex is not None and not isinstance(ex, list):
        ex = [ex]
    ex_list = normalize_excluded_allergen_list([*(ex or [])])
    ex_lex_raw = data.get("excluded_lexical")
    if ex_lex_raw is None:
        ex_lex_raw = data.get("excluded_menu_terms")
    if ex_lex_raw is not None and not isinstance(ex_lex_raw, list):
        ex_lex_raw = [ex_lex_raw]
    ex_lex = normalize_excluded_lexical_terms([*(ex_lex_raw or [])])
    max_k = _f_or_none(data.get("max_kcal", data.get("max_energy", data.get("energy"))))
    min_k = _f_or_none(data.get("min_kcal", data.get("min_energy")))
    compare_metrics = _normalize_compare_metrics(data.get("compare_metrics"))
    if max_k is not None and min_k is not None and max_k < min_k:
        min_k, max_k = max_k, min_k
    return {
        "compare_metrics": compare_metrics,
        "intent": intent,
        "search_query": " ".join(str(sq).split()),
        "excluded_allergens": ex_list,
        "excluded_lexical": ex_lex,
        "max_kcal": max_k,
        "min_kcal": min_k,
    }
