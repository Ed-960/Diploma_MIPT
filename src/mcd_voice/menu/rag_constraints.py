"""
Извлечение диет-ограничений из **свободного** текста реплики (English)
в токены, совпадающие с полем ``allergens`` в Chroma.

Дополняет RAG-фильтр ``$not_contains`` для реалистичного сценария (без
заранее сданного JSON-профиля) и **объединяется** с токенами из профиля.
Расширяемо: новые пары в ``_WORD_TO_CHROMA`` / ``_COMPOUND_FREE``.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

# Подстроки, как в Chroma ``metadata['allergens']`` (mcd)
_CHROMA_ALLERGEN_SUBSTR: frozenset[str] = frozenset(
    {
        "Milk",
        "Egg",
        "Fish",
        "Nuts",
        "Cereal containing gluten",
        "Soya",
        "Sulphites",
    }
)

_WORD_TO_CHROMA: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?i)\b(?:lactose|dairy|milk|cheese|creamy|cream|yogurt|butter|paneer)\b"), "Milk"),
    (re.compile(r"(?i)\b(?:peanuts?|tree nuts?|almonds?|hazelnuts?|walnuts?|cashews?|nuts?)\b"), "Nuts"),
    (re.compile(r"(?i)\b(?:gluten|wheat|barley|rye)\b"), "Cereal containing gluten"),
    (re.compile(r"(?i)\b(?:shell ?fish|shrimp|prawns?|crab|lobster|fish|salmon)\b"), "Fish"),
    (re.compile(r"(?i)\b(?:eggs?|omelette)\b"), "Egg"),
    (re.compile(r"(?i)\b(?:soya|soy)\b"), "Soya"),
    (re.compile(r"(?i)\bsulphites?\b"), "Sulphites"),
]

_COMPOUND_FREE: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?i)gluten[- ]free"), "Cereal containing gluten"),
    (re.compile(r"(?i)(?:dairy|lactose|milk)[- ]free"), "Milk"),
    (re.compile(r"(?i)nut[- ]free|peanut[- ]free"), "Nuts"),
    (re.compile(r"(?i)egg[- ]free"), "Egg"),
]

_NEG_IN_WINDOW = re.compile(
    r"(?is)(?:\bno\b|without|avoid|allergic|allerg(?:y|ies)|intolerant|can't have|cannot have|"
    r"don'?t (?:want|eat|get)|\bfree of\b|[- ]free\b|non-?dairy|lactose[- ]free|"
    r"gluten[- ]free|nut[- ]free|vegan|plant-based)"
)

_VEGAN_RE = re.compile(
    r"(?i)(?:\bvegan\b|\bplant[- ]based\b|nothing (?:of )?animal|fully plant)"
)


def _has_dietary_negation_context(s: str, start: int, end: int) -> bool:
    lo = max(0, start - 95)
    win = s[lo : end + 30]
    return _NEG_IN_WINDOW.search(win) is not None


def _looks_like_extra_topping(matched: str, pre: str) -> bool:
    w = (matched or "").lower()
    if w not in ("cheese", "cream", "butter", "milk"):
        return False
    tail = (pre or "")[-50:].lower()
    for cue in ("extra ", "add ", "more ", "double "):
        if cue in tail and "no" not in tail and "without" not in tail and "allerg" not in tail:
            return True
    return False


def extract_utterance_chroma_allergen_exclusions(text: str) -> list[str]:
    if not (text or "").strip():
        return []
    s = (text or "").strip()
    out: set[str] = set()

    if _VEGAN_RE.search(s):
        out.update(["Milk", "Egg", "Fish"])

    for pat, token in _COMPOUND_FREE:
        if pat.search(s) and token in _CHROMA_ALLERGEN_SUBSTR:
            out.add(token)

    for pat, token in _WORD_TO_CHROMA:
        if token not in _CHROMA_ALLERGEN_SUBSTR:
            continue
        for m in pat.finditer(s):
            if not _has_dietary_negation_context(s, m.start(), m.end()):
                continue
            if _looks_like_extra_topping(m.group(0), s[: m.start()]):
                continue
            out.add(token)

    return sorted(out)


def merge_rag_allergen_blacklist(
    profile_base: list[str],
    utterance_texts: Sequence[str],
) -> tuple[list[str], dict[str, Any]]:
    u: set[str] = set(profile_base)
    u_ex: set[str] = set()
    for t in utterance_texts:
        for x in extract_utterance_chroma_allergen_exclusions(t or ""):
            u_ex.add(x)
    u |= u_ex
    return (sorted(u), {"utterance_allergen_exclusions": sorted(u_ex)})
