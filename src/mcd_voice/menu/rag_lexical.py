"""
Универсальная лексическая блокировка для RAG: то, что не выражено токенами Chroma ``allergens``.

Мини-LLM возвращает список коротких фраз/слов; позиция отбрасывается, если любая из них
встречается в объединённом тексте name / description / ingredients / tag (границы слова
для однословных терминов).
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Sequence

from mcd_voice.menu.dataset import load_menu_from_json

_MAX_TERMS = 12
_MAX_TERM_LEN = 64
_TERM_OK = re.compile(
    r"^[a-z0-9][a-z0-9 \-]{0,62}[a-z0-9]$|^[a-z0-9]{2}$",
    flags=re.IGNORECASE,
)
# Однословные «шумы» — дают ложные вхождения в описаниях (unknown, nothing, …).
_SKIP_SINGLE_WORDS = frozenset(
    {
        "no",
        "not",
        "or",
        "and",
        "the",
        "a",
        "an",
        "any",
        "all",
        "with",
        "without",
    }
)


def normalize_excluded_lexical_terms(raw: Sequence[Any] | None) -> list[str]:
    if not raw:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for x in raw:
        s = re.sub(r"\s+", " ", str(x or "").strip())
        if not s:
            continue
        low = s.lower()
        if len(low) < 2 or len(low) > _MAX_TERM_LEN:
            continue
        if " " not in low and low in _SKIP_SINGLE_WORDS:
            continue
        if not _TERM_OK.match(low):
            continue
        if low in seen:
            continue
        seen.add(low)
        out.append(low)
        if len(out) >= _MAX_TERMS:
            break
    return out


def _row_blob_from_fields(row: dict[str, Any]) -> str:
    parts = [
        str(row.get("name") or ""),
        str(row.get("description") or ""),
        str(row.get("ingredients") or ""),
        str(row.get("tag") or ""),
    ]
    return " ".join(parts).lower()


@lru_cache(maxsize=1)
def _menu_name_blob_lower() -> dict[str, str]:
    _, _, metas = load_menu_from_json()
    d: dict[str, str] = {}
    for m in metas:
        name = str(m.get("name") or "").strip()
        if not name:
            continue
        parts = [
            name,
            str(m.get("description") or ""),
            str(m.get("ingredients") or ""),
            str(m.get("tag") or ""),
        ]
        d[name] = " ".join(parts).lower()
    return d


def row_exclusion_blob(row: dict[str, Any]) -> str:
    """Текст для проверки исключений (полные метаданные или fallback по имени из меню)."""
    ing = str(row.get("ingredients") or "").strip()
    des = str(row.get("description") or "").strip()
    if ing or des:
        return _row_blob_from_fields(row)
    name = str(row.get("name") or "").strip()
    return _menu_name_blob_lower().get(name, name.lower())


def _term_in_blob(term: str, blob: str) -> bool:
    t = (term or "").strip().lower()
    if not t or not blob:
        return False
    if " " in t:
        return t in blob
    return (
        re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", blob, flags=re.IGNORECASE)
        is not None
    )


def row_violates_excluded_lexical(row: dict[str, Any], terms: Sequence[str]) -> bool:
    if not terms:
        return False
    blob = row_exclusion_blob(row)
    return any(_term_in_blob(t, blob) for t in terms)


def filter_rows_by_excluded_lexical(
    rows: list[dict[str, Any]],
    terms: Sequence[str],
) -> list[dict[str, Any]]:
    if not terms:
        return list(rows)
    return [r for r in rows if not row_violates_excluded_lexical(r, terms)]


def chroma_fetch_n_for_lexical(top_k: int, total_docs: int, has_lexical: bool) -> int:
    if not has_lexical:
        return min(top_k, max(1, total_docs))
    return min(max(top_k * 10, top_k + 24), max(1, total_docs))
