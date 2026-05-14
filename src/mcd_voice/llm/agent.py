"""
Агенты ClientAgent и CashierAgent поверх OpenAI-compatible Chat Completions.

Конфигурация через переменные окружения:
  API_PROVIDER   — "openai" (по умолчанию) или "ollama".
  API_MODEL      — основная модель для диалога (приоритетнее DEFAULT_MODEL).
  REWRITE_MODEL  — маленькая/быстрая модель для query rewriting перед RAG.
                   Используется из .env как основной источник для mini-LLM.
                   Если не задана, используется API_MODEL.
  LLM_API_KEY    — ключ (любой OpenAI-compatible провайдер: xAI, OpenAI, Groq, …).
  LLM_BASE_URL   — base URL …/v1 (облако или локальный Ollama).
  Дубликаты не нужны: OPENAI_API_KEY, XAI_API_KEY, OPENAI_BASE_URL, XAI_BASE_URL, OLLAMA_URL
  поддерживаются только для старых .env и не рекомендуются к новым проектам.
  RAG_DISTANCE_THRESHOLD  — жёсткий порог уверенного попадания (косинусное расстояние, 0..∞).
  RAG_SOFT_DISTANCE_MAX   — мягкий порог: если лучший хит хуже жёсткого, но не хуже этого,
                             позиции всё равно подставляются в контекст с предупреждением.
  RAG_MAX_PROMPT_LINES    — максимум строк меню в system prompt после порога по distance
                             (по умолчанию 15; порядок = релевантность Chroma). 0 или отрицательное
                             — без ограничения (старое поведение).
  OPENROUTER_PROVIDER_IGNORE — только для LLM_BASE_URL с openrouter.ai: список slug через запятую
                             для provider.ignore (см. OpenRouter routing). Не задано → по умолчанию
                             Venice/venice (частые 403 «Venice.ai is at capacity» на free-маршрутах).
                             Значение none|false|0|off|- отключает передачу ignore.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from openai import APITimeoutError, OpenAI, OpenAIError

from mcd_voice.config import MCD_JSON_PATH
from mcd_voice.llm.prompts import get_cashier_system_prompt, get_client_system_prompt
from mcd_voice.menu.graph_rag import search_menu_graph
from mcd_voice.menu.rag_constraints import merge_rag_allergen_blacklist
from mcd_voice.menu.rag_structured import get_rag_json_system_prompt, parse_rag_json_response
from mcd_voice.menu.search import search_menu
from mcd_voice.profile.generator import get_group_allergen_blacklist
from mcd_voice.text_normalization import normalize_item_text as _normalize_item_text

# ── Константы ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "gpt-4o"
_RAG_CATALOG_TOP_K_FALLBACK = 42


def _load_menu_catalog_top_k() -> int:
    """Use the current mcd.json size for broad retrieval without touching Chroma."""
    try:
        rows = json.loads(MCD_JSON_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _RAG_CATALOG_TOP_K_FALLBACK
    if not isinstance(rows, list):
        return _RAG_CATALOG_TOP_K_FALLBACK
    return max(_RAG_CATALOG_TOP_K_FALLBACK, len(rows))


# Full-catalog retrieval: fetch at least as many rows as the current menu file.
# We intentionally retrieve broadly, then filter by distance/allergens.
RAG_CATALOG_TOP_K = _load_menu_catalog_top_k()
# Experimental narrow setup for A/B checks (not used by default).
RAG_EXPERIMENT_TOP_K = 5
# Backward-compatible alias used across scripts/tests.
RAG_FULL_TOP_K = RAG_CATALOG_TOP_K

RAG_DISTANCE_THRESHOLD: float = 0.60
try:
    RAG_SOFT_DISTANCE_MAX: float = float(
        os.environ.get("RAG_SOFT_DISTANCE_MAX") or "1.15"
    )
except (TypeError, ValueError):
    RAG_SOFT_DISTANCE_MAX = 1.15


def _parse_rag_max_prompt_lines() -> int | None:
    """Сколько уникальных позиций меню включать в промпт; None = без лимита."""
    raw = (os.environ.get("RAG_MAX_PROMPT_LINES") or "15").strip()
    try:
        n = int(raw)
    except ValueError:
        return 15
    if n <= 0:
        return None
    return n


RAG_MAX_PROMPT_LINES: int | None = _parse_rag_max_prompt_lines()
RAG_MODE_VECTOR = "vector"
RAG_MODE_GRAPH = "graph"


def resolve_rag_mode_from_env() -> str:
    """Читает ``RAG_MODE`` из окружения: ``vector`` (по умолчанию) или ``graph``."""
    raw = (os.environ.get("RAG_MODE") or RAG_MODE_VECTOR).strip().lower()
    if raw == RAG_MODE_GRAPH:
        return RAG_MODE_GRAPH
    return RAG_MODE_VECTOR


def merge_graph_retrieval_query(
    rewrite_query: str,
    client_utterance: str,
    secondary_queries: Sequence[str],
) -> str:
    """Склеивает сырой текст клиента, переписанный запрос и вторичные запросы для graph-RAG.

    Переписанный mini-LLM запрос (например «dairy-free burger options») часто теряет
    слова «burger» / «no milk» из реплики; без объединения lexical seeds уводят в
    нерелевантные узлы графа.
    """
    chunks: list[str] = []
    seen: set[str] = set()
    for part in (client_utterance, rewrite_query, *secondary_queries):
        p = str(part or "").strip()
        if not p:
            continue
        key = p.casefold()
        if key in seen:
            continue
        seen.add(key)
        chunks.append(p)
    return "\n".join(chunks)


HistoryEntry = dict[str, str]


def _use_lexical_exclusions() -> bool:
    """
    Post-filter by lexical terms is optional and disabled by default.
    Default vector-only behavior: rely on embeddings + metadata filters.
    """
    v = (os.environ.get("RAG_USE_LEXICAL_EXCLUDE") or "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def _use_turn_orchestrator() -> bool:
    v = (os.environ.get("LLM_TURN_ORCHESTRATOR") or "0").strip().lower()
    return v not in ("0", "false", "no", "off")

def _was_name_already_discussed(name: str, history: list[HistoryEntry]) -> bool:
    n = _normalize_item_text(name)
    if not n:
        return False
    for h in history[-24:]:
        t = _normalize_item_text(h.get("text") or "")
        if not t:
            continue
        if n in t:
            return True
    return False


def _rag_intent(spec: dict[str, Any] | None) -> str:
    raw = str((spec or {}).get("intent") or "lookup").strip().lower()
    if raw in {"lookup", "alternatives", "details", "calorie_tune", "compare"}:
        return raw
    return "lookup"


def _rag_json_list(spec: dict[str, Any] | None, key: str) -> list[str]:
    raw = (spec or {}).get(key) or []
    if not isinstance(raw, list):
        raw = [raw]
    out: list[str] = []
    for item in raw:
        s = str(item or "").strip()
        if s:
            out.append(s)
    return out


_RESTRICTION_OVERRIDE_RE = re.compile(
    r"\b("
    r"anyway|still want|i still want|i want it|i want that|that'?s fine|thats fine|"
    r"it'?s fine|its fine|that'?s okay|thats okay|it'?s okay|its okay|"
    r"just add|add it|i'?ll take it|ill take it|take it anyway|"
    r"yes,?\s*(please|that|it)|no,?\s*(i|we)"
    r")\b",
    re.I,
)

_WARNING_CUE_RE = re.compile(
    r"\b("
    r"contains?|has|not suitable|dietary|allergen|allergy|warning|"
    r"milk|dairy|lactose|gluten|egg|nuts?|fish|soya|soy|sulphites?|sulfites?"
    r")\b",
    re.I,
)

_RAG_DUMP_CUE_RE = re.compile(
    r"\b("
    r"bite-sized|a classic|crispy|juicy|breaded|topped with|served with|"
    r"comes with|delicious|refreshing|signature sauce|best fit is"
    r")\b",
    re.I,
)
_RAG_DUMP_ADD_RE = re.compile(
    r"\bwould you like to add (?:it|that) to your order\b",
    re.I,
)


def _cashier_warned_about_requested_item(
    history: list[HistoryEntry],
    requested_items: Sequence[str],
) -> bool:
    if not requested_items:
        return False
    for entry in reversed(history[-10:]):
        if entry.get("speaker") != "cashier":
            continue
        text = entry.get("text") or ""
        if not _WARNING_CUE_RE.search(text):
            continue
        if any(_matches_menu_name_in_text(item, text) for item in requested_items):
            return True
    return False


def _detect_restriction_override(
    client_text: str,
    history: list[HistoryEntry],
    requested_items: Sequence[str],
    *,
    explicit_override: bool = False,
) -> bool:
    warned = _cashier_warned_about_requested_item(history, requested_items)
    if explicit_override and warned:
        return True
    if not _RESTRICTION_OVERRIDE_RE.search(client_text or ""):
        return False
    return warned


_NON_FOOD_CLIENT_UTTERANCE_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(yes|yeah|yep|correct|right|okay|ok)\s*[.!?]*\s*$", re.I),
    re.compile(r"^\s*(okay|ok),?\s*(that'?s|thats)\s+all(?:,?\s*thanks?)?\s*[.!?]*\s*$", re.I),
    re.compile(r"^\s*(okay|ok|yes|yeah|yep),?\s*thanks?(?:\s+you)?\s*[.!?]*\s*$", re.I),
    re.compile(r"^\s*(yes|yeah|yep),?\s*(that'?s|thats)\s+right\s*[.!?]*\s*$", re.I),
    re.compile(r"^\s*(that'?s|thats)\s+(right|correct|fine)\s*[.!?]*\s*$", re.I),
    re.compile(r"^\s*(that'?s|thats)\s+all(?:,?\s*thanks?)?\s*[.!?]*\s*$", re.I),
    re.compile(r"^\s*all\s+set(?:,?\s*thanks?)?\s*[.!?]*\s*$", re.I),
    re.compile(r"^\s*thanks?(?:\s+you)?\s*[.!?]*\s*$", re.I),
)


_ORDER_INTENT_RE = re.compile(
    r"\b("
    r"order|add|take|want|need|get|have|choose|pick|recommend|suggest|"
    r"burger|sandwich|nuggets?|fries|salad|drink|coffee|tea|cola|coke|sprite|fanta|"
    r"meal|combo|side|dessert|kcal|calories?|nutrition|protein|fat|carbs|sugar|sodium"
    r")\b",
    re.I,
)

# Customer challenges *what the cashier said* — not a menu lookup (avoids bogus
# "details" + deterministic catalog dumps when RAG returns an unrelated first hit).
_SERVICE_META_QUESTION_RE = re.compile(
    r"\b("
    r"why\s+did\s+you\s+(say|ask|tell)|"
    r"why\s+do\s+you\s+ask|"
    r"why\s+you\s+ask(ed)?(\s+me)?|"
    r"so\s+why\s+(did\s+)?you\s+ask|"
    r"why\s+would\s+you\s+say|"
    r"what\s+do\s+you\s+mean\s+by"
    r")\b",
    re.I,
)

# Explicit ingredients / nutrition question — safe to use deterministic meal-details reply.
_ITEM_DETAILS_CUE_RE = re.compile(
    r"\b("
    r"what'?s\s+in(\s+the)?|what\s+is\s+in(\s+the)?|"
    r"ingredients?|"
    r"allergens?|nutrition(al)?|"
    r"calories?\s*(in|for)|how\s+many\s+calories|"
    r"\bkcal\b|"
    r"describe\s+(the|it|that)?|"
    r"tell\s+me\s+(more\s+)?about|"
    r"what'?s\s+inside|"
    r"made\s+(of|with)|"
    r"(is|does)\s+it\s+(contain|have)|"
    r"is\s+it\s+(spicy|vegan|vegetarian|gluten)"
    r")\b",
    re.I,
)

_ITEM_FOLLOWUP_CUE_RE = re.compile(
    r"\b("
    r"it|that|this|those|one|"
    r"contains?|include|has|have|with|without|"
    r"ingredients?|allergens?|made\s+(?:of|with)|"
    r"only\s+if|is\s+it|does\s+it"
    r")\b",
    re.I,
)
_NUTRITION_CUE_RE = re.compile(
    r"\b("
    r"nutrition(?:al)?|macros?|"
    r"protein|carbs?|carbohydrates?|fat|fats|saturated\s+fat|trans\s+fat|"
    r"sodium|cholesterol|sugar|added\s+sugar|total\s+sugar|"
    r"grams?\b|mg\b"
    r")\b",
    re.I,
)
_NUTRITION_FOLLOWUP_CUE_RE = re.compile(
    r"\b("
    r"like\s+(?:this|that|these|those)|"
    r"same|those|them|both(?:\s+of\s+them)?"
    r")\b",
    re.I,
)


def _is_non_food_client_utterance(text: str) -> bool:
    """True for short confirmation/closure phrases where RAG should be skipped."""
    t = (text or "").strip()
    if not t:
        return False
    # Generic order/menu/nutrition intent should keep RAG enabled.
    if _ORDER_INTENT_RE.search(t):
        return False
    if any(rx.match(t) for rx in _NON_FOOD_CLIENT_UTTERANCE_RES):
        return True
    # Support combined confirmations: "Yeah, that's right. That's all, thanks."
    parts = [p.strip() for p in re.split(r"[.!?]+", t) if p.strip()]
    if not parts:
        return False
    return all(any(rx.match(p) for rx in _NON_FOOD_CLIENT_UTTERANCE_RES) for p in parts)


def _extract_names_from_rag_context(rag_context: str) -> list[str]:
    """Extract menu names from '- Name (...)' lines in RAG context."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in (rag_context or "").splitlines():
        line = raw.strip()
        if not line.startswith("- "):
            continue
        name = line[2:].split("(", maxsplit=1)[0].strip()
        if not name:
            continue
        low = name.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(name)
    return out


def _matches_menu_name_in_text(name: str, text: str) -> bool:
    n = _normalize_item_text(name)
    t = _normalize_item_text(text)
    if not n or not t:
        return False
    return n in t


def _generic_name_tokens(rows: Sequence[dict[str, Any]]) -> set[str]:
    """Tokens shared by several menu item names are too broad for item grounding."""
    names_by_token: dict[str, set[str]] = {}
    for row in rows:
        name = _normalize_item_text(str(row.get("name") or ""))
        if not name:
            continue
        unique_tokens = {tok for tok in name.split() if len(tok) >= 4}
        for tok in unique_tokens:
            names_by_token.setdefault(tok, set()).add(name)
    return {tok for tok, names in names_by_token.items() if len(names) > 1}


def _row_matches_text(
    row: dict[str, Any],
    text: str,
    *,
    generic_tokens: set[str] | None = None,
) -> bool:
    """Match exact menu names plus distinctive name tokens like Oreo/Tikki."""
    t = _normalize_item_text(text)
    if not t:
        return False
    name = str(row.get("name") or "")
    if _matches_menu_name_in_text(name, text):
        return True
    generic = generic_tokens or set()
    tokens = [
        tok
        for tok in _normalize_item_text(name).split()
        if len(tok) >= 4 and tok not in generic
    ]
    return any(re.search(rf"\b{re.escape(tok)}\b", t) for tok in tokens)


def _looks_like_item_followup(text: str) -> bool:
    """True for pronoun/attribute follow-ups where recent named items matter."""
    return bool(_ITEM_FOLLOWUP_CUE_RE.search(text or ""))


def _append_unique_names(out: list[str], names: Sequence[str], *, max_names: int) -> None:
    seen = {_normalize_item_text(x) for x in out}
    for raw in names:
        name = str(raw or "").strip()
        key = _normalize_item_text(name)
        if not name or not key or key in seen:
            continue
        out.append(name)
        seen.add(key)
        if len(out) >= max_names:
            return


def _grounding_target_names(
    client_text: str,
    history: list[HistoryEntry] | None,
    rows: Sequence[dict[str, Any]],
    spec_requested_items: Sequence[str],
    *,
    max_names: int = 4,
) -> list[str]:
    """
    Menu rows that must be grounded even if vector distance is above the hard threshold.

    This prevents follow-up answers about already named items from relying on model memory
    when the item lost the current semantic ranking (e.g. "does it contain potato?").
    """
    generic_tokens = _generic_name_tokens(rows)
    out: list[str] = []
    _append_unique_names(out, spec_requested_items, max_names=max_names)

    direct = [
        str(r.get("name") or "")
        for r in rows
        if _row_matches_text(r, client_text, generic_tokens=generic_tokens)
    ]
    _append_unique_names(out, direct, max_names=max_names)

    if len(out) >= max_names or not _looks_like_item_followup(client_text):
        return out[:max_names]

    recent_names: list[str] = []
    for entry in reversed((history or [])[-8:]):
        text = entry.get("text") or ""
        for row in rows:
            name = str(row.get("name") or "")
            if name and _row_matches_text(row, text, generic_tokens=generic_tokens):
                recent_names.append(name)
    _append_unique_names(out, recent_names, max_names=max_names)
    return out[:max_names]


def _grounded_rows_for_names(
    rows: Sequence[dict[str, Any]],
    names: Sequence[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for target in names:
        target_norm = _normalize_item_text(target)
        if not target_norm:
            continue
        matches = [
            r for r in rows
            if _normalize_item_text(str(r.get("name") or "")) == target_norm
        ]
        if not matches:
            generic_tokens = _generic_name_tokens(rows)
            matches = [
                r for r in rows
                if _row_matches_text(r, target, generic_tokens=generic_tokens)
            ]
        if not matches:
            continue
        best = min(matches, key=lambda r: float(r.get("distance") or 9.0))
        name = str(best.get("name") or "").strip()
        key = _normalize_item_text(name)
        if not name or key in seen:
            continue
        out.append(best)
        seen.add(key)
    return out


def _render_grounded_rows(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return ""
    lines = [
        "Grounded menu data for named/follow-up items (use these fields; do not guess):",
    ]
    for row in rows:
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        ingredients = str(row.get("ingredients") or "").strip() or "not listed"
        description = str(row.get("description") or "").strip() or "not listed"
        allergens = ", ".join(str(a) for a in (row.get("allergens") or []) if str(a).strip())
        if not allergens:
            allergens = "none listed"
        lines.append(
            f"* {name}: ingredients: {ingredients}; allergens: {allergens}; description: {description}"
        )
    return "\n".join(lines) if len(lines) > 1 else ""


def _allergen_hits(
    allergens: Sequence[str] | None,
    blacklist: Sequence[str] | None,
) -> list[str]:
    if not allergens or not blacklist:
        return []
    bl = {str(x).strip().lower() for x in blacklist if str(x).strip()}
    if not bl:
        return []
    hits: list[str] = []
    for a in allergens:
        token = str(a).strip()
        if token and token.lower() in bl and token not in hits:
            hits.append(token)
    return hits


def _collect_allergen_excluded_candidates(
    *,
    query: str,
    shown_rows: Sequence[dict[str, Any]],
    blacklist: Sequence[str],
    top_k: int,
    max_energy: float | None,
    min_energy: float | None,
    excluded_lexical: Sequence[str] | None,
    mentioned_terms: Sequence[str] = (),
) -> list[dict[str, Any]]:
    if not blacklist:
        return []
    try:
        unfiltered_rows = search_menu(
            query,
            allergens_blacklist=None,
            top_k=max(top_k, RAG_CATALOG_TOP_K),
            max_energy=max_energy,
            min_energy=min_energy,
            excluded_lexical=excluded_lexical,
        )
    except Exception:
        return []

    shown_names = {str(r.get("name") or "") for r in shown_rows}
    mention_text = " ".join([query, *[str(x) for x in mentioned_terms]])
    by_name: dict[str, dict[str, Any]] = {}
    for row in unfiltered_rows:
        name = str(row.get("name") or "")
        if not name or name in shown_names:
            continue
        hits = _allergen_hits(row.get("allergens"), blacklist)
        if not hits:
            continue
        existing = by_name.get(name)
        if existing is None:
            by_name[name] = {
                "name": name,
                "distance": float(row.get("distance") or 0.0),
                "allergens": hits,
                "mentioned_in_query": _matches_menu_name_in_text(name, mention_text),
            }
            continue
        existing["distance"] = min(existing["distance"], float(row.get("distance") or 0.0))
        merged = list(existing.get("allergens") or [])
        for h in hits:
            if h not in merged:
                merged.append(h)
        existing["allergens"] = merged
        existing["mentioned_in_query"] = bool(existing["mentioned_in_query"]) or _matches_menu_name_in_text(name, mention_text)

    out = sorted(
        by_name.values(),
        key=lambda r: (
            0 if bool(r.get("mentioned_in_query")) else 1,
            float(r.get("distance") or 0.0),
            str(r.get("name") or ""),
        ),
    )
    return out[:3]


def _render_excluded_constraints_block(
    excluded_rows: Sequence[dict[str, Any]],
) -> str:
    if not excluded_rows:
        return ""
    lines = [
        "Items requiring dietary warning this turn (they exist on menu but may conflict with the customer's stated restrictions):",
    ]
    for row in excluded_rows:
        name = str(row.get("name") or "").strip()
        allergens = ", ".join(str(a) for a in (row.get("allergens") or []) if str(a).strip())
        if not name:
            continue
        if allergens:
            lines.append(f"* {name} (contains: {allergens})")
        else:
            lines.append(f"* {name}")
    return "\n".join(lines) if len(lines) > 1 else ""


def _normalize_rewrite_output(text: str) -> str:
    """
    Normalize mini-LLM rewrite output:
    - strip trailing punctuation,
    - collapse whitespace,
    - guard against meaningless one-word outputs like 'all'.
    """
    cleaned = (text or "").strip().replace(",", " ")
    # Remove label-like punctuation from mini-LLM outputs (e.g. "food type: burgers").
    cleaned = re.sub(r"[:|/\\]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"[.!?,;:]+$", "", cleaned).strip()
    if not cleaned:
        return "general menu items"
    low = cleaned.lower()
    if low in {
        "all", "ok", "okay", "yes", "yeah", "thanks", "thank you",
        "ok thanks", "okay thanks", "ok thank you", "okay thank you",
        "thats all", "that's all", "thats all thanks", "that's all thanks",
    }:
        return "general menu items"
    return cleaned


def _should_skip_rag(client_text: str, search_query: str) -> bool:
    """Decide whether current turn has no food intent and RAG should be skipped."""
    if _is_service_meta_question(client_text):
        return True
    if _is_non_food_client_utterance(client_text):
        return True
    q = (search_query or "").strip().lower()
    if not q:
        return True
    if q == "general menu items":
        return True
    courtesy_tokens = {
        "ok", "okay", "thanks", "thank", "you", "yes", "yeah", "yep",
        "right", "correct", "all", "thats", "that's",
    }
    q_tokens = [tok for tok in re.split(r"\s+", q) if tok]
    if q_tokens and all(tok in courtesy_tokens for tok in q_tokens):
        return True
    return False


_RECOMMENDATION_INTENT_RE = re.compile(
    r"\b(recommend|suggest|suggestion|what\s+should\s+i\s+(?:get|have|order)|"
    r"what\s+(?:do|would)\s+you\s+recommend|(?:some|any)thing\s+to\s+(?:eat|drink)|"
    r"sth\s+to\s+(?:eat|drink|it))\b",
    re.I,
)
_MENU_BROWSE_INTENT_RE = re.compile(
    r"\b("
    r"what\s+(?:kind|kinds|type|types)\s+of|what\s+do\s+you\s+have|"
    r"what(?:'s| is)\s+on\s+the\s+menu|show\s+me|list|options?|"
    r"anything\s+to\s+(?:eat|drink)|(?:some|any)thing\s+to\s+(?:eat|drink)|"
    r"that'?s\s+all\s+you\s+have|is\s+that\s+all|what\s+else|"
    r"anything\s+else|any\s+other|more\s+(?:options?|choices?)"
    r")\b",
    re.I,
)
_CONFIRM_ORDER_CUE_RE = re.compile(
    r"\b("
    r"yes|yeah|yep|"
    r"i(?:\s+would)?\s+like|"
    r"i'?ll\s+have|"
    r"i\s+want|"
    r"can\s+i\s+have|"
    r"add|"
    r"take"
    r")\b",
    re.I,
)
_CLAUSE_SPLIT_RE = re.compile(
    r"(?:[.!?;]+|\b(?:and then|and also|also|plus|and)\b)",
    re.I,
)
_FOOD_INTENT_RE = re.compile(
    r"\b(eat|food|meal|burger|sandwich|wrap|nuggets?|fries|salad|side|snack)\b",
    re.I,
)
_DRINK_INTENT_RE = re.compile(
    r"\b(drink|beverage|coffee|tea|coke|cola|sprite|water|juice|frappe|smoothie)\b",
    re.I,
)
_COFFEE_INTENT_RE = re.compile(r"\b(coffee|espresso|frappe|macchiato|mocha)\b", re.I)
_GENERIC_RECOMMENDATION_QUERIES = {
    "general menu items",
    "recommended menu items",
    "recommended menu options",
    "menu recommendations",
    "menu recommendation",
}


def _recent_client_texts(history: list[HistoryEntry], *, limit: int = 6) -> list[str]:
    out: list[str] = []
    for entry in reversed(history or []):
        if entry.get("speaker") != "client":
            continue
        text = str(entry.get("text") or "").strip()
        if text:
            out.append(text)
        if len(out) >= limit:
            break
    return list(reversed(out))


def _recommendation_search_query(
    client_text: str,
    history: list[HistoryEntry],
) -> str | None:
    """Turn vague browse/recommendation/follow-up requests into category-rich vector queries."""
    current = client_text or ""
    clauses = _split_intent_clauses(current)
    browse_clauses = [
        c for c in clauses
        if (_RECOMMENDATION_INTENT_RE.search(c) or _MENU_BROWSE_INTENT_RE.search(c))
    ]
    current_focus = " ".join(browse_clauses) if browse_clauses else current
    recent = " ".join(_recent_client_texts(history, limit=4))
    haystack = f"{recent} {current_focus}".strip()
    if not (
        _RECOMMENDATION_INTENT_RE.search(haystack)
        or _MENU_BROWSE_INTENT_RE.search(haystack)
    ):
        return None

    current_has_food = bool(_FOOD_INTENT_RE.search(current_focus))
    history_has_food = bool(_FOOD_INTENT_RE.search(haystack))
    has_drink = bool(_DRINK_INTENT_RE.search(haystack))
    has_coffee = bool(_COFFEE_INTENT_RE.search(haystack))

    if current_has_food or (history_has_food and has_coffee):
        return "main food items burgers chicken sandwiches wraps fries sides"
    if has_coffee and not current_has_food:
        return "coffee drinks hot cold espresso frappe"
    # Drink suggestions must not be replaced by a "mains" query just because earlier
    # turns mentioned burgers (haystack history_has_food); that made "suggest drinks"
    # retrieve burgers from the menu slice.
    if has_drink and not current_has_food:
        if not history_has_food or _DRINK_INTENT_RE.search(current_focus):
            return "drinks beverages coffee tea soda water juice"
    if history_has_food:
        return "main food items burgers chicken sandwiches wraps sides"
    return "popular menu items burgers chicken sandwiches sides drinks dessert"


def _is_menu_browse_request(
    client_text: str,
    history: list[HistoryEntry],
    rag_spec: dict[str, Any] | None,
) -> bool:
    current = client_text or ""
    if _MENU_BROWSE_INTENT_RE.search(current):
        return True
    if _RECOMMENDATION_INTENT_RE.search(current):
        return True
    # Do not drag prior "more options" intent into a fresh confirmation/order turn.
    if _CONFIRM_ORDER_CUE_RE.search(current):
        return False
    if _rag_intent(rag_spec) == "alternatives":
        return True
    return False


def _is_confirm_plus_browse_utterance(
    client_text: str,
    rag_spec: dict[str, Any] | None,
) -> bool:
    """True when one utterance confirms an item and asks a browse/recommend follow-up."""
    text = (client_text or "").strip()
    if not text:
        return False
    requested = _rag_json_list(rag_spec, "requested_items")
    if not requested:
        return False
    clauses = _split_intent_clauses(text)
    has_browse_clause = any(_is_menu_browse_request(c, [], rag_spec) for c in clauses)
    if not has_browse_clause:
        return False
    has_confirm_clause = False
    for c in clauses:
        if not _CONFIRM_ORDER_CUE_RE.search(c):
            continue
        if any(_matches_menu_name_in_text(item, c) for item in requested):
            has_confirm_clause = True
            break
    if not has_confirm_clause:
        # Fallback: model extracted requested_items and we have a global confirm cue.
        has_confirm_clause = bool(_CONFIRM_ORDER_CUE_RE.search(text))
    return has_confirm_clause


def _split_intent_clauses(text: str) -> list[str]:
    raw = [part.strip() for part in _CLAUSE_SPLIT_RE.split(text or "") if part.strip()]
    return raw or ([text.strip()] if (text or "").strip() else [])


def _derive_secondary_search_queries(
    client_text: str,
    history: list[HistoryEntry],
    primary_query: str,
    rag_spec: dict[str, Any] | None,
) -> list[str]:
    """Split compound utterances and derive one extra RAG query per clause."""
    clauses = _split_intent_clauses(client_text)
    if len(clauses) <= 1:
        return []
    out: list[str] = []
    seen: set[str] = {str(primary_query or "").strip().lower()}
    for clause in clauses:
        c = clause.strip()
        if not c:
            continue
        cq = _recommendation_search_query(c, history)
        if cq is None:
            cq = _normalize_rewrite_output(c)
            if _should_skip_rag(c, cq):
                continue
        low = cq.strip().lower()
        if not low or low in seen:
            continue
        seen.add(low)
        out.append(cq)
    return out


def _format_menu_names(names: Sequence[str], *, limit: int = 8) -> str:
    picked = [name for name in names if str(name).strip()][:limit]
    if not picked:
        return ""
    if len(picked) == 1:
        return picked[0]
    if len(picked) == 2:
        return f"{picked[0]} and {picked[1]}"
    return f"{', '.join(picked[:-1])}, and {picked[-1]}"


def _format_requested_item_ack(rag_spec: dict[str, Any] | None) -> str:
    requested = _rag_json_list(rag_spec, "requested_items")
    if not requested:
        return ""
    rendered = _format_menu_names(requested, limit=2)
    if not rendered:
        return ""
    return f"Got it, {rendered}. "


def _filter_menu_names_for_rag_spec(
    names: Sequence[str],
    rag_spec: dict[str, Any] | None,
) -> list[str]:
    """Apply lightweight deterministic post-filters implied by rag_spec."""
    out = [str(name).strip() for name in names if str(name).strip()]
    if not out:
        return []
    spec = rag_spec or {}
    search_query = _normalize_item_text(str(spec.get("search_query") or ""))
    excluded_raw = _rag_json_list(spec, "excluded_lexical")
    excluded_tokens = {
        tok
        for phrase in excluded_raw
        for tok in _normalize_item_text(phrase).split()
        if tok
    }

    # If user explicitly asked for burgers, keep burger-like options first.
    if re.search(r"\b(?:ham)?burgers?\b", search_query):
        burgers_only = [
            name
            for name in out
            if re.search(r"\b(?:ham)?burger\b", _normalize_item_text(name))
        ]
        if burgers_only:
            out = burgers_only

    if excluded_tokens:
        without_excluded = []
        for name in out:
            nn = _normalize_item_text(name)
            if any(re.search(rf"\b{re.escape(tok)}\b", nn) for tok in excluded_tokens):
                continue
            without_excluded.append(name)
        if without_excluded:
            out = without_excluded
    return out


def _is_service_meta_question(text: str) -> bool:
    """True when the customer is questioning the cashier's wording, not ordering."""
    t = (text or "").strip()
    if not t:
        return False
    if _SERVICE_META_QUESTION_RE.search(t):
        return True
    return False


def _wants_menu_item_details(text: str) -> bool:
    """True when the customer is asking for ingredients/nutrition, not a meta-remark."""
    t = (text or "").strip()
    if not t:
        return False
    if _is_service_meta_question(t):
        return False
    return bool(_ITEM_DETAILS_CUE_RE.search(t))


def _wants_full_nutrition_context(
    client_text: str,
    rag_spec: dict[str, Any] | None,
) -> bool:
    if any(isinstance(m, dict) for m in (rag_spec or {}).get("compare_metrics", [])):
        return True
    if _NUTRITION_CUE_RE.search(client_text or ""):
        return True
    restrictions = {str(x).strip().lower() for x in _rag_json_list(rag_spec, "restrictions")}
    if "sugar" in restrictions:
        return True
    return False


def _effective_rag_top_k(base_top_k: int, *, include_full_nutrition: bool) -> int:
    """Always search across the full menu catalog (no top-k slice)."""
    _ = include_full_nutrition
    return max(base_top_k, RAG_CATALOG_TOP_K)


@lru_cache(maxsize=1)
def _load_menu_rows_for_nutrition() -> tuple[dict[str, Any], ...]:
    try:
        rows = json.loads(MCD_JSON_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    if not isinstance(rows, list):
        return ()
    out: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            out.append(row)
    return tuple(out)


def _extract_numeric_range(text: str) -> tuple[float, float] | None:
    t = (text or "").lower()
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:-|to|–)\s*(\d+(?:\.\d+)?)\s*(?:g|gram|grams|mg|kcal)?",
        t,
    )
    if not m:
        return None
    lo = float(m.group(1))
    hi = float(m.group(2))
    if hi < lo:
        lo, hi = hi, lo
    return (lo, hi)


def _detect_requested_nutrient_field(text: str) -> str | None:
    t = (text or "").lower()
    pairs = [
        (r"\bprotein\b", "protein"),
        (r"\badded sugar\b", "added_sugar"),
        (r"\btotal sugar\b", "total_sugar"),
        (r"\bsugar\b", "total_sugar"),
        (r"\bcarbs?\b|\bcarbohydrates?\b", "carbs"),
        (r"\bsodium\b", "sodium"),
        (r"\bcholesterol\b|\bchol\b", "chol"),
        (r"\bsaturated fat\b|\bsat fat\b", "sat_fat"),
        (r"\btrans fat\b", "trans_fat"),
        (r"\bfats?\b", "total_fat"),
        (r"\bcalories?\b|\bkcal\b|\benergy\b", "energy"),
    ]
    for patt, field in pairs:
        if re.search(patt, t):
            return field
    return None


def _is_burger_like_name(name: str) -> bool:
    n = _normalize_item_text(name)
    return bool(re.search(r"\b(?:ham)?burger\b", n))


def _is_coffee_like_name(name: str) -> bool:
    n = _normalize_item_text(name)
    return bool(re.search(r"\bcoffee\b|\bespresso\b|\bfrappe\b", n))


def _scope_label_and_matcher(text: str) -> tuple[str, Any]:
    t = (text or "").lower()
    rules: list[tuple[str, re.Pattern[str], Any]] = [
        ("burgers", re.compile(r"\b(?:ham)?burgers?\b", re.I), _is_burger_like_name),
        ("coffee drinks", re.compile(r"\bcoffee|espresso|frappe|latte|macchiato|mocha\b", re.I), _is_coffee_like_name),
        ("fries", re.compile(r"\bfries|hash\s*browns?\b", re.I), lambda n: bool(re.search(r"\bfries\b|\bhash\s*browns?\b", _normalize_item_text(n)))),
        ("wraps", re.compile(r"\bwraps?\b", re.I), lambda n: bool(re.search(r"\bwraps?\b", _normalize_item_text(n)))),
        ("sandwiches", re.compile(r"\bsandwich(?:es)?\b", re.I), lambda n: bool(re.search(r"\bsandwich(?:es)?\b", _normalize_item_text(n)))),
        ("nuggets", re.compile(r"\bnuggets?\b", re.I), lambda n: bool(re.search(r"\bnuggets?\b", _normalize_item_text(n)))),
    ]
    for label, cue_re, matcher in rules:
        if cue_re.search(t):
            return label, matcher
    return "menu items", lambda _n: True


def _is_more_options_request(text: str) -> bool:
    return bool(
        re.search(
            r"\b(more|other|another|options?|choices?|alternatives?)\b",
            text or "",
            re.I,
        )
    )


def _recent_client_nutrition_constraint(
    history: Sequence[HistoryEntry] | None,
) -> tuple[str, tuple[float, float]] | None:
    for entry in reversed(history or []):
        if entry.get("speaker") != "client":
            continue
        prev_text = str(entry.get("text") or "").strip()
        if not prev_text:
            continue
        field = _detect_requested_nutrient_field(prev_text)
        value_range = _extract_numeric_range(prev_text)
        if field and value_range:
            return field, value_range
    return None


def _deterministic_full_catalog_nutrition_reply(
    client_text: str,
    history: Sequence[HistoryEntry] | None = None,
) -> str | None:
    text = (client_text or "").strip()
    if not text:
        return None
    t = text.lower()
    wants_more_options = _is_more_options_request(t)
    followup_like_previous = bool(_NUTRITION_FOLLOWUP_CUE_RE.search(text))
    has_nutrition_cue = bool(_NUTRITION_CUE_RE.search(text))
    # Support follow-ups like "more options like this?" by inheriting the last
    # explicit nutrition constraint from recent client turns.
    if not has_nutrition_cue and not (wants_more_options and followup_like_previous):
        return None
    rows = _load_menu_rows_for_nutrition()
    if not rows:
        return None

    parts: list[str] = []
    nutrient_field = _detect_requested_nutrient_field(text)
    value_range = _extract_numeric_range(text)
    if (not nutrient_field or not value_range) and wants_more_options and followup_like_previous:
        prev_constraint = _recent_client_nutrition_constraint(history)
        if prev_constraint:
            prev_field, prev_range = prev_constraint
            nutrient_field = nutrient_field or prev_field
            value_range = value_range or prev_range
    history_text = " ".join(
        str(entry.get("text") or "")
        for entry in (history or [])[-8:]
        if entry.get("speaker") == "client"
    ).lower()
    scope_label, scope_matcher = _scope_label_and_matcher(f"{t} {history_text}")

    if nutrient_field and value_range:
        lo, hi = value_range
        candidates: list[tuple[str, float]] = []
        for row in rows:
            name = str(row.get("name") or "").strip()
            if not name or not scope_matcher(name):
                continue
            try:
                val = float(row.get(nutrient_field) or 0.0)
            except (TypeError, ValueError):
                continue
            candidates.append((name, val))
        # If scope inference is too narrow, fallback to full catalog.
        if not candidates:
            for row in rows:
                name = str(row.get("name") or "").strip()
                if not name:
                    continue
                try:
                    val = float(row.get(nutrient_field) or 0.0)
                except (TypeError, ValueError):
                    continue
                candidates.append((name, val))
            scope_label = "menu items"
        if candidates:
            mid = (lo + hi) / 2.0
            in_range = [(n, v) for (n, v) in candidates if lo <= v <= hi]
            if in_range:
                in_range.sort(key=lambda nv: abs(nv[1] - mid))
                if wants_more_options:
                    options = [f"{name} (~{val:.2f} g)" for name, val in in_range[:5]]
                    rendered = _format_menu_names(options, limit=5)
                    parts.append(
                        f"Sure — options around {lo:.0f}-{hi:.0f} g {nutrient_field.replace('_', ' ')} "
                        f"for {scope_label} are {rendered}."
                    )
                else:
                    best_name, best_val = in_range[0]
                    parts.append(
                        f"Yes, {best_name} is about {best_val:.2f} g {nutrient_field.replace('_', ' ')}."
                    )
            else:
                nearest_name, nearest_val = min(candidates, key=lambda nv: abs(nv[1] - mid))
                parts.append(
                    f"I don't see {scope_label} in the {lo:.0f}-{hi:.0f} g range. "
                    f"The closest is {nearest_name} at about {nearest_val:.2f} g {nutrient_field.replace('_', ' ')}."
                )

    if "coffee" in t and re.search(r"\bno sugar\b|\bsugar[- ]?free\b|\bwithout sugar\b", t):
        coffees: list[tuple[str, float, float]] = []
        for row in rows:
            name = str(row.get("name") or "").strip()
            if not name or not _is_coffee_like_name(name):
                continue
            try:
                total_sugar = float(row.get("total_sugar") or 0.0)
                added_sugar = float(row.get("added_sugar") or 0.0)
            except (TypeError, ValueError):
                continue
            if total_sugar <= 0.5 and added_sugar <= 0.5:
                coffees.append((name, total_sugar, added_sugar))
        if coffees:
            unique_names: list[str] = []
            seen: set[str] = set()
            for name, _ts, _as in coffees:
                low = name.lower()
                if low in seen:
                    continue
                seen.add(low)
                unique_names.append(name)
            rendered = _format_menu_names(unique_names, limit=3)
            if rendered:
                parts.append(
                    f"For no-sugar coffee, {rendered} fit (about 0 g sugar)."
                )

    if not parts:
        return None
    if wants_more_options:
        parts.append("Which one would you like?")
    else:
        parts.append("Would you like me to add those to your order?")
    return " ".join(parts)


def _sanitize_cashier_response(text: str, *, allow_calories: bool = True) -> str:
    """Remove formatting/emoji artifacts forbidden by cashier prompt."""
    cleaned = (text or "").strip()
    if not cleaned:
        return cleaned
    # Markdown-ish emphasis markers.
    cleaned = cleaned.replace("**", "").replace("__", "")
    # Strip common emoji ranges while preserving plain ASCII punctuation/text.
    cleaned = re.sub(
        r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U0001F1E6-\U0001F1FF]",
        "",
        cleaned,
    )
    # Light touch: fix robotic screen-talk only (do not rewrite honest "we don't have"
    # — RAG "no matching items" turns need plain natural English).
    replacements = {
        "on my screen": "on the menu",
    }
    low = cleaned.lower()
    for src, dst in replacements.items():
        if src in low:
            cleaned = re.sub(re.escape(src), dst, cleaned, flags=re.I)
            low = cleaned.lower()
    cleaned = re.sub(r"\bfoodwould\b", "food would", cleaned, flags=re.I)

    # Remove RAG catalog dump fragments while keeping normal warning/confirm sentences.
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", cleaned) if p.strip()]
    if len(parts) > 1:
        filtered: list[str] = []
        for sent in parts:
            lower = sent.lower()
            if _RAG_DUMP_ADD_RE.search(lower):
                continue
            if ":" in sent and len(sent) > 70 and not _WARNING_CUE_RE.search(lower):
                continue
            if _RAG_DUMP_CUE_RE.search(lower) and not _WARNING_CUE_RE.search(lower):
                continue
            filtered.append(sent)
        if filtered:
            cleaned = " ".join(filtered)

    if not allow_calories:
        # Remove unsolicited calories from free-form cashier output.
        cleaned = re.sub(r"\(\s*~?\s*\d[\d.,]*(?:\s*[–-]\s*\d[\d.,]*)?\s*kcal\s*\)", "", cleaned, flags=re.I)
        cleaned = re.sub(r"~?\s*\d[\d.,]*(?:\s*[–-]\s*\d[\d.,]*)?\s*kcal", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# ── LLM-клиент ────────────────────────────────────────────────────────────────

def _normalize_base_url(url: str) -> str:
    base = url.strip().rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    return base


def _first_env(*keys: str) -> str:
    """Первое непустое значение среди переменных окружения."""
    for k in keys:
        v = (os.environ.get(k) or "").strip()
        if v:
            return v
    return ""


def _cloud_api_key() -> str:
    return _first_env("LLM_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY")


def _cloud_base_url() -> str:
    base = _first_env("LLM_BASE_URL", "OPENAI_BASE_URL", "XAI_BASE_URL")
    if base:
        return base
    # Старый сценарий: только XAI_API_KEY без LLM_/OPENAI_ — дефолтный хост xAI.
    if (
        _first_env("XAI_API_KEY")
        and not _first_env("LLM_API_KEY")
        and not _first_env("OPENAI_API_KEY")
    ):
        return "https://api.x.ai/v1"
    return ""


def _base_url_is_openrouter() -> bool:
    """True если облачный base URL указывает на OpenRouter (там есть provider routing)."""
    return "openrouter.ai" in (_cloud_base_url() or "").lower()


def _openrouter_provider_ignore_list() -> list[str] | None:
    """
    Список slug провайдеров OpenRouter для поля provider.ignore.

    ENV:
      OPENROUTER_PROVIDER_IGNORE — через запятую, например: venice,deepinfra
      Пусто (не задано) при LLM_BASE_URL=openrouter → по умолчанию ['venice'],
      чтобы реже ловить 403 «Venice.ai is at capacity» на бесплатных маршрутах.
      none|false|0|off|- — не передавать provider.ignore.
    """
    if not _base_url_is_openrouter():
        return None
    raw = (os.environ.get("OPENROUTER_PROVIDER_IGNORE") or "").strip()
    if raw.lower() in {"none", "false", "0", "off", "-"}:
        return None
    if not raw:
        # slug в метадатах OpenRouter встречается как Venice; в ignore обычно lowercase
        return ["Venice", "venice"]
    out: list[str] = []
    for part in raw.split(","):
        p = part.strip()
        if p and p not in out:
            out.append(p)
    return out or None


def _openrouter_extra_body() -> dict[str, Any] | None:
    """Доп. тело запроса для OpenRouter (Chat Completions)."""
    ign = _openrouter_provider_ignore_list()
    if not ign:
        return None
    return {"provider": {"ignore": ign}}


def _ollama_base_url() -> str:
    return _first_env("LLM_BASE_URL", "OLLAMA_URL", "OPENAI_BASE_URL")


def _resolve_model(explicit: str | None) -> str:
    """Явный аргумент > API_MODEL из env > DEFAULT_MODEL."""
    return explicit or os.environ.get("API_MODEL") or DEFAULT_MODEL


def _build_openai_client(timeout: float = 60.0) -> OpenAI:
    provider = (os.environ.get("API_PROVIDER") or "openai").strip().lower()

    if provider == "ollama":
        raw_url = _ollama_base_url()
        if not raw_url:
            raise RuntimeError(
                "Для API_PROVIDER=ollama задайте LLM_BASE_URL "
                "(например http://127.0.0.1:11434/v1)."
            )
        api_key = _first_env("LLM_API_KEY", "OPENAI_API_KEY") or "ollama"
        return OpenAI(
            api_key=api_key,
            base_url=_normalize_base_url(raw_url),
            timeout=timeout,
        )

    api_key = _cloud_api_key()
    if not api_key:
        raise RuntimeError(
            "Задайте LLM_API_KEY для облака, либо API_PROVIDER=ollama и LLM_BASE_URL. "
            "При необходимости поддерживаются устаревшие OPENAI_* / XAI_* / OLLAMA_URL в .env."
        )
    kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout}
    raw_base = _cloud_base_url()
    if raw_base:
        kwargs["base_url"] = _normalize_base_url(raw_base)
    return OpenAI(**kwargs)


def ensure_llm_credentials() -> None:
    """Проверяет env до параллельной генерации; иначе тот же RuntimeError, что при первом вызове LLM."""
    _build_openai_client()


def get_llm_runtime_config() -> dict[str, str]:
    """Текущая runtime-конфигурация LLM из env (для логов и отладки)."""
    provider = (os.environ.get("API_PROVIDER") or "openai").strip().lower()
    model = _resolve_model(None)
    if provider == "ollama":
        raw_url = _ollama_base_url()
    else:
        raw_url = _cloud_base_url()
    return {
        "provider": provider,
        "model": model,
        "base_url": _normalize_base_url(raw_url) if raw_url else "",
    }


# ── Низкоуровневые helpers ────────────────────────────────────────────────────

def _call_llm(
    client: OpenAI,
    model: str,
    system: str,
    messages: list[dict[str, str]],
    temperature: float = 0.8,
) -> str:
    """Один Chat Completions вызов; бросает RuntimeError при пустом ответе."""
    payload = [{"role": "system", "content": system}, *messages]
    timeout_err: APITimeoutError | None = None
    openai_err: OpenAIError | None = None
    for attempt in range(3):
        try:
            create_kwargs: dict[str, Any] = {
                "model": model,
                "messages": payload,
                "temperature": temperature,
            }
            xb = _openrouter_extra_body()
            if xb is not None:
                create_kwargs["extra_body"] = xb
            resp = client.chat.completions.create(**create_kwargs)
            choices = getattr(resp, "choices", None) or []
            if not choices:
                raise RuntimeError("Malformed LLM response: no choices returned.")
            first = choices[0]
            msg = getattr(first, "message", None)
            content = getattr(msg, "content", None)
            if isinstance(content, list):
                parts = [
                    str(getattr(part, "text", "") or "")
                    for part in content
                    if getattr(part, "text", None)
                ]
                content = " ".join(p for p in parts if p).strip()
            elif content is None:
                content = ""
            else:
                content = str(content).strip()
            if not content:
                raise RuntimeError("Malformed LLM response: empty content.")
            return content
        except APITimeoutError as exc:
            timeout_err = exc
            if attempt < 2:
                # brief backoff for transient local Ollama stalls
                time.sleep(0.7 * (attempt + 1))
                continue
            raise RuntimeError(f"Таймаут OpenAI API: {exc}") from exc
        except OpenAIError as exc:
            openai_err = exc
            # Free/cloud routes often emit transient 429/5xx; small retry helps
            # reduce hard dialog failures during batch dataset generation.
            err_text = str(exc).lower()
            is_retryable = (
                "429" in err_text
                or "rate limit" in err_text
                or "timeout" in err_text
                or "temporarily unavailable" in err_text
                or "503" in err_text
                or "502" in err_text
                or "500" in err_text
            )
            if is_retryable and attempt < 2:
                time.sleep(1.0 * (attempt + 1))
                continue
            raise RuntimeError(f"Ошибка OpenAI API: {exc}") from exc
    # unreachable, but keeps type checkers happy
    if timeout_err is not None:
        raise RuntimeError(f"Таймаут OpenAI API: {timeout_err}")
    raise RuntimeError(f"Ошибка OpenAI API: {openai_err}")


def _rag_json_user_message(
    rag_text: str,
    history: list[HistoryEntry] | None,
) -> str:
    """
    Mini-LLM user payload: optional recent dialog tail + current client line.

    Omits the last history entry when it is the same client line as rag_text
    to avoid duplicate text.
    """
    text = (rag_text or "").strip()
    if not text:
        return text
    entries = list(history or [])
    if entries and entries[-1].get("speaker") == "client":
        last_t = str(entries[-1].get("text") or "").strip()
        if last_t == text:
            entries = entries[:-1]
    if not entries:
        return text
    lines: list[str] = []
    for e in entries[-14:]:
        sp = str(e.get("speaker") or "")
        msg = str(e.get("text") or "").strip().replace("\n", " ")
        if not msg:
            continue
        label = "Customer" if sp == "client" else "Cashier"
        if len(msg) > 240:
            msg = msg[:237] + "..."
        lines.append(f"{label}: {msg}")
    if not lines:
        return text
    max_tail = 2800
    while len("\n".join(lines)) + len(text) > max_tail and len(lines) > 1:
        lines.pop(0)
    return (
        "Recent dialog (newest last; context only):\n"
        + "\n".join(lines)
        + "\n\nCurrent customer message:\n"
        + text
    )


def _rewrite_rag_structured_json(
    rag_text: str,
    client: OpenAI,
    model: str,
    *,
    history: list[HistoryEntry] | None = None,
    llm_trace: list[dict[str, Any]] | None = None,
    trace_meta: dict[str, Any] | None = None,
    trace_verbose: bool = False,
) -> tuple[dict[str, Any] | None, str | None]:
    """
    Один вызов mini-LLM: JSON с intent, search_query, excluded_lexical,
    max/min kcal.
    Успех: (spec, None). Иначе (None, error_message).
    """
    t0 = time.perf_counter()
    user_content = _rag_json_user_message(rag_text, history)
    try:
        raw = _call_llm(
            client,
            model,
            get_rag_json_system_prompt(),
            [{"role": "user", "content": user_content}],
            temperature=0.0,
        )
        spec = parse_rag_json_response(raw)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        _trace(
            llm_trace,
            {
                "event": "rag_json_rewrite",
                **(trace_meta or {}),
                "model": model,
                "kind": "rag_structured_json",
                "rag_json_duration_ms": round(dt_ms, 2),
                **(
                    {
                        "raw_response": raw,
                        "spec": spec,
                    }
                    if trace_verbose
                    else {
                        "raw_preview": _preview(raw),
                    }
                ),
            },
        )
        return (spec, None)
    except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        _trace(
            llm_trace,
            {
                "event": "rag_json_parse_error",
                **(trace_meta or {}),
                "model": model,
                "error": str(exc)[:200],
                "rag_json_duration_ms": round(dt_ms, 2),
            },
        )
        return (None, str(exc))
    except (RuntimeError, OSError) as exc:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        _trace(
            llm_trace,
            {
                "event": "rag_json_llm_error",
                **(trace_meta or {}),
                "model": model,
                "error": str(exc)[:200],
                "rag_json_duration_ms": round(dt_ms, 2),
            },
        )
        return (None, str(exc))


def _use_rag_json_rewrite() -> bool:
    v = (os.environ.get("RAG_JSON_REWRITE") or "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _history_to_messages(
    history: list[HistoryEntry],
    *,
    my_role: str,
) -> list[dict[str, str]]:
    """
    Преобразует внутреннюю историю в формат OpenAI messages.
    Реплики говорящего агента → assistant, остальные → user.
    """
    return [
        {
            "role": "assistant" if entry.get("speaker") == my_role else "user",
            "content": entry.get("text") or "",
        }
        for entry in history
        if entry.get("text") is not None
    ]


# ── ClientAgent ───────────────────────────────────────────────────────────────

class ClientAgent:
    """Агент-клиент: генерирует реплики на основе профиля и истории диалога."""

    def __init__(
        self,
        model: str | None = None,
        timeout: float = 60.0,
        *,
        trace_verbose: bool = False,
    ) -> None:
        self.model = _resolve_model(model)
        self._client = _build_openai_client(timeout)
        self._trace_verbose = trace_verbose

    def generate_response(
        self,
        profile: dict[str, Any],
        history: list[HistoryEntry],
        *,
        llm_trace: list[dict[str, Any]] | None = None,
    ) -> str:
        system = get_client_system_prompt(profile)
        if not history:
            seed = (
                "You just pulled up to the drive-through speaker. "
                "Say one short opening line as the customer."
            )
            messages: list[dict[str, str]] = [{"role": "user", "content": seed}]
        else:
            messages = _history_to_messages(history, my_role="client")
        try:
            t0 = time.perf_counter()
            response = _call_llm(self._client, self.model, system, messages)
            dt_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as exc:
            _trace(
                llm_trace,
                _llm_error_payload(
                    agent="client",
                    model=self.model,
                    messages=messages,
                    system=system,
                    error=str(exc),
                    verbose=self._trace_verbose,
                ),
            )
            raise
        _trace(
            llm_trace,
            _llm_call_payload(
                agent="client",
                model=self.model,
                system=system,
                messages=messages,
                response=response,
                duration_ms=dt_ms,
                verbose=self._trace_verbose,
            ),
        )
        return response


# ── CashierAgent ──────────────────────────────────────────────────────────────

class CashierAgent:
    """
    Агент-кассир: формирует ответы с учётом истории, заказа и RAG по меню.

    По умолчанию в системный промпт попадают психотип и состав группы из profile
    (удобно для симуляции). При realistic_cashier=True кассир видит только общие
    правила и реплики из диалога; фильтр аллергенов в RAG по профилю отключается.
    """

    def __init__(
        self,
        model: str | None = None,
        timeout: float = 60.0,
        rag_top_k: int = RAG_FULL_TOP_K,
        distance_threshold: float = RAG_DISTANCE_THRESHOLD,
        rewrite_model: str | None = None,
        rag_mode: str = RAG_MODE_VECTOR,
        *,
        trace_verbose: bool = False,
        realistic_cashier: bool = True,
        rag_max_prompt_lines: int | None = None,
        full_menu_context: bool = False,
    ) -> None:
        self.model = _resolve_model(model)
        # Always query broad vector slice by default; very small top_k leads
        # to artificial menu narrowing (e.g. only fries/coffee).
        if rag_top_k <= 0:
            self.rag_top_k = 0
        else:
            self.rag_top_k = max(int(rag_top_k), RAG_CATALOG_TOP_K)
        self.distance_threshold = distance_threshold
        self.rag_max_prompt_lines = (
            rag_max_prompt_lines
            if rag_max_prompt_lines is not None
            else RAG_MAX_PROMPT_LINES
        )
        mode = (rag_mode or RAG_MODE_VECTOR).strip().lower()
        if mode not in (RAG_MODE_VECTOR, RAG_MODE_GRAPH):
            raise ValueError(f"Unsupported rag_mode: {rag_mode!r}")
        self.rag_mode = mode
        self._client = _build_openai_client(timeout)
        self._trace_verbose = trace_verbose
        self._realistic_cashier = realistic_cashier
        self._full_menu_context = bool(full_menu_context)
        # Mini-LLM для query rewriting: используем REWRITE_MODEL из .env.
        # Параметр rewrite_model оставлен для обратной совместимости, но не применяется.
        _ = rewrite_model
        self._rewrite_model = os.environ.get("REWRITE_MODEL") or self.model

    # ── Публичный интерфейс ───────────────────────────────────────────────────

    def generate_response(
        self,
        profile: dict[str, Any],
        history: list[HistoryEntry],
        order_state: dict[str, Any],
        query: str | None = None,
        *,
        rag_trace: list[dict[str, Any]] | None = None,
        rag_meta: dict[str, Any] | None = None,
        llm_trace: list[dict[str, Any]] | None = None,
    ) -> str:
        if _use_turn_orchestrator():
            from mcd_voice.llm.turn_orchestrator import run_cashier_turn

            return run_cashier_turn(
                agent=self,
                profile=profile,
                history=history,
                order_state=order_state,
                query=query,
                rag_trace=rag_trace,
                rag_meta=rag_meta,
                llm_trace=llm_trace,
            )
        client_text = query or _last_client_text(history)
        rag_context, rag_spec = self._resolve_rag_context(
            client_text,
            profile,
            history,
            rag_trace=rag_trace,
            rag_meta=rag_meta,
            llm_trace=llm_trace,
        )
        intent = _rag_intent(rag_spec)
        allow_calories = (
            intent == "calorie_tune"
            or any(
                str(m.get("field")) == "energy"
                for m in (rag_spec or {}).get("compare_metrics", [])
                if isinstance(m, dict)
            )
            or (rag_spec or {}).get("max_kcal") is not None
            or (rag_spec or {}).get("min_kcal") is not None
        )
        allow_full_nutrition = _wants_full_nutrition_context(client_text, rag_spec)
        macro_reply = (
            self._deterministic_compare_reply(profile, client_text, rag_spec)
            if intent == "compare"
            else None
        )
        if macro_reply:
            _trace(
                llm_trace,
                {
                    "event": "deterministic_compare_reply",
                    **(rag_meta or {}),
                },
            )
            return macro_reply
        meal_details = (
            self._deterministic_meal_details_reply(client_text, rag_context)
            if intent == "details" and _wants_menu_item_details(client_text)
            else None
        )
        if meal_details:
            _trace(
                llm_trace,
                {
                    "event": "deterministic_meal_details_reply",
                    **(rag_meta or {}),
                },
            )
            return meal_details
        tune_reply = (
            self._deterministic_calorie_tuning_reply(profile, order_state, client_text)
            if intent == "calorie_tune" and not self._realistic_cashier
            else None
        )
        if tune_reply:
            _trace(
                llm_trace,
                {
                    "event": "deterministic_calorie_tuning_reply",
                    **(rag_meta or {}),
                    **(
                        {"target_kcal": profile.get("calApprValue")}
                        if not self._realistic_cashier
                        else {}
                    ),
                },
            )
            return tune_reply
        nutrition_reply = _deterministic_full_catalog_nutrition_reply(client_text, history)
        if nutrition_reply:
            _trace(
                llm_trace,
                {
                    "event": "deterministic_full_catalog_nutrition_reply",
                    **(rag_meta or {}),
                },
            )
            return nutrition_reply
        menu_browse_reply = self._deterministic_menu_browse_reply(
            client_text,
            history,
            rag_context,
            rag_spec,
        )
        if menu_browse_reply:
            _trace(
                llm_trace,
                {
                    "event": "deterministic_menu_browse_reply",
                    **(rag_meta or {}),
                },
            )
            return menu_browse_reply
        system = self._build_system(
            profile,
            order_state,
            rag_context,
            allow_calories=allow_calories,
            allow_full_nutrition=allow_full_nutrition,
            finalize_requested=bool((rag_spec or {}).get("finalize")),
            override_restriction=bool((rag_spec or {}).get("override_restriction")),
        )
        messages = _history_to_messages(history, my_role="cashier")
        try:
            t0 = time.perf_counter()
            response = _call_llm(self._client, self.model, system, messages)
            dt_ms = (time.perf_counter() - t0) * 1000.0
        except RuntimeError as exc:
            # Provider can occasionally return a malformed/empty payload.
            # Keep the voice session alive with a natural retry prompt.
            _trace(
                llm_trace,
                _llm_error_payload(
                    agent="cashier",
                    model=self.model,
                    messages=messages,
                    system=system,
                    error=str(exc),
                    verbose=self._trace_verbose,
                ),
            )
            return "Sorry, I didn't catch that clearly. Could you repeat your order?"
        except Exception as exc:
            _trace(
                llm_trace,
                _llm_error_payload(
                    agent="cashier",
                    model=self.model,
                    messages=messages,
                    system=system,
                    error=str(exc),
                    verbose=self._trace_verbose,
                ),
            )
            raise
        response = _sanitize_cashier_response(response, allow_calories=allow_calories)
        _trace(
            llm_trace,
            _llm_call_payload(
                agent="cashier",
                model=self.model,
                system=system,
                messages=messages,
                response=response,
                duration_ms=dt_ms,
                verbose=self._trace_verbose,
            ),
        )
        return response

    def _deterministic_menu_browse_reply(
        self,
        client_text: str,
        history: list[HistoryEntry],
        rag_context: str,
        rag_spec: dict[str, Any] | None,
    ) -> str | None:
        if not _is_menu_browse_request(client_text, history, rag_spec):
            return None
        names = _extract_names_from_rag_context(rag_context)
        names = _filter_menu_names_for_rag_spec(names, rag_spec)
        if not names:
            return None
        rendered = _format_menu_names(names, limit=8)
        if not rendered:
            return None
        ack = _format_requested_item_ack(rag_spec)
        if _RECOMMENDATION_INTENT_RE.search(client_text or ""):
            first = names[0]
            rest = _format_menu_names(names[1:], limit=4)
            if rest:
                return (
                    f"{ack}I'd suggest {first}. Other options are {rest}. "
                    "Which one would you like?"
                )
            return f"{ack}I'd suggest {first}. Would you like that?"
        return f"{ack}We have {rendered}. Which one would you like?"

    def _deterministic_meal_details_reply(self, client_text: str, rag_context: str) -> str | None:
        if not (client_text or "").strip():
            return None
        names = _extract_names_from_rag_context(rag_context)
        if not names:
            return None
        target = names[0]
        rows = search_menu(target, top_k=min(max(self.rag_top_k, 8), RAG_CATALOG_TOP_K))
        if not rows:
            return None
        norm_target = re.sub(r"[^a-z0-9]+", " ", target.lower()).strip()
        picked = None
        for r in rows:
            n = re.sub(r"[^a-z0-9]+", " ", str(r.get("name") or "").lower()).strip()
            if n == norm_target:
                picked = r
                break
        if picked is None:
            picked = rows[0]
        name = str(picked.get("name") or target).strip()
        ingredients = str(picked.get("ingredients") or "").strip()
        description = str(picked.get("description") or "").strip()
        allergens = picked.get("allergens") or []
        allergens_text = ", ".join(str(x) for x in allergens if str(x).strip()) or "none listed"
        kcal = float(picked.get("energy") or 0.0)
        parts: list[str] = []
        if description:
            parts.append(f"{name}: {description}")
        else:
            parts.append(f"{name}: this is one of our available menu items.")
        if ingredients:
            parts.append(f"Ingredients: {ingredients}.")
        parts.append(f"Allergens: {allergens_text}.")
        if kcal > 0:
            parts.append(f"Approx. {kcal:.0f} kcal.")
        parts.append("Would you like to add it to your order?")
        return " ".join(parts)

    def _deterministic_calorie_tuning_reply(
        self,
        profile: dict[str, Any],
        order_state: dict[str, Any],
        client_text: str,
    ) -> str | None:
        if not (client_text or "").strip():
            return None
        try:
            target = float(profile.get("calApprValue") or 0.0)
        except (TypeError, ValueError):
            return None
        if target <= 0:
            return None
        persons = order_state.get("persons", []) or []
        current = float(sum(float(p.get("total_energy") or 0.0) for p in persons))
        delta = round(target - current, 1)
        if abs(delta) <= 40:
            return (
                f"You are already very close to your target: about {current:.0f} kcal "
                f"vs target {target:.0f} kcal. We can keep the order as-is."
            )
        base: list[str] = [] if self._realistic_cashier else get_group_allergen_blacklist(profile)
        blacklist, _ = merge_rag_allergen_blacklist(base, [client_text])
        rows = search_menu(
            "light sides drinks snacks simple add-on options",
            allergens_blacklist=blacklist or None,
            top_k=max(self.rag_top_k, RAG_CATALOG_TOP_K),
        )
        if not rows:
            return None
        current_items = {
            str(it.get("name", "")).strip()
            for p in persons
            for it in (p.get("items", []) or [])
            if str(it.get("name", "")).strip()
        }
        dedup: dict[str, dict[str, Any]] = {}
        for r in rows:
            name = str(r.get("name") or "").strip()
            if not name:
                continue
            if name in dedup and float(dedup[name].get("distance", 9.0)) <= float(r.get("distance", 9.0)):
                continue
            dedup[name] = r
        candidates = [
            r for r in dedup.values()
            if float(r.get("energy") or 0.0) > 0.0 and str(r.get("name") or "").strip() not in current_items
        ]
        if not candidates:
            candidates = [r for r in dedup.values() if float(r.get("energy") or 0.0) > 0.0]
        if not candidates:
            return None
        if delta > 0:
            best = min(candidates, key=lambda r: abs(float(r.get("energy") or 0.0) - delta))
            add_kcal = float(best.get("energy") or 0.0)
            return (
                f"To stay near your target with minimal change, add 1 {best['name']} "
                f"(~{add_kcal:.0f} kcal). This moves the order from ~{current:.0f} to "
                f"~{current + add_kcal:.0f} kcal."
            )
        reduce = abs(delta)
        best_low = min(candidates, key=lambda r: abs(float(r.get("energy") or 0.0) - reduce))
        low_kcal = float(best_low.get("energy") or 0.0)
        return (
            f"You are above target by about {reduce:.0f} kcal. Minimal tweak: replace one higher-calorie "
            f"item with {best_low['name']} (~{low_kcal:.0f} kcal) to bring total closer to "
            f"{target:.0f} kcal."
        )

    def _deterministic_compare_reply(
        self,
        profile: dict[str, Any],
        client_text: str,
        rag_spec: dict[str, Any] | None,
    ) -> str | None:
        metrics = [
            m for m in (rag_spec or {}).get("compare_metrics", [])
            if isinstance(m, dict) and m.get("field") and m.get("goal")
        ]
        if not metrics:
            return None
        search_query = str((rag_spec or {}).get("search_query") or "").strip()
        if not search_query:
            return None
        base: list[str] = [] if self._realistic_cashier else get_group_allergen_blacklist(profile)
        blacklist, _ = merge_rag_allergen_blacklist(base, [client_text])
        rows = search_menu(
            search_query,
            allergens_blacklist=blacklist or None,
            max_energy=float((rag_spec or {}).get("max_kcal"))
            if (rag_spec or {}).get("max_kcal") is not None else None,
            min_energy=float((rag_spec or {}).get("min_kcal"))
            if (rag_spec or {}).get("min_kcal") is not None else None,
            top_k=max(self.rag_top_k, RAG_CATALOG_TOP_K),
        )
        if not rows:
            return None
        uniq: dict[str, dict[str, Any]] = {}
        for r in rows:
            name = str(r.get("name") or "").strip()
            if not name:
                continue
            cur = uniq.get(name)
            if cur is None or float(r.get("distance") or 9.0) < float(cur.get("distance") or 9.0):
                uniq[name] = r
        cands = list(uniq.values())
        if not cands:
            return None
        def _metric_val(row: dict[str, Any], field: str) -> float:
            try:
                return float(row.get(field) or 0.0)
            except (TypeError, ValueError):
                return 0.0

        def _sort_key(row: dict[str, Any]) -> tuple[float, ...]:
            vals: list[float] = []
            for m in metrics:
                fld = str(m.get("field"))
                goal = str(m.get("goal"))
                val = _metric_val(row, fld)
                vals.append(-val if goal == "max" else val)
            vals.append(float(row.get("distance") or 9.0))
            return tuple(vals)

        cands.sort(key=_sort_key)
        best = cands[0]
        alt = cands[1] if len(cands) > 1 else None
        field_units = {
            "energy": "kcal",
            "protein": "g",
            "total_fat": "g",
            "sat_fat": "g",
            "trans_fat": "g",
            "chol": "mg",
            "carbs": "g",
            "total_sugar": "g",
            "added_sugar": "g",
            "sodium": "mg",
        }
        field_labels = {
            "energy": "energy",
            "protein": "protein",
            "total_fat": "fat",
            "sat_fat": "saturated fat",
            "trans_fat": "trans fat",
            "chol": "cholesterol",
            "carbs": "carbs",
            "total_sugar": "total sugar",
            "added_sugar": "added sugar",
            "sodium": "sodium",
        }

        def _metrics_text(row: dict[str, Any]) -> str:
            parts: list[str] = []
            for m in metrics:
                fld = str(m.get("field"))
                v = _metric_val(row, fld)
                parts.append(f"{field_labels.get(fld, fld)} ~{v:.0f} {field_units.get(fld, '')}".strip())
            return ", ".join(parts)

        msg = f"Best fit is {best['name']}: {_metrics_text(best)}."
        if alt:
            msg += f" Another option is {alt['name']}: {_metrics_text(alt)}."
        msg += " Would you like to go with the first one?"
        return msg

    # ── RAG ───────────────────────────────────────────────────────────────────

    def _resolve_rag_context(
        self,
        client_text: str,
        profile: dict[str, Any],
        history: list[HistoryEntry],
        *,
        rag_trace: list[dict[str, Any]] | None,
        rag_meta: dict[str, Any] | None,
        llm_trace: list[dict[str, Any]] | None,
    ) -> tuple[str, dict[str, Any] | None]:
        """
        Запускает семантический поиск по меню и возвращает контекст для промпта.

        Нет keyword-фильтрации: запрос всегда идёт в Chroma.
        Если семантическое расстояние слишком большое — контекст пустой
        (модель не засоряется нерелевантным меню).
        Если клиент ещё не говорил (приветствие) — контекст меню не подставляется.
        """
        base_trace = {**(rag_meta or {}), "client_query": client_text[:800]}
        rag_text = (client_text or "").strip()

        if not rag_text:
            _trace(
                rag_trace,
                {
                    **base_trace,
                    "event": "rag_skipped_no_client_query",
                    "retrieval_mode": "none" if self._full_menu_context else self.rag_mode,
                },
            )
            return ("", None)

        if self._full_menu_context:
            context = _full_mcd_json_context()
            _trace(
                rag_trace,
                {
                    **base_trace,
                    "event": "full_menu_context",
                    "retrieval_mode": "none",
                    "source": str(MCD_JSON_PATH),
                    "menu_chars": len(context),
                    "context_preview": _preview(context),
                },
            )
            return (context, None)

        if self.rag_top_k <= 0:
            _trace(
                rag_trace,
                {
                    **base_trace,
                    "event": "rag_disabled",
                    "rag_top_k": 0,
                    "retrieval_mode": self.rag_mode,
                },
            )
            return ("", None)

        rag_json_spec: dict[str, Any] | None = None
        if _use_rag_json_rewrite():
            jspec, _jerr = _rewrite_rag_structured_json(
                rag_text,
                self._client,
                self._rewrite_model,
                history=history,
                llm_trace=llm_trace,
                trace_meta=rag_meta,
                trace_verbose=self._trace_verbose,
            )
            if jspec is not None:
                rag_json_spec = dict(jspec)
                requested = _rag_json_list(rag_json_spec, "requested_items")
                rag_json_spec["override_restriction"] = _detect_restriction_override(
                    client_text,
                    history,
                    requested,
                    explicit_override=bool(rag_json_spec.get("override_restriction")),
                )
        if rag_json_spec is not None:
            search_query = str(rag_json_spec.get("search_query", "")).strip()
            if not search_query:
                rag_json_spec = None
        if rag_json_spec is None:
            search_query = _normalize_rewrite_output(rag_text)
        rec_query = _recommendation_search_query(client_text, history)
        if rec_query and (
            search_query.strip().lower() in _GENERIC_RECOMMENDATION_QUERIES
            or _is_menu_browse_request(client_text, [], rag_json_spec)
        ):
            search_query = rec_query
            if rag_json_spec is not None:
                rag_json_spec["search_query"] = search_query
        is_fallback = False
        if _should_skip_rag(client_text, search_query):
            _trace(
                rag_trace,
                {
                    **base_trace,
                    "event": "rag_skipped_non_food",
                    "search_query": search_query,
                    "rewrite_model": self._rewrite_model,
                    "retrieval_mode": self.rag_mode,
                },
            )
            return ("", rag_json_spec)

        retrieval_spec = dict(rag_json_spec) if isinstance(rag_json_spec, dict) else None
        retrieval_texts: list[str]
        if client_text and not is_fallback:
            retrieval_texts = [
                str(rag_text),
                str(search_query),
                *_recent_client_texts(history),
            ]
        else:
            retrieval_texts = [str(search_query)]
        ctexts = tuple(dict.fromkeys(retrieval_texts))
        rj = locals().get("rag_json_spec")
        prefer_novel = _rag_intent(rj) == "alternatives"
        include_full_nutrition = _wants_full_nutrition_context(
            client_text,
            retrieval_spec,
        )
        context, info = self._do_rag(
            search_query,
            profile,
            client_utterance=rag_text,
            search_queries=_derive_secondary_search_queries(
                client_text,
                history,
                search_query,
                retrieval_spec if client_text and not is_fallback else None,
            ),
            rag_constraint_texts=ctexts,
            rag_json_spec=retrieval_spec if client_text and not is_fallback else None,
            history=history,
            prefer_novel=prefer_novel,
            include_full_nutrition=include_full_nutrition,
            rag_trace=rag_trace,
        )
        _trace(rag_trace, {
            **base_trace,
            "event": "rag",
            "search_query": search_query,
            "rewrite_model": self._rewrite_model,
            **({"fallback": True} if is_fallback else {}),
            "retrieval_mode": self.rag_mode,
            "full_nutrition_context": bool(include_full_nutrition),
            **(
                {
                    "rag_json_spec": {
                        "intent": rj.get("intent"),
                        "compare_metrics": rj.get("compare_metrics", []),
                        "excluded_lexical": rj.get("excluded_lexical", []),
                        "max_kcal": rj.get("max_kcal"),
                        "min_kcal": rj.get("min_kcal"),
                        "restrictions": rj.get("restrictions", []),
                        "requested_items": rj.get("requested_items", []),
                        "override_restriction": bool(rj.get("override_restriction")),
                        "finalize": bool(rj.get("finalize")),
                    }
                }
                if rj
                else {}
            ),
            **info,
            "context_preview": _preview(context),
        })
        return (context, rj if isinstance(rj, dict) else None)

    def _do_rag(
        self,
        text: str,
        profile: dict[str, Any],
        *,
        client_utterance: str = "",
        search_queries: Sequence[str] = (),
        rag_constraint_texts: Sequence[str] = (),
        rag_json_spec: dict[str, Any] | None = None,
        history: list[HistoryEntry] | None = None,
        prefer_novel: bool = False,
        include_full_nutrition: bool = False,
        rag_trace: list[dict[str, Any]] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Поиск по меню с фильтрацией по аллергенам группы.
        Возвращает (текст для system prompt, метаданные для rag_trace).
        """
        if self.rag_mode == RAG_MODE_GRAPH:
            return self._do_graph_rag(
                text,
                profile,
                client_utterance=client_utterance,
                search_queries=search_queries,
                rag_constraint_texts=rag_constraint_texts,
                rag_json_spec=rag_json_spec,
                include_full_nutrition=include_full_nutrition,
            )

        base: list[str] = (
            []
            if self._realistic_cashier
            else get_group_allergen_blacklist(profile)
        )
        spec_restrictions = _rag_json_list(rag_json_spec, "restrictions")
        spec_requested_items = _rag_json_list(rag_json_spec, "requested_items")
        override_restriction = bool((rag_json_spec or {}).get("override_restriction"))
        blacklist, cmeta = merge_rag_allergen_blacklist(
            base,
            rag_constraint_texts,
            explicit_restrictions=spec_restrictions,
        )
        search_blacklist = [] if override_restriction else blacklist
        if rag_json_spec is not None:
            cmeta = {
                **cmeta,
                "rag_json_excluded_ignored": list(
                    rag_json_spec.get("excluded_allergens", []) or []
                ),
                "rag_json_lexical": list(rag_json_spec.get("excluded_lexical", []) or []),
                "rag_json_restrictions": spec_restrictions,
                "rag_json_requested_items": spec_requested_items,
                "rag_json_finalize": bool(rag_json_spec.get("finalize")),
                "rag_json_override_restriction": override_restriction,
            }
        if override_restriction and blacklist:
            cmeta = {
                **cmeta,
                "allergen_blacklist_suppressed_for_override": list(blacklist),
            }
        max_e = rag_json_spec.get("max_kcal") if rag_json_spec else None
        min_e = rag_json_spec.get("min_kcal") if rag_json_spec else None
        lex_e = (
            list(rag_json_spec.get("excluded_lexical") or [])
            if (rag_json_spec and _use_lexical_exclusions())
            else []
        )
        effective_top_k = _effective_rag_top_k(
            self.rag_top_k,
            include_full_nutrition=include_full_nutrition,
        )
        query_batch: list[str] = []
        for q in [text, *search_queries]:
            qq = str(q or "").strip()
            if not qq:
                continue
            if qq not in query_batch:
                query_batch.append(qq)
        if not query_batch:
            query_batch = [str(text or "").strip() or "general menu items"]
        chroma_buf: list[dict[str, Any]] = []
        rows_all: list[dict[str, Any]] = []
        for i, q in enumerate(query_batch):
            rows_i = search_menu(
                q,
                allergens_blacklist=search_blacklist or None,
                top_k=effective_top_k,
                max_energy=float(max_e) if max_e is not None else None,
                min_energy=float(min_e) if min_e is not None else None,
                excluded_lexical=lex_e or None,
                chroma_trace=chroma_buf if (rag_trace is not None and i == 0) else None,
            )
            rows_all.extend(rows_i)
        rows_all.sort(key=lambda r: float(r.get("distance") or 9.0))
        dedup_rows: list[dict[str, Any]] = []
        seen_row_keys: set[tuple[str, str, float, str]] = set()
        for r in rows_all:
            key = (
                str(r.get("name") or ""),
                str(r.get("serving_size") or ""),
                float(r.get("energy") or 0.0),
                str(r.get("ingredients") or ""),
            )
            if key in seen_row_keys:
                continue
            seen_row_keys.add(key)
            dedup_rows.append(r)
        rows = dedup_rows
        excluded_by_constraints = (
            []
            if override_restriction
            else _collect_allergen_excluded_candidates(
                query=query_batch[0],
                shown_rows=rows,
                blacklist=blacklist,
                top_k=effective_top_k,
                max_energy=float(max_e) if max_e is not None else None,
                min_energy=float(min_e) if min_e is not None else None,
                excluded_lexical=lex_e or None,
                mentioned_terms=[*rag_constraint_texts, *spec_requested_items],
            )
        )
        excluded_block = _render_excluded_constraints_block(excluded_by_constraints)
        grounding_targets = _grounding_target_names(
            " ".join(str(x) for x in rag_constraint_texts),
            history,
            rows,
            spec_requested_items,
        )
        grounded_rows = _grounded_rows_for_names(rows, grounding_targets)
        grounded_block = _render_grounded_rows(grounded_rows)
        if prefer_novel and rows:
            seen: list[dict[str, Any]] = []
            fresh: list[dict[str, Any]] = []
            for r in rows:
                nm = str(r.get("name") or "")
                if _was_name_already_discussed(nm, history or []):
                    seen.append(r)
                else:
                    fresh.append(r)
            if fresh:
                rows = [*fresh, *seen]
        for ev in chroma_buf:
            _trace(rag_trace, ev)
        soft_max = RAG_SOFT_DISTANCE_MAX
        info: dict[str, Any] = {
            "retrieval_mode": RAG_MODE_VECTOR,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "metric": "cosine_distance",
            "top_k": effective_top_k,
            "distance_threshold": self.distance_threshold,
            "soft_distance_max": soft_max,
            "allergen_blacklist_tokens": list(search_blacklist),
            "prefer_novel": bool(prefer_novel),
            "query_batch": list(query_batch),
            **cmeta,
            "candidates": [
                {"name": r["name"], "distance": r["distance"], "energy": r["energy"]}
                for r in rows
            ],
            "excluded_by_constraints": [
                {
                    "name": str(r.get("name") or ""),
                    "distance": float(r.get("distance") or 0.0),
                    "allergens": list(r.get("allergens") or []),
                    "mentioned_in_query": bool(r.get("mentioned_in_query")),
                }
                for r in excluded_by_constraints
            ],
            "grounding_targets": list(grounding_targets),
            "grounded_rows": [
                {
                    "name": str(r.get("name") or ""),
                    "distance": float(r.get("distance") or 0.0),
                }
                for r in grounded_rows
            ],
        }

        if not rows:
            info["outcome"] = "no_chroma_hits"
            base_context = "(no matching menu items found for this request)"
            if excluded_block:
                return base_context + "\n\n" + excluded_block, info
            return base_context, info

        best = rows[0]["distance"]
        info["best_distance"] = best

        if best <= self.distance_threshold:
            lines, used = _render_rows(
                rows,
                max_dist=self.distance_threshold,
                max_lines=self.rag_max_prompt_lines,
                include_full_nutrition=include_full_nutrition,
            )
            if self.rag_max_prompt_lines is not None:
                info["rag_prompt_line_cap"] = self.rag_max_prompt_lines
                info["rag_prompt_lines_included"] = len(lines)
            info.update(outcome="injected", injected_hits=used)
            context = "\n".join(lines)
            if excluded_block:
                context += "\n\n" + excluded_block
            if grounded_block:
                context += "\n\n" + grounded_block
            return context, info

        if best <= soft_max:
            lines, used = _render_rows(
                rows,
                max_dist=soft_max,
                max_lines=self.rag_max_prompt_lines,
                include_full_nutrition=include_full_nutrition,
            )
            if lines:
                if self.rag_max_prompt_lines is not None:
                    info["rag_prompt_line_cap"] = self.rag_max_prompt_lines
                    info["rag_prompt_lines_included"] = len(lines)
                info.update(outcome="injected_soft", injected_hits=used)
                context = _SOFT_PREFIX + "\n".join(lines)
                if excluded_block:
                    context += "\n\n" + excluded_block
                if grounded_block:
                    context += "\n\n" + grounded_block
                return context, info

        info["outcome"] = "above_threshold"
        base_context = "(no matching menu items found — closest match too distant)"
        if excluded_block:
            return base_context + "\n\n" + excluded_block, info
        if grounded_block:
            return base_context + "\n\n" + grounded_block, info
        return base_context, info

    def _do_graph_rag(
        self,
        text: str,
        profile: dict[str, Any],
        *,
        client_utterance: str = "",
        search_queries: Sequence[str] = (),
        rag_constraint_texts: Sequence[str] = (),
        rag_json_spec: dict[str, Any] | None = None,
        include_full_nutrition: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        base: list[str] = (
            []
            if self._realistic_cashier
            else get_group_allergen_blacklist(profile)
        )
        spec_restrictions = _rag_json_list(rag_json_spec, "restrictions")
        spec_requested_items = _rag_json_list(rag_json_spec, "requested_items")
        override_restriction = bool((rag_json_spec or {}).get("override_restriction"))
        blacklist, cmeta = merge_rag_allergen_blacklist(
            base,
            rag_constraint_texts,
            explicit_restrictions=spec_restrictions,
        )
        search_blacklist = [] if override_restriction else blacklist
        if rag_json_spec is not None:
            cmeta = {
                **cmeta,
                "rag_json_excluded_ignored": list(
                    rag_json_spec.get("excluded_allergens", []) or []
                ),
                "rag_json_lexical": list(rag_json_spec.get("excluded_lexical", []) or []),
                "rag_json_restrictions": spec_restrictions,
                "rag_json_requested_items": spec_requested_items,
                "rag_json_finalize": bool(rag_json_spec.get("finalize")),
                "rag_json_override_restriction": override_restriction,
            }
        if override_restriction and blacklist:
            cmeta = {
                **cmeta,
                "allergen_blacklist_suppressed_for_override": list(blacklist),
            }
        max_e = rag_json_spec.get("max_kcal") if rag_json_spec else None
        min_e = rag_json_spec.get("min_kcal") if rag_json_spec else None
        lex_e = (
            list(rag_json_spec.get("excluded_lexical") or [])
            if rag_json_spec
            else []
        )
        effective_top_k = _effective_rag_top_k(
            self.rag_top_k,
            include_full_nutrition=include_full_nutrition,
        )
        spec_extra: list[str] = []
        if rag_json_spec:
            for x in _rag_json_list(rag_json_spec, "requested_items"):
                t = str(x or "").strip()
                if t:
                    spec_extra.append(t)
        graph_query = merge_graph_retrieval_query(
            text,
            client_utterance,
            (*search_queries, *spec_extra),
        )
        rows, graph_info = search_menu_graph(
            graph_query,
            allergens_blacklist=search_blacklist or None,
            top_k=effective_top_k,
            max_energy=float(max_e) if max_e is not None else None,
            min_energy=float(min_e) if min_e is not None else None,
            excluded_lexical=lex_e or None,
        )
        info: dict[str, Any] = {
            "top_k": effective_top_k,
            "allergen_blacklist_tokens": list(search_blacklist),
            **cmeta,
            "graph_retrieval_query": graph_query,
            "candidates": [
                {"name": r["name"], "distance": r["distance"], "energy": r["energy"]}
                for r in rows
            ],
            **graph_info,
        }
        if not rows:
            info["outcome"] = "no_graph_hits"
            return "(no matching menu items found for this request)", info

        lines, used = _render_rows(
            rows,
            max_dist=1.0,
            max_lines=self.rag_max_prompt_lines,
            include_full_nutrition=include_full_nutrition,
        )
        if self.rag_max_prompt_lines is not None:
            info["rag_prompt_line_cap"] = self.rag_max_prompt_lines
            info["rag_prompt_lines_included"] = len(lines)
        info.update(outcome="injected_graph", injected_hits=used)
        return "\n".join(lines), info

    def _build_system(
        self,
        profile: dict[str, Any],
        order_state: dict[str, Any],
        rag_context: str,
        *,
        allow_calories: bool,
        allow_full_nutrition: bool = False,
        finalize_requested: bool = False,
        override_restriction: bool = False,
    ) -> str:
        """Собирает системный промпт: базовый + опциональные блоки RAG и заказа."""
        prompt_profile = None if self._realistic_cashier else profile
        base = get_cashier_system_prompt(
            prompt_profile,
            realistic=self._realistic_cashier,
        )
        extras: list[str] = []
        if rag_context:
            context_payload = (
                rag_context
                if (self._full_menu_context or allow_calories or allow_full_nutrition)
                else _hide_kcal_in_rag_context(rag_context)
            )
            if self._full_menu_context:
                extras.append(
                    "Full mcd.json menu context for this turn. Use this JSON as the complete "
                    "menu source; do not use retrieval, vector DB assumptions, or invented items:\n"
                    f"{context_payload}"
                )
            else:
                if allow_full_nutrition:
                    extras.append(
                        "Menu data slice for this turn (each line includes product name plus "
                        "nutrition fields available in data: kcal, protein, fats, carbs, sugars, "
                        "sodium, allergens, and ingredients when listed):\n"
                        f"{context_payload}"
                    )
                else:
                    extras.append(
                        "Menu data slice for this turn (each line: product name, kcal estimate, "
                        "allergen tags, approximate added/total sugar, and ingredients when listed — "
                        "not full nutrition):\n"
                        f"{context_payload}"
                    )
        visible_order_state = (
            _cashier_visible_order_state(order_state)
            if self._realistic_cashier
            else order_state
        )
        if any(p.get("items") for p in visible_order_state.get("persons", [])):
            extras.append(
                "Current order state:\n"
                + json.dumps(visible_order_state, ensure_ascii=False)
            )
        if finalize_requested and any(
            p.get("items") for p in visible_order_state.get("persons", [])
        ):
            extras.append(
                "Client finalization signal:\n"
                "The customer indicated this is all. Give one concise final order "
                "confirmation/readback using only the current order state. Do not add "
                "new items or ask another open-ended upsell question."
            )
        if override_restriction:
            extras.append(
                "Dietary warning override:\n"
                "The customer indicated they still want the requested item after a "
                "dietary warning. Serve it without repeating the warning or blocking "
                "the order."
            )
        if not extras:
            return base
        return base + "\n\n--- Context ---\n" + "\n\n".join(extras)


# ── Приватные helpers ─────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _full_mcd_json_context() -> str:
    """Raw menu JSON for Non-RAG runs: no retrieval, same menu every LLM turn."""
    return MCD_JSON_PATH.read_text(encoding="utf-8")


def _last_client_text(history: list[HistoryEntry]) -> str:
    for entry in reversed(history):
        if entry["speaker"] == "client":
            return entry["text"]
    return ""


def _cashier_visible_order_state(order_state: dict[str, Any]) -> dict[str, Any]:
    """
    Redact hidden profile-derived people/restrictions before prompting cashier.

    In realistic mode, the cashier may remember items already spoken in the
    conversation, but must not see empty companions or dietary flags seeded from
    the synthetic profile.
    """
    visible_persons: list[dict[str, Any]] = []
    for person in order_state.get("persons", []) or []:
        items = list(person.get("items") or [])
        if not items:
            continue
        redacted: dict[str, Any] = {
            "role": person.get("role"),
            "label": person.get("label"),
            "items": items,
        }
        if person.get("total_energy") is not None:
            redacted["total_energy"] = person.get("total_energy")
        if person.get("allergens"):
            redacted["allergens"] = list(person.get("allergens") or [])
        visible_persons.append(redacted)

    return {
        "persons": visible_persons,
        "order_complete": bool(order_state.get("order_complete", False)),
    }


def _render_rows(
    rows: list[dict[str, Any]],
    *,
    max_dist: float,
    max_lines: int | None = None,
    include_full_nutrition: bool = False,
) -> tuple[list[str], list[dict[str, Any]]]:
    """
    Форматирует строки меню для промпта и список использованных хитов.

    mcd.json содержит несколько вариантов одного блюда (разные размеры порций).
    Дедупликация по имени объединяет их в одну строку с диапазоном калорий,
    чтобы не занимать лишние слоты top_k одинаковыми позициями.

    max_lines: обрезка после порога по distance (порядок = релевантность в rows).
    """
    # Собираем квалифицированные строки (в пределах порога)
    by_name: dict[str, dict[str, Any]] = {}
    name_energies: dict[str, list[float]] = {}
    name_added_sugar: dict[str, list[float]] = {}
    name_total_sugar: dict[str, list[float]] = {}
    metric_fields = ("protein", "total_fat", "sat_fat", "trans_fat", "chol", "carbs", "sodium")
    name_metrics: dict[str, dict[str, list[float]]] = {}
    order: list[str] = []  # сохраняем порядок первого вхождения
    for r in rows:
        if r["distance"] > max_dist:
            continue
        name = r["name"]
        if name not in by_name:
            by_name[name] = r
            name_energies[name] = []
            name_added_sugar[name] = []
            name_total_sugar[name] = []
            name_metrics[name] = {field: [] for field in metric_fields}
            order.append(name)
        name_energies[name].append(float(r["energy"]))
        if r.get("added_sugar") is not None:
            name_added_sugar[name].append(float(r["added_sugar"]))
        if r.get("total_sugar") is not None:
            name_total_sugar[name].append(float(r["total_sugar"]))
        for field in metric_fields:
            if r.get(field) is not None:
                name_metrics[name][field].append(float(r.get(field)))

    lines: list[str] = []
    used: list[dict[str, Any]] = []
    for name in order:
        r = by_name[name]
        allergens = ", ".join(r["allergens"]) if r["allergens"] else "none listed"
        energies = name_energies[name]
        if len(energies) > 1:
            lo, hi = round(min(energies), 1), round(max(energies), 1)
            energy_str = f"~{lo}–{hi}"
        else:
            energy_str = f"~{energies[0]}"
        sugar_parts: list[str] = []
        added_vals = name_added_sugar[name]
        total_vals = name_total_sugar[name]
        if added_vals:
            if len(added_vals) > 1:
                sugar_parts.append(
                    f"added sugar: ~{round(min(added_vals), 1)}–{round(max(added_vals), 1)} g"
                )
            else:
                sugar_parts.append(f"added sugar: ~{round(added_vals[0], 1)} g")
        else:
            sugar_parts.append("added sugar: unknown")
        if total_vals:
            if len(total_vals) > 1:
                sugar_parts.append(
                    f"total sugar: ~{round(min(total_vals), 1)}–{round(max(total_vals), 1)} g"
                )
            else:
                sugar_parts.append(f"total sugar: ~{round(total_vals[0], 1)} g")
        else:
            sugar_parts.append("total sugar: unknown")
        ingredients = str(r.get("ingredients") or "").strip()
        ingredient_part = f"; ingredients: {ingredients}" if ingredients else ""
        if include_full_nutrition:
            nutrition_parts: list[str] = []
            labels = {
                "protein": "protein",
                "total_fat": "fat",
                "sat_fat": "sat fat",
                "trans_fat": "trans fat",
                "chol": "cholesterol",
                "carbs": "carbs",
                "sodium": "sodium",
            }
            units = {
                "protein": "g",
                "total_fat": "g",
                "sat_fat": "g",
                "trans_fat": "g",
                "chol": "mg",
                "carbs": "g",
                "sodium": "mg",
            }
            for field in metric_fields:
                vals = name_metrics[name][field]
                if not vals:
                    continue
                if len(vals) > 1:
                    lo = round(min(vals), 1)
                    hi = round(max(vals), 1)
                    nutrition_parts.append(f"{labels[field]}: ~{lo}–{hi} {units[field]}")
                else:
                    nutrition_parts.append(f"{labels[field]}: ~{round(vals[0], 1)} {units[field]}")
            if nutrition_parts:
                ingredient_part = f"; {'; '.join(nutrition_parts)}{ingredient_part}"
        lines.append(
            f"- {name} ({energy_str} kcal, allergens: {allergens}; "
            f"{'; '.join(sugar_parts)}{ingredient_part})"
        )
        used.append({"name": name, "distance": r["distance"]})
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        used = used[:max_lines]
    return lines, used


def _hide_kcal_in_rag_context(rag_context: str) -> str:
    out_lines: list[str] = []
    for raw in (rag_context or "").splitlines():
        line = re.sub(
            r"~?\d[\d.,]*(?:\s*[–-]\s*\d[\d.,]*)?\s*kcal,\s*",
            "",
            raw,
            flags=re.I,
        )
        line = re.sub(
            r"\(\s*~?\d[\d.,]*(?:\s*[–-]\s*\d[\d.,]*)?\s*kcal\s*\)",
            "",
            line,
            flags=re.I,
        )
        out_lines.append(re.sub(r"\s+", " ", line).strip())
    return "\n".join(out_lines)


_SOFT_PREFIX = (
    "Semantic match is weak; below are real menu rows. "
    "Use exact names and calories; do NOT claim the menu is empty or has no "
    "suitable items if the list is non-empty.\n\n"
)


def _llm_call_payload(
    *,
    agent: str,
    model: str,
    system: str,
    messages: list[dict[str, str]],
    response: str,
    duration_ms: float,
    verbose: bool,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "event": "llm_call",
        "agent": agent,
        "model": model,
        "kind": "dialog_llm",
        "messages_count": len(messages),
        "duration_ms": round(duration_ms, 2),
    }
    if verbose:
        base["system"] = system
        base["messages"] = [{"role": m["role"], "content": m["content"]} for m in messages]
        base["response"] = response
    else:
        base["system_preview"] = _preview(system)
        base["messages_preview"] = _messages_preview(messages)
        base["response_preview"] = _preview(response)
    return base


def _llm_error_payload(
    *,
    agent: str,
    model: str,
    messages: list[dict[str, str]],
    system: str,
    error: str,
    verbose: bool,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "event": "llm_error",
        "agent": agent,
        "model": model,
        "messages_count": len(messages),
        "error": error,
    }
    if verbose:
        base["system"] = system
        base["messages"] = [{"role": m["role"], "content": m["content"]} for m in messages]
    else:
        base["system_preview"] = _preview(system)
        base["messages_preview"] = _messages_preview(messages)
    return base


def _preview(text: str | None, limit: int = 1200) -> str:
    if not text:
        return ""
    return text[:limit] + "…" if len(text) > limit else text


def _messages_preview(
    messages: list[dict[str, str]],
    *,
    max_items: int = 3,
) -> list[dict[str, str]]:
    tail = messages[-max_items:]
    return [{"role": m["role"], "content": _preview(m["content"], 280)} for m in tail]


def _trace(
    rag_trace: list[dict[str, Any]] | None,
    event: dict[str, Any],
) -> None:
    if rag_trace is not None:
        rag_trace.append(event)
