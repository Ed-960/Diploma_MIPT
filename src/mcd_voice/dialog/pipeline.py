"""
Конвейер одного диалога: REG → LLM-агенты → RAG → эвристика заказа → валидация.

Поток:
  1. Кассир приветствует (drive-through).
  2. Цикл: клиент → кассир → парсинг заказа → проверка завершения.
  3. По окончании — валидация per-person (allergen_violation, calorie_warning…).

order_state имеет многопользовательскую структуру: массив persons.
Агенты получают полную историю диалога на каждом ходу.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import time
from collections import Counter
from typing import Any, Callable

from mcd_voice.dialog.catalog import MenuCatalog
from mcd_voice.llm import CashierAgent, ClientAgent
from mcd_voice.llm.agent import _build_openai_client, _call_llm, _resolve_model
from mcd_voice.profile import ProfileGenerator
from mcd_voice.profile.generator import get_allergen_blacklist
from mcd_voice.text_normalization import normalize_item_text


# ── Эвристики завершения ──────────────────────────────────────────────

_CASHIER_FINALIZE_PATTERNS = re.compile(
    r"your total|that will be|that.?ll be|order is ready|anything else\?|"
    r"order is confirmed|confirm(ed)? your order|enjoy your meal|have a great|have a nice|here you go|"
    r"see you|drive.?through|pull.?forward",
    re.IGNORECASE,
)

_CLIENT_CONFIRM_PATTERNS = re.compile(
    r"\bthat.?s all\b|\bthat.?s it\b|\bno thanks\b|\bnope\b|"
    r"\bjust that\b|\bnothing else\b|\bi.?m good\b|\bbye\b|"
    r"\bthank you\b|\bthanks\b|\ball set\b|\bsounds good\b|\bperfect\b|"
    r"\bgot it\b|\bpickup now\b|\border ready\b|"
    r"\byes\b|\byeah\b|\byep\b|\bcorrect\b|\bexactly\b",
    re.IGNORECASE,
)

_CLIENT_FAREWELL_PATTERNS = re.compile(
    r"\byou too\b|\bbye\b|\bsee you\b|\bgood day\b|\bgood night\b",
    re.IGNORECASE,
)

# ── CoT-утечки ────────────────────────────────────────────────────────

_COT_LEAK_PATTERNS = re.compile(
    r"\$\$|\\\[|\\\(|\\boxed\{|"
    r"the user is a real customer|"
    r"real customer at a mcdonald|"
    r"the goal is to find|"
    r"given the constraints|"
    r"step[- ]by[- ]step|"
    r"let me (think|reason|analyze|consider)|"
    r"in (this|the) scenario",
    re.IGNORECASE,
)


def _is_cot_leak(text: str) -> bool:
    """True если модель «протекла» — вставила рассуждения или инструкции в реплику."""
    return bool(_COT_LEAK_PATTERNS.search(text))


# ── Детектор зацикливания ─────────────────────────────────────────────

_STALL_DENIAL_RE = re.compile(
    r"(i.?m sorry|we don.?t have|i don.?t have|not available|"
    r"could you (tell me|let me know)|what type of food)",
    re.IGNORECASE,
)


def _is_stalled(history: list[dict[str, str]], window: int = 4) -> bool:
    """
    True если последние `window` реплик кассира содержат одну и ту же
    фразу-отказ — кассир крутится на месте без прогресса.
    """
    cashier_msgs = [
        t["text"] for t in history if t.get("speaker") == "cashier"
    ]
    recent = cashier_msgs[-window:]
    if len(recent) < window:
        return False
    return all(_STALL_DENIAL_RE.search(m) for m in recent)

ProgressCallback = Callable[[dict[str, Any]], None]


def _cashier_signals_end(text: str) -> bool:
    return bool(_CASHIER_FINALIZE_PATTERNS.search(text))


def _client_confirms_end(text: str) -> bool:
    return bool(_CLIENT_CONFIRM_PATTERNS.search(text))


def _client_says_farewell(text: str) -> bool:
    return bool(_CLIENT_FAREWELL_PATTERNS.search(text))


def _order_parser_reason_stats(
    llm_trace: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Сводка по причинам fallback mini-LLM order parser за диалог."""
    rewrite_fallback_reasons: Counter[str] = Counter()
    deterministic_fallback_reasons: Counter[str] = Counter()
    rewrite_calls = 0
    rewrite_fallbacks = 0
    deterministic_fallbacks = 0
    for ev in llm_trace or []:
        if not isinstance(ev, dict):
            continue
        kind = str(ev.get("event") or "")
        if kind == "order_json_rewrite":
            rewrite_calls += 1
            if ev.get("fallback_used"):
                rewrite_fallbacks += 1
                reason = str(ev.get("fallback_reason") or "unknown")
                rewrite_fallback_reasons[reason] += 1
            continue
        if kind == "order_json_fallback_to_deterministic":
            deterministic_fallbacks += 1
            reason = str(ev.get("fallback_reason") or "unknown")
            deterministic_fallback_reasons[reason] += 1
    return {
        "total_order_parser_events": rewrite_calls + deterministic_fallbacks,
        "rewrite_calls": rewrite_calls,
        "rewrite_fallbacks": rewrite_fallbacks,
        "deterministic_fallbacks": deterministic_fallbacks,
        "rewrite_fallback_reasons": dict(rewrite_fallback_reasons),
        "deterministic_fallback_reasons": dict(deterministic_fallback_reasons),
    }


# ── Парсинг количества блюд ──────────────────────────────────────────

def _normalize_item_text(text: str) -> str:
    """
    Нормализация названий для мягкого матчинга:
    убираем trademark-символы и пунктуацию, схлопываем пробелы.
    """
    return normalize_item_text(text)


_RESTRICTION_CUE_RE = re.compile(
    r"\b(allerg\w*|intoleran\w*|without|avoid|exclude|free from|"
    r"no|not|dont|can't|cannot|must not)\b",
    re.IGNORECASE,
)
_ORDER_CUE_RE = re.compile(
    r"\b(i(?:\s+would)?\s+like|i\s+want|i\s+need|i\s+take|i\s+ll\s+have|"
    r"can\s+i\s+have|give\s+me|add|order|for\s+me)\b",
    re.IGNORECASE,
)


def _is_restriction_mention(lower_text: str, idx: int, name_norm: str) -> bool:
    """
    True when an item mention is likely a dietary restriction, not an order.
    Example: "I have allergy on milk" should not add menu item "Milk".
    """
    left = lower_text[max(0, idx - 40):idx]
    ctx = lower_text[max(0, idx - 40): min(len(lower_text), idx + len(name_norm) + 24)]
    if not _RESTRICTION_CUE_RE.search(ctx):
        return False
    # Explicit ordering cue near the mention should still count as an order.
    if _ORDER_CUE_RE.search(left):
        return False
    return True


def parse_order_from_text(
    text: str,
    menu_names: list[str],
) -> list[tuple[str, int]]:
    """
    Извлекает (item_name, quantity) из текста.
    Ищет «N <item_name>» или просто «<item_name>» (qty=1).
    """
    lower = _normalize_item_text(text)
    found: list[tuple[str, int]] = []
    seen: set[str] = set()

    for name in menu_names:
        name_norm = _normalize_item_text(name)
        if len(name_norm) < 4:
            continue
        idx = lower.find(name_norm)
        if idx == -1 or name in seen:
            continue
        if _is_restriction_mention(lower, idx, name_norm):
            continue
        seen.add(name)

        qty = 1
        prefix = lower[max(0, idx - 24):idx].strip()
        m = re.search(r"(\d{1,2})\s*$", prefix)
        if m:
            qty = int(m.group(1))
            if qty < 1:
                qty = 1
            if qty > 20:
                qty = 1

        found.append((name, qty))

    # Also parse short natural aliases ("fries", "nuggets", "coke") even when
    # one full menu name has already matched.
    seen_names = {name for name, _ in found}
    alias_patterns = _build_alias_patterns(menu_names)
    for rx, target_name in alias_patterns:
        if target_name in seen_names:
            continue
        m = rx.search(lower)
        if not m:
            continue
        target_norm = _normalize_item_text(target_name)
        if _is_restriction_mention(lower, m.start(), target_norm):
            continue
        qty = 1
        prefix = lower[max(0, m.start() - 24):m.start()].strip()
        qty_match = re.search(r"(\d{1,2})\s*$", prefix)
        if qty_match:
            qty = max(1, min(20, int(qty_match.group(1))))
        found.append((target_name, qty))
        seen_names.add(target_name)
    return found


def _build_alias_patterns(menu_names: list[str]) -> list[tuple[re.Pattern[str], str]]:
    """Подбирает безопасные алиасы к конкретным пунктам меню."""
    pairs: list[tuple[str, str]] = []

    def pick(*tokens: str) -> str | None:
        for name in menu_names:
            if not name:
                continue
            n = _normalize_item_text(name)
            if all(t in n for t in tokens):
                return name
        return None

    def pick_exact(name_fragment: str) -> str | None:
        """Match item containing fragment but prefer exact/shorter names."""
        candidates = [
            n for n in menu_names
            if n and name_fragment.lower() in _normalize_item_text(n)
        ]
        if not candidates:
            return None
        return min(candidates, key=len)

    def pick_plain_fries() -> str | None:
        """Match plain fries, excluding Cheesy Fries."""
        for name in menu_names:
            if not name:
                continue
            n = _normalize_item_text(name)
            if "fries" in n and "cheesy" not in n:
                return name
        return None

    mapping = [
        (r"\bblack coffee\b", pick("black", "coffee")),
        (r"\bcold coffee\b", pick("cold", "coffee")),
        (r"\biced tea\b", pick("iced", "tea")),
        (r"\bcheesy fries\b", pick("cheesy", "fries")),
        (r"\b(plain |regular |just )?fries\b", pick_plain_fries()),
        (r"\bnuggets\b", pick("mcnuggets")),
        (r"\bmcveggie\b", pick("mcveggie")),
        (r"\bbig mac\b", pick_exact("big mac")),
        (r"\bquarter pounder\b", pick_exact("quarter pounder")),
        (r"\bmcdouble\b", pick_exact("mcdouble")),
        (r"\bcheeseburger\b", pick_exact("cheeseburger")),
        (r"\bhamburger\b", pick_exact("hamburger")),
        (r"\bhappy meal\b", pick_exact("happy meal")),
        (r"\begg mcmuffin\b", pick_exact("egg mcmuffin")),
        (r"\bhash browns?\b", pick_exact("hash brown")),
        (r"\bhotcakes\b", pick_exact("hotcakes")),
        (r"\bapple (slices|pie)\b", pick_exact("apple")),
        (r"\bcoke\b", pick_exact("coca-cola")),
        (r"\bdiet coke\b", pick_exact("diet coke")),
        (r"\bdr\.?\s*pepper\b", pick_exact("dr pepper")),
        (r"\bfanta\b", pick_exact("fanta")),
        (r"\bsweet tea\b", pick_exact("sweet tea")),
        (r"\blemonade\b", pick_exact("lemonade")),
        (r"\bsprite\b", pick_exact("sprite")),
        (r"\borange juice\b", pick_exact("orange juice")),
        (r"\b(bottled )?water\b", pick_exact("bottled water")),
        (r"\bhot chocolate\b", pick_exact("hot chocolate")),
        (r"\b(mocha|caramel) frappe\b", pick_exact("frappe")),
        (r"\b(chocolate|vanilla|strawberry) shake\b", pick_exact("shake")),
        (r"\bsmoothie\b", pick_exact("smoothie")),
        (r"\bcaesar salad\b", pick_exact("caesar")),
        (r"\bmozzarella sticks?\b", pick_exact("mozzarella")),
    ]
    for patt, target in mapping:
        if target:
            pairs.append((patt, target))
    return [(re.compile(p, re.IGNORECASE), t) for p, t in pairs]


_UNAVAILABLE_CUES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bi don't have\b", re.I),
    re.compile(r"\bwe don't have\b", re.I),
    re.compile(r"\b(?:i|we)\s+(?:am|are)\s+not seeing\b", re.I),
    re.compile(r"\bi'm\s+not\s+seeing\b", re.I),
    re.compile(r"\bnot (currently )?in (our )?menu\b", re.I),
    re.compile(r"\bnot available\b", re.I),
    re.compile(r"\bcan'?t (offer|do)\b", re.I),
    re.compile(r"\bcan'?t do that\b", re.I),
    re.compile(r"\bcan'?t recommend\b", re.I),
    re.compile(r"\b(?:has|contains)\s+added\s+sugar\b", re.I),
)


def _extract_unavailable_items(
    text: str,
    menu_names: list[str],
) -> set[str]:
    """
    Heuristic extraction of menu items that cashier explicitly marks unavailable.
    Used to clean stale items from order_state after customer changes choices.
    """
    if not text:
        return set()
    raw = (text or "").lower()
    if not any(rx.search(raw) for rx in _UNAVAILABLE_CUES):
        return set()

    unavailable: set[str] = set()
    for name in menu_names:
        name_norm = _normalize_item_text(name)
        if len(name_norm) < 4:
            continue
        name_pat = r"\b" + r"\W+".join(re.escape(tok) for tok in name_norm.split()) + r"\b"
        before_item = re.compile(
            rf"\b(?:i|we)\s+don['’]?t\s+have\b[^.!?;\n]{{0,30}}{name_pat}"
            rf"|\b(?:i|we)\s+(?:am|are)\s+not\s+seeing\b[^.!?;\n]{{0,30}}{name_pat}"
            rf"|\bi['’]?m\s+not\s+seeing\b[^.!?;\n]{{0,30}}{name_pat}"
            rf"|\bnot\s+available\b[^.!?;\n]{{0,30}}{name_pat}"
            rf"|\bcan['’]?t\s+(?:offer|recommend|do)\b[^.!?;\n]{{0,30}}{name_pat}",
            re.I,
        )
        after_item = re.compile(
            rf"{name_pat}[^.!?;\n]{{0,24}}"
            rf"(?:isn['’]?t\s+listed|not\s+available|not\s+in\s+(?:our\s+)?menu"
            rf"|(?:i|we)\s+can['’]?t\s+(?:offer|recommend|do)(?:\s+that)?"
            rf"|(?:has|contains)\s+added\s+sugar[^.!?;\n]{{0,40}}\binstead\b)",
            re.I,
        )
        if before_item.search(raw) or after_item.search(raw):
            unavailable.add(name)
    return unavailable


def _restriction_flags_from_profile(profile_or_restrictions: dict[str, Any]) -> dict[str, bool]:
    flags = {
        "noMilk": bool(profile_or_restrictions.get("noMilk")),
        "noFish": bool(profile_or_restrictions.get("noFish")),
        "noNuts": bool(profile_or_restrictions.get("noNuts")),
        "noEggs": bool(profile_or_restrictions.get("noEggs")),
        "noGluten": bool(profile_or_restrictions.get("noGluten")),
        "noBeef": bool(profile_or_restrictions.get("noBeef")),
        "isVegan": bool(profile_or_restrictions.get("isVegan")),
        "noSugar": bool(profile_or_restrictions.get("noSugar")),
    }
    if flags["isVegan"]:
        # Vegan profile implies avoiding core animal-derived categories.
        flags["noMilk"] = True
        flags["noFish"] = True
        flags["noEggs"] = True
        flags["noBeef"] = True
    return flags


def _item_restriction_violations(
    item_name: str,
    restriction_flags: dict[str, bool],
    restriction_map: dict[str, dict[str, bool]],
) -> set[str]:
    meta = restriction_map.get(item_name)
    if not meta:
        return set()
    violated: set[str] = set()
    for key, enabled in restriction_flags.items():
        if enabled and meta.get(key, False):
            violated.add(key)
    return violated


# ── Назначение блюд персонам ─────────────────────────────────────────

_NUM_WORDS: dict[int, str] = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
}


def _label_to_aliases(label: str) -> set[str]:
    raw = (label or "").strip().lower()
    if not raw:
        return set()
    out = {raw}
    s = re.sub(r"[_\-]+", " ", raw)
    s = re.sub(r"\s+", " ", s).strip()
    out.add(s)
    m = re.search(r"\b(\d+)\b", s)
    if m:
        n = int(m.group(1))
        out.add(s.replace(m.group(1), f" {n} ").strip())
        if n in _NUM_WORDS:
            out.add(s.replace(m.group(1), _NUM_WORDS[n]).strip())
    return {x for x in out if x}


def _detect_target_person(text: str) -> str:
    """
    Lightweight public detector kept for tests/backward compatibility.
    Main runtime routing is resolved in _resolve_target_indices() using persons.
    """
    t = (text or "").lower()
    if re.search(r"\bfor (me|myself)\b|\bi.?ll have\b|\bi want\b|\bi.?d like\b|\bдля меня\b", t):
        return "self"
    if re.search(r"\b(wife|husband|spouse)\b", t):
        return "spouse"
    if re.search(r"\b(youngest|little|smaller)\b", t):
        return "child_youngest"
    if re.search(r"\b(oldest|elder|bigger|older)\b", t):
        return "child_oldest"
    if re.search(r"\b\d{1,2}\s*[- ]?\s*year\s*[- ]?\s*old\b", t):
        return "child_generic"
    if re.search(r"\b(kid|child|son|daughter|children|kids)\b|\b(child|kid)\s*(one|1|first|two|2|second)\b", t):
        return "child_generic"
    if re.search(r"\bfriend\b", t):
        return "friend_generic"
    return "self"


def _resolve_person_index(
    target: str,
    persons: list[dict[str, Any]],
) -> int:
    """Маппит текстовый ключ на индекс в массиве persons."""
    if target == "self":
        return 0
    if target == "spouse":
        for i, p in enumerate(persons):
            if p.get("role") == "spouse":
                return i
    if target == "child_youngest":
        children = [(i, p) for i, p in enumerate(persons) if p.get("role") == "child"]
        if children:
            return min(children, key=lambda x: x[1].get("age", 99))[0]
    if target == "child_oldest":
        children = [(i, p) for i, p in enumerate(persons) if p.get("role") == "child"]
        if children:
            return max(children, key=lambda x: x[1].get("age", 0))[0]
    if target in ("child_generic", "friend_generic"):
        role_key = "child" if "child" in target else "friend"
        for i, p in enumerate(persons):
            if p.get("role") == role_key and not p.get("items"):
                return i
        for i, p in enumerate(persons):
            if p.get("role") == role_key:
                return i
    return 0


def _resolve_target_indices(
    text: str,
    persons: list[dict[str, Any]],
) -> list[int]:
    """
    Возвращает список индексов персон для фразы заказа.
    Поддерживает массовые формулировки: for both / for everyone / for my kids.
    """
    lower = (text or "").lower()
    all_indices = list(range(len(persons)))
    child_indices = [i for i, p in enumerate(persons) if p.get("role") == "child"]
    friend_indices = [i for i, p in enumerate(persons) if p.get("role") == "friend"]
    spouse_indices = [i for i, p in enumerate(persons) if p.get("role") == "spouse"]

    if (
        "for everyone" in lower
        or "for all of us" in lower
        or "for us all" in lower
        or "for all" in lower
    ):
        return all_indices
    if "for both" in lower and len(all_indices) >= 2:
        return all_indices[:2]
    if "for my kids" in lower or "for the kids" in lower or "for my children" in lower:
        return child_indices or [0]
    if "for my friends" in lower or "for the friends" in lower:
        return friend_indices or [0]
    if "for me and my friend" in lower:
        if friend_indices:
            return [0, friend_indices[0]]
        return [0]
    age_match = re.search(r"\b(\d{1,2})\s*[- ]?\s*year\s*[- ]?\s*old\b", lower)
    if age_match and child_indices:
        target_age = int(age_match.group(1))
        for i in child_indices:
            age = persons[i].get("age")
            if isinstance(age, int) and age == target_age:
                return [i]
    if re.search(r"\b(youngest|little|smaller)\b", lower) and child_indices:
        return [min(child_indices, key=lambda i: persons[i].get("age", 99))]
    if re.search(r"\b(oldest|elder|bigger|older)\b", lower) and child_indices:
        return [max(child_indices, key=lambda i: persons[i].get("age", 0))]
    if re.search(r"\b(spouse|wife|husband)\b", lower) and spouse_indices:
        return [spouse_indices[0]]

    hits: list[tuple[int, int]] = []
    for idx, p in enumerate(persons):
        role = str(p.get("role") or "").lower()
        aliases: set[str] = set()
        aliases.update(_label_to_aliases(str(p.get("label") or "")))
        if idx == 0:
            aliases.update({"me", "myself"})
        if role == "child":
            child_no = child_indices.index(idx) + 1 if idx in child_indices else None
            if len(child_indices) == 1:
                aliases.update({"child", "kid", "son", "daughter"})
            if child_no is not None:
                aliases.update({f"child {child_no}", f"kid {child_no}"})
                if child_no in _NUM_WORDS:
                    w = _NUM_WORDS[child_no]
                    aliases.update({f"child {w}", f"kid {w}"})
            age = p.get("age")
            if isinstance(age, int) and age > 0:
                aliases.update({f"{age} year old", f"{age}-year-old"})
        elif role == "friend":
            friend_no = friend_indices.index(idx) + 1 if idx in friend_indices else None
            if len(friend_indices) == 1:
                aliases.add("friend")
            if friend_no is not None:
                aliases.add(f"friend {friend_no}")
                if friend_no in _NUM_WORDS:
                    aliases.add(f"friend {_NUM_WORDS[friend_no]}")
        elif role == "spouse":
            aliases.update({"spouse", "wife", "husband"})
        for alias in aliases:
            a = alias.strip()
            if len(a) < 2:
                continue
            patt = r"\b" + re.escape(a).replace(r"\ ", r"\s+") + r"\b"
            m = re.search(patt, lower, re.I)
            if m:
                hits.append((m.start(), idx))
    if hits:
        hits.sort(key=lambda x: x[0])
        out: list[int] = []
        seen: set[int] = set()
        for _, i in hits:
            if i in seen:
                continue
            seen.add(i)
            out.append(i)
        if out:
            return out

    # Universal fallback: if no explicit person marker is found,
    # treat the segment as customer's own order.
    return [0]


def _split_order_segments(text: str) -> list[str]:
    """
    Split complex multi-person order into segments likely tied to different people.
    Keeps default behavior for simple one-person phrases.
    """
    if not (text or "").strip():
        return []
    split_re = re.compile(
        r"(?:(?<=[.;])\s+|,\s+)"
        r"(?=(?:and\s+)?(?:for\s+(?:my|the|me|myself)|"
        r"child\s+\w+|kid\s+\w+|my\s+friend|friend\s+\w+|"
        r"my\s+wife|my\s+husband|spouse|child[_\s-]?\d+|friend[_\s-]?\d+))",
        re.I,
    )
    parts = [p.strip() for p in split_re.split(text) if p.strip()]
    return parts or [text]


_ORDER_JSON_REWRITE_SYSTEM = """You extract structured order assignments from one customer utterance.
Return JSON only.

Schema:
{
  "orders": [
    {
      "target": "self | spouse | child_1 | child_2 | ... | friend_1 | friend_2 | ... | everyone | both",
      "items": [{"name": "menu item name", "quantity": 1}]
    }
  ]
}

Rules:
- Use exact menu item names from MENU_NAMES list only.
- target:
  - self for customer
  - spouse for spouse
  - child_N / friend_N when explicitly referenced by number/age/label
  - everyone for all persons
  - both for customer + one explicitly referenced person
- If item is not clearly ordered (restriction-only phrase), do not include it.
- quantity must be integer >= 1.
- Output valid JSON only.
"""


def _normalize_menu_lookup(menu_names: list[str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for n in menu_names:
        key = _normalize_item_text(n)
        if key and key not in lookup:
            lookup[key] = n
    return lookup


def _map_item_name_to_menu(name: str, menu_lookup: dict[str, str]) -> str | None:
    k = _normalize_item_text(name)
    if not k:
        return None
    if k in menu_lookup:
        return menu_lookup[k]
    # soft contains fallback for minor punctuation/spacing differences
    for nk, original in menu_lookup.items():
        if k in nk or nk in k:
            return original
    return None


def _is_main_item_name(name: str) -> bool:
    n = (name or "").lower()
    return bool(
        re.search(
            r"\b(big mac|burger|hamburger|wrap|sandwich|nuggets?|mcchicken|"
            r"mcdouble|quarter pounder|maharaja|filet|mcspicy|paneer)\b",
            n,
        )
    )


# ── Построение multi-person order_state ───────────────────────────────

def build_initial_order_state(profile: dict[str, Any]) -> dict[str, Any]:
    """Создаёт order_state с массивом persons из профиля."""
    persons: list[dict[str, Any]] = [
        {
            "role": "self",
            "label": "customer",
            "items": [],
            "total_energy": 0.0,
            "allergens": [],
        },
    ]
    for comp in profile.get("companions", []):
        role = comp.get("role", "friend")
        label = comp.get("label", f"{role}_?")
        persons.append({
            "role": role,
            "label": label,
            "age": comp.get("age"),
            "restrictions": comp.get("restrictions", {}),
            "items": [],
            "total_energy": 0.0,
            "allergens": [],
        })
    return {"persons": persons, "order_complete": False}


# ── DialogPipeline ────────────────────────────────────────────────────

class DialogPipeline:
    """Оркестрация одного синтетического диалога."""

    def __init__(
        self,
        *,
        max_turns: int = 20,
        model: str | None = None,
        profile_generator: ProfileGenerator | None = None,
        menu_catalog: MenuCatalog | None = None,
        client_agent: ClientAgent | None = None,
        cashier_agent: CashierAgent | None = None,
        progress_callback: ProgressCallback | None = None,
        collect_rag_trace: bool = False,
        collect_llm_trace: bool = False,
        emit_trace_progress: bool = False,
        trace_verbose: bool = False,
        realistic_cashier: bool = True,
    ) -> None:
        self.max_turns = max_turns
        # При отсутствии явной модели берем API_MODEL из env.
        self.model = _resolve_model(model)
        self._profiles = profile_generator or ProfileGenerator()
        self._catalog = menu_catalog or MenuCatalog()
        self._client_agent = client_agent
        self._cashier_agent = cashier_agent
        self._progress_callback = progress_callback
        self._collect_rag_trace = collect_rag_trace
        self._collect_llm_trace = collect_llm_trace
        self._emit_trace_progress = emit_trace_progress
        self._trace_verbose = trace_verbose
        self._realistic_cashier = realistic_cashier
        self._order_json_enabled = (
            (os.environ.get("ORDER_JSON_REWRITE") or "1").strip().lower()
            not in {"0", "false", "off", "no"}
        )
        self._order_json_model = os.environ.get("REWRITE_MODEL") or self.model
        self._order_json_client = None
        self._order_json_disabled_reason: str | None = None

    def run(
        self,
        profile: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any], dict[str, Any]]:
        self._emit_progress(
            "prepare",
            max_turns=self.max_turns,
            message="Подготовка профиля, меню и order_state.",
        )
        if profile is None:
            profile = self._profiles.generate()

        (
            menu_names,
            energy_by_name,
            allergen_map,
            restriction_map,
        ) = self._load_catalog_runtime_index()

        client = self._client_agent or ClientAgent(
            model=self.model, trace_verbose=self._trace_verbose,
        )
        cashier = self._cashier_agent or CashierAgent(
            model=self.model,
            trace_verbose=self._trace_verbose,
            realistic_cashier=self._realistic_cashier,
        )

        history: list[dict[str, str]] = []
        order_state = build_initial_order_state(profile)
        rag_trace: list[dict[str, Any]] | None = (
            [] if self._collect_rag_trace else None
        )
        llm_trace: list[dict[str, Any]] | None = (
            [] if self._collect_llm_trace else None
        )
        n_rag = 0
        n_llm = 0
        cot_leak_count = 0
        stall_detected = False
        loop_detected = False

        self._emit_progress(
            "greeting_start",
            message="Кассир формирует приветствие.",
        )
        cashier_kwargs: dict[str, Any] = {
            "rag_trace": rag_trace,
            "rag_meta": {"call": "greeting"},
        }
        if llm_trace is not None and _accepts_kwarg(cashier.generate_response, "llm_trace"):
            cashier_kwargs["llm_trace"] = llm_trace
        greeting = cashier.generate_response(profile, history, order_state, **cashier_kwargs)
        history.append({"speaker": "cashier", "text": greeting})
        self._emit_progress(
            "greeting_done",
            history_len=len(history),
            message="Приветствие готово.",
        )
        n_rag, n_llm = self._flush_trace_delta(
            rag_trace, llm_trace, n_rag, n_llm, label="greeting",
        )

        cashier_signaled = False

        for turn in range(1, self.max_turns + 1):
            self._emit_progress(
                "turn_start",
                turn=turn,
                max_turns=self.max_turns,
                history_len=len(history),
                message=f"Ход {turn}/{self.max_turns}.",
            )
            self._emit_progress(
                "client_thinking",
                turn=turn,
                max_turns=self.max_turns,
                message="Клиент формирует ответ.",
            )
            client_kwargs: dict[str, Any] = {}
            if llm_trace is not None and _accepts_kwarg(client.generate_response, "llm_trace"):
                client_kwargs["llm_trace"] = llm_trace
            client_msg = client.generate_response(profile, history, **client_kwargs)
            if _is_cot_leak(client_msg):
                cot_leak_count += 1
            history.append({"speaker": "client", "text": client_msg})
            self._emit_progress(
                "client_done",
                turn=turn,
                max_turns=self.max_turns,
                history_len=len(history),
                text=client_msg,
                message="Клиент ответил.",
            )
            n_rag, n_llm = self._flush_trace_delta(
                rag_trace, llm_trace, n_rag, n_llm, label="client", turn=turn,
            )

            if cashier_signaled and (
                _client_confirms_end(client_msg) or _client_says_farewell(client_msg)
            ):
                order_state["order_complete"] = True
                self._emit_progress(
                    "finished",
                    turn=turn,
                    max_turns=self.max_turns,
                    history_len=len(history),
                    reason="client_confirmed_end",
                    message="Клиент подтвердил/закрыл диалог после финальной реплики кассира.",
                )
                break

            self._emit_progress(
                "cashier_thinking",
                turn=turn,
                max_turns=self.max_turns,
                message="Кассир формирует ответ.",
            )
            cashier_kwargs = {
                "rag_trace": rag_trace,
                "rag_meta": {"call": "turn", "turn": turn},
            }
            if llm_trace is not None and _accepts_kwarg(cashier.generate_response, "llm_trace"):
                cashier_kwargs["llm_trace"] = llm_trace
            cashier_msg = cashier.generate_response(
                profile,
                history,
                order_state,
                **cashier_kwargs,
            )
            if _is_cot_leak(cashier_msg):
                cot_leak_count += 1
            history.append({"speaker": "cashier", "text": cashier_msg})
            self._emit_progress(
                "cashier_done",
                turn=turn,
                max_turns=self.max_turns,
                history_len=len(history),
                text=cashier_msg,
                message="Кассир ответил.",
            )
            n_rag, n_llm = self._flush_trace_delta(
                rag_trace, llm_trace, n_rag, n_llm, label="cashier", turn=turn,
            )

            if _is_stalled(history):
                stall_detected = True
                self._emit_progress(
                    "finished",
                    turn=turn,
                    max_turns=self.max_turns,
                    history_len=len(history),
                    reason="stall_detected",
                    message="Диалог прерван: кассир зациклился на отказах без прогресса.",
                )
                break

            if _is_looping_tail(history) or _has_cashier_hard_repeat(history, repeat=3):
                loop_detected = True
                # Если заказ уже собран, считаем диалог завершённым, чтобы не тянуть до max_turns.
                if validate_dialog(
                    profile,
                    order_state,
                    history,
                    menu_names=menu_names,
                    restriction_map=restriction_map,
                )["total_items"] > 0:
                    order_state["order_complete"] = True
                self._emit_progress(
                    "finished",
                    turn=turn,
                    max_turns=self.max_turns,
                    history_len=len(history),
                    reason="loop_detected",
                    message="Диалог прерван: обнаружен повторяющийся хвост реплик.",
                )
                break

            # Обновляем заказ только по реплике клиента: так не добавляем
            # предложения кассира, которые клиент не подтверждал.
            structured_orders = self._parse_structured_orders(
                client_msg,
                menu_names,
                order_state.get("persons", []),
                llm_trace=llm_trace,
                trace_meta={"event_scope": "order_parser", "turn": turn},
            )
            self._update_order(
                client_msg,
                menu_names,
                order_state,
                energy_by_name,
                allergen_map,
                structured_orders=structured_orders,
                llm_trace=llm_trace,
                trace_meta={"event_scope": "order_parser", "turn": turn},
            )
            self._enforce_restriction_safety(
                profile, order_state, energy_by_name, allergen_map, restriction_map,
            )

            # Если кассир явно сообщил, что некоторые позиции недоступны/не подходят,
            # убираем эти позиции из уже накопленного заказа.
            self._remove_unavailable_from_order(
                cashier_msg, menu_names, order_state, energy_by_name, allergen_map,
            )
            self._enforce_restriction_safety(
                profile, order_state, energy_by_name, allergen_map, restriction_map,
            )

            cashier_signaled = _cashier_signals_end(cashier_msg)
            if cashier_signaled:
                # Финальная реплика кассира обычно содержит итоговый состав заказа;
                # это безопаснее, чем парсить любые промежуточные предложения.
                self._update_order(
                    cashier_msg, menu_names, order_state, energy_by_name, allergen_map,
                )
                self._enforce_restriction_safety(
                    profile,
                    order_state,
                    energy_by_name,
                    allergen_map,
                    restriction_map,
                )
                # Защита от бесконечного хвоста "Yes." -> "Order confirmed."
                # Если клиент уже дал краткое подтверждение, завершаем сразу.
                if _is_yes_only(client_msg):
                    order_state["order_complete"] = True
                    self._emit_progress(
                        "finished",
                        turn=turn,
                        max_turns=self.max_turns,
                        history_len=len(history),
                        reason="cashier_finalized_after_yes",
                        message="Завершено: кассир подтвердил заказ после краткого yes клиента.",
                    )
                    break

        else:
            self._emit_progress(
                "finished",
                turn=self.max_turns,
                max_turns=self.max_turns,
                history_len=len(history),
                reason="max_turns_reached",
                message="Диалог остановлен по лимиту max_turns.",
            )

        flags = validate_dialog(
            profile,
            order_state,
            history,
            menu_names=menu_names,
            restriction_map=restriction_map,
        )
        if cot_leak_count:
            flags = {**flags, "cot_leak_count": cot_leak_count}
        if stall_detected:
            flags = {**flags, "stall_detected": True}
        if loop_detected:
            flags = {**flags, "loop_detected": True}
        error_localization = localize_errors(history, flags)
        if error_localization:
            flags = {**flags, "error_localization": error_localization}
        if rag_trace is not None:
            flags = {**flags, "rag_trace": rag_trace}
        if llm_trace is not None:
            flags = {**flags, "llm_trace": llm_trace}
            order_parser_stats = _order_parser_reason_stats(llm_trace)
            if order_parser_stats.get("total_order_parser_events", 0) > 0:
                flags = {**flags, "order_parser_stats": order_parser_stats}
                self._emit_progress(
                    "order_parser_summary",
                    summary=order_parser_stats,
                )
        return history, profile, order_state, flags

    def _build_allergen_map(self) -> dict[str, list[str]]:
        """
        Backward-compatible helper kept for tests and old integrations.
        Runtime pipeline now gets allergen_map from MenuCatalog.load_runtime_index().
        """
        load_rt = getattr(self._catalog, "load_runtime_index", None)
        if callable(load_rt):
            _, _, allergen_map, _ = load_rt()
            return allergen_map
        return {}

    def _load_catalog_runtime_index(
        self,
    ) -> tuple[
        list[str],
        dict[str, float],
        dict[str, list[str]],
        dict[str, dict[str, bool]],
    ]:
        """
        Runtime catalog accessor with backward compatibility for legacy test stubs.
        Preferred API: load_runtime_index() -> (names, energy, allergen_map, restriction_map).
        Legacy API: load() -> (names, energy).
        """
        load_rt = getattr(self._catalog, "load_runtime_index", None)
        if callable(load_rt):
            return load_rt()
        load_legacy = getattr(self._catalog, "load", None)
        if callable(load_legacy):
            menu_names, energy_by_name = load_legacy()
            allergen_map = self._build_allergen_map()
            restriction_map: dict[str, dict[str, bool]] = {}
            return menu_names, energy_by_name, allergen_map, restriction_map
        raise AttributeError("Menu catalog must provide load_runtime_index() or load().")

    @staticmethod
    def _remove_unavailable_from_order(
        text: str,
        menu_names: list[str],
        order_state: dict[str, Any],
        energy_by_name: dict[str, float],
        allergen_map: dict[str, list[str]],
    ) -> None:
        names_to_drop = _extract_unavailable_items(text, menu_names)
        if not names_to_drop:
            return

        persons = order_state.get("persons", [])
        for p in persons:
            items = p.get("items", [])
            p["items"] = [it for it in items if it.get("name") not in names_to_drop]

        # Recompute aggregates after removal.
        for p in persons:
            p["total_energy"] = round(
                sum(
                    energy_by_name.get(it["name"], 0) * it.get("quantity", 1)
                    for it in p.get("items", [])
                ),
                2,
            )
            all_ag: set[str] = set()
            for it in p.get("items", []):
                all_ag.update(allergen_map.get(it["name"], []))
            p["allergens"] = sorted(all_ag)

    @staticmethod
    def _enforce_restriction_safety(
        profile: dict[str, Any],
        order_state: dict[str, Any],
        energy_by_name: dict[str, float],
        allergen_map: dict[str, list[str]],
        restriction_map: dict[str, dict[str, bool]],
    ) -> None:
        """
        Hard safety pass for synthetic data:
        remove ordered items that violate person-specific allergen blacklist.
        """
        persons = order_state.get("persons", [])
        if not persons:
            return

        for p in persons:
            role = p.get("role")
            if role == "self":
                restriction_flags = _restriction_flags_from_profile(profile)
            else:
                restriction_flags = _restriction_flags_from_profile(p.get("restrictions", {}))
            if not any(restriction_flags.values()):
                continue

            safe_items: list[dict[str, Any]] = []
            for it in p.get("items", []):
                name = it.get("name", "")
                if _item_restriction_violations(name, restriction_flags, restriction_map):
                    continue
                safe_items.append(it)
            p["items"] = safe_items

        # Recompute aggregates after filtering.
        for p in persons:
            p["total_energy"] = round(
                sum(
                    energy_by_name.get(it["name"], 0) * it.get("quantity", 1)
                    for it in p.get("items", [])
                ),
                2,
            )
            all_ag: set[str] = set()
            for it in p.get("items", []):
                all_ag.update(allergen_map.get(it["name"], []))
            p["allergens"] = sorted(all_ag)

    @staticmethod
    def _target_indices_from_token(
        token: str,
        persons: list[dict[str, Any]],
    ) -> list[int]:
        t = (token or "").strip().lower()
        if not t:
            return []
        all_indices = list(range(len(persons)))
        child_indices = [i for i, p in enumerate(persons) if p.get("role") == "child"]
        friend_indices = [i for i, p in enumerate(persons) if p.get("role") == "friend"]
        spouse_indices = [i for i, p in enumerate(persons) if p.get("role") == "spouse"]
        if t in {"self", "customer", "me", "myself"}:
            return [0] if persons else []
        if t == "everyone":
            return all_indices
        if t == "both":
            return all_indices[:2]
        if t == "spouse":
            return spouse_indices[:1]
        if t.startswith("child_"):
            try:
                idx = int(t.split("_", 1)[1]) - 1
            except ValueError:
                return child_indices[:1]
            return [child_indices[idx]] if 0 <= idx < len(child_indices) else child_indices[:1]
        if t.startswith("friend_"):
            try:
                idx = int(t.split("_", 1)[1]) - 1
            except ValueError:
                return friend_indices[:1]
            return [friend_indices[idx]] if 0 <= idx < len(friend_indices) else friend_indices[:1]
        # label/alias fallback
        for i, p in enumerate(persons):
            aliases = _label_to_aliases(str(p.get("label") or ""))
            if t in aliases:
                return [i]
        return []

    def _parse_structured_orders(
        self,
        text: str,
        menu_names: list[str],
        persons: list[dict[str, Any]],
        *,
        llm_trace: list[dict[str, Any]] | None = None,
        trace_meta: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Optional mini-LLM parser: extract structured (target, items[]) assignments.
        Always returns [] on errors, so deterministic parser can fallback safely.
        """
        if not self._order_json_enabled:
            if llm_trace is not None:
                llm_trace.append(
                    {
                        "event": "order_json_disabled",
                        **(trace_meta or {}),
                        "model": self._order_json_model,
                        "reason": "env_disabled",
                    }
                )
            return []
        if self._order_json_disabled_reason:
            if llm_trace is not None:
                llm_trace.append(
                    {
                        "event": "order_json_disabled",
                        **(trace_meta or {}),
                        "model": self._order_json_model,
                        "reason": self._order_json_disabled_reason,
                    }
                )
            return []
        if not (text or "").strip() or not menu_names or not persons:
            return []
        if self._order_json_client is None:
            try:
                self._order_json_client = _build_openai_client(timeout=20.0)
            except Exception as exc:
                self._order_json_disabled_reason = str(exc)[:200]
                if llm_trace is not None:
                    llm_trace.append(
                        {
                            "event": "order_json_client_error",
                            **(trace_meta or {}),
                            "model": self._order_json_model,
                            "error": self._order_json_disabled_reason,
                        }
                    )
                return []
        persons_desc = []
        for i, p in enumerate(persons):
            persons_desc.append(
                {
                    "index": i,
                    "label": p.get("label"),
                    "role": p.get("role"),
                    "age": p.get("age"),
                }
            )
        user_payload = {
            "utterance": text,
            "persons": persons_desc,
            "menu_names": menu_names,
        }
        t0 = time.perf_counter()
        try:
            raw = _call_llm(
                self._order_json_client,
                self._order_json_model,
                _ORDER_JSON_REWRITE_SYSTEM,
                [{"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}],
                temperature=0.0,
            )
            data = json.loads(raw)
        except Exception as exc:
            if llm_trace is not None:
                llm_trace.append(
                    {
                        "event": "order_json_parse_error",
                        **(trace_meta or {}),
                        "model": self._order_json_model,
                        "error": str(exc)[:200],
                        "duration_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                    }
                )
            return []
        orders = data.get("orders") if isinstance(data, dict) else None
        if not isinstance(orders, list):
            if llm_trace is not None:
                llm_trace.append(
                    {
                        "event": "order_json_parse_error",
                        **(trace_meta or {}),
                        "model": self._order_json_model,
                        "error": "orders_missing_or_not_list",
                        "duration_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                    }
                )
            return []
        menu_lookup = _normalize_menu_lookup(menu_names)
        out: list[dict[str, Any]] = []
        rows_with_targets = 0
        rows_with_raw_items = 0
        rows_with_mapped_items = 0
        for row in orders:
            if not isinstance(row, dict):
                continue
            indices = self._target_indices_from_token(str(row.get("target") or ""), persons)
            if not indices:
                continue
            rows_with_targets += 1
            raw_items = row.get("items")
            if not isinstance(raw_items, list):
                continue
            rows_with_raw_items += 1
            items: list[tuple[str, int]] = []
            for it in raw_items:
                if not isinstance(it, dict):
                    continue
                mapped = _map_item_name_to_menu(str(it.get("name") or ""), menu_lookup)
                if not mapped:
                    continue
                try:
                    qty = int(it.get("quantity") or 1)
                except (TypeError, ValueError):
                    qty = 1
                qty = max(1, min(20, qty))
                items.append((mapped, qty))
            if items:
                rows_with_mapped_items += 1
                out.append({"target_indices": indices, "items": items})
        fallback_reason: str | None = None
        if not out:
            if not orders:
                fallback_reason = "empty_orders"
            elif rows_with_targets == 0:
                fallback_reason = "no_valid_targets"
            elif rows_with_raw_items == 0:
                fallback_reason = "items_missing_or_not_list"
            elif rows_with_mapped_items == 0:
                fallback_reason = "no_mapped_items"
            else:
                fallback_reason = "no_valid_structured_rows"
        if llm_trace is not None:
            llm_trace.append(
                {
                    "event": "order_json_rewrite",
                    **(trace_meta or {}),
                    "model": self._order_json_model,
                    "duration_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                    "orders_count": len(out),
                    "fallback_used": len(out) == 0,
                    "fallback_reason": fallback_reason,
                }
            )
        return out

    def _emit_progress(self, stage: str, **payload: Any) -> None:
        if self._progress_callback is None:
            return
        event = {"stage": stage, **payload}
        self._progress_callback(event)

    def _flush_trace_delta(
        self,
        rag_trace: list[dict[str, Any]] | None,
        llm_trace: list[dict[str, Any]] | None,
        n_rag: int,
        n_llm: int,
        *,
        label: str,
        turn: int | None = None,
    ) -> tuple[int, int]:
        """Отправляет в progress_callback новые события rag_trace / llm_trace (для консоли)."""
        if not self._emit_trace_progress or self._progress_callback is None:
            return (
                len(rag_trace) if rag_trace is not None else n_rag,
                len(llm_trace) if llm_trace is not None else n_llm,
            )
        new_rag = rag_trace[n_rag:] if rag_trace is not None else []
        new_llm = llm_trace[n_llm:] if llm_trace is not None else []
        if not new_rag and not new_llm:
            return (
                len(rag_trace) if rag_trace is not None else n_rag,
                len(llm_trace) if llm_trace is not None else n_llm,
            )
        payload: dict[str, Any] = {
            "label": label,
            "rag_events": list(new_rag),
            "llm_events": list(new_llm),
        }
        if turn is not None:
            payload["turn"] = turn
        self._emit_progress("trace_delta", **payload)
        return (
            len(rag_trace) if rag_trace is not None else n_rag,
            len(llm_trace) if llm_trace is not None else n_llm,
        )

    # ── Внутренние методы ─────────────────────────────────────────────

    @staticmethod
    def _update_order(
        text: str,
        menu_names: list[str],
        order_state: dict[str, Any],
        energy_by_name: dict[str, float],
        allergen_map: dict[str, list[str]],
        *,
        structured_orders: list[dict[str, Any]] | None = None,
        llm_trace: list[dict[str, Any]] | None = None,
        trace_meta: dict[str, Any] | None = None,
    ) -> None:
        persons = order_state["persons"]
        group_size = max(1, len(persons))
        applied = False
        if structured_orders:
            for row in structured_orders:
                target_indices = list(row.get("target_indices") or [])
                parsed = list(row.get("items") or [])
                if not target_indices or not parsed:
                    continue
                distribute_to_many = len(target_indices) > 1
                for idx in target_indices:
                    if idx >= len(persons):
                        continue
                    person = persons[idx]
                    for name, qty in parsed:
                        qty = int(qty or 1)
                        if distribute_to_many:
                            qty = 1
                        else:
                            qty = _apply_group_quantity_hint(text, qty, group_size)
                        existing = next((it for it in person["items"] if it["name"] == name), None)
                        applied = True
                        if existing:
                            existing["quantity"] = max(existing["quantity"], qty)
                        else:
                            person["items"].append({"name": name, "quantity": qty})
        if not applied:
            if structured_orders is not None and llm_trace is not None:
                fallback_reason = (
                    "empty_structured_orders"
                    if len(structured_orders) == 0
                    else "structured_orders_not_applied"
                )
                llm_trace.append(
                    {
                        "event": "order_json_fallback_to_deterministic",
                        **(trace_meta or {}),
                        "structured_orders_count": len(structured_orders),
                        "fallback_reason": fallback_reason,
                    }
                )
            segments = _split_order_segments(text)
            for seg in segments:
                parsed = parse_order_from_text(seg, menu_names)
                if not parsed:
                    continue
                target_indices = _resolve_target_indices(seg, persons)
                has_instead_cue = bool(re.search(r"\binstead\b", seg, re.I))
                # Если заказ адресован нескольким людям ("for both", "for everyone"),
                # каждый получает по 1 штуке — не умножаем quantity на одного.
                distribute_to_many = len(target_indices) > 1
                for idx in target_indices:
                    if idx >= len(persons):
                        continue
                    person = persons[idx]
                    # Replacement intent: "I'll take X instead."
                    # Keep sides/drinks, but swap out previously collected mains.
                    if has_instead_cue:
                        parsed_names = {name for name, _ in parsed}
                        parsed_has_main = any(_is_main_item_name(name) for name in parsed_names)
                        if parsed_has_main:
                            person["items"] = [
                                it
                                for it in person.get("items", [])
                                if not (
                                    _is_main_item_name(it.get("name", ""))
                                    and it.get("name") not in parsed_names
                                )
                            ]
                    for name, qty in parsed:
                        if distribute_to_many:
                            qty = 1
                        else:
                            qty = _apply_group_quantity_hint(seg, qty, group_size)
                        existing = next((it for it in person["items"] if it["name"] == name), None)
                        if existing:
                            existing["quantity"] = max(existing["quantity"], qty)
                        else:
                            person["items"].append({"name": name, "quantity": qty})

        # Пересчёт energy и allergens для каждой персоны
        for p in persons:
            p["total_energy"] = round(
                sum(
                    energy_by_name.get(it["name"], 0) * it["quantity"]
                    for it in p["items"]
                ), 2,
            )
            all_ag: set[str] = set()
            for it in p["items"]:
                all_ag.update(allergen_map.get(it["name"], []))
            p["allergens"] = sorted(all_ag)


# ── Валидация ─────────────────────────────────────────────────────────

def validate_dialog(
    profile: dict[str, Any],
    order_state: dict[str, Any],
    history: list[dict[str, str]],
    *,
    menu_names: list[str] | None = None,
    restriction_map: dict[str, dict[str, bool]] | None = None,
) -> dict[str, Any]:
    """
    Валидация per-person + общая.

    Возвращает:
      per_person: [{label, allergen_violation, calorie_ok}, ...]
      allergen_violation: объединённый список нарушений
      calorie_warning: общие калории > 1.5× calApprValue
      empty_order: ни одного блюда
      total_items: общее число позиций
      total_energy: общая энергия
      turns: количество реплик
      hallucination: есть позиции, которых нет в меню
      incomplete_order: при наличии детей не все дети получили позиции
    """
    persons = order_state.get("persons", [])
    rmap = restriction_map or {}
    profile_blacklist = set(get_allergen_blacklist(profile))

    per_person: list[dict[str, Any]] = []
    allergen_violation_per_person: list[dict[str, Any]] = []
    restriction_violation_per_person: list[dict[str, Any]] = []
    all_violations: set[str] = set()
    all_restriction_violations: set[str] = set()
    total_energy = 0.0
    total_items = 0
    ordered_item_names: list[str] = []

    for p in persons:
        label = p.get("label", "?")
        role = p.get("role", "?")
        p_allergens = set(p.get("allergens", []))
        p_energy = p.get("total_energy", 0.0)
        p_items = sum(it.get("quantity", 1) for it in p.get("items", []))
        ordered_item_names.extend(
            str(it.get("name", "")).strip()
            for it in p.get("items", [])
            if str(it.get("name", "")).strip()
        )

        # Blacklist: self → профиль, компаньоны → их restrictions
        if role == "self":
            bl = profile_blacklist
            restriction_flags = _restriction_flags_from_profile(profile)
        else:
            bl = set(get_allergen_blacklist(p.get("restrictions", {})))
            restriction_flags = _restriction_flags_from_profile(p.get("restrictions", {}))

        violation = bl & p_allergens
        if violation:
            all_violations.update(violation)
            allergen_violation_per_person.append(
                {
                    "label": label,
                    "role": role,
                    "allergens": sorted(violation),
                }
            )

        restriction_violation: set[str] = set()
        for it in p.get("items", []):
            name = str(it.get("name", "")).strip()
            if not name:
                continue
            restriction_violation.update(
                _item_restriction_violations(name, restriction_flags, rmap)
            )
        if restriction_violation:
            all_restriction_violations.update(restriction_violation)
            restriction_violation_per_person.append(
                {
                    "label": label,
                    "role": role,
                    "restrictions": sorted(restriction_violation),
                }
            )

        per_person.append({
            "label": label,
            "role": role,
            "items_count": p_items,
            "total_energy": p_energy,
            "allergen_violation": sorted(violation),
            "restriction_violation": sorted(restriction_violation),
        })
        total_energy += p_energy
        total_items += p_items

    cal_target = profile.get("calApprValue", 2200)
    companion_slots = [p for p in persons if p.get("role") != "self"]
    companions_without_items = sum(
        1 for p in companion_slots if not p.get("items")
    )
    under_target_warning = (
        total_items > 0 and cal_target > 0 and total_energy < cal_target * 0.35
    )
    child_persons = [p for p in persons if p.get("role") == "child"]
    all_children_ordered = all(
        len(p.get("items", [])) > 0 for p in child_persons
    )
    incomplete_order = bool(
        profile.get("childQuant", 0) > 0 and child_persons and not all_children_ordered
    )
    menu_names_set = set(menu_names or [])
    hallucinated_items = (
        sorted({name for name in ordered_item_names if name not in menu_names_set})
        if menu_names_set else []
    )

    return {
        "per_person": per_person,
        "allergen_violation": sorted(all_violations),
        "allergen_violation_per_person": allergen_violation_per_person,
        "restriction_violation": sorted(all_restriction_violations),
        "restriction_violation_per_person": restriction_violation_per_person,
        "calorie_warning": total_energy > cal_target * 1.5,
        "under_target_warning": under_target_warning,
        "empty_order": total_items == 0,
        "incomplete_order": incomplete_order,
        "hallucination": bool(hallucinated_items),
        "hallucinated_items": hallucinated_items,
        "total_items": total_items,
        "total_energy": round(total_energy, 2),
        "calorie_target": cal_target,
        "companions_without_items": companions_without_items,
        "turns": len(history),
    }


def localize_errors(
    history: list[dict[str, str]],
    flags: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Heuristic per-turn localization for post-hoc diploma analysis.

    Output row:
      {turn, speaker, error_type, excerpt, ...optional fields}
    """
    errors: list[dict[str, Any]] = []
    violation_rows = flags.get("allergen_violation_per_person", [])
    violating_tokens = {
        token.lower()
        for row in violation_rows
        for token in row.get("allergens", [])
        if token
    }
    restriction_rows = flags.get("restriction_violation_per_person", [])
    violating_restrictions = {
        token.lower()
        for row in restriction_rows
        for token in row.get("restrictions", [])
        if token
    }

    for i, turn in enumerate(history):
        text = turn.get("text", "")
        speaker = turn.get("speaker", "")
        excerpt = text[:120]
        if _is_cot_leak(text):
            errors.append(
                {
                    "turn": i,
                    "speaker": speaker,
                    "error_type": "cot_leak",
                    "excerpt": excerpt,
                }
            )

        if speaker == "cashier" and violating_tokens:
            lower_text = text.lower()
            for token in sorted(violating_tokens):
                if token in lower_text:
                    errors.append(
                        {
                            "turn": i,
                            "speaker": speaker,
                            "error_type": "allergen_suggestion",
                            "allergen": token,
                            "excerpt": excerpt,
                        }
                    )
        if speaker == "cashier" and violating_restrictions:
            lower_text = text.lower()
            for token in sorted(violating_restrictions):
                if token.replace("no", "") in lower_text or token in lower_text:
                    errors.append(
                        {
                            "turn": i,
                            "speaker": speaker,
                            "error_type": "restriction_suggestion",
                            "restriction": token,
                            "excerpt": excerpt,
                        }
                    )

    if flags.get("stall_detected") and history:
        errors.append(
            {
                "turn": len(history) - 1,
                "speaker": history[-1].get("speaker", ""),
                "error_type": "stall_detected",
                "excerpt": (history[-1].get("text") or "")[:120],
            }
        )
    if flags.get("loop_detected") and history:
        errors.append(
            {
                "turn": len(history) - 1,
                "speaker": history[-1].get("speaker", ""),
                "error_type": "loop_detected",
                "excerpt": (history[-1].get("text") or "")[:120],
            }
        )
    for item_name in flags.get("hallucinated_items", []):
        errors.append(
            {
                "turn": None,
                "speaker": "system",
                "error_type": "hallucinated_item",
                "item": item_name,
                "excerpt": item_name,
            }
        )
    return errors


# ── Удобные функции ──────────────────────────────────────────────────

def simulate_dialog(
    profile: dict[str, Any] | None = None,
    *,
    max_turns: int = 20,
    model: str | None = None,
    client_agent: ClientAgent | None = None,
    cashier_agent: CashierAgent | None = None,
    progress_callback: ProgressCallback | None = None,
    collect_rag_trace: bool = False,
    collect_llm_trace: bool = False,
    emit_trace_progress: bool = False,
    trace_verbose: bool = False,
    realistic_cashier: bool = True,
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Функциональный фасад: один диалог с валидацией."""
    pipeline = DialogPipeline(
        max_turns=max_turns,
        model=model,
        client_agent=client_agent,
        cashier_agent=cashier_agent,
        progress_callback=progress_callback,
        collect_rag_trace=collect_rag_trace,
        collect_llm_trace=collect_llm_trace,
        emit_trace_progress=emit_trace_progress,
        trace_verbose=trace_verbose,
        realistic_cashier=realistic_cashier,
    )
    return pipeline.run(profile=profile)


def generate_dialog(
    max_turns: int = 20,
    model: str | None = None,
    progress_callback: ProgressCallback | None = None,
    collect_rag_trace: bool = False,
    collect_llm_trace: bool = False,
    emit_trace_progress: bool = False,
    trace_verbose: bool = False,
    realistic_cashier: bool = True,
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    return simulate_dialog(
        max_turns=max_turns,
        model=model,
        progress_callback=progress_callback,
        collect_rag_trace=collect_rag_trace,
        collect_llm_trace=collect_llm_trace,
        emit_trace_progress=emit_trace_progress,
        trace_verbose=trace_verbose,
        realistic_cashier=realistic_cashier,
    )


def _accepts_kwarg(func: Callable[..., Any], key: str) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    if key in signature.parameters:
        return True
    return any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )


def _apply_group_quantity_hint(text: str, qty: int, group_size: int) -> int:
    """If client says 'for everyone/all of us', lift qty to group size."""
    if qty != 1 or group_size <= 1:
        return qty
    lower = text.lower()
    if (
        "for everyone" in lower
        or "for all of us" in lower
        or "for all" in lower
        or "for us all" in lower
    ):
        return group_size
    return qty


def _is_yes_only(text: str) -> bool:
    normalized = _normalize_item_text(text)
    return normalized in {"yes", "yes thanks", "yes thank you", "yeah", "yep", "correct"}


def _is_looping_tail(history: list[dict[str, str]]) -> bool:
    """
    Detects short repetitive tail patterns like:
      client "hurry"  -> cashier "almost there" (x3+)
      client "you too" -> cashier "have a great day" (x3+)
    """
    if len(history) < 6:
        return False

    tail = history[-6:]
    expected = ["client", "cashier", "client", "cashier", "client", "cashier"]
    if [x.get("speaker") for x in tail] != expected:
        return False

    client_msgs = [_normalize_item_text(tail[i].get("text", "")) for i in (0, 2, 4)]
    cashier_msgs = [_normalize_item_text(tail[i].get("text", "")) for i in (1, 3, 5)]

    if not all(client_msgs) or not all(cashier_msgs):
        return False
    if len(set(client_msgs)) != 1 or len(set(cashier_msgs)) != 1:
        return False

    # Guard against long semantic replies: loop tails are usually short acknowledgements.
    return len(client_msgs[0].split()) <= 6 and len(cashier_msgs[0].split()) <= 10


def _has_cashier_hard_repeat(history: list[dict[str, str]], repeat: int = 3) -> bool:
    cashier_msgs = [
        _normalize_item_text(t.get("text", ""))
        for t in history
        if t.get("speaker") == "cashier"
    ]
    if len(cashier_msgs) < repeat:
        return False
    tail = cashier_msgs[-repeat:]
    return bool(tail[0]) and len(set(tail)) == 1


def print_dialog(
    history: list[dict[str, str]],
    profile: dict[str, Any],
    order: dict[str, Any],
    flags: dict[str, Any] | None = None,
) -> None:
    print("\n=== Профиль ===")
    print(json.dumps(profile, ensure_ascii=False, indent=2))
    print("\n=== Диалог ===")
    for line in history:
        who = "Клиент" if line["speaker"] == "client" else "Кассир"
        print(f"\n{who}: {line['text']}")
    print("\n=== Заказ (per-person) ===")
    for p in order.get("persons", []):
        label = p.get("label", "?")
        items = p.get("items", [])
        if items:
            items_str = ", ".join(f"{it['name']} x{it['quantity']}" for it in items)
        else:
            items_str = "(пусто)"
        print(f"  {label}: {items_str}  [{p.get('total_energy', 0)} kcal]")
    if flags:
        print("\n=== Флаги валидации ===")
        print(json.dumps(flags, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        h, p, o, f = simulate_dialog(max_turns=10)
        print_dialog(h, p, o, f)
    except RuntimeError as e:
        print(f"Ошибка: {e}")
        print("Проверьте LLM_API_KEY / LLM_BASE_URL (или Ollama) и наличие chroma_db после load_chroma.py.")
