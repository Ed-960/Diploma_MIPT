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
import re
from typing import Any, Callable

from mcd_voice.dialog.catalog import MenuCatalog
from mcd_voice.llm import CashierAgent, ClientAgent
from mcd_voice.llm.agent import _resolve_model
from mcd_voice.profile import ProfileGenerator
from mcd_voice.profile.generator import get_allergen_blacklist


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


# ── Парсинг количества блюд ──────────────────────────────────────────

def _normalize_item_text(text: str) -> str:
    """
    Нормализация названий для мягкого матчинга:
    убираем trademark-символы и пунктуацию, схлопываем пробелы.
    """
    s = text.lower()
    s = re.sub(r"[®™℠]", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


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

    # Fallback: короткие естественные упоминания ("fries", "nuggets", "black coffee"),
    # когда полное название из меню не произнесено.
    if found:
        return found

    alias_patterns = _build_alias_patterns(menu_names)
    for rx, target_name in alias_patterns:
        m = rx.search(lower)
        if not m:
            continue
        qty = 1
        prefix = lower[max(0, m.start() - 24):m.start()].strip()
        qty_match = re.search(r"(\d{1,2})\s*$", prefix)
        if qty_match:
            qty = max(1, min(20, int(qty_match.group(1))))
        found.append((target_name, qty))
        break
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


# ── Назначение блюд персонам ─────────────────────────────────────────

_PERSON_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bfor (my |the )?wife\b", re.I), "spouse"),
    (re.compile(r"\bfor (my |the )?husband\b", re.I), "spouse"),
    (re.compile(r"\bfor (my |the )?(youngest|little|smaller)", re.I), "child_youngest"),
    (re.compile(r"\bfor (my |the )?(oldest|elder|bigger|older)", re.I), "child_oldest"),
    (re.compile(r"\bfor (my |the )?(kid|child|son|daughter|children|kids)\b", re.I), "child_generic"),
    (re.compile(r"\bfor (my |the )?friend\b", re.I), "friend_generic"),
    (re.compile(r"\bfor (me|myself)\b", re.I), "self"),
    (re.compile(r"\b(i.?ll have|i want|i.?d like|для меня)\b", re.I), "self"),
]


def _detect_target_person(text: str) -> str:
    """Определяет, для кого заказ по тексту. Возвращает ключ или 'self'."""
    for pattern, label in _PERSON_PATTERNS:
        if pattern.search(text):
            return label
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
    lower = text.lower()
    all_indices = list(range(len(persons)))
    child_indices = [i for i, p in enumerate(persons) if p.get("role") == "child"]
    friend_indices = [i for i, p in enumerate(persons) if p.get("role") == "friend"]

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

    target_key = _detect_target_person(text)
    return [_resolve_person_index(target_key, persons)]


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
        realistic_cashier: bool = False,
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

        menu_names, energy_by_name = self._catalog.load()
        allergen_map = self._build_allergen_map()

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
                if validate_dialog(profile, order_state, history)["total_items"] > 0:
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
            self._update_order(
                client_msg, menu_names, order_state, energy_by_name, allergen_map,
            )

            cashier_signaled = _cashier_signals_end(cashier_msg)
            if cashier_signaled:
                # Финальная реплика кассира обычно содержит итоговый состав заказа;
                # это безопаснее, чем парсить любые промежуточные предложения.
                self._update_order(
                    cashier_msg, menu_names, order_state, energy_by_name, allergen_map,
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

        flags = validate_dialog(profile, order_state, history)
        if cot_leak_count:
            flags = {**flags, "cot_leak_count": cot_leak_count}
        if stall_detected:
            flags = {**flags, "stall_detected": True}
        if loop_detected:
            flags = {**flags, "loop_detected": True}
        if rag_trace is not None:
            flags = {**flags, "rag_trace": rag_trace}
        if llm_trace is not None:
            flags = {**flags, "llm_trace": llm_trace}
        return history, profile, order_state, flags

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

    def _build_allergen_map(self) -> dict[str, list[str]]:
        from mcd_voice.config import MCD_JSON_PATH
        from mcd_voice.menu.parsing import parse_allergy_field

        with open(MCD_JSON_PATH, "r", encoding="utf-8") as f:
            items: list[dict[str, Any]] = json.load(f)
        return {
            it["name"]: parse_allergy_field(it.get("allergy"))
            for it in items
            if it.get("name")
        }

    @staticmethod
    def _update_order(
        text: str,
        menu_names: list[str],
        order_state: dict[str, Any],
        energy_by_name: dict[str, float],
        allergen_map: dict[str, list[str]],
    ) -> None:
        parsed = parse_order_from_text(text, menu_names)
        if not parsed:
            return

        persons = order_state["persons"]
        group_size = max(1, len(persons))
        target_indices = _resolve_target_indices(text, persons)

        # Если заказ адресован нескольким людям ("for both", "for everyone"),
        # каждый получает по 1 штуке — не умножаем quantity на одного.
        distribute_to_many = len(target_indices) > 1

        for idx in target_indices:
            if idx >= len(persons):
                continue
            person = persons[idx]
            for name, qty in parsed:
                if distribute_to_many:
                    qty = 1
                else:
                    qty = _apply_group_quantity_hint(text, qty, group_size)
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
    """
    persons = order_state.get("persons", [])
    profile_blacklist = set(get_allergen_blacklist(profile))

    per_person: list[dict[str, Any]] = []
    all_violations: set[str] = set()
    total_energy = 0.0
    total_items = 0

    for p in persons:
        label = p.get("label", "?")
        role = p.get("role", "?")
        p_allergens = set(p.get("allergens", []))
        p_energy = p.get("total_energy", 0.0)
        p_items = sum(it.get("quantity", 1) for it in p.get("items", []))

        # Blacklist: self → профиль, компаньоны → их restrictions
        if role == "self":
            bl = profile_blacklist
        else:
            bl = set(get_allergen_blacklist(p.get("restrictions", {})))

        violation = bl & p_allergens
        if violation:
            all_violations.update(violation)

        per_person.append({
            "label": label,
            "role": role,
            "items_count": p_items,
            "total_energy": p_energy,
            "allergen_violation": sorted(violation),
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

    return {
        "per_person": per_person,
        "allergen_violation": sorted(all_violations),
        "calorie_warning": total_energy > cal_target * 1.5,
        "under_target_warning": under_target_warning,
        "empty_order": total_items == 0,
        "total_items": total_items,
        "total_energy": round(total_energy, 2),
        "calorie_target": cal_target,
        "companions_without_items": companions_without_items,
        "turns": len(history),
    }


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
    realistic_cashier: bool = False,
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
    realistic_cashier: bool = False,
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
        print("Проверьте OPENAI_API_KEY и наличие chroma_db после load_chroma.py.")
