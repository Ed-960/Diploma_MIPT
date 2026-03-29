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

import json
import re
from typing import Any

from mcd_voice.dialog.catalog import MenuCatalog
from mcd_voice.llm import CashierAgent, ClientAgent
from mcd_voice.llm.agent import _resolve_model
from mcd_voice.profile import ProfileGenerator
from mcd_voice.profile.generator import get_allergen_blacklist


# ── Эвристики завершения ──────────────────────────────────────────────

_CASHIER_FINALIZE_PATTERNS = re.compile(
    r"your total|that will be|that.?ll be|order is ready|anything else\?|"
    r"enjoy your meal|have a great|have a nice|here you go|"
    r"ваш заказ|итого|приятного аппетита|что-нибудь ещё\?",
    re.IGNORECASE,
)

_CLIENT_CONFIRM_PATTERNS = re.compile(
    r"\bthat.?s all\b|\bthat.?s it\b|\bno thanks\b|\bnope\b|"
    r"\bjust that\b|\bnothing else\b|\bi.?m good\b|\bbye\b|"
    r"\bthank you\b|\bthanks\b|"
    r"\bэто всё\b|\bнет, спасибо\b|\bвсё\b|\bспасибо\b|\bдо свидания\b",
    re.IGNORECASE,
)


def _cashier_signals_end(text: str) -> bool:
    return bool(_CASHIER_FINALIZE_PATTERNS.search(text))


def _client_confirms_end(text: str) -> bool:
    return bool(_CLIENT_CONFIRM_PATTERNS.search(text))


# ── Парсинг количества блюд ──────────────────────────────────────────

_QTY_PATTERN = re.compile(r"\b(\d{1,2})\s+", re.IGNORECASE)


def parse_order_from_text(
    text: str,
    menu_names: list[str],
) -> list[tuple[str, int]]:
    """
    Извлекает (item_name, quantity) из текста.
    Ищет «N <item_name>» или просто «<item_name>» (qty=1).
    """
    lower = text.lower()
    found: list[tuple[str, int]] = []
    seen: set[str] = set()

    for name in menu_names:
        if len(name) < 4:
            continue
        name_lower = name.lower()
        idx = lower.find(name_lower)
        if idx == -1 or name in seen:
            continue
        seen.add(name)

        qty = 1
        prefix = lower[max(0, idx - 10):idx].strip()
        m = re.search(r"(\d{1,2})\s*$", prefix)
        if m:
            qty = int(m.group(1))
            if qty < 1:
                qty = 1
            if qty > 20:
                qty = 1

        found.append((name, qty))
    return found


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
            if p["role"] == "spouse":
                return i
    if target == "child_youngest":
        children = [(i, p) for i, p in enumerate(persons) if p["role"] == "child"]
        if children:
            return min(children, key=lambda x: x[1].get("age", 99))[0]
    if target == "child_oldest":
        children = [(i, p) for i, p in enumerate(persons) if p["role"] == "child"]
        if children:
            return max(children, key=lambda x: x[1].get("age", 0))[0]
    if target in ("child_generic", "friend_generic"):
        role_key = "child" if "child" in target else "friend"
        for i, p in enumerate(persons):
            if p["role"] == role_key and not p.get("items"):
                return i
        for i, p in enumerate(persons):
            if p["role"] == role_key:
                return i
    return 0


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
        persons.append({
            "role": comp["role"],
            "label": comp["label"],
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
    ) -> None:
        self.max_turns = max_turns
        # При отсутствии явной модели берем API_MODEL из env.
        self.model = _resolve_model(model)
        self._profiles = profile_generator or ProfileGenerator()
        self._catalog = menu_catalog or MenuCatalog()
        self._client_agent = client_agent
        self._cashier_agent = cashier_agent

    def run(
        self,
        profile: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any], dict[str, Any]]:
        if profile is None:
            profile = self._profiles.generate()

        menu_names, energy_by_name = self._catalog.load()
        allergen_map = self._build_allergen_map()

        client = self._client_agent or ClientAgent(model=self.model)
        cashier = self._cashier_agent or CashierAgent(model=self.model)

        history: list[dict[str, str]] = []
        order_state = build_initial_order_state(profile)

        greeting = cashier.generate_response(profile, history, order_state)
        history.append({"speaker": "cashier", "text": greeting})

        cashier_signaled = False

        for turn in range(1, self.max_turns + 1):
            client_msg = client.generate_response(profile, history)
            history.append({"speaker": "client", "text": client_msg})

            if cashier_signaled and _client_confirms_end(client_msg):
                break

            cashier_msg = cashier.generate_response(profile, history, order_state)
            history.append({"speaker": "cashier", "text": cashier_msg})

            # Обновляем заказ по обеим репликам (клиент может назвать блюдо,
            # кассир может подтвердить или предложить)
            self._update_order(
                client_msg, menu_names, order_state, energy_by_name, allergen_map,
            )
            self._update_order(
                cashier_msg, menu_names, order_state, energy_by_name, allergen_map,
            )

            cashier_signaled = _cashier_signals_end(cashier_msg)

        flags = validate_dialog(profile, order_state, history)
        return history, profile, order_state, flags

    # ── Внутренние методы ─────────────────────────────────────────────

    def _build_allergen_map(self) -> dict[str, list[str]]:
        from mcd_voice.config import MCD_JSON_PATH
        from mcd_voice.menu.parsing import parse_allergy_field

        with open(MCD_JSON_PATH, "r", encoding="utf-8") as f:
            items: list[dict[str, Any]] = json.load(f)
        return {it["name"]: parse_allergy_field(it.get("allergy")) for it in items}

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
        target_key = _detect_target_person(text)
        target_idx = _resolve_person_index(target_key, persons)

        person = persons[target_idx]
        for name, qty in parsed:
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

    return {
        "per_person": per_person,
        "allergen_violation": sorted(all_violations),
        "calorie_warning": total_energy > cal_target * 1.5,
        "empty_order": total_items == 0,
        "total_items": total_items,
        "total_energy": round(total_energy, 2),
        "calorie_target": cal_target,
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
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Функциональный фасад: один диалог с валидацией."""
    pipeline = DialogPipeline(
        max_turns=max_turns,
        model=model,
        client_agent=client_agent,
        cashier_agent=cashier_agent,
    )
    return pipeline.run(profile=profile)


def generate_dialog(
    max_turns: int = 20,
    model: str | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    return simulate_dialog(max_turns=max_turns, model=model)


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
