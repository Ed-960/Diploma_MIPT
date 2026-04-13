"""
REG: стохастическая генерация профилей клиентов (JSON).

Вероятности основаны на российской статистике:
  - Росстат (2024): демография, пол, возрастные группы.
  - Известия / генетические исследования: 61% населения РФ —
    непереносимость лактозы (ферментативная).
  - CMD (Центр молекулярной диагностики): пищевые аллергии
    (рыба 0.5–2%, орехи 0.5–2%, яйца 1–2% у детей / <1% у взрослых,
    глютен 0.3–1%).
  - ВЦИОМ (2025): репродуктивные планы (childQuant).
  - Оценка рынка веганской продукции: ~3–5% (isVegan).

Только стандартная библиотека: random, json.
"""

from __future__ import annotations

import json
import random
from typing import Any, Literal

from mcd_voice.profile.decision_graph import PROFILE_DECISION_GRAPH

# ── Типы ──────────────────────────────────────────────────────────────

Sex = Literal["male", "female"]
Language = Literal["EN", "RU"]
Psycho = Literal[
    "friendly",
    "impatient",
    "indecisive",
    "polite_and_respectful",
    "regular",
]

# ── Константы распределений ───────────────────────────────────────────

_SEX_WEIGHTS: dict[Sex, float] = {"male": 0.46, "female": 0.54}

_AGE_GROUPS: list[tuple[int, int, float]] = [
    (18, 30, 0.25),
    (31, 55, 0.50),
    (56, 80, 0.25),
]

_PSYCHO_WEIGHTS: list[tuple[Psycho, float]] = [
    ("regular", 0.30),
    ("friendly", 0.20),
    ("impatient", 0.20),
    ("polite_and_respectful", 0.15),
    ("indecisive", 0.15),
]

_LANG_WEIGHTS: dict[Language, float] = {"EN": 1.0, "RU": 0.0}

_CAL_BASE: dict[Sex, int] = {"male": 2200, "female": 1800}
_CAL_STD = 300
_CAL_MIN, _CAL_MAX = 800, 3500
_OVERWEIGHT_PROB = 0.40
_OVERWEIGHT_REDUCTION = 200

# Маргинальные доли для взрослого; совместное распределение задаётся в
# _sample_adult_dietary_flags (сначала ветка isVegan, затем условные броски).
_DIETARY_PROBS: dict[str, float] = {
    "noMilk":   0.61,
    "noFish":   0.02,
    "noNuts":   0.02,
    "noEggs":   0.01,
    "noGluten": 0.01,
    "noBeef":   0.04,
    "isVegan":  0.04,
    "noSugar":  0.08,
}

# У детей аллергии чаще: повышенные вероятности для разнообразия
_CHILD_DIETARY_PROBS: dict[str, float] = {
    "noMilk":   0.10,
    "noEggs":   0.06,
    "noNuts":   0.05,
    "noGluten": 0.03,
}

# Веса для ~50% solo: P(0 children) * P(0 friends) = 0.60 * 0.83 ≈ 0.50
_CHILD_WEIGHTS = [(0, 0.60), (1, 0.20), (2, 0.12), (3, 0.08)]
_FRIEND_WEIGHTS = [(0, 0.83), (1, 0.12), (2, 0.05)]

_CHILD_NAMES = ["child_1", "child_2", "child_3"]
_FRIEND_NAMES = ["friend_1", "friend_2"]

# ── Маппинг профильных флагов → токены аллергенов в mcd.json ─────────

_FLAG_TO_ALLERGEN: dict[str, str] = {
    "noMilk":   "Milk",
    "noFish":   "Fish",
    "noNuts":   "Nuts",
    "noEggs":   "Egg",
    "noGluten": "Cereal containing gluten",
}

_VEGAN_EXTRA_TOKENS = ("Milk", "Egg", "Fish")


# ── Генератор ─────────────────────────────────────────────────────────

class ProfileGenerator:
    """Генератор профилей с настраиваемым RNG."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random

    def generate(self) -> dict[str, Any]:
        """
        Случайный профиль с массивом companions.

        Логика семплирования эквивалентна обходу формального decision graph
        (Algorithm 2): на каждом шаге выбирается значение атрибута по заданным
        вероятностям, а итоговый профиль собирается из последовательности таких
        «бросков монетки». Граф экспортируется отдельно для документации.
        """
        r = self._rng

        sex: Sex = r.choices(
            list(_SEX_WEIGHTS.keys()), list(_SEX_WEIGHTS.values()),
        )[0]  # type: ignore[assignment]
        age = self._sample_age()
        psycho: Psycho = r.choices(
            [p for p, _ in _PSYCHO_WEIGHTS], [w for _, w in _PSYCHO_WEIGHTS],
        )[0]  # type: ignore[assignment]
        language: Language = r.choices(
            list(_LANG_WEIGHTS.keys()), list(_LANG_WEIGHTS.values()),
        )[0]  # type: ignore[assignment]

        cal = self._sample_calories(sex)

        flags = self._sample_adult_dietary_flags()

        child_quant = r.choices(
            [v for v, _ in _CHILD_WEIGHTS], [w for _, w in _CHILD_WEIGHTS],
        )[0]
        friends_quant = r.choices(
            [v for v, _ in _FRIEND_WEIGHTS], [w for _, w in _FRIEND_WEIGHTS],
        )[0]

        companions = self._generate_companions(child_quant, friends_quant)

        return {
            "sex": sex,
            "age": age,
            "psycho": psycho,
            "language": language,
            "calApprValue": cal,
            **flags,
            "childQuant": child_quant,
            "friendsQuant": friends_quant,
            "companions": companions,
        }

    def generate_via_graph(self) -> dict[str, Any]:
        """
        Альтернативная генерация через явный обход PROFILE_DECISION_GRAPH.

        Используется для верификации дипломной формализации (Algorithm 2):
        структура решений берётся из графа, а числовые сэмплы (возраст, калории)
        и ограничения — из тех же распределений, что и в generate().
        """
        r = self._rng
        node = PROFILE_DECISION_GRAPH

        sex = self._choice_from_outcomes(node.outcomes)  # male/female
        age_node = node.children[sex]
        age_bucket = self._choice_from_outcomes(age_node.outcomes)
        age = self._sample_age_from_bucket(age_bucket)

        psycho_node = age_node.children["next"]
        psycho = self._choice_from_outcomes(psycho_node.outcomes)

        language_node = psycho_node.children["next"]
        language = self._choice_from_outcomes(language_node.outcomes)

        # Узел калорий в графе документирует распределение, сам сэмпл считаем функцией.
        cal = self._sample_calories(sex)  # type: ignore[arg-type]

        adult_flags_node = language_node.children["next"].children["next"]
        vegan_branch = self._choice_from_outcomes(adult_flags_node.outcomes) == "isVegan=True"
        flags = self._sample_adult_dietary_flags(vegan_override=vegan_branch)

        companions_node = node.children["companions"]
        child_quant = self._choice_child_quant(companions_node.outcomes)
        friends_quant = self._choice_friends_quant(companions_node.outcomes)
        companions = self._generate_companions(child_quant, friends_quant)

        return {
            "sex": sex,
            "age": age,
            "psycho": psycho,
            "language": language,
            "calApprValue": cal,
            **flags,
            "childQuant": child_quant,
            "friendsQuant": friends_quant,
            "companions": companions,
        }

    def _sample_adult_dietary_flags(self, vegan_override: bool | None = None) -> dict[str, bool]:
        """
        Пищевые ограничения взрослого (или друга с тем же распределением).

        1) С вероятностью ``isVegan`` из ``_DIETARY_PROBS`` клиент — веган: тогда
           ``noMilk``, ``noFish``, ``noBeef``, ``noEggs`` всегда True (животные
           продукты исключены); ``noNuts``, ``noGluten``, ``noSugar`` семплируются
           **независимо** с теми же маргиналами — возможны веган + орехи и т.д.

        2) Иначе ``isVegan`` False; остальные семь флагов — **независимые** броски
           по своим вероятностям (как в статистике по отдельным ограничениям).

        Маргинальные доли совпадают с прежней схемой «все независимо + принудительная
        правка при вегане», но порядок семплирования отражает смысл полей.
        """
        r = self._rng
        p_vegan = _DIETARY_PROBS["isVegan"]
        flags: dict[str, bool] = {}
        is_vegan = (r.random() < p_vegan) if vegan_override is None else vegan_override
        if is_vegan:
            flags["isVegan"] = True
            flags["noMilk"] = True
            flags["noFish"] = True
            flags["noBeef"] = True
            flags["noEggs"] = True
            flags["noNuts"] = r.random() < _DIETARY_PROBS["noNuts"]
            flags["noGluten"] = r.random() < _DIETARY_PROBS["noGluten"]
            flags["noSugar"] = r.random() < _DIETARY_PROBS["noSugar"]
        else:
            flags["isVegan"] = False
            for key, prob in _DIETARY_PROBS.items():
                if key == "isVegan":
                    continue
                flags[key] = r.random() < prob
        return flags

    def _choice_from_outcomes(self, outcomes: dict[str, float]) -> str:
        keys = list(outcomes.keys())
        weights = list(outcomes.values())
        return self._rng.choices(keys, weights)[0]

    def _sample_age_from_bucket(self, bucket: str) -> int:
        bounds = {
            "18-30": (18, 30),
            "31-55": (31, 55),
            "56-80": (56, 80),
        }
        lo, hi = bounds.get(bucket, (18, 80))
        return self._rng.randint(lo, hi)

    def _choice_child_quant(self, outcomes: dict[str, float]) -> int:
        items = [(k, v) for k, v in outcomes.items() if k.startswith("childQuant=")]
        vals = [int(k.split("=", 1)[1]) for k, _ in items]
        ws = [w for _, w in items]
        return self._rng.choices(vals, ws)[0]

    def _choice_friends_quant(self, outcomes: dict[str, float]) -> int:
        items = [(k, v) for k, v in outcomes.items() if k.startswith("friendsQuant=")]
        vals = [int(k.split("=", 1)[1]) for k, _ in items]
        ws = [w for _, w in items]
        return self._rng.choices(vals, ws)[0]

    # ── Компаньоны ────────────────────────────────────────────────────

    def _generate_companions(
        self, child_quant: int, friends_quant: int,
    ) -> list[dict[str, Any]]:
        companions: list[dict[str, Any]] = []
        r = self._rng
        for i in range(child_quant):
            child_age = r.randint(3, 14)
            restrictions: dict[str, bool] = {}
            for key, prob in _CHILD_DIETARY_PROBS.items():
                restrictions[key] = r.random() < prob
            companions.append({
                "role": "child",
                "label": _CHILD_NAMES[i] if i < len(_CHILD_NAMES) else f"child_{i+1}",
                "age": child_age,
                "restrictions": restrictions,
            })
        for i in range(friends_quant):
            friend_flags = self._sample_adult_dietary_flags()
            companions.append({
                "role": "friend",
                "label": _FRIEND_NAMES[i] if i < len(_FRIEND_NAMES) else f"friend_{i+1}",
                "restrictions": friend_flags,
            })
        return companions

    # ── Семплирование ─────────────────────────────────────────────────

    def _sample_age(self) -> int:
        r = self._rng
        groups, weights = zip(*[(g, w) for *g, w in _AGE_GROUPS])
        lo, hi = r.choices(list(groups), list(weights))[0]
        return r.randint(lo, hi)

    def _sample_calories(self, sex: Sex) -> int:
        r = self._rng
        base = _CAL_BASE[sex]
        cal = int(round(r.gauss(base, _CAL_STD)))
        if r.random() < _OVERWEIGHT_PROB:
            cal -= r.randint(0, _OVERWEIGHT_REDUCTION)
        return max(_CAL_MIN, min(_CAL_MAX, cal))


# ── Функции уровня модуля ─────────────────────────────────────────────

_default_generator = ProfileGenerator()


def generate_profile() -> dict[str, Any]:
    return _default_generator.generate()


def profile_to_json(profile: dict[str, Any]) -> str:
    return json.dumps(profile, ensure_ascii=False, indent=2)


def get_allergen_blacklist(profile_or_flags: dict[str, Any]) -> list[str]:
    """
    Флаги → токены аллергенов.
    Принимает полный профиль ИЛИ dict ограничений компаньона (restrictions).
    """
    tokens: list[str] = []
    for flag, token in _FLAG_TO_ALLERGEN.items():
        if profile_or_flags.get(flag):
            if token not in tokens:
                tokens.append(token)
    if profile_or_flags.get("isVegan"):
        for t in _VEGAN_EXTRA_TOKENS:
            if t not in tokens:
                tokens.append(t)
    return tokens


def get_group_allergen_blacklist(profile: dict[str, Any]) -> list[str]:
    """
    Объединённый blacklist для всей группы (клиент + все companions).
    Используется при RAG-запросе, чтобы не предложить ничего опасного
    ни для кого из компании.
    """
    tokens = set(get_allergen_blacklist(profile))
    for comp in profile.get("companions", []):
        restr = comp.get("restrictions", {})
        tokens.update(get_allergen_blacklist(restr))
    return sorted(tokens)


def _describe_restrictions(flags: dict[str, Any]) -> str:
    """Краткое описание ограничений (для клиента или компаньона)."""
    parts: list[str] = []
    if flags.get("isVegan"):
        parts.append("vegan")
    else:
        if flags.get("noMilk"):
            parts.append("lactose intolerant")
        if flags.get("noEggs"):
            parts.append("egg allergy")
        if flags.get("noFish"):
            parts.append("fish allergy")
        if flags.get("noNuts"):
            parts.append("nut allergy")
        if flags.get("noGluten"):
            parts.append("gluten intolerant")
        if flags.get("noBeef"):
            parts.append("avoids beef")
    if flags.get("noSugar"):
        parts.append("avoids sugar")
    return ", ".join(parts) if parts else "no restrictions"


def generate_text_description(profile: dict[str, Any]) -> str:
    """
    Человекочитаемое описание всей группы для system prompt LLM.

    Включает данные каждого компаньона (возраст, ограничения).
    """
    age = profile.get("age", 30)
    sex = profile.get("sex", "male")
    psycho = profile.get("psycho", "regular")
    cal = profile.get("calApprValue", 2000)

    parts: list[str] = [
        f"{age}-year-old {sex}, {psycho} personality.",
        f"Dietary: {_describe_restrictions(profile)}.",
    ]

    companions = profile.get("companions", [])
    if companions:
        group_size = 1 + len(companions)
        parts.append(f"Ordering for a group of {group_size}:")
        parts.append(f"  - Self ({_describe_restrictions(profile)}).")
        for comp in companions:
            role = comp["role"]
            label = comp["label"]
            restr = _describe_restrictions(comp.get("restrictions", {}))
            if role == "child":
                child_age = comp.get("age", "?")
                parts.append(f"  - {label} (child, age {child_age}, {restr}).")
            else:
                parts.append(f"  - {label} (friend, {restr}).")
    else:
        parts.append("Ordering alone.")

    parts.append(f"Calorie target: ~{cal} kcal/meal.")
    return "\n".join(parts)


if __name__ == "__main__":
    print("Пример: 5 случайных профилей\n")
    for i in range(1, 6):
        p = generate_profile()
        print(f"--- Профиль {i} ---")
        print(profile_to_json(p))
        print(f"  blacklist (self): {get_allergen_blacklist(p)}")
        print(f"  blacklist (group): {get_group_allergen_blacklist(p)}")
        print(f"  description:\n{generate_text_description(p)}")
        print()
