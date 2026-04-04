"""Каталог блюд из mcd.json (имена и калории для эвристики заказа)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from mcd_voice.config import MCD_JSON_PATH


class MenuCatalog:
    """Загрузка меню с диска; используется конвейером диалога."""

    def __init__(self, json_path: Path | None = None) -> None:
        self._path = json_path or MCD_JSON_PATH

    def load(self) -> tuple[list[str], dict[str, float]]:
        """
        Возвращает:
          - уникальный список названий (порядок первого вхождения);
          - словарь name → средняя калорийность по всем размерам.

        mcd.json содержит несколько записей с одинаковым именем (разные размеры
        порций). Дедупликация имён предотвращает многократный матчинг одного
        блюда в parse_order_from_text; усреднение даёт нейтральную оценку.
        """
        with open(self._path, "r", encoding="utf-8") as f:
            items: list[dict[str, Any]] = json.load(f)

        seen_names: list[str] = []
        energy_acc: dict[str, list[float]] = defaultdict(list)

        for it in items:
            name = it.get("name")
            if not name:
                continue
            if name not in energy_acc:
                seen_names.append(name)
            try:
                energy_acc[name].append(float(it.get("energy") or 0))
            except (TypeError, ValueError):
                energy_acc[name].append(0.0)

        energy = {
            name: round(sum(vals) / len(vals), 2)
            for name, vals in energy_acc.items()
        }
        return seen_names, energy
