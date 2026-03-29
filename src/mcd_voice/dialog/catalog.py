"""Каталог блюд из mcd.json (имена и калории для эвристики заказа)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcd_voice.config import MCD_JSON_PATH


class MenuCatalog:
    """Загрузка меню с диска; используется конвейером диалога."""

    def __init__(self, json_path: Path | None = None) -> None:
        self._path = json_path or MCD_JSON_PATH

    def load(self) -> tuple[list[str], dict[str, float]]:
        with open(self._path, "r", encoding="utf-8") as f:
            items: list[dict[str, Any]] = json.load(f)
        names = [it["name"] for it in items]
        energy = {it["name"]: float(it.get("energy", 0)) for it in items}
        return names, energy
