"""
Сохранение, загрузка и агрегация диалогов — только JSON.

save_dialog       → dialogs/dialog_0001.json
load_dialog       ← dialogs/dialog_0001.json
load_all_dialogs  ← все dialog_*.json
aggregate_stats   → dialogs/summary.json
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_TM_RE = re.compile(r"[®™℠]")


def _strip_trademarks(history: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return a copy of history with ®/™/℠ removed from spoken text."""
    return [
        {**turn, "text": _TM_RE.sub("", turn["text"])}
        for turn in history
    ]


def save_dialog(
    dialog_id: int,
    profile: dict[str, Any],
    history: list[dict[str, str]],
    order_state: dict[str, Any],
    flags: dict[str, Any],
    output_dir: str | Path = "dialogs",
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    record: dict[str, Any] = {
        "dialog_id": dialog_id,
        "profile": profile,
        "history": _strip_trademarks(history),
        "final_order": order_state,
        "validation_flags": flags,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    path = out / f"dialog_{dialog_id:04d}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return path


def load_dialog(
    dialog_id: int,
    input_dir: str | Path = "dialogs",
) -> dict[str, Any]:
    path = Path(input_dir) / f"dialog_{dialog_id:04d}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_dialogs(input_dir: str | Path = "dialogs") -> list[dict[str, Any]]:
    d = Path(input_dir)
    if not d.is_dir():
        return []
    files = sorted(d.glob("dialog_*.json"))
    out: list[dict[str, Any]] = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            out.append(json.load(f))
    return out


# ── JSON-сводка ──────────────────────────────────────────────────────

def _summarize_record(rec: dict[str, Any]) -> dict[str, Any]:
    """Извлекает плоскую сводку из одного dialog JSON."""
    profile = rec.get("profile", {})
    flags = rec.get("validation_flags", {})
    companions = profile.get("companions", [])

    final_order = rec.get("final_order", {})
    return {
        "dialog_id": rec.get("dialog_id"),
        "sex": profile.get("sex"),
        "age": profile.get("age"),
        "psycho": profile.get("psycho"),
        "language": profile.get("language"),
        "calApprValue": profile.get("calApprValue"),
        "isVegan": profile.get("isVegan", False),
        "noMilk": profile.get("noMilk", False),
        "noFish": profile.get("noFish", False),
        "noNuts": profile.get("noNuts", False),
        "noEggs": profile.get("noEggs", False),
        "noGluten": profile.get("noGluten", False),
        "noBeef": profile.get("noBeef", False),
        "noSugar": profile.get("noSugar", False),
        "childQuant": profile.get("childQuant", 0),
        "friendsQuant": profile.get("friendsQuant", 0),
        "group_size": 1 + len(companions),
        "order_complete": final_order.get("order_complete", False),
        "turns": flags.get("turns", 0),
        "total_items": flags.get("total_items", 0),
        "total_energy": flags.get("total_energy", 0),
        "calorie_target": flags.get("calorie_target", profile.get("calApprValue")),
        "calorie_warning": flags.get("calorie_warning", False),
        "allergen_violation": flags.get("allergen_violation", []),
        "empty_order": flags.get("empty_order", False),
        "per_person": flags.get("per_person", []),
    }


def aggregate_stats(
    input_dir: str | Path = "dialogs",
    output_file: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    Проходит по всем dialog_*.json, собирает сводку в список словарей.
    Если output_file задан — сохраняет в summary.json.
    Возвращает список сводок.
    """
    dialogs = load_all_dialogs(input_dir)
    summaries = [_summarize_record(d) for d in dialogs]

    if output_file is None:
        output_file = Path(input_dir) / "summary.json"
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    return summaries
