"""Сборка записей меню из mcd.json в формат для Chroma (ids, documents, metadatas)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcd_voice.config import MCD_JSON_PATH
from mcd_voice.menu.parsing import (
    allergens_for_chroma_metadata,
    build_embedding_text,
    parse_allergy_field,
)


def load_menu_from_json(
    json_path: str | Path | None = None,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """
    Читает JSON меню и возвращает три списка для collection.add:
    ids, documents, metadatas.
    """
    path = Path(json_path) if json_path is not None else MCD_JSON_PATH
    with open(path, "r", encoding="utf-8") as f:
        menu_items: list[dict[str, Any]] = json.load(f)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for idx, item in enumerate(menu_items):
        doc_text = build_embedding_text(item)
        allergens_list = parse_allergy_field(item.get("allergy"))
        allergens_chroma = allergens_for_chroma_metadata(allergens_list)

        def _f(key: str) -> float:
            """Числовое поле → float, None/пустое → 0.0."""
            v = item.get(key)
            try:
                return float(v) if v is not None else 0.0
            except (TypeError, ValueError):
                return 0.0

        def _s(key: str) -> str:
            """Строковое поле → str, None/отсутствие → ''."""
            v = item.get(key)
            if v is None:
                return ""
            return str(v).strip()

        meta: dict[str, Any] = {
            # идентификация
            "name":         item.get("name") or "",
            "category":     _s("category"),
            "serving_size": _s("serving_size"),
            "tag":          _s("tag"),
            "description":  _s("description"),
            "ingredients":  _s("ingredients"),
            # аллергены (список строк для фильтрации через $not_contains)
            "allergens":    allergens_chroma,
            # нутриенты (все числовые поля из mcd.json)
            "energy":       _f("energy"),
            "protein":      _f("protein"),
            "total_fat":    _f("total_fat"),
            "sat_fat":      _f("sat_fat"),
            "trans_fat":    _f("trans_fat"),
            "chol":         _f("chol"),
            "carbs":        _f("carbs"),
            "total_sugar":  _f("total_sugar"),
            "added_sugar":  _f("added_sugar"),
            "sodium":       _f("sodium"),
        }
        ids.append(str(idx))
        documents.append(doc_text)
        metadatas.append(meta)

    return ids, documents, metadatas
