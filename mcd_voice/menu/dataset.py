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

        meta: dict[str, Any] = {
            "name": item["name"],
            "energy": item.get("energy", 0),
            "allergens": allergens_chroma,
            "category": item["category"] if "category" in item else "",
        }
        ids.append(str(idx))
        documents.append(doc_text)
        metadatas.append(meta)

    return ids, documents, metadatas
