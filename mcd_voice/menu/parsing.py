"""Парсинг полей меню из JSON и подготовка текстов для эмбеддингов."""

from __future__ import annotations

from typing import Any

from mcd_voice.config import NO_ALLERGEN_SENTINEL


def parse_allergy_field(allergy_raw: Any) -> list[str]:
    """
    Разбор поля allergy: список аллергенов без «No Allergens», без лишних пробелов.
    """
    if allergy_raw is None:
        return []
    text = str(allergy_raw).strip()
    if not text or text == "No Allergens":
        return []
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return [p for p in parts if p != "No Allergens"]


def build_embedding_text(item: dict[str, Any]) -> str:
    """Текст для эмбеддинга: name, затем при наличии description, ingredients, tag."""
    parts: list[str] = [item["name"]]
    desc = item.get("description")
    if desc:
        parts.append(str(desc).strip())
    ing = item.get("ingredients")
    if ing and str(ing).strip():
        parts.append(str(ing).strip())
    tag = item.get("tag")
    if tag and str(tag).strip():
        parts.append(str(tag).strip())
    return ". ".join(parts)


def allergens_for_chroma_metadata(allergens_list: list[str]) -> list[str]:
    """Список для поля metadata['allergens']; пустой список в Chroma недопустим."""
    return allergens_list if allergens_list else [NO_ALLERGEN_SENTINEL]


def allergens_meta_to_display(allergens_val: Any) -> str:
    """Строка для вывода из значения метаданных Chroma."""
    if isinstance(allergens_val, list):
        parts = [x for x in allergens_val if x != NO_ALLERGEN_SENTINEL]
        return ", ".join(parts) if parts else "нет"
    if isinstance(allergens_val, str) and allergens_val.strip():
        return allergens_val.strip()
    return "нет"


def allergens_meta_to_list(allergens_val: Any) -> list[str]:
    """Нормализация allergens из ответа Chroma в список строк."""
    if isinstance(allergens_val, list):
        return [x for x in allergens_val if x != NO_ALLERGEN_SENTINEL]
    if isinstance(allergens_val, str) and allergens_val.strip():
        return [p.strip() for p in allergens_val.split(",") if p.strip()]
    return []
