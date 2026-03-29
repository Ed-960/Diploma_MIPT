"""Конвейер диалога (клиент + кассир + RAG + валидация + сохранение)."""

from mcd_voice.dialog.allergens import profile_to_allergen_blacklist
from mcd_voice.dialog.catalog import MenuCatalog
from mcd_voice.dialog.pipeline import (
    DialogPipeline,
    build_initial_order_state,
    generate_dialog,
    parse_order_from_text,
    print_dialog,
    simulate_dialog,
    validate_dialog,
)
from mcd_voice.dialog.save_dialog import (
    aggregate_stats,
    load_all_dialogs,
    load_dialog,
    save_dialog,
)

__all__ = [
    "DialogPipeline",
    "MenuCatalog",
    "build_initial_order_state",
    "generate_dialog",
    "load_all_dialogs",
    "load_dialog",
    "parse_order_from_text",
    "print_dialog",
    "profile_to_allergen_blacklist",
    "save_dialog",
    "aggregate_stats",
    "simulate_dialog",
    "validate_dialog",
]
