"""Пакет дипломного проекта: RAG по меню McDonald's, далее — диалоги и валидация.

Импорты ленивые (PEP 562), чтобы `python -m mcd_voice.menu.chroma` не подгружал
поиск до запуска нужного модуля и не вызывал RuntimeWarning.
"""

from __future__ import annotations

__all__ = ["load_menu_from_json", "search_menu"]


def __getattr__(name: str):
    if name == "load_menu_from_json":
        from mcd_voice.menu.dataset import load_menu_from_json

        return load_menu_from_json
    if name == "search_menu":
        from mcd_voice.menu.search import search_menu

        return search_menu
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
