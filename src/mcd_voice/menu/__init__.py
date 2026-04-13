"""Меню: парсинг JSON, загрузка в Chroma, семантический поиск.

Без жадных импортов — иначе `python -m mcd_voice.menu.chroma` даёт RuntimeWarning.
"""

from __future__ import annotations

__all__ = [
    "configure_hf_cache",
    "ingest_menu_clear_existing",
    "load_menu_from_json",
    "search_menu",
    "search_menu_graph",
]


def __getattr__(name: str):
    if name == "configure_hf_cache":
        from mcd_voice.menu.chroma import configure_hf_cache

        return configure_hf_cache
    if name == "ingest_menu_clear_existing":
        from mcd_voice.menu.chroma import ingest_menu_clear_existing

        return ingest_menu_clear_existing
    if name == "load_menu_from_json":
        from mcd_voice.menu.dataset import load_menu_from_json

        return load_menu_from_json
    if name == "search_menu":
        from mcd_voice.menu.search import search_menu

        return search_menu
    if name == "search_menu_graph":
        from mcd_voice.menu.graph_rag import search_menu_graph

        return search_menu_graph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
