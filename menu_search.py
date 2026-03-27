"""
Обратная совместимость: семантический поиск по меню.

Предпочтительно: `from mcd_voice.menu.search import search_menu`
или `from mcd_voice import search_menu`.
"""

from mcd_voice.menu.chroma import configure_hf_cache
from mcd_voice.menu.search import search_menu

__all__ = ["search_menu"]

if __name__ == "__main__":
    configure_hf_cache()
    rows = search_menu(
        "chicken without milk",
        allergens_blacklist=["Milk"],
        top_k=5,
    )
    for i, row in enumerate(rows, start=1):
        print(f"{i}. {row['name']}")
        print(f"   energy: {row['energy']}, distance: {row['distance']:.4f}")
        print(f"   allergens: {row['allergens']}")
