"""Демо search_menu. Запуск: python scripts/menu_search_demo.py"""

import _bootstrap

_bootstrap.ensure_src()

from mcd_voice.menu.chroma import configure_hf_cache
from mcd_voice.menu.search import search_menu

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
