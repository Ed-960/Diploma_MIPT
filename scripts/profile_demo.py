"""Демо REG. Запуск: python scripts/profile_demo.py"""

import _bootstrap

_bootstrap.ensure_src()

from mcd_voice.profile import (
    generate_profile,
    generate_text_description,
    get_allergen_blacklist,
    profile_to_json,
)

if __name__ == "__main__":
    print("Пример: 10 случайных профилей\n")
    for i in range(1, 11):
        p = generate_profile()
        print(f"--- Профиль {i} ---")
        print(profile_to_json(p))
        bl = get_allergen_blacklist(p)
        print(f"  Blacklist (для RAG): {bl}")
        print(f"  Описание: {generate_text_description(p)}")
        print()
