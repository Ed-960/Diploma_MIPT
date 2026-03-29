"""
Демо конвейера диалога.
Запуск: python scripts/dialog_demo.py

Требуется OPENAI_API_KEY и загруженная Chroma (scripts/load_chroma.py).
"""

import _bootstrap

_bootstrap.ensure_src()

from mcd_voice.dialog import simulate_dialog, print_dialog

if __name__ == "__main__":
    try:
        h, p, o, f = simulate_dialog(max_turns=10)
        print_dialog(h, p, o, f)
    except RuntimeError as e:
        print(f"Ошибка: {e}")
        print(
            "Проверьте OPENAI_API_KEY и наличие chroma_db после load_chroma.py."
        )
