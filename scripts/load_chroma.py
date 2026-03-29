"""Загрузка mcd.json в Chroma. Запуск: python scripts/load_chroma.py (из корня репо)."""

import _bootstrap

_bootstrap.ensure_src()

from mcd_voice.menu.chroma import main

if __name__ == "__main__":
    main()
