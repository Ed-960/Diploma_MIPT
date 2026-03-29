"""Загрузка mcd.json в Chroma. Запуск: python scripts/load_chroma.py (из корня репо)."""

from __future__ import annotations

import argparse

import _bootstrap

_bootstrap.ensure_src()

from mcd_voice.menu.chroma import main


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Загрузить mcd.json в Chroma (эмбеддинги MiniLM).",
    )
    p.add_argument(
        "--no-demo",
        action="store_true",
        help="Не выполнять демо-поиск в конце (быстрее выход после индексации).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(run_demo=not args.no_demo)
