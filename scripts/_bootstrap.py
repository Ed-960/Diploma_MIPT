"""Подключает `src/` к sys.path, если пакет не установлен (`pip install -e .`)."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_src() -> None:
    root = Path(__file__).resolve().parent.parent
    src = root / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
