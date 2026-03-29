"""Bootstrap для скриптов: `src/` в sys.path + автозагрузка `.env`."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _parse_env_line(line: str) -> tuple[str, str] | None:
    raw = line.strip()
    if not raw or raw.startswith("#"):
        return None
    if raw.startswith("export "):
        raw = raw[len("export "):].strip()
    if "=" not in raw:
        return None
    key, value = raw.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None
    # Поддержка простых quoted значений.
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    return key, value


def load_dotenv(dotenv_path: Path, *, override: bool = False) -> None:
    """
    Загружает переменные окружения из .env без внешних зависимостей.
    По умолчанию НЕ перезаписывает уже выставленные переменные.
    """
    if not dotenv_path.exists():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(line)
        if not parsed:
            continue
        key, value = parsed
        if key in os.environ and not override:
            continue
        os.environ[key] = value


def ensure_src() -> None:
    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env")
    src = root / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
