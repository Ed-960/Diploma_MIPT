"""Пути и константы проекта (единая точка конфигурации)."""

from __future__ import annotations

from pathlib import Path

# Корень репозитория: .../src/mcd_voice/config.py → вверх на два уровня
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

CHROMA_DIR = PROJECT_ROOT / "chroma_db"
MCD_JSON_PATH = PROJECT_ROOT / "mcd.json"
HF_CACHE_DIR = PROJECT_ROOT / ".cache" / "huggingface"

# Эмбеддинги
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Chroma
COLLECTION_MENU = "menu"
CHROMA_METADATA_COSINE = {"hnsw:space": "cosine"}

# Меню / метаданные
NO_ALLERGEN_SENTINEL = "__none__"
