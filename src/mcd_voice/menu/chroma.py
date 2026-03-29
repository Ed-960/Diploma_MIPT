"""
Подключение к Chroma, кэш Hugging Face, загрузка коллекции menu.

Зависимости:
    pip install chromadb sentence-transformers
  или: pip install -r requirements.txt
"""

from __future__ import annotations

import os
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions

from mcd_voice.config import (
    CHROMA_DIR,
    CHROMA_METADATA_COSINE,
    COLLECTION_MENU,
    EMBEDDING_MODEL_ID,
    HF_CACHE_DIR,
)
from mcd_voice.menu.dataset import load_menu_from_json
from mcd_voice.menu.parsing import allergens_meta_to_display


def configure_hf_cache() -> None:
    """Кэш моделей HF в каталоге проекта (до создания SentenceTransformer)."""
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
    os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR))


def _cuda_runtime_info() -> str:
    """Короткая диагностика CUDA для логов."""
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return "CUDA: unavailable"
        name = torch.cuda.get_device_name(0)
        count = torch.cuda.device_count()
        return f"CUDA: available ({count} device(s), first={name})"
    except Exception:
        return "CUDA: unknown (torch diagnostics unavailable)"


def resolve_embedding_device() -> str:
    """
    Определяет девайс для SentenceTransformerEmbeddingFunction.

    ENV:
      EMBEDDING_DEVICE=auto|cuda|cpu (default: auto)
    """
    raw = os.environ.get("EMBEDDING_DEVICE", "auto").strip().lower()
    if raw not in {"auto", "cuda", "cpu"}:
        print(
            f"  → EMBEDDING_DEVICE={raw!r} не распознан, используем auto.",
            flush=True,
        )
        raw = "auto"

    if raw == "cpu":
        return "cpu"
    if raw == "cuda":
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                return "cuda"
            print(
                "  → Запрошен EMBEDDING_DEVICE=cuda, но CUDA недоступна; fallback на cpu.",
                flush=True,
            )
            return "cpu"
        except Exception:
            print(
                "  → Запрошен EMBEDDING_DEVICE=cuda, но torch/CUDA не обнаружены; fallback на cpu.",
                flush=True,
            )
            return "cpu"

    # auto
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def get_embedding_function(
    *, device: str | None = None
) -> embedding_functions.SentenceTransformerEmbeddingFunction:
    selected = device or resolve_embedding_device()
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_ID,
        device=selected,
    )


def get_menu_collection() -> Collection:
    """Коллекция menu с той же embedding function, что при индексации."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    ef = get_embedding_function()
    return client.get_collection(name=COLLECTION_MENU, embedding_function=ef)


def get_or_create_menu_collection() -> Collection:
    """Создание коллекции при первой загрузке меню."""
    configure_hf_cache()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    device = resolve_embedding_device()
    print(
        f"  → EMBEDDING_DEVICE={os.environ.get('EMBEDDING_DEVICE', 'auto')} -> фактически: {device}.",
        flush=True,
    )
    print(f"  → {_cuda_runtime_info()}", flush=True)
    print(
        "  → Загрузка SentenceTransformer (MiniLM) в память… "
        "Часто это самый долгий шаг: при первом запуске ещё и скачивание в "
        f"{HF_CACHE_DIR} — терпение, консоль может не печатать прогресс.",
        flush=True,
    )
    ef = get_embedding_function(device=device)
    print("  → MiniLM загружен.", flush=True)
    return client.get_or_create_collection(
        name=COLLECTION_MENU,
        embedding_function=ef,
        metadata=CHROMA_METADATA_COSINE,
    )


def ingest_menu_clear_existing(
    bundle: tuple[list[str], list[str], list[dict[str, Any]]] | None = None,
) -> int:
    """
    Перезаписывает коллекцию: удаляет старые id, добавляет данные из mcd.json
    (или переданный bundle из load_menu_from_json).
    Возвращает число записей в коллекции.
    """
    print(
        "Инициализация Chroma и модели эмбеддингов…",
        flush=True,
    )
    configure_hf_cache()
    collection = get_or_create_menu_collection()
    ids, documents, metadatas = (
        bundle if bundle is not None else load_menu_from_json()
    )
    print(
        f"Индексация {len(ids)} позиций (эмбеддинг батчами в Chroma)…",
        flush=True,
    )
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
    batch_size = int(os.environ.get("EMBEDDING_BATCH_SIZE", "16"))
    if batch_size <= 0:
        batch_size = 16
    total = len(ids)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"  → Индексация: {end}/{total}", flush=True)
    print("Индексация завершена.", flush=True)
    return collection.count()


def main(*, run_demo: bool = True) -> None:
    bundle = load_menu_from_json()
    print(f"Загружено позиций меню: {len(bundle[0])}")
    count = ingest_menu_clear_existing(bundle)
    print(f"Данные загружены в коллекцию '{COLLECTION_MENU}'. Всего записей: {count}")

    if not run_demo:
        return

    collection = get_menu_collection()
    print("\n=== Пример поиска (query + where) ===")
    query = "что есть из курицы без молока"
    print("(демо-запрос к индексу, ещё один проход эмбеддинга для текста запроса)…")
    results = collection.query(
        query_texts=[query],
        n_results=3,
        where={"allergens": {"$not_contains": "Milk"}},
    )
    for i, (meta, dist) in enumerate(
        zip(results["metadatas"][0], results["distances"][0]),
        start=1,
    ):
        name = meta.get("name", "?")
        energy = meta.get("energy", "")
        ag = allergens_meta_to_display(meta.get("allergens"))
        print(f"\n{i}. {name}")
        print(f"   Калории: {energy} ккал")
        print(f"   Аллергены: {ag}")
        print(f"   (distance: {dist:.4f})")
    print("\nГотово.")


if __name__ == "__main__":
    main()
