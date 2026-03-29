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


def get_embedding_function() -> embedding_functions.SentenceTransformerEmbeddingFunction:
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_ID,
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
    ef = get_embedding_function()
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
    configure_hf_cache()
    collection = get_or_create_menu_collection()
    ids, documents, metadatas = (
        bundle if bundle is not None else load_menu_from_json()
    )
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return collection.count()


def main() -> None:
    bundle = load_menu_from_json()
    print(f"Загружено позиций меню: {len(bundle[0])}")
    count = ingest_menu_clear_existing(bundle)
    print(f"Данные загружены в коллекцию '{COLLECTION_MENU}'. Всего записей: {count}")

    collection = get_menu_collection()
    print("\n=== Пример поиска (query + where) ===")
    query = "что есть из курицы без молока"
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


if __name__ == "__main__":
    main()
