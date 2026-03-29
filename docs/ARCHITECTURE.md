# Архитектура репозитория

Код приложения — только в **`src/mcd_voice/`**. В корне — данные, конфиги, скрипты запуска, тесты.

```
diploma_project/
├── README.md
├── PROJECT_CONTEXT.md
├── pyproject.toml              # зависимости + pytest (pythonpath=src)
├── requirements.txt
├── mcd.json                    # данные меню (42 позиции)
├── chroma_db/                  # после scripts/load_chroma.py
├── src/
│   └── mcd_voice/              # ← весь прикладной Python
│       ├── config.py           # пути, константы
│       ├── menu/               # RAG: Chroma, поиск, фильтры
│       │   ├── chroma.py       # PersistentClient, загрузка
│       │   ├── dataset.py      # JSON → текст для эмбеддингов
│       │   ├── parsing.py      # парсинг аллергенов
│       │   ├── search.py       # search_menu(), build_where()
│       │   └── search_checks.py
│       ├── profile/            # REG: генератор профилей
│       │   └── generator.py    # ProfileGenerator, companions
│       ├── llm/                # LLM-агенты + промпты
│       │   ├── agent.py        # ClientAgent, CashierAgent
│       │   └── prompts.py      # system prompts (психотип, группа)
│       └── dialog/             # конвейер диалога
│           ├── catalog.py      # MenuCatalog (имена + калории)
│           ├── pipeline.py     # DialogPipeline, validate_dialog
│           ├── save_dialog.py  # save/load JSON, aggregate_stats
│           └── allergens.py    # совместимость
├── scripts/                    # точки входа (не библиотека)
│   ├── _bootstrap.py           # sys.path без pip install
│   ├── load_chroma.py          # загрузка mcd.json → Chroma
│   ├── test_menu_search.py     # ручные проверки RAG
│   ├── menu_search_demo.py     # демо поиска
│   ├── profile_demo.py         # демо REG
│   ├── agents_demo.py          # демо LLM-агентов
│   ├── dialog_demo.py          # один диалог (API)
│   ├── generate_dataset.py     # массовая генерация (RAG/non-RAG)
│   └── compare_rag.py          # сравнение RAG vs non-RAG
└── tests/
    ├── test_search.py           # RAG: семантика, фильтры
    ├── test_profile_generator.py # REG: профили, blacklist
    ├── test_pipeline_helpers.py  # парсинг, персоны, валидация
    └── test_save_dialog.py      # save/load/aggregate JSON
```

## Слои внутри `mcd_voice`

| Подпакет | Роль |
|----------|------|
| `menu` | JSON → эмбеддинги → Chroma → `search_menu` с фильтрами |
| `profile` | `ProfileGenerator`, companions (дети/друзья с ограничениями) |
| `llm` | `ClientAgent`, `CashierAgent` с адаптацией к психотипу и RAG |
| `dialog` | `DialogPipeline`, multi-person order, валидация, сохранение JSON |

## Импорты

После `pip install -e .` из любого места:

```python
from mcd_voice.menu.search import search_menu
from mcd_voice.profile import generate_profile, get_group_allergen_blacklist
from mcd_voice.llm import ClientAgent, CashierAgent
from mcd_voice.dialog import DialogPipeline, simulate_dialog, aggregate_stats
```
