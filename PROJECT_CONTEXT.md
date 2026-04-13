# Проект: Симулятор диалогов для голосового ассистента ресторана быстрого питания

Документ — **единый контекст** для разработки и для ИИ (Cursor): архитектура диплома, что уже в коде, что планируется.

## Цель проекта

Разработка голосового ассистента для ресторанов быстрого питания (на примере McDonald's) с персонализированными диалогами на английском без экрана. Комбинация **RAG** (меню) и **REG** (профили клиентов), генерация множества синтетических диалогов (целевой масштаб — порядка 3000), анализ ошибок и итерация по промптам.

---

## Структура пакета (Python)

Весь прикладной код — в **`src/mcd_voice/`** (слои: **menu** → RAG, **profile** → REG, **llm** → агенты, **dialog** → конвейер). Запуск сценариев — **`scripts/`** (не смешивать с библиотекой). Дерево: **`docs/ARCHITECTURE.md`**.

| Путь | Назначение |
|------|------------|
| `src/mcd_voice/config.py` | Пути, константы Chroma/эмбеддингов. |
| `src/mcd_voice/menu/` | Парсинг меню, Chroma, `search_menu`, `build_where`, тестовые проверки. |
| `src/mcd_voice/profile/` | `ProfileGenerator`, `generate_profile()`, companions — REG. |
| `src/mcd_voice/llm/` | `ClientAgent`, `CashierAgent`, системные промпты (психотип, группа). |
| `src/mcd_voice/dialog/` | `DialogPipeline`, multi-person order_state, валидация per-person, сохранение/загрузка JSON, `aggregate_stats`. |
| `scripts/load_chroma.py` | Загрузка меню в Chroma. |
| `scripts/generate_dataset.py` | Массовая генерация диалогов (RAG/non-RAG). |
| `scripts/compare_rag.py` | Сравнение RAG vs non-RAG по summary.json. |
| `scripts/profile_demo.py`, `agents_demo.py`, `dialog_demo.py` | Демо REG / LLM / диалог. |
| `scripts/test_menu_search.py` | Ручные проверки + `--check` / `--demo`. |
| `tests/` | pytest (55 тестов): RAG, REG, парсинг, валидация, save/load. |

Импорт (после `pip install -e .`): `from mcd_voice.dialog import simulate_dialog`, `from mcd_voice.profile import generate_profile` и т.д.

## Реализованные артефакты (данные и окружение)

| Файл | Назначение |
|------|------------|
| `mcd.json` | Меню: **42** позиции. |
| `requirements.txt` | `chromadb`, `sentence-transformers`, `openai`. |
| `chroma_db/` | Persistent-БД Chroma (генерируется `scripts/load_chroma.py`). |
| `.cache/huggingface/` | Кэш модели эмбеддингов (в проекте, чтобы не зависеть от `~/.cache`). |

### RAG / меню

- **Текст документа для эмбеддинга:** `name` + `description`, `ingredients`, `tag` (пустые поля пропускаются).
- **Модель эмбеддингов:** `sentence-transformers/all-MiniLM-L6-v2` (384-мерные векторы).
- **Chroma:** `PersistentClient`, каталог `./chroma_db`, коллекция `menu`, `metadata={"hnsw:space": "cosine"}`.
- **Метаданные `allergens`:** список строк. Пустой → маркер `__none__`. Фильтр `$not_contains`.
- **Distance threshold:** `0.60` — если ближайший результат дальше, кассир отвечает «нет в меню».

---

## Архитектура (реализовано)

### 1. Меню (RAG)

Семантический поиск с фильтрами: аллергены (blacklist), энергетическая ценность (min/max). Distance threshold для fallback.

### 2. Генератор профилей (REG)

Стохастический граф на основе российской статистики (Росстат 2024, ВЦИОМ 2025, CMD):
- **sex**: male 46%, female 54%.
- **age**: young (18–30) 25%, middle (31–55) 50%, senior (56–80) 25%.
- **psycho**: regular 50%, friendly 15%, impatient 15%, polite_and_respectful 10%, indecisive 10%.
- **language**: RU 90%, EN 10%.
- **calApprValue**: N(2200, 300) муж. / N(1800, 300) жен.; overweight-коррекция.
- **Пищевые ограничения**: noMilk 61%, noSugar 8%, noBeef 4%, isVegan 4%, noFish 2%, noNuts 2%, noEggs 1%, noGluten 1%.
- **childQuant/friendsQuant**: распределения ВЦИОМ.
- **companions**: массив детей (возраст 3–14, свои ограничения с повышенными вероятностями: noMilk 5%, noEggs 3%, noNuts 2%, noGluten 1%) и друзей (полный набор ограничений).

Функции: `generate_profile()`, `get_allergen_blacklist()`, `get_group_allergen_blacklist()`, `generate_text_description()`.

### 3. LLM-агенты

- **ClientAgent**: знает профиль, companions, заказывает за всю группу.
- **CashierAgent**: адаптируется к психотипу клиента, знает состав группы (дети + ограничения), поочерёдно опрашивает, использует RAG с distance threshold, fallback «нет в меню».
- Полная история диалога передаётся обоим агентам на каждом ходу.

### 4. Конвейер диалога (multi-person)

- **order_state**: массив `persons` (self + companions), каждый со своими items, energy, allergens.
- **Парсинг количеств**: `"3 Big Mac"` → quantity=3.
- **Детекция персоны**: `"for my wife"` → spouse, `"for the oldest"` → child_oldest.
- **Эвристики завершения**: regex-паттерны для кассира и клиента.
- Кассир начинает диалог (drive-through greeting).

### 5. Валидация (per-person)

- `allergen_violation`: проверка blacklist каждого участника по его ограничениям.
- `calorie_warning`: общая энергия > 1.5× calApprValue.
- `empty_order`: ни одной позиции.
- `total_items`, `total_energy`, `turns`.

### 6. Сохранение и анализ (только JSON)

- Каждый диалог → `dialog_NNNN.json` (profile, history, order_state, flags).
- `aggregate_stats()` → `summary.json` (плоские сводки для анализа).
- `compare_rag.py` → сравнение двух наборов, группировка, интерпретация, диаграмма.

---

## Структура данных

### Профиль клиента (пример)

```json
{
  "sex": "male",
  "age": 38,
  "psycho": "impatient",
  "language": "EN",
  "calApprValue": 2150,
  "noMilk": true,
  "companions": [
    {"role": "child", "label": "child_1", "age": 5,
     "restrictions": {"noMilk": true, "noEggs": false, "noNuts": false, "noGluten": false}},
    {"role": "friend", "label": "friend_1",
     "restrictions": {"noMilk": false, "isVegan": true}}
  ]
}
```

### Заказ (multi-person)

```json
{
  "persons": [
    {"role": "self", "label": "customer", "items": [{"name": "McChicken®", "quantity": 2}],
     "total_energy": 800, "allergens": ["Cereal containing gluten"]},
    {"role": "child", "label": "child_1", "items": [...], ...}
  ],
  "order_complete": false
}
```

---

## Технологии

- Python 3.10+
- ChromaDB, sentence-transformers (`all-MiniLM-L6-v2`)
- LLM через OpenAI-compatible SDK: в `.env` нейтрально **`LLM_API_KEY`**, **`LLM_BASE_URL`** (облако или Ollama); см. `.env.example`. Старые имена (`OPENAI_*`, `XAI_*`, `OLLAMA_URL`) поддерживаются.
- json; matplotlib (опционально, для диаграмм)

---

## Как запустить

```bash
cd diploma_project
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Вариант 1: облако (xAI / Groq / OpenAI — любой OpenAI-compatible /v1)
# export API_PROVIDER=openai
# export LLM_API_KEY=...
# export LLM_BASE_URL=https://api.x.ai/v1
# export API_MODEL=grok-3-mini

# Вариант 2: локальный Ollama
# export API_PROVIDER=ollama
# export LLM_BASE_URL=http://127.0.0.1:11434/v1
# export API_MODEL=qwen3:1.7b

# 1. Загрузить меню в Chroma
python scripts/load_chroma.py

# 2. Тесты (55 тестов, без API)
pytest tests/ -v

# 3. Демо
python scripts/profile_demo.py
python scripts/dialog_demo.py        # один диалог (API)

# 4. Массовая генерация
python scripts/generate_dataset.py --num_dialogs 100 --output_dir dialogs_rag
python scripts/generate_dataset.py --num_dialogs 100 --output_dir dialogs_norag --no_rag

# 5. Сравнение
python scripts/compare_rag.py --rag_dir dialogs_rag --norag_dir dialogs_norag
```

---

## Примечания для Cursor / ИИ

- Опираться на этот файл и на фактические имена модулей.
- Прикладной код — только в **`src/mcd_voice/`**, скрипты — **`scripts/`**.
- Маппинг флагов → токены аллергенов: единственный источник `_FLAG_TO_ALLERGEN` в `generator.py`.
- CSV не используется — только JSON (summary.json, dialog_*.json).
