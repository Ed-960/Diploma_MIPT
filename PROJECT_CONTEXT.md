# Проект: Симулятор диалогов для голосового ассистента ресторана быстрого питания

Документ — **единый контекст** для разработки и для ИИ (Cursor): архитектура диплома, что уже в коде, что планируется.

## Цель проекта

Разработка голосового ассистента для ресторанов быстрого питания (на примере McDonald’s) с персонализированными диалогами на английском без экрана. Комбинация **RAG** (меню) и **REG** (профили клиентов), генерация множества синтетических диалогов (целевой масштаб — порядка 3000), анализ ошибок и итерация по промптам.

---

## Структура пакета (Python)

Основная логика в пакете **`mcd_voice/`**; в корне — тонкие CLI-обёртки для привычного запуска.

| Путь | Назначение |
|------|------------|
| `mcd_voice/config.py` | Пути (`PROJECT_ROOT`, `CHROMA_DIR`, `mcd.json`), константы модели, `NO_ALLERGEN_SENTINEL`, имя коллекции. |
| `mcd_voice/menu/parsing.py` | Разбор `allergy`, сборка текста для эмбеддинга, нормализация allergens для вывода. |
| `mcd_voice/menu/dataset.py` | `load_menu_from_json()` → `ids`, `documents`, `metadatas`. |
| `mcd_voice/menu/chroma.py` | Кэш HF, клиент Chroma, `ingest_menu_clear_existing()`, `main()` загрузки. |
| `mcd_voice/menu/search.py` | `search_menu()`, `build_where()`: аллергены + опционально `max_energy` / `min_energy` (ккал в метаданных). |
| `load_menu_to_chroma.py` | Обёртка: `from mcd_voice.menu.chroma import main`. |
| `menu_search.py` | Обратная совместимость: реэкспорт `search_menu`. |
| `test_menu_search.py` | Ручные проверки (импорт из `mcd_voice.menu.search`). |

Импорт для конвейера: `from mcd_voice import search_menu` или `from mcd_voice.menu.search import search_menu`.

## Реализованные артефакты (данные и окружение)

| Файл | Назначение |
|------|------------|
| `mcd.json` | Меню: сейчас **42** позиции (число может меняться при правках файла). |
| `requirements.txt` | `chromadb`, `sentence-transformers`. |
| `chroma_db/` | Persistent-БД Chroma (генерируется при запуске загрузки). |
| `.cache/huggingface/` | Кэш модели эмбеддингов (в проекте, чтобы не зависеть от `~/.cache`). |

### RAG / меню (фактическая реализация)

- **Текст документа для эмбеддинга:** `name` + при наличии `description`, `ingredients`, `tag` (пустые поля пропускаются).
- **Поле `allergy` в JSON:** разбор по запятой, исключение `No Allergens`.
- **Модель эмбеддингов:** `sentence-transformers/all-MiniLM-L6-v2` (384-мерные векторы).
- **Chroma:** `PersistentClient`, каталог `./chroma_db` относительно корня проекта, коллекция `menu`, **`metadata={"hnsw:space": "cosine"}`**.
- **Метаданные `allergens`:** список строк (токены как в `mcd.json`). Пустой список в Chroma недопустим — используется маркер `__none__`. Фильтр `where={"allergens": {"$not_contains": "Milk"}}` рассчитан на **список**, не на строку с запятыми.
- **Токены для чёрного списка** в поиске должны совпадать с данными (например, `Cereal containing gluten`, а не условное `Gluten`).

---

## Архитектура (диплом + план)

### 1. Меню (RAG) — см. таблицу выше

### 2. Генератор профилей (REG)

Стохастический граф: пол, возраст, психотип, язык, ограничения (`isVegan`, `noFish`, `noMilk`, …), калории, `childQuant`, `friendsQuant`. Выход — JSON профиля.

### 3. LLM-агенты

Клиент и кассир с системными промптами; единый вызов LLM (конфиг модели). **В репозитории класс `Agent` пока может отсутствовать** — запланировано.

### 4. Конвейер диалога

Профиль → агенты → цикл: реплика клиента → при необходимости RAG → реплика кассира → обновление заказа → условия выхода → валидация → сохранение.

### 5. Валидация и аналитика

- `allergen_violation`, `calorie_warning`, `hallucination`.
- **`incomplete_order` / отсутствие «детских» позиций при `childQuant > 0`:** не обязана трактоваться как жёсткая ошибка. В реальности дети могут есть общие блюда; удобнее считать метрикой качества (например, предлагал ли кассир детское меню) или ввести флаг вроде `no_kids_menu_offered`. Упрощённая модель: один агент-клиент формулирует заказ за группу; полноценный поочерёдный опрос сопровождающих — возможное **расширение** (в профиле уже есть `childQuant`, `friendsQuant`; при необходимости — поля вроде `childAges`).

### 6. Масштабирование

Пакетная генерация диалогов, кэширование эмбеддингов и при необходимости асинхронные вызовы LLM.

---

## Структура данных (эскизы)

### Позиция меню (`mcd.json`)

Поля включают `name`, `description`, `ingredients`, `tag`, `allergy`, `energy`, макронутриенты, `serving_size` и т.д. Поля **`category` в текущем файле нет** — в метаданных Chroma задаётся пустая строка.

### Профиль клиента (пример)

```json
{
  "sex": "male",
  "age": 32,
  "psycho": "impatient",
  "language": "EN",
  "calApprValue": 2200,
  "isVegan": false,
  "noMilk": true,
  "childQuant": 2,
  "friendsQuant": 0
}
```

### Заказ (пример)

```json
{
  "items": [{ "name": "Big Mac", "quantity": 1, "modifications": [] }],
  "total_energy": 530,
  "allergens_in_order": ["Cereal containing gluten", "Milk", "Soya"]
}
```

---

## Технологии

- Python 3.10+
- ChromaDB, sentence-transformers (`all-MiniLM-L6-v2`)
- OpenAI / litellm — для LLM (план)
- json, csv; при необходимости pandas, argparse, визуализация

---

## Этапы разработки

| Статус | Этап |
|--------|------|
| Сделано | Загрузка `mcd.json` в Chroma, косинус, метаданные, `menu_search.py`, тесты `test_menu_search.py` |
| Далее | REG, агенты, конвейер диалога, валидация, массовая генерация |
| Опционально | Docker, другие эмбеддинги (OpenAI), фильтры по калориям/категориям |

---

## Как запустить (текущий код)

```bash
cd /path/to/diploma_project
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python load_menu_to_chroma.py    # создать/обновить chroma_db
python test_menu_search.py       # демо + проверки (или --check / --demo)
pip install -r requirements-dev.txt
pytest tests/ -q                 # pytest (нужна загруженная Chroma)
```

Ключи API хранить в переменных окружения, не в коде.

---

## Docker

**Не обязателен.** Для диплома достаточно `venv` + `requirements.txt`. Docker полезен для воспроизводимости и демо «одной командой»; при желании — простой `Dockerfile` на официальном образе Python.

---

## Примечания для Cursor / ИИ

- Опираться на этот файл и на фактические имена модулей в репозитории.
- Новые модули — по смыслу: профиль, агенты, пайплайн, валидация (имена могут отличаться от ранних набросков).
- При повторной загрузке меню `load_menu_to_chroma.py` удаляет старые id коллекции перед `add`, дубликатов по id нет.
- Для интеграции RAG в конвейер: `from mcd_voice import search_menu` (или `mcd_voice.menu.search`), маппинг флагов профиля (`noMilk`, …) в токены аллергенов как в `mcd.json`.
