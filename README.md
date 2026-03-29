# Симулятор диалогов: голосовой ассистент fast-food (диплом МФТИ)

## Где что лежит

| Каталог / файл | Содержимое |
|----------------|------------|
| **`src/mcd_voice/`** | Python-пакет: меню (RAG), профили (REG), LLM-агенты, конвейер диалога |
| **`scripts/`** | Запускаемые сценарии (генерация, сравнение, демо) |
| **`tests/`** | 55 pytest-тестов |
| **`docs/`** | Архитектура |
| **`mcd.json`** | Данные меню (42 позиции) |
| **`chroma_db/`** | Векторная БД (генерируется `scripts/load_chroma.py`) |

## Быстрый старт

```bash
cd diploma_project
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Сценарии использования

### 1. Загрузка меню в Chroma (обязательно первый шаг)

```bash
python scripts/load_chroma.py
```

### 2. Тесты (без API-ключа)

```bash
pytest tests/ -v
```

### 3. Демо профиля (без API)

```bash
python scripts/profile_demo.py
```

### 4. Один диалог (нужен LLM API)

```bash
# Вариант 1: облачный OpenAI
export OPENAI_API_KEY=sk-...
python scripts/dialog_demo.py
```

```bash
# Вариант 2: локальный Ollama через OpenAI-compatible API
export API_PROVIDER=ollama
export API_MODEL=qwen3:1.7b
export OLLAMA_URL=http://localhost:11434/v1/chat/completions
python scripts/dialog_demo.py
```

Примечание: `OLLAMA_URL` можно указывать как `http://localhost:11434/v1`
или полным путём до `.../v1/chat/completions` — код сам нормализует URL.

### 5. Массовая генерация диалогов

```bash
# RAG-версия (кассир с доступом к меню)
python scripts/generate_dataset.py --num_dialogs 100 --output_dir dialogs_rag

# non-RAG-версия (кассир без доступа к меню)
python scripts/generate_dataset.py --num_dialogs 100 --output_dir dialogs_norag --no_rag

# Параметры:
#   --model gpt-4o-mini    модель (по умолчанию gpt-4o-mini)
#   --max_turns 20         макс. ходов в диалоге
```

### 6. Сравнение RAG vs non-RAG

```bash
python scripts/compare_rag.py --rag_dir dialogs_rag --norag_dir dialogs_norag
```

Результат: таблица в консоль + `rag_comparison.json` + `rag_comparison.png` (если установлен matplotlib).

---

Подробности: **`PROJECT_CONTEXT.md`**, дерево: **`docs/ARCHITECTURE.md`**.
