set OLLAMA_NUM_PARALLEL=8 && ollama serve

# Vector RAG (Chroma)

make dataset-rag-from-profiles NUM=20 PROFILES_FILE=profiles_1000.json OUT_RAG=dialogs_rag PRINT_TRACE=1 TRACE_VERBOSE=1 WORKERS=8
make dataset-rag-from-profiles NUM=20 PROFILES_FILE=profiles_1000.json OUT_RAG=dialogs_rag PRINT_TRACE=1 TRACE_VERBOSE=1 WORKERS=8 SHUFFLE_PROFILES=1 REALISTIC_CASHIER=1

# Graph RAG (та же схема переменных, каталог по умолчанию OUT_GRAPH_RAG=dialogs_graph_rag)

make dataset-rag-graph-from-profiles NUM=20 PROFILES_FILE=profiles_1000.json OUT_GRAPH_RAG=dialogs_graph_rag PRINT_TRACE=1 TRACE_VERBOSE=1 WORKERS=8
make dataset-rag-graph-from-profiles NUM=20 PROFILES_FILE=profiles_1000.json OUT_GRAPH_RAG=dialogs_graph_rag PRINT_TRACE=1 TRACE_VERBOSE=1 WORKERS=8 SHUFFLE_PROFILES=1 REALISTIC_CASHIER=1

# Визуал графа меню (graph-RAG) -> docs/menu_graph_rag.mmd, menu_graph_rag_data.json, menu_graph_rag.html

make visualize-menu-graph
make visualize-menu-graph-open

# Полный граф (все рёбра по min-weight, без лимита 380)

make visualize-menu-graph-full
make visualize-menu-graph-full-open

# Подграф вокруг блюд: переменные FOCUS, FOCUS_HOPS, FOCUS_STYLE, MENU_GRAPH_MAX_EDGES (см. make help)

make visualize-menu-graph-focus
make visualize-menu-graph-focus-open

# Примеры фокуса (имена как в меню JSON)

make visualize-menu-graph-focus FOCUS="Big Mac"
make visualize-menu-graph-focus-open FOCUS="McChicken,Apple Pie"
make visualize-menu-graph-focus FOCUS="Big Mac,Quarter Pounder with Cheese" FOCUS_HOPS=0
make visualize-menu-graph-focus FOCUS="Big Mac" FOCUS_STYLE=induced
make visualize-menu-graph-focus-open FOCUS="Big Mac" MENU_GRAPH_MAX_EDGES=40

make voice-browser-api
make voice-browser-ollama
make voice-browser-norag-api
make voice-browser-norag-ollama

//
REWRITE_MODEL — лёгкая и быстрая
Задача — коротко переписать запрос под RAG, не «ведёт диалог».

meta-llama/llama-3.2-3b-instruct:free — лучший баланс «мало параметров / нормально следует инструкции» для rewrite.
google/gemma-3-4b-it:free — чуть крупнее, часто стабильнее на мелких формулировках.
liquid/lfm-2.5-1.2b-instruct:free — самая лёгкая; может слабее держать русский и капризничать на сложных формулировках.
Практично начать с Llama 3.2 3B, при проблемах с качеством rewrite — перейти на Gemma 3 4B.

API_MODEL — основной диалог
Нужна сила и предсказуемость (в т.ч. русский).

meta-llama/llama-3.3-70b-instruct:free — проверенный универсальный вариант для чата.
qwen/qwen3-next-80b-a3b-instruct:free — сильная многоязычная линейка, часто хорош на русском.
z-ai/glm-4.5-air:free — компактнее 70B+, но часто ок по качеству/латентности.
openai/gpt-oss-120b:free или openai/gpt-oss-20b:free — если хотите попробовать новые OSS; 120B может быть тяжелее по ответу.
Разумный старт: llama-3.3-70b-instruct:free или qwen3-next-80b, если важнее русский.

//

максимум логов:
make voice-browser-api VOICE_FLAGS="--trace-all"
чуть легче (trace без full snapshots/payload):
make voice-browser-api VOICE_FLAGS="--print-trace --trace-verbose"
То же для Ollama:

make voice-browser-ollama VOICE_FLAGS="--trace-all"

Non-RAG (без vector DB / Chroma, полный mcd.json отправляется в LLM каждый ход):
make voice-browser-api VOICE_FLAGS="--no-rag --trace-all"
make voice-browser-ollama VOICE_FLAGS="--no-rag --trace-all"
make voice-browser-norag-api
make voice-browser-norag-ollama

Генерация Non-RAG из профилей:
make dataset-norag-from-profiles-api \
 NUM=40 \
 PROFILES_FILE=profiles_1000.json \
 OUT_NORAG=dialogs_norag \
 PRINT_TRACE=1 \
 TRACE_VERBOSE=1 \
 WORKERS=8 \
 SHUFFLE_PROFILES=1 \
 SEED=42 \
 2>&1 | tee generate_dataset.log

make dataset-rag-from-profiles-api \
 NUM=40 \
 PROFILES_FILE=profiles_1000.json \
 OUT_RAG=dialogs_rag \
 PRINT_TRACE=1 \
 TRACE_VERBOSE=1 \
 WORKERS=8 \
 2>&1 | tee generate_dataset.log

Истории диалогов (как «соединяются»)

# Склеить все history из dialogs*rag/dialog*\*.json → dialogs_rag/merged_histories.json (без API)

python scripts/merge_dialog_histories.py --dialogs_dir dialogs_rag --out dialogs_rag/merged_histories.json

# то же, читаемый текст: --format txt --out dialogs_rag/merged_histories.txt

python scripts/audit_history_llm_judge.py --dialogs_dir dialogs_rag --out dialogs_rag/history_judge.jsonl
python scripts/audit_history_llm_judge.py --dialogs_dir dialogs_rag --limit 5 --dry_run

//
//

make dataset-rag-from-profiles-api \
 NUM=40 \
 PROFILES_FILE=profiles_1000.json \
 OUT_RAG=dialogs_rag \
 PRINT_TRACE=1 \
 TRACE_VERBOSE=1 \
 WORKERS=8 \
 SHUFFLE_PROFILES=1 \
 SEED=42 \
 2>&1 | tee generate_dataset.log

make voice-browser-api VOICE_FLAGS="--trace-all"

//
//

# Added voice commands:

make voice-browser-api VOICE_FLAGS="--no-rag --trace-all"
make voice-browser-ollama VOICE_FLAGS="--no-rag --trace-all"

# Also added shortcuts:

make voice-browser-norag-api
make voice-browser-norag-ollama

# For dataset generation from profiles:

make dataset-norag-from-profiles-api NUM=40 PROFILES_FILE=profiles_1000.json OUT_NORAG=dialogs_norag PRINT_TRACE=1 TRACE_VERBOSE=1 WORKERS=8
make dataset-norag-from-profiles-ollama NUM=40 PROFILES_FILE=profiles_1000.json OUT_NORAG=dialogs_norag PRINT_TRACE=1 TRACE_VERBOSE=1 WORKERS=8

//
//

# Graph-RAG из профилей (аналог vector, выход по умолчанию OUT_GRAPH_RAG=dialogs_graph_rag):

make dataset-rag-graph-from-profiles-api \
 NUM=40 \
 PROFILES_FILE=profiles_1000.json \
 OUT_GRAPH_RAG=dialogs_graph_rag \
 PRINT_TRACE=1 \
 TRACE_VERBOSE=1 \
 WORKERS=8 \
 SHUFFLE_PROFILES=1 \
 SEED=42 \
 2>&1 | tee generate_dataset_graph.log

make dataset-rag-graph-from-profiles-api NUM=40 PROFILES_FILE=profiles_1000.json OUT_GRAPH_RAG=dialogs_graph_rag PRINT_TRACE=1 TRACE_VERBOSE=1 WORKERS=8

make dataset-rag-graph-from-profiles-ollama NUM=40 PROFILES_FILE=profiles_1000.json OUT_GRAPH_RAG=dialogs_graph_rag PRINT_TRACE=1 TRACE_VERBOSE=1 WORKERS=8

# Голосовой браузер сейчас использует vector-RAG у кассира; graph-RAG — через generate_dataset / dialog_demo с --rag_mode graph.

//

make voice-browser-api-graph
make voice-browser-ollama-graph
make voice-browser-api-graph VOICE_FLAGS="--trace-all"

// Эксперименты
Полный прогон (как ваш первый пример):

make question-experiment-norag
По умолчанию: OUT_QUESTIONS=experiments/no_rag_questions, MAX_QUESTIONS=0 (все вопросы), MAX_DIALOG_TURNS=4. Подробные трассы LLM/RAG (`--trace_verbose`) включены всегда для этих целей `make`.

Свой каталог / лимит:

make question-experiment-norag OUT_QUESTIONS=experiments/no_rag_questions MAX_QUESTIONS=60 MAX_DIALOG_TURNS=4
Облако / Ollama (как dataset-norag-api — через apply_llm_mode):

make question-experiment-norag-api
make question-experiment-norag-ollama
Короткий smoke (60 вопросов → experiments/no_rag_questions_smoke):

make question-experiment-norag-smoke
make question-experiment-norag-smoke-api
make question-experiment-norag-smoke-ollama
Дополнительно: CASHIER_MODEL=..., CLIENT_MODEL=..., JUDGE_MODEL=...

Справка: make help — блок «Эксперимент по вопросам».

.venv-wsl/bin/python scripts/apply_llm_mode.py ollama .venv-wsl/bin/python scripts/run_question_experiment.py --output_dir experiments/no_rag_questions --max_dialog_turns 4 --trace_verbose


//

make question-experiment-vector-api OUT_QUESTIONS_VECTOR=experiments/my_vec_rag MAX_QUESTIONS=100
make question-experiment-vector-ollama OUT_QUESTIONS_VECTOR=experiments/my_vec_rag MAX_QUESTIONS=100
make question-experiment-vector-ollama OUT_QUESTIONS_VECTOR=experiments/my_vec_rag

//

Фильтр по категории
В поле JSON "category" (как в questions/simple_questions.json: simple, allergy, diet, lexical, mixed, group).

Make — переменная QUESTION_CATEGORY: одна категория или несколько через запятую (без пробелов вокруг запятой надёжнее):

make question-experiment-vector-ollama \
  OUT_QUESTIONS_VECTOR=experiments/my_vec_rag \
  QUESTION_CATEGORY=simple \
  MAX_QUESTIONS=50
Несколько категорий:

make question-experiment-vector-ollama \
  OUT_QUESTIONS_VECTOR=experiments/my_vec_rag \
  QUESTION_CATEGORY=simple,diet \
  MAX_QUESTIONS=100
То же для no-RAG:

make question-experiment-norag-ollama \
  OUT_QUESTIONS=experiments/my_norag \
  QUESTION_CATEGORY=allergy \
  MAX_QUESTIONS=30
Порядок действий скрипта: загрузить все файлы из банка → отфильтровать по категориям → при MAX_QUESTIONS>0 взять только первые N из отфильтрованного списка.

CLI напрямую
python scripts/run_question_experiment.py \
  --retrieval_mode vector \
  --output_dir experiments/my_vec_rag \
  --categories simple \
  --max_questions 50 \
  --trace_verbose

  //

make question-experiment-norag-ollama-simple   MAX_QUESTIONS=50 OUT_QUESTIONS=experiments/norag_simple_50
make question-experiment-norag-ollama-allergy MAX_QUESTIONS=50 OUT_QUESTIONS=experiments/norag_allergy_50
make question-experiment-norag-ollama-diet    MAX_QUESTIONS=50 OUT_QUESTIONS=experiments/norag_diet_50
make question-experiment-norag-ollama-lexical MAX_QUESTIONS=50 OUT_QUESTIONS=experiments/norag_lexical_50
make question-experiment-norag-ollama-mixed   MAX_QUESTIONS=50 OUT_QUESTIONS=experiments/norag_mixed_50
make question-experiment-norag-ollama-group   MAX_QUESTIONS=50 OUT_QUESTIONS=experiments/norag_group_50

make question-experiment-vector-ollama-simple   MAX_QUESTIONS=50 OUT_QUESTIONS_VECTOR=experiments/vec_rag_simple_50
make question-experiment-vector-ollama-allergy MAX_QUESTIONS=50 OUT_QUESTIONS_VECTOR=experiments/vec_rag_allergy_50
make question-experiment-vector-ollama-diet   MAX_QUESTIONS=50 OUT_QUESTIONS_VECTOR=experiments/vec_rag_diet_50
make question-experiment-vector-ollama-lexical MAX_QUESTIONS=50 OUT_QUESTIONS_VECTOR=experiments/vec_rag_lexical_50
make question-experiment-vector-ollama-mixed   MAX_QUESTIONS=50 OUT_QUESTIONS_VECTOR=experiments/vec_rag_mixed_50
make question-experiment-vector-ollama-group   MAX_QUESTIONS=50 OUT_QUESTIONS_VECTOR=experiments/vec_rag_group_50