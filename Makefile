# mcd-voice-diploma — удобные цели для тестов и сценариев.
# Запуск из корня репозитория: make help
#
# Интерпретатор: по умолчанию локальный venv. Переопределение:
#   make test PY=python
#   make test PY=.venv/Scripts/python.exe   (Windows, без make OS)

ifeq ($(OS),Windows_NT)
  PY ?= .venv/Scripts/python.exe
else
  PY ?= .venv/bin/python
endif

NUM        ?= 100
OUT_RAG    ?= dialogs_rag
OUT_GRAPH_RAG ?= dialogs_graph_rag
OUT_NORAG  ?= dialogs_norag
OUT_TRACE  ?= dialogs_trace
TURNS      ?= 20
RAG_DIR    ?= $(OUT_RAG)
GRAPH_DIR  ?= $(OUT_GRAPH_RAG)
NORAG_DIR  ?= $(OUT_NORAG)
TRACE_FLAGS ?= --rag_trace --llm_trace --print_trace
# Непустое значение (например 1) — консольный лог RAG/LLM и трассы в JSON (см. generate_dataset.py).
PRINT_TRACE ?=
# Непустое — полные промпты/ответы и сырой Chroma в трассе (--trace_verbose).
TRACE_VERBOSE ?=
TRACE_VERBOSE_FLAG = $(if $(TRACE_VERBOSE),--trace_verbose,)
VOICE_FLAGS ?=
PROFILES_FILE ?= profiles_1000.json
SEED ?=
CLIENT_MODEL ?=
CASHIER_MODEL ?=
WORKERS ?= 1
# Непустое — случайная выборка NUM профилей из PROFILES_FILE без повторений (--shuffle_profiles).
SHUFFLE_PROFILES ?=
# Непустое — кассир без скрытого профиля и без RAG-фильтра аллергенов по профилю (--realistic_cashier).
REALISTIC_CASHIER ?=
REALISTIC_FLAG = $(if $(REALISTIC_CASHIER),--realistic_cashier,)
# Вариативность клиентского промпта: high | normal | off (см. scripts/generate_dataset.py --client_variation).
CLIENT_VARIATION ?= high
CLIENT_VARIATION_FLAG = --client_variation $(CLIENT_VARIATION)
# Выходной файл для make export-ai (один файл со всем кодом для ИИ).
AI_EXPORT ?= allProject_forAI_Test.txt

# Визуал графа меню (scripts/visualize_menu_graph.py): FOCUS, FOCUS_HOPS, FOCUS_STYLE, MENU_GRAPH_MAX_EDGES
FOCUS ?= Big Mac
FOCUS_HOPS ?= 1
FOCUS_STYLE ?= star
MENU_GRAPH_MAX_EDGES ?=

# Принудительный API_PROVIDER для дочернего процесса (Windows-совместимо).
_LLM_API := $(PY) scripts/apply_llm_mode.py api $(PY)
_LLM_OLLAMA := $(PY) scripts/apply_llm_mode.py ollama $(PY)

.PHONY: help install install-dev test test-v chroma \
	demo-profile demo-dialog demo-dialog-api demo-dialog-ollama demo-dialog-trace \
	demo-dialog-trace-api demo-dialog-trace-ollama \
	demo-agents demo-agents-api demo-agents-ollama demo-menu-search \
	voice-browser voice-browser-api voice-browser-ollama \
	dataset-rag dataset-rag-api dataset-rag-ollama \
	dataset-rag-vector dataset-rag-vector-api dataset-rag-vector-ollama \
	dataset-rag-graph dataset-rag-graph-api dataset-rag-graph-ollama \
	dataset-norag dataset-norag-api dataset-norag-ollama \
	dataset-trace dataset-trace-api dataset-trace-ollama \
	dataset-trace-10 dataset-trace-10-api dataset-trace-10-ollama \
	dataset-trace-20 dataset-trace-20-api dataset-trace-20-ollama \
	dataset-trace-50 dataset-trace-50-api dataset-trace-50-ollama \
	dataset-trace-100 dataset-trace-100-api dataset-trace-100-ollama \
	profiles-gen dataset-rag-from-profiles dataset-rag-from-profiles-api dataset-rag-from-profiles-ollama \
	dataset-rag-graph-from-profiles dataset-rag-graph-from-profiles-api dataset-rag-graph-from-profiles-ollama \
	visualize-menu-graph visualize-menu-graph-open \
	visualize-menu-graph-full visualize-menu-graph-full-open \
	visualize-menu-graph-focus visualize-menu-graph-focus-open \
	visualize-profile-graph compare-rag export-ai venv

help:
	@echo "mcd-voice-diploma — make targets"
	@echo ""
	@echo "  make install-dev     pip install -e .[dev]  (нужен существующий venv)"
	@echo "  make venv            создать .venv (python -m venv), затем: make install-dev"
	@echo "  make test            pytest tests/ -q"
	@echo "  make test-v          pytest tests/ -v"
	@echo "  make chroma          загрузить меню в Chroma"
	@echo ""
	@echo "Демо:"
	@echo "  make demo-profile"
	@echo "  make demo-dialog          # LLM: как в .env (API_PROVIDER)"
	@echo "  make demo-dialog-api      # принудительно облако (API_PROVIDER=openai)"
	@echo "  make demo-dialog-ollama   # принудительно Ollama (API_PROVIDER=ollama)"
	@echo "  make demo-dialog-trace   dialog_demo с --print_trace (консоль + JSON трассы)"
	@echo "  make demo-dialog-trace-api / demo-dialog-trace-ollama"
	@echo "  make demo-agents"
	@echo "  make demo-agents-api / demo-agents-ollama"
	@echo "  make demo-menu-search"
	@echo "  make voice-browser-api    # микрофон браузера + LLM, облако (см. .env LLM_*)"
	@echo "  make voice-browser-ollama # то же, локальный Ollama"
	@echo "  make voice-browser        # alias на voice-browser-api"
	@echo "    + логи: make voice-browser-api VOICE_FLAGS='--trace-all'"
	@echo "    + умеренно: VOICE_FLAGS='--print-trace --trace-verbose'"
	@echo "  (make demo-dialog-* — только консольный текст, без микрофона.)"
	@echo ""
	@echo "Генерация (переменные: NUM, TURNS, OUT_RAG, OUT_GRAPH_RAG, OUT_NORAG, OUT_TRACE):"
	@echo "  make dataset-rag         NUM=$(NUM) OUT_RAG=$(OUT_RAG) TURNS=$(TURNS)   # alias vector"
	@echo "  make dataset-rag-vector  NUM=$(NUM) OUT_RAG=$(OUT_RAG) TURNS=$(TURNS)"
	@echo "  make dataset-rag-graph   NUM=$(NUM) OUT_GRAPH_RAG=$(OUT_GRAPH_RAG) TURNS=$(TURNS)"
	@echo "  make dataset-norag       NUM=$(NUM) OUT_NORAG=$(OUT_NORAG) TURNS=$(TURNS)"
	@echo "  make dataset-trace       1 диалог: rag+llm в JSON + print_trace в консоль"
	@echo "  make dataset-trace-10    10 диалогов -> $(OUT_TRACE)_10"
	@echo "  make dataset-trace-20    20 -> $(OUT_TRACE)_20"
	@echo "  make dataset-trace-50    50 -> $(OUT_TRACE)_50"
	@echo "  make dataset-trace-100   100 -> $(OUT_TRACE)_100"
	@echo "  make profiles-gen        NUM=$(NUM) PROFILES_FILE=$(PROFILES_FILE) SEED=42"
	@echo "  make dataset-rag-from-profiles NUM=$(NUM) PROFILES_FILE=$(PROFILES_FILE) PRINT_TRACE=1 TRACE_VERBOSE=1   # vector RAG"
	@echo "  make dataset-rag-graph-from-profiles NUM=$(NUM) OUT_GRAPH_RAG=$(OUT_GRAPH_RAG) ...   # graph RAG (как выше)"
	@echo "  make dataset-rag-from-profiles ... SHUFFLE_PROFILES=1 SEED=42   # профили из файла — случайная выборка"
	@echo "  make dataset-rag-from-profiles ... CLIENT_VARIATION=high|normal|off   # вариативность клиента (по умолчанию high)"
	@echo "  make dataset-rag ... REALISTIC_CASHIER=1   # реалистичный кассир (без скрытого профиля)"
	@echo "  Для каждой цели dataset-* есть суффиксы -api и -ollama (принудительный LLM, как demo-dialog-api)."
	@echo ""
	@echo "Граф меню (graph-RAG) -> docs/menu_graph_rag.* :"
	@echo "  make visualize-menu-graph              # демо ~380 сильнейших рёбер"
	@echo "  make visualize-menu-graph-open         # то же + браузер"
	@echo "  make visualize-menu-graph-full         # все рёбра (max-edges 0)"
	@echo "  make visualize-menu-graph-full-open"
	@echo "  make visualize-menu-graph-focus        # подграф: FOCUS (по умолчанию Big Mac), FOCUS_HOPS, FOCUS_STYLE"
	@echo "  make visualize-menu-graph-focus-open"
	@echo "    пример: make visualize-menu-graph-focus FOCUS='McChicken,Apple Pie' FOCUS_HOPS=1"
	@echo "    только выбранные и рёбра между ними: FOCUS_HOPS=0 и 2+ названия в FOCUS"
	@echo "    плотнее: FOCUS_STYLE=induced; лимит рёбер: MENU_GRAPH_MAX_EDGES=40"
	@echo "  make visualize-profile-graph  # граф семплинга профилей -> docs/profile_decision_graph.mmd"
	@echo ""
	@echo "  make compare-rag         RAG_DIR=$(OUT_RAG) NORAG_DIR=$(OUT_NORAG)"
	@echo ""
	@echo "Экспорт кода в один файл (для ИИ / ревью):"
	@echo "  make export-ai           -> $(AI_EXPORT)  (скрипт scripts/export_all_project_for_ai.py)"
	@echo "  make export-ai AI_EXPORT=my_dump.txt"
	@echo ""
	@echo "PY=$(PY)"

venv:
	python -m venv .venv

install-dev:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e ".[dev]"

install: install-dev

test:
	$(PY) -m pytest tests/ -q

test-v:
	$(PY) -m pytest tests/ -v

chroma:
	$(PY) scripts/load_chroma.py

demo-profile:
	$(PY) scripts/profile_demo.py

demo-dialog:
	$(PY) scripts/dialog_demo.py --max_turns $(TURNS)

demo-dialog-api:
	$(_LLM_API) scripts/dialog_demo.py --max_turns $(TURNS)

demo-dialog-ollama:
	$(_LLM_OLLAMA) scripts/dialog_demo.py --max_turns $(TURNS)

demo-dialog-trace:
	$(PY) scripts/dialog_demo.py --max_turns $(TURNS) --print_trace

demo-dialog-trace-api:
	$(_LLM_API) scripts/dialog_demo.py --max_turns $(TURNS) --print_trace

demo-dialog-trace-ollama:
	$(_LLM_OLLAMA) scripts/dialog_demo.py --max_turns $(TURNS) --print_trace

demo-agents:
	$(PY) scripts/agents_demo.py

demo-agents-api:
	$(_LLM_API) scripts/agents_demo.py

demo-agents-ollama:
	$(_LLM_OLLAMA) scripts/agents_demo.py

demo-menu-search:
	$(PY) scripts/menu_search_demo.py

voice-browser-api:
	$(_LLM_API) scripts/voice_browser_server.py $(VOICE_FLAGS)

voice-browser-ollama:
	$(_LLM_OLLAMA) scripts/voice_browser_server.py $(VOICE_FLAGS)

voice-browser: voice-browser-api

dataset-rag:
	$(PY) scripts/generate_dataset.py --num_dialogs $(NUM) --output_dir $(OUT_RAG) --rag_mode vector --max_turns $(TURNS) --workers $(WORKERS) $(REALISTIC_FLAG) $(if $(CLIENT_MODEL),--client_model $(CLIENT_MODEL),) $(if $(CASHIER_MODEL),--cashier_model $(CASHIER_MODEL),) $(CLIENT_VARIATION_FLAG) $(TRACE_VERBOSE_FLAG)

dataset-rag-api:
	$(_LLM_API) scripts/generate_dataset.py --num_dialogs $(NUM) --output_dir $(OUT_RAG) --rag_mode vector --max_turns $(TURNS) --workers $(WORKERS) $(REALISTIC_FLAG) $(if $(CLIENT_MODEL),--client_model $(CLIENT_MODEL),) $(if $(CASHIER_MODEL),--cashier_model $(CASHIER_MODEL),) $(CLIENT_VARIATION_FLAG) $(TRACE_VERBOSE_FLAG)

dataset-rag-ollama:
	$(_LLM_OLLAMA) scripts/generate_dataset.py --num_dialogs $(NUM) --output_dir $(OUT_RAG) --rag_mode vector --max_turns $(TURNS) --workers $(WORKERS) $(REALISTIC_FLAG) $(if $(CLIENT_MODEL),--client_model $(CLIENT_MODEL),) $(if $(CASHIER_MODEL),--cashier_model $(CASHIER_MODEL),) $(CLIENT_VARIATION_FLAG) $(TRACE_VERBOSE_FLAG)

dataset-rag-vector:
	$(PY) scripts/generate_dataset.py --num_dialogs $(NUM) --output_dir $(OUT_RAG) --rag_mode vector --max_turns $(TURNS) --workers $(WORKERS) $(REALISTIC_FLAG) $(if $(CLIENT_MODEL),--client_model $(CLIENT_MODEL),) $(if $(CASHIER_MODEL),--cashier_model $(CASHIER_MODEL),) $(CLIENT_VARIATION_FLAG) $(TRACE_VERBOSE_FLAG)

dataset-rag-vector-api:
	$(_LLM_API) scripts/generate_dataset.py --num_dialogs $(NUM) --output_dir $(OUT_RAG) --rag_mode vector --max_turns $(TURNS) --workers $(WORKERS) $(REALISTIC_FLAG) $(if $(CLIENT_MODEL),--client_model $(CLIENT_MODEL),) $(if $(CASHIER_MODEL),--cashier_model $(CASHIER_MODEL),) $(CLIENT_VARIATION_FLAG) $(TRACE_VERBOSE_FLAG)

dataset-rag-vector-ollama:
	$(_LLM_OLLAMA) scripts/generate_dataset.py --num_dialogs $(NUM) --output_dir $(OUT_RAG) --rag_mode vector --max_turns $(TURNS) --workers $(WORKERS) $(REALISTIC_FLAG) $(if $(CLIENT_MODEL),--client_model $(CLIENT_MODEL),) $(if $(CASHIER_MODEL),--cashier_model $(CASHIER_MODEL),) $(CLIENT_VARIATION_FLAG) $(TRACE_VERBOSE_FLAG)

dataset-rag-graph:
	$(PY) scripts/generate_dataset.py --num_dialogs $(NUM) --output_dir $(OUT_GRAPH_RAG) --rag_mode graph --max_turns $(TURNS) --workers $(WORKERS) $(REALISTIC_FLAG) $(if $(CLIENT_MODEL),--client_model $(CLIENT_MODEL),) $(if $(CASHIER_MODEL),--cashier_model $(CASHIER_MODEL),) $(CLIENT_VARIATION_FLAG) $(TRACE_VERBOSE_FLAG)

dataset-rag-graph-api:
	$(_LLM_API) scripts/generate_dataset.py --num_dialogs $(NUM) --output_dir $(OUT_GRAPH_RAG) --rag_mode graph --max_turns $(TURNS) --workers $(WORKERS) $(REALISTIC_FLAG) $(if $(CLIENT_MODEL),--client_model $(CLIENT_MODEL),) $(if $(CASHIER_MODEL),--cashier_model $(CASHIER_MODEL),) $(CLIENT_VARIATION_FLAG) $(TRACE_VERBOSE_FLAG)

dataset-rag-graph-ollama:
	$(_LLM_OLLAMA) scripts/generate_dataset.py --num_dialogs $(NUM) --output_dir $(OUT_GRAPH_RAG) --rag_mode graph --max_turns $(TURNS) --workers $(WORKERS) $(REALISTIC_FLAG) $(if $(CLIENT_MODEL),--client_model $(CLIENT_MODEL),) $(if $(CASHIER_MODEL),--cashier_model $(CASHIER_MODEL),) $(CLIENT_VARIATION_FLAG) $(TRACE_VERBOSE_FLAG)

dataset-norag:
	$(PY) scripts/generate_dataset.py --num_dialogs $(NUM) --output_dir $(OUT_NORAG) --no_rag --max_turns $(TURNS) --workers $(WORKERS) $(REALISTIC_FLAG) $(if $(CLIENT_MODEL),--client_model $(CLIENT_MODEL),) $(if $(CASHIER_MODEL),--cashier_model $(CASHIER_MODEL),) $(CLIENT_VARIATION_FLAG) $(TRACE_VERBOSE_FLAG)

dataset-norag-api:
	$(_LLM_API) scripts/generate_dataset.py --num_dialogs $(NUM) --output_dir $(OUT_NORAG) --no_rag --max_turns $(TURNS) --workers $(WORKERS) $(REALISTIC_FLAG) $(if $(CLIENT_MODEL),--client_model $(CLIENT_MODEL),) $(if $(CASHIER_MODEL),--cashier_model $(CASHIER_MODEL),) $(CLIENT_VARIATION_FLAG) $(TRACE_VERBOSE_FLAG)

dataset-norag-ollama:
	$(_LLM_OLLAMA) scripts/generate_dataset.py --num_dialogs $(NUM) --output_dir $(OUT_NORAG) --no_rag --max_turns $(TURNS) --workers $(WORKERS) $(REALISTIC_FLAG) $(if $(CLIENT_MODEL),--client_model $(CLIENT_MODEL),) $(if $(CASHIER_MODEL),--cashier_model $(CASHIER_MODEL),) $(CLIENT_VARIATION_FLAG) $(TRACE_VERBOSE_FLAG)

dataset-trace:
	$(PY) scripts/generate_dataset.py --num_dialogs 1 --output_dir $(OUT_TRACE) --max_turns $(TURNS) $(REALISTIC_FLAG) $(TRACE_FLAGS) $(CLIENT_VARIATION_FLAG) $(TRACE_VERBOSE_FLAG)

dataset-trace-api:
	$(_LLM_API) scripts/generate_dataset.py --num_dialogs 1 --output_dir $(OUT_TRACE) --max_turns $(TURNS) $(REALISTIC_FLAG) $(TRACE_FLAGS) $(CLIENT_VARIATION_FLAG) $(TRACE_VERBOSE_FLAG)

dataset-trace-ollama:
	$(_LLM_OLLAMA) scripts/generate_dataset.py --num_dialogs 1 --output_dir $(OUT_TRACE) --max_turns $(TURNS) $(REALISTIC_FLAG) $(TRACE_FLAGS) $(CLIENT_VARIATION_FLAG) $(TRACE_VERBOSE_FLAG)

dataset-trace-10:
	$(PY) scripts/generate_dataset.py --num_dialogs 10 --output_dir $(OUT_TRACE)_10 --max_turns $(TURNS) $(REALISTIC_FLAG) $(TRACE_FLAGS) $(CLIENT_VARIATION_FLAG) $(TRACE_VERBOSE_FLAG)

dataset-trace-10-api:
	$(_LLM_API) scripts/generate_dataset.py --num_dialogs 10 --output_dir $(OUT_TRACE)_10 --max_turns $(TURNS) $(REALISTIC_FLAG) $(TRACE_FLAGS) $(CLIENT_VARIATION_FLAG) $(TRACE_VERBOSE_FLAG)

dataset-trace-10-ollama:
	$(_LLM_OLLAMA) scripts/generate_dataset.py --num_dialogs 10 --output_dir $(OUT_TRACE)_10 --max_turns $(TURNS) $(REALISTIC_FLAG) $(TRACE_FLAGS) $(CLIENT_VARIATION_FLAG) $(TRACE_VERBOSE_FLAG)

dataset-trace-20:
	$(PY) scripts/generate_dataset.py --num_dialogs 20 --output_dir $(OUT_TRACE)_20 --max_turns $(TURNS) $(REALISTIC_FLAG) $(TRACE_FLAGS) $(CLIENT_VARIATION_FLAG) $(TRACE_VERBOSE_FLAG)

dataset-trace-20-api:
	$(_LLM_API) scripts/generate_dataset.py --num_dialogs 20 --output_dir $(OUT_TRACE)_20 --max_turns $(TURNS) $(REALISTIC_FLAG) $(TRACE_FLAGS) $(CLIENT_VARIATION_FLAG) $(TRACE_VERBOSE_FLAG)

dataset-trace-20-ollama:
	$(_LLM_OLLAMA) scripts/generate_dataset.py --num_dialogs 20 --output_dir $(OUT_TRACE)_20 --max_turns $(TURNS) $(REALISTIC_FLAG) $(TRACE_FLAGS) $(CLIENT_VARIATION_FLAG) $(TRACE_VERBOSE_FLAG)

dataset-trace-50:
	$(PY) scripts/generate_dataset.py --num_dialogs 50 --output_dir $(OUT_TRACE)_50 --max_turns $(TURNS) $(REALISTIC_FLAG) $(TRACE_FLAGS) $(CLIENT_VARIATION_FLAG) $(TRACE_VERBOSE_FLAG)

dataset-trace-50-api:
	$(_LLM_API) scripts/generate_dataset.py --num_dialogs 50 --output_dir $(OUT_TRACE)_50 --max_turns $(TURNS) $(REALISTIC_FLAG) $(TRACE_FLAGS) $(CLIENT_VARIATION_FLAG) $(TRACE_VERBOSE_FLAG)

dataset-trace-50-ollama:
	$(_LLM_OLLAMA) scripts/generate_dataset.py --num_dialogs 50 --output_dir $(OUT_TRACE)_50 --max_turns $(TURNS) $(REALISTIC_FLAG) $(TRACE_FLAGS) $(CLIENT_VARIATION_FLAG) $(TRACE_VERBOSE_FLAG)

dataset-trace-100:
	$(PY) scripts/generate_dataset.py --num_dialogs 100 --output_dir $(OUT_TRACE)_100 --max_turns $(TURNS) $(REALISTIC_FLAG) $(TRACE_FLAGS) $(CLIENT_VARIATION_FLAG) $(TRACE_VERBOSE_FLAG)

dataset-trace-100-api:
	$(_LLM_API) scripts/generate_dataset.py --num_dialogs 100 --output_dir $(OUT_TRACE)_100 --max_turns $(TURNS) $(REALISTIC_FLAG) $(TRACE_FLAGS) $(CLIENT_VARIATION_FLAG) $(TRACE_VERBOSE_FLAG)

dataset-trace-100-ollama:
	$(_LLM_OLLAMA) scripts/generate_dataset.py --num_dialogs 100 --output_dir $(OUT_TRACE)_100 --max_turns $(TURNS) $(REALISTIC_FLAG) $(TRACE_FLAGS) $(CLIENT_VARIATION_FLAG) $(TRACE_VERBOSE_FLAG)

profiles-gen:
	$(PY) scripts/generate_profiles.py --num_profiles $(NUM) --output_file $(PROFILES_FILE) $(if $(SEED),--seed $(SEED),)

dataset-rag-from-profiles:
	$(PY) scripts/generate_dataset.py --num_dialogs $(NUM) --output_dir $(OUT_RAG) --rag_mode vector --max_turns $(TURNS) --workers $(WORKERS) --profiles_file $(PROFILES_FILE) $(if $(SHUFFLE_PROFILES),--shuffle_profiles,) $(if $(SEED),--seed $(SEED),) $(REALISTIC_FLAG) $(if $(CLIENT_MODEL),--client_model $(CLIENT_MODEL),) $(if $(CASHIER_MODEL),--cashier_model $(CASHIER_MODEL),) $(if $(PRINT_TRACE),$(TRACE_FLAGS),) $(CLIENT_VARIATION_FLAG) $(TRACE_VERBOSE_FLAG)

dataset-rag-from-profiles-api:
	$(_LLM_API) scripts/generate_dataset.py --num_dialogs $(NUM) --output_dir $(OUT_RAG) --rag_mode vector --max_turns $(TURNS) --workers $(WORKERS) --profiles_file $(PROFILES_FILE) $(if $(SHUFFLE_PROFILES),--shuffle_profiles,) $(if $(SEED),--seed $(SEED),) $(REALISTIC_FLAG) $(if $(CLIENT_MODEL),--client_model $(CLIENT_MODEL),) $(if $(CASHIER_MODEL),--cashier_model $(CASHIER_MODEL),) $(if $(PRINT_TRACE),$(TRACE_FLAGS),) $(CLIENT_VARIATION_FLAG) $(TRACE_VERBOSE_FLAG)

dataset-rag-from-profiles-ollama:
	$(_LLM_OLLAMA) scripts/generate_dataset.py --num_dialogs $(NUM) --output_dir $(OUT_RAG) --rag_mode vector --max_turns $(TURNS) --workers $(WORKERS) --profiles_file $(PROFILES_FILE) $(if $(SHUFFLE_PROFILES),--shuffle_profiles,) $(if $(SEED),--seed $(SEED),) $(REALISTIC_FLAG) $(if $(CLIENT_MODEL),--client_model $(CLIENT_MODEL),) $(if $(CASHIER_MODEL),--cashier_model $(CASHIER_MODEL),) $(if $(PRINT_TRACE),$(TRACE_FLAGS),) $(CLIENT_VARIATION_FLAG) $(TRACE_VERBOSE_FLAG)

dataset-rag-graph-from-profiles:
	$(PY) scripts/generate_dataset.py --num_dialogs $(NUM) --output_dir $(OUT_GRAPH_RAG) --rag_mode graph --max_turns $(TURNS) --workers $(WORKERS) --profiles_file $(PROFILES_FILE) $(if $(SHUFFLE_PROFILES),--shuffle_profiles,) $(if $(SEED),--seed $(SEED),) $(REALISTIC_FLAG) $(if $(CLIENT_MODEL),--client_model $(CLIENT_MODEL),) $(if $(CASHIER_MODEL),--cashier_model $(CASHIER_MODEL),) $(if $(PRINT_TRACE),$(TRACE_FLAGS),) $(CLIENT_VARIATION_FLAG) $(TRACE_VERBOSE_FLAG)

dataset-rag-graph-from-profiles-api:
	$(_LLM_API) scripts/generate_dataset.py --num_dialogs $(NUM) --output_dir $(OUT_GRAPH_RAG) --rag_mode graph --max_turns $(TURNS) --workers $(WORKERS) --profiles_file $(PROFILES_FILE) $(if $(SHUFFLE_PROFILES),--shuffle_profiles,) $(if $(SEED),--seed $(SEED),) $(REALISTIC_FLAG) $(if $(CLIENT_MODEL),--client_model $(CLIENT_MODEL),) $(if $(CASHIER_MODEL),--cashier_model $(CASHIER_MODEL),) $(if $(PRINT_TRACE),$(TRACE_FLAGS),) $(CLIENT_VARIATION_FLAG) $(TRACE_VERBOSE_FLAG)

dataset-rag-graph-from-profiles-ollama:
	$(_LLM_OLLAMA) scripts/generate_dataset.py --num_dialogs $(NUM) --output_dir $(OUT_GRAPH_RAG) --rag_mode graph --max_turns $(TURNS) --workers $(WORKERS) --profiles_file $(PROFILES_FILE) $(if $(SHUFFLE_PROFILES),--shuffle_profiles,) $(if $(SEED),--seed $(SEED),) $(REALISTIC_FLAG) $(if $(CLIENT_MODEL),--client_model $(CLIENT_MODEL),) $(if $(CASHIER_MODEL),--cashier_model $(CASHIER_MODEL),) $(if $(PRINT_TRACE),$(TRACE_FLAGS),) $(CLIENT_VARIATION_FLAG) $(TRACE_VERBOSE_FLAG)

visualize-menu-graph:
	$(PY) scripts/visualize_menu_graph.py

visualize-menu-graph-open:
	$(PY) scripts/visualize_menu_graph.py --open

visualize-menu-graph-full:
	$(PY) scripts/visualize_menu_graph.py --max-edges 0

visualize-menu-graph-full-open:
	$(PY) scripts/visualize_menu_graph.py --max-edges 0 --open

visualize-menu-graph-focus:
	$(PY) scripts/visualize_menu_graph.py --focus "$(FOCUS)" --focus-hops $(FOCUS_HOPS) --focus-style $(FOCUS_STYLE) $(if $(MENU_GRAPH_MAX_EDGES),--max-edges $(MENU_GRAPH_MAX_EDGES),)

visualize-menu-graph-focus-open:
	$(PY) scripts/visualize_menu_graph.py --focus "$(FOCUS)" --focus-hops $(FOCUS_HOPS) --focus-style $(FOCUS_STYLE) $(if $(MENU_GRAPH_MAX_EDGES),--max-edges $(MENU_GRAPH_MAX_EDGES),) --open

visualize-profile-graph:
	$(PY) scripts/visualize_profile_graph.py

compare-rag:
	$(PY) scripts/compare_rag.py --rag_dir $(RAG_DIR) --norag_dir $(NORAG_DIR)

export-ai:
	$(PY) scripts/export_all_project_for_ai.py --output $(AI_EXPORT)
