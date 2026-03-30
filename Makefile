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
OUT_NORAG  ?= dialogs_norag
OUT_TRACE  ?= dialogs_trace
TURNS      ?= 20
RAG_DIR    ?= $(OUT_RAG)
NORAG_DIR  ?= $(OUT_NORAG)

.PHONY: help install install-dev test test-v chroma \
	demo-profile demo-dialog demo-dialog-trace demo-agents demo-menu-search \
	dataset-rag dataset-norag dataset-trace dataset-trace-10 dataset-trace-20 \
	dataset-trace-50 dataset-trace-100 compare-rag venv

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
	@echo "  make demo-dialog"
	@echo "  make demo-dialog-trace   dialog_demo с --rag_trace --llm_trace"
	@echo "  make demo-agents"
	@echo "  make demo-menu-search"
	@echo ""
	@echo "Генерация (переменные: NUM, TURNS, OUT_RAG, OUT_NORAG, OUT_TRACE):"
	@echo "  make dataset-rag         NUM=$(NUM) OUT_RAG=$(OUT_RAG) TURNS=$(TURNS)"
	@echo "  make dataset-norag       NUM=$(NUM) OUT_NORAG=$(OUT_NORAG) TURNS=$(TURNS)"
	@echo "  make dataset-trace       1 диалог с rag_trace+llm_trace -> $(OUT_TRACE)"
	@echo "  make dataset-trace-10    10 диалогов -> $(OUT_TRACE)_10"
	@echo "  make dataset-trace-20    20 -> $(OUT_TRACE)_20"
	@echo "  make dataset-trace-50    50 -> $(OUT_TRACE)_50"
	@echo "  make dataset-trace-100   100 -> $(OUT_TRACE)_100"
	@echo ""
	@echo "  make compare-rag         RAG_DIR=$(OUT_RAG) NORAG_DIR=$(OUT_NORAG)"
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

demo-dialog-trace:
	$(PY) scripts/dialog_demo.py --max_turns $(TURNS) --rag_trace --llm_trace

demo-agents:
	$(PY) scripts/agents_demo.py

demo-menu-search:
	$(PY) scripts/menu_search_demo.py

dataset-rag:
	$(PY) scripts/generate_dataset.py --num_dialogs $(NUM) --output_dir $(OUT_RAG) --max_turns $(TURNS)

dataset-norag:
	$(PY) scripts/generate_dataset.py --num_dialogs $(NUM) --output_dir $(OUT_NORAG) --no_rag --max_turns $(TURNS)

dataset-trace:
	$(PY) scripts/generate_dataset.py --num_dialogs 1 --output_dir $(OUT_TRACE) --max_turns $(TURNS) --rag_trace --llm_trace

dataset-trace-10:
	$(PY) scripts/generate_dataset.py --num_dialogs 10 --output_dir $(OUT_TRACE)_10 --max_turns $(TURNS) --rag_trace --llm_trace

dataset-trace-20:
	$(PY) scripts/generate_dataset.py --num_dialogs 20 --output_dir $(OUT_TRACE)_20 --max_turns $(TURNS) --rag_trace --llm_trace

dataset-trace-50:
	$(PY) scripts/generate_dataset.py --num_dialogs 50 --output_dir $(OUT_TRACE)_50 --max_turns $(TURNS) --rag_trace --llm_trace

dataset-trace-100:
	$(PY) scripts/generate_dataset.py --num_dialogs 100 --output_dir $(OUT_TRACE)_100 --max_turns $(TURNS) --rag_trace --llm_trace

compare-rag:
	$(PY) scripts/compare_rag.py --rag_dir $(RAG_DIR) --norag_dir $(NORAG_DIR)
