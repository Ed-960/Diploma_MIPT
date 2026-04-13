#!/usr/bin/env bash
set -euo pipefail

# Full diploma experiment run: RAG vs non-RAG in one command.
SEED="${SEED:-42}"
N="${N:-10000}"
MODEL="${MODEL:-gpt-4o-mini}"
MAX_TURNS="${MAX_TURNS:-20}"
RAG_DIR="${RAG_DIR:-dialogs_rag}"
NORAG_DIR="${NORAG_DIR:-dialogs_norag}"
REPORT="${REPORT:-experiment_report.json}"

echo "=== Generating RAG dataset ==="
python scripts/generate_dataset.py \
  --num_dialogs "${N}" \
  --output_dir "${RAG_DIR}" \
  --model "${MODEL}" \
  --seed "${SEED}" \
  --max_turns "${MAX_TURNS}"

echo "=== Generating non-RAG dataset ==="
python scripts/generate_dataset.py \
  --num_dialogs "${N}" \
  --output_dir "${NORAG_DIR}" \
  --no_rag \
  --model "${MODEL}" \
  --seed "${SEED}" \
  --max_turns "${MAX_TURNS}"

echo "=== Comparing datasets ==="
python scripts/compare_rag.py \
  --rag_dir "${RAG_DIR}" \
  --norag_dir "${NORAG_DIR}" \
  --output "${REPORT}"

echo "Done. Report: ${REPORT}"
