#!/usr/bin/env python3
"""Run single-turn experiments over prepared question banks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import _bootstrap

_bootstrap.ensure_src()

from mcd_voice.llm import ensure_llm_credentials, get_llm_runtime_config
from mcd_voice.dialog.question_experiment import (
    filter_questions_by_categories,
    load_question_banks,
    parse_category_filter,
    run_question_dialog_experiment,
    save_dialogs_by_category,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _default_question_paths() -> list[str]:
    base = PROJECT_ROOT / "questions"
    return [
        str(base / "simple_questions.json"),
        str(base / "allergy_questions.json"),
        str(base / "diet_questions.json"),
        str(base / "lexical_questions.json"),
        str(base / "mixed_questions.json"),
        str(base / "group_questions.json"),
    ]


def _resolve_paths(paths: list[str]) -> list[str]:
    out: list[str] = []
    for path in paths:
        p = Path(path)
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()
        out.append(str(p))
    return out


def _configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass


def main() -> int:
    _configure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description=(
            "Question-driven dialogue benchmark over prepared question banks."
        )
    )
    parser.add_argument(
        "--question_files",
        nargs="+",
        default=_default_question_paths(),
        help="Paths to JSON files with generated questions.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/no_rag_questions",
        help="Directory for detailed rows and summary JSON files.",
    )
    parser.add_argument(
        "--cashier_model",
        type=str,
        default=None,
        help="LLM model for cashier (defaults to API_MODEL/.env).",
    )
    parser.add_argument(
        "--client_model",
        type=str,
        default=None,
        help="LLM model for client clarifications (defaults to API_MODEL/.env).",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default=None,
        help="LLM model for post-dialog judge (defaults to API_MODEL/.env).",
    )
    parser.add_argument(
        "--retrieval_mode",
        choices=["none", "vector"],
        default="none",
        help=(
            "Menu grounding mode for cashier: "
            "'none' = no RAG retrieval; "
            "'vector' = standard vector RAG path (no full menu dump)."
        ),
    )
    parser.add_argument(
        "--disable_question_grounding",
        action="store_true",
        help=(
            "Disable compact machine-built grounding block from question bank row. "
            "By default it is enabled only for --retrieval_mode none."
        ),
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        default=0,
        help="Limit total questions (0 = use all remaining after filters).",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        metavar="LIST",
        help=(
            "Only run rows with these question categories "
            '(comma or space separated, matching JSON \"category\", e.g. "simple" or "simple,diet").'
        ),
    )
    parser.add_argument(
        "--trace_verbose",
        action="store_true",
        help="Store verbose llm/rag traces for each question.",
    )
    parser.add_argument(
        "--max_dialog_turns",
        type=int,
        default=4,
        help="Max cashier turns for one question dialog (clarification loop bound).",
    )
    parser.add_argument(
        "--client_nudge_on_miss",
        action="store_true",
        help=(
            "Legacy: after each cashier turn, if the expected item was not acknowledged, inject a "
            "scripted client line to reinforce the order. Default is off: end after cashier unless "
            "the cashier asks a clarifying question (then the client LLM answers)."
        ),
    )
    args = parser.parse_args()

    ensure_llm_credentials()
    question_files = _resolve_paths(args.question_files)
    questions = load_question_banks(question_files)
    category_filter = parse_category_filter(args.categories)
    if category_filter:
        before = len(questions)
        questions = filter_questions_by_categories(questions, category_filter)
        print(
            f"[question-experiment] category filter {category_filter!r}: "
            f"{before} -> {len(questions)} questions",
            flush=True,
        )
    if not questions:
        print("No valid questions loaded (after filters).", file=sys.stderr)
        return 1

    total_run = len(questions) if args.max_questions <= 0 else min(len(questions), args.max_questions)
    cfg = get_llm_runtime_config()
    print(
        "[question-experiment] LLM "
        f"provider={cfg['provider']} model={cfg['model']} base_url={cfg['base_url']}",
        flush=True,
    )
    if (cfg.get("provider") or "").strip().lower() == "ollama":
        bu = (cfg.get("base_url") or "").lower()
        if bu and "127.0.0.1" not in bu and "localhost" not in bu:
            print(
                "[question-experiment] WARN: API_PROVIDER=ollama, но LLM_BASE_URL не указывает на локальный "
                "Ollama (ожидается что-то вроде http://127.0.0.1:11434/v1). Сейчас запросы всё равно идут на "
                f"{cfg.get('base_url')!r}. Для OpenRouter в .env используйте "
                "`make question-experiment-norag-api` или выставьте API_PROVIDER=openai.",
                file=sys.stderr,
                flush=True,
            )
    print(
        f"[question-experiment] loaded {len(questions)} questions; will run {total_run} "
        f"(max_dialog_turns={max(1, args.max_dialog_turns)}, trace_verbose={args.trace_verbose}, "
        f"client_nudge_on_miss={args.client_nudge_on_miss}"
        f"{f', categories={category_filter!r}' if category_filter else ''})",
        flush=True,
    )

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    question_grounding_enabled = (
        False if args.disable_question_grounding else (args.retrieval_mode == "none")
    )
    rows = run_question_dialog_experiment(
        questions,
        cashier_model=args.cashier_model,
        client_model=args.client_model,
        judge_model=args.judge_model,
        retrieval_mode=args.retrieval_mode,
        use_question_grounding=question_grounding_enabled,
        max_questions=args.max_questions,
        max_dialog_turns=max(1, args.max_dialog_turns),
        trace_verbose=args.trace_verbose,
        incremental_save_dir=str(out_dir),
        client_nudge_on_miss=args.client_nudge_on_miss,
    )
    artifacts = save_dialogs_by_category(rows, output_dir=str(out_dir))
    summary = artifacts["summary"]
    summary["mode"] = (
        "vector_rag_question_bank"
        if args.retrieval_mode == "vector"
        else "no_rag_question_bank_grounding"
    )
    summary["retrieval_mode"] = args.retrieval_mode
    summary["question_grounding_enabled"] = question_grounding_enabled
    summary["question_files"] = question_files
    summary["cashier_model"] = args.cashier_model
    summary["client_model"] = args.client_model
    summary["judge_model"] = args.judge_model
    summary["client_nudge_on_miss"] = args.client_nudge_on_miss
    summary["category_filter"] = category_filter if category_filter else None
    summary["generated_at_unix"] = int(time.time())
    summary["elapsed_sec"] = round(time.time() - started, 3)
    with open(artifacts["summary_path"], "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Questions processed: {len(rows)}")
    print(f"Rows saved to: {artifacts['rows_path']}")
    print(f"Summary saved to: {artifacts['summary_path']}")
    print(f"Per-category dialogs root: {artifacts['by_category_root']}")
    print("Key metrics:")
    print(f"  Success@1: {summary.get('success_at_1', 0.0):.3f}")
    print(f"  Success@3: {summary.get('success_at_3', 0.0):.3f}")
    print(f"  Success@5: {summary.get('success_at_5', 0.0):.3f}")
    print(f"  Hallucination rate: {summary.get('hallucination_rate', 0.0):.3f}")
    print(f"  Constraint violation rate: {summary.get('constraint_violation_rate', 0.0):.3f}")
    print(f"  Need-to-specify rate: {summary.get('need_to_specify_rate', 0.0):.3f}")
    print(f"  Empty response rate: {summary.get('empty_response_rate', 0.0):.3f}")
    latency = summary.get("latency_ms", {})
    print(f"  Latency median (ms): {float(latency.get('median', 0.0)):.1f}")
    print(f"  Latency p95 (ms): {float(latency.get('p95', 0.0)):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
