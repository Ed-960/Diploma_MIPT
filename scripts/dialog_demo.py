"""
Демо конвейера диалога.
Запуск: python scripts/dialog_demo.py

Требуется OPENAI_API_KEY и загруженная Chroma (scripts/load_chroma.py).
"""

import argparse

import _bootstrap

_bootstrap.ensure_src()

from mcd_voice.dialog import simulate_dialog, print_dialog
from mcd_voice.dialog.trace_format import (
    format_trace_event_pretty,
    summarize_llm_event,
    summarize_rag_event,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Демо одного полного диалога.")
    parser.add_argument(
        "--max_turns",
        type=int,
        default=10,
        help="Максимум ходов в одном диалоге (по умолчанию 10).",
    )
    parser.add_argument(
        "--rag_trace",
        action="store_true",
        help="Печатать в блоке флагов события RAG (validation_flags.rag_trace).",
    )
    parser.add_argument(
        "--llm_trace",
        action="store_true",
        help="Печатать в блоке флагов вызовы mini-LLM/LLM (validation_flags.llm_trace).",
    )
    parser.add_argument(
        "--print_trace",
        action="store_true",
        help="Печатать в консоль шаги RAG/mini-LLM/LLM (включает rag_trace+llm_trace).",
    )
    parser.add_argument(
        "--trace_verbose",
        action="store_true",
        help="В консоли — полные JSON событий (промпты, ответы, Chroma documents).",
    )
    parser.add_argument(
        "--realistic_cashier",
        action="store_true",
        help="Кассир не получает скрытый профиль и RAG без фильтра аллергенов по профилю.",
    )
    return parser.parse_args()


def _make_progress_printer(*, print_trace: bool, trace_verbose: bool):
    def _progress_printer(event: dict[str, object]) -> None:
        stage = event["stage"]
        if stage == "prepare":
            print("[prepare] Подготовка профиля, меню и order_state…", flush=True)
        elif stage == "greeting_start":
            print("[greeting] Кассир формирует приветствие…", flush=True)
        elif stage == "turn_start":
            print(
                f"[turn {event['turn']}/{event['max_turns']}] Начало хода.",
                flush=True,
            )
        elif stage == "client_thinking":
            print("  Клиент думает…", flush=True)
        elif stage == "cashier_thinking":
            print("  Кассир думает…", flush=True)
        elif stage == "trace_delta" and print_trace:
            label = event.get("label", "")
            turn = event.get("turn")
            turn_s = f" turn={turn}" if turn is not None else ""
            print(f"  --- trace: {label}{turn_s} ---", flush=True)
            for ev in event.get("rag_events") or []:
                block = (
                    format_trace_event_pretty(ev)
                    if trace_verbose
                    else summarize_rag_event(ev)
                )
                for line in block.splitlines():
                    print(f"    | {line}", flush=True)
            for ev in event.get("llm_events") or []:
                block = (
                    format_trace_event_pretty(ev)
                    if trace_verbose
                    else summarize_llm_event(ev)
                )
                for line in block.splitlines():
                    print(f"    | {line}", flush=True)
        elif stage == "finished":
            print(f"[done] {event['message']}", flush=True)

    return _progress_printer


if __name__ == "__main__":
    args = _parse_args()
    collect_rag = args.rag_trace or args.print_trace
    collect_llm = args.llm_trace or args.print_trace
    try:
        h, p, o, f = simulate_dialog(
            max_turns=args.max_turns,
            progress_callback=_make_progress_printer(
                print_trace=args.print_trace,
                trace_verbose=args.trace_verbose,
            ),
            collect_rag_trace=collect_rag,
            collect_llm_trace=collect_llm,
            emit_trace_progress=args.print_trace,
            trace_verbose=args.trace_verbose,
            realistic_cashier=args.realistic_cashier,
        )
        print_dialog(h, p, o, f)
    except RuntimeError as e:
        print(f"Ошибка: {e}")
        print(
            "Проверьте OPENAI_API_KEY и наличие chroma_db после load_chroma.py."
        )
