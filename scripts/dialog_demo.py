"""
Демо конвейера диалога.
Запуск: python scripts/dialog_demo.py

Требуется OPENAI_API_KEY и загруженная Chroma (scripts/load_chroma.py).
"""

import argparse

import _bootstrap

_bootstrap.ensure_src()

from mcd_voice.dialog import simulate_dialog, print_dialog


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
    return parser.parse_args()


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
    elif stage == "finished":
        print(f"[done] {event['message']}", flush=True)


if __name__ == "__main__":
    args = _parse_args()
    try:
        h, p, o, f = simulate_dialog(
            max_turns=args.max_turns,
            progress_callback=_progress_printer,
            collect_rag_trace=args.rag_trace,
            collect_llm_trace=args.llm_trace,
        )
        print_dialog(h, p, o, f)
    except RuntimeError as e:
        print(f"Ошибка: {e}")
        print(
            "Проверьте OPENAI_API_KEY и наличие chroma_db после load_chroma.py."
        )
