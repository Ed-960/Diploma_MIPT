"""
Массовая генерация синтетических диалогов для дипломного исследования.

Запуск (из корня репозитория):
  python scripts/generate_dataset.py --num_dialogs 100 --output_dir dialogs_rag
  python scripts/generate_dataset.py --num_dialogs 100 --output_dir dialogs_norag --no_rag
  python scripts/generate_dataset.py --num_dialogs 5 --model gpt-4o-mini   # тест

Для сравнения RAG vs non-RAG:
  1) сгенерировать два набора (с --no_rag и без);
  2) scripts/compare_rag.py.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback

import _bootstrap

_bootstrap.ensure_src()

from mcd_voice.dialog.pipeline import simulate_dialog
from mcd_voice.dialog.save_dialog import aggregate_stats, save_dialog
from mcd_voice.llm import CashierAgent, ClientAgent
from mcd_voice.profile import ProfileGenerator


def _find_next_id(output_dir: str) -> int:
    """Return the next available dialog_id (max existing + 1, or 1 if empty)."""
    from pathlib import Path
    import re
    d = Path(output_dir)
    if not d.is_dir():
        return 1
    max_id = 0
    for f in d.glob("dialog_*.json"):
        m = re.search(r"dialog_(\d+)\.json$", f.name)
        if m:
            max_id = max(max_id, int(m.group(1)))
    return max_id + 1


def _make_dialog_progress_printer(dialog_id: int, total: int):
    def _printer(event: dict[str, object]) -> None:
        stage = event["stage"]
        prefix = f"  [dialog {dialog_id}/{total}]"
        if stage == "greeting_start":
            print(f"{prefix} greeting...", flush=True)
        elif stage == "turn_start":
            print(
                f"{prefix} turn {event['turn']}/{event['max_turns']}",
                flush=True,
            )
        elif stage == "finished":
            print(f"{prefix} done: {event['message']}", flush=True)

    return _printer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Массовая генерация синтетических диалогов (диплом).",
    )
    parser.add_argument(
        "--num_dialogs", type=int, default=100,
        help="Количество диалогов (по умолчанию 100).",
    )
    parser.add_argument(
        "--output_dir", type=str, default="dialogs",
        help="Каталог для сохранения JSON-файлов (по умолчанию dialogs/).",
    )
    parser.add_argument(
        "--no_rag", action="store_true",
        help="Отключить RAG у кассира (для сравнения с RAG-версией).",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Модель LLM; если не указана, берётся из API_MODEL/.env.",
    )
    parser.add_argument(
        "--max_turns", type=int, default=20,
        help="Максимум ходов в одном диалоге (по умолчанию 20).",
    )
    parser.add_argument(
        "--rag_trace",
        action="store_true",
        help="Сохранять в validation_flags.rag_trace события RAG (удлиняет JSON).",
    )
    parser.add_argument(
        "--llm_trace",
        action="store_true",
        help="Сохранять в validation_flags.llm_trace вызовы mini-LLM/LLM (preview).",
    )
    args = parser.parse_args()

    num = args.num_dialogs
    out_dir = args.output_dir
    model = args.model
    rag_top_k = 0 if args.no_rag else 3
    mode_label = "non-RAG" if args.no_rag else "RAG"

    gen = ProfileGenerator()
    client_agent = ClientAgent(model=model)
    cashier_agent = CashierAgent(model=model, rag_top_k=rag_top_k)
    print(
        f"=== Генерация {num} диалогов "
        f"({mode_label}, model={client_agent.model}) ==="
    )
    print(
        f"    output_dir={out_dir}  max_turns={args.max_turns}"
        f"  rag_trace={args.rag_trace}  llm_trace={args.llm_trace}"
    )
    print()

    start_id = _find_next_id(out_dir)
    if start_id > 1:
        print(f"    В «{out_dir}» уже есть файлы; нумерация с {start_id}.")

    ok = 0
    errors = 0
    t0 = time.time()

    try:
        for i in range(num):
            dialog_id = start_id + i
            profile = gen.generate()
            try:
                history, profile, order, flags = simulate_dialog(
                    profile=profile,
                    max_turns=args.max_turns,
                    client_agent=client_agent,
                    cashier_agent=cashier_agent,
                    progress_callback=_make_dialog_progress_printer(i + 1, num),
                    collect_rag_trace=args.rag_trace,
                    collect_llm_trace=args.llm_trace,
                )
                save_dialog(dialog_id, profile, history, order, flags, output_dir=out_dir)
                ok += 1
            except Exception:
                errors += 1
                print(f"  [!] Ошибка в диалоге {dialog_id} (#{i+1}/{num}):")
                traceback.print_exc(limit=2, file=sys.stdout)

            if (i + 1) % 10 == 0 or (i + 1) == num:
                done = i + 1
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                rate_s = f"{rate:.4f}" if rate < 0.01 else f"{rate:.2f}"
                print(
                    f"  [{done}/{num}]  ok={ok}  errors={errors}  "
                    f"elapsed={elapsed:.1f}s  rate={rate_s} d/s"
                )

    except KeyboardInterrupt:
        print(f"\n  Прервано пользователем на диалоге {dialog_id} (#{i+1}/{num}).")

    elapsed_total = time.time() - t0
    print(f"\n=== Итого ===")
    print(f"  Успешно: {ok}  Ошибок: {errors}  Время: {elapsed_total:.1f}s")

    # JSON-сводка по всем dialog_*.json в output_dir (не только этот запуск).
    if ok > 0:
        summaries = aggregate_stats(out_dir)
        print(
            f"  summary.json: {len(summaries)} записей "
            f"(все файлы в «{out_dir}»; в этом запуске ок={ok})."
        )
        _print_stats(summaries)


def _print_stats(summaries: list[dict]) -> None:
    total = len(summaries)
    if total == 0:
        return

    allergen_v = sum(1 for s in summaries if s.get("allergen_violation"))
    calorie_w = sum(1 for s in summaries if s.get("calorie_warning"))
    empty_o = sum(1 for s in summaries if s.get("empty_order"))
    turns_list = [s.get("turns", 0) for s in summaries]
    avg_turns = sum(turns_list) / total

    print(f"\n=== Статистика ({total} диалогов) ===")
    print(f"  allergen_violation: {allergen_v} ({allergen_v/total:.1%})")
    print(f"  calorie_warning:    {calorie_w} ({calorie_w/total:.1%})")
    print(f"  empty_order:        {empty_o} ({empty_o/total:.1%})")
    print(f"  avg turns/dialog:   {avg_turns:.1f}")


if __name__ == "__main__":
    main()
