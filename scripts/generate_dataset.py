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
        "--model", type=str, default="gpt-4o-mini",
        help="Модель OpenAI (по умолчанию gpt-4o-mini).",
    )
    parser.add_argument(
        "--max_turns", type=int, default=20,
        help="Максимум ходов в одном диалоге (по умолчанию 20).",
    )
    args = parser.parse_args()

    num = args.num_dialogs
    out_dir = args.output_dir
    model = args.model
    rag_top_k = 0 if args.no_rag else 3
    mode_label = "non-RAG" if args.no_rag else "RAG"

    print(f"=== Генерация {num} диалогов ({mode_label}, model={model}) ===")
    print(f"    output_dir={out_dir}  max_turns={args.max_turns}")
    print()

    gen = ProfileGenerator()
    client_agent = ClientAgent(model=model)
    cashier_agent = CashierAgent(model=model, rag_top_k=rag_top_k)

    ok = 0
    errors = 0
    t0 = time.time()

    try:
        for dialog_id in range(1, num + 1):
            profile = gen.generate()
            try:
                history, profile, order, flags = simulate_dialog(
                    profile=profile,
                    max_turns=args.max_turns,
                    client_agent=client_agent,
                    cashier_agent=cashier_agent,
                )
                save_dialog(dialog_id, profile, history, order, flags, output_dir=out_dir)
                ok += 1
            except Exception:
                errors += 1
                print(f"  [!] Ошибка в диалоге {dialog_id}:")
                traceback.print_exc(limit=2, file=sys.stdout)

            if dialog_id % 10 == 0 or dialog_id == num:
                elapsed = time.time() - t0
                rate = dialog_id / elapsed if elapsed > 0 else 0
                print(
                    f"  [{dialog_id}/{num}]  ok={ok}  errors={errors}  "
                    f"elapsed={elapsed:.1f}s  rate={rate:.2f} d/s"
                )

    except KeyboardInterrupt:
        print(f"\n  Прервано пользователем на диалоге {dialog_id}.")

    elapsed_total = time.time() - t0
    print(f"\n=== Итого ===")
    print(f"  Успешно: {ok}  Ошибок: {errors}  Время: {elapsed_total:.1f}s")

    # JSON-сводка
    if ok > 0:
        summaries = aggregate_stats(out_dir)
        print(f"  summary.json: {len(summaries)} записей")
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
