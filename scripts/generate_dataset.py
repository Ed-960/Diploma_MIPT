"""
Массовая генерация синтетических диалогов для дипломного исследования.

Запуск (из корня репозитория):
  python scripts/generate_dataset.py --num_dialogs 100 --output_dir dialogs_rag
  python scripts/generate_dataset.py --num_dialogs 100 --output_dir dialogs_norag --no_rag
  python scripts/generate_dataset.py --num_dialogs 5 --model gpt-4o-mini   # тест
  python scripts/generate_dataset.py --profiles_file profiles.json --num_dialogs 50 --shuffle_profiles

Для сравнения RAG vs non-RAG:
  1) сгенерировать два набора (с --no_rag и без);
  2) scripts/compare_rag.py.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import random
import sys
import threading
import time
import traceback
from typing import Any

import _bootstrap

_bootstrap.ensure_src()

from mcd_voice.dialog.pipeline import simulate_dialog
from mcd_voice.dialog.trace_format import (
    format_trace_event_pretty,
    summarize_llm_event,
    summarize_rag_event,
)
from mcd_voice.dialog.save_dialog import aggregate_stats, save_dialog
from mcd_voice.llm import CashierAgent, ClientAgent, get_llm_runtime_config
from mcd_voice.llm.agent import (
    RAG_FULL_TOP_K,
    _resolve_model as _resolve_llm_model,
)
from mcd_voice.profile import ProfileGenerator


def _configure_utf8_stdio() -> None:
    """
    Make console output robust on Windows terminals with legacy code pages.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass


def _load_profiles(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("profiles_file must contain a JSON array of profiles")
    profiles = [p for p in data if isinstance(p, dict)]
    if not profiles:
        raise ValueError("profiles_file contains no valid profiles")
    return profiles


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


def _make_dialog_progress_printer(
    dialog_id: int,
    total: int,
    *,
    print_trace: bool = False,
    trace_verbose: bool = False,
):
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
        elif stage == "trace_delta" and print_trace:
            label = event.get("label", "")
            turn = event.get("turn")
            turn_s = f" turn={turn}" if turn is not None else ""
            print(f"{prefix} --- trace: {label}{turn_s} ---", flush=True)
            for ev in event.get("rag_events") or []:
                block = (
                    format_trace_event_pretty(ev)
                    if trace_verbose
                    else summarize_rag_event(ev)
                )
                for line in block.splitlines():
                    print(f"{prefix}   | {line}", flush=True)
            for ev in event.get("llm_events") or []:
                block = (
                    format_trace_event_pretty(ev)
                    if trace_verbose
                    else summarize_llm_event(ev)
                )
                for line in block.splitlines():
                    print(f"{prefix}   | {line}", flush=True)
        elif stage == "finished":
            print(f"{prefix} done: {event['message']}", flush=True)

    return _printer


def _run_one_dialog(
    *,
    idx: int,
    dialog_id: int,
    total: int,
    profile: dict,
    out_dir: str,
    max_turns: int,
    client_model: str | None,
    cashier_model: str | None,
    rag_top_k: int,
    collect_rag: bool,
    collect_llm: bool,
    print_trace: bool,
    trace_verbose: bool,
    print_lock: threading.Lock,
    realistic_cashier: bool,
    retry_on_loop: int,
) -> None:
    # Создаём агентов внутри воркера: меньше shared-state, стабильнее при параллели.
    client_agent = ClientAgent(model=client_model, trace_verbose=trace_verbose)
    cashier_agent = CashierAgent(
        model=cashier_model,
        rag_top_k=rag_top_k,
        trace_verbose=trace_verbose,
        realistic_cashier=realistic_cashier,
    )
    progress = _make_dialog_progress_printer(
        idx + 1,
        total,
        print_trace=print_trace,
        trace_verbose=trace_verbose,
    )

    def _thread_safe_progress(event: dict[str, object]) -> None:
        with print_lock:
            progress(event)

    attempts = 0
    while True:
        history, profile_out, order, flags = simulate_dialog(
            profile=profile,
            max_turns=max_turns,
            client_agent=client_agent,
            cashier_agent=cashier_agent,
            progress_callback=_thread_safe_progress,
            collect_rag_trace=collect_rag,
            collect_llm_trace=collect_llm,
            emit_trace_progress=print_trace,
            trace_verbose=trace_verbose,
            realistic_cashier=realistic_cashier,
        )
        reached_limit = flags.get("turns", 0) >= max_turns * 2
        likely_loop = bool(
            flags.get("loop_detected")
            or flags.get("stall_detected")
            or (reached_limit and not order.get("order_complete", False))
        )
        if not likely_loop or attempts >= retry_on_loop:
            break
        attempts += 1
        with print_lock:
            print(
                f"  [dialog {idx + 1}/{total}] retry {attempts}/{retry_on_loop}: loop/stall detected",
                flush=True,
            )
    if attempts:
        flags = {**flags, "regen_retries": attempts}
    save_dialog(dialog_id, profile_out, history, order, flags, output_dir=out_dir)


def main() -> None:
    _configure_utf8_stdio()
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
        "--client_model", type=str, default=None,
        help="Отдельная модель для клиента (если не задано, используется --model/API_MODEL).",
    )
    parser.add_argument(
        "--cashier_model", type=str, default=None,
        help="Отдельная модель для кассира (если не задано, используется --model/API_MODEL).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Seed для воспроизводимой генерации профилей (on-the-fly).",
    )
    parser.add_argument(
        "--profiles_file", type=str, default=None,
        help="JSON-массив профилей; если задан, диалоги строятся по нему (см. --shuffle_profiles).",
    )
    parser.add_argument(
        "--shuffle_profiles",
        action="store_true",
        help="С --profiles_file: брать NUM профилей случайной выборкой без повторений, "
        "а не подряд с начала файла. С --seed порядок воспроизводим.",
    )
    parser.add_argument(
        "--max_turns", type=int, default=20,
        help="Максимум ходов в одном диалоге (по умолчанию 20).",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Число параллельных воркеров (потоки). 1 = последовательный режим.",
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
    parser.add_argument(
        "--print_trace",
        action="store_true",
        help="Печатать в консоль шаги RAG/mini-LLM/LLM (включает сбор rag_trace+llm_trace).",
    )
    parser.add_argument(
        "--trace_verbose",
        action="store_true",
        help="Полные тела промптов/ответов и сырые Chroma (сильно удлиняет JSON при rag/llm trace).",
    )
    parser.add_argument(
        "--realistic_cashier",
        action="store_true",
        help="Кассир без скрытого профиля: не видит психотип/группу/ограничения до реплик клиента; "
        "RAG без фильтра аллергенов по профилю.",
    )
    parser.add_argument(
        "--retry_on_loop", type=int, default=1,
        help="Сколько раз перегенерировать диалог при loop/stall/max_turns без завершения.",
    )
    args = parser.parse_args()

    num = args.num_dialogs
    out_dir = args.output_dir
    default_model = args.model
    client_model = args.client_model or default_model
    cashier_model = args.cashier_model or default_model
    rag_top_k = 0 if args.no_rag else RAG_FULL_TOP_K
    mode_label = "non-RAG" if args.no_rag else "RAG"
    workers = max(1, args.workers)

    collect_rag = args.rag_trace or args.print_trace
    collect_llm = args.llm_trace or args.print_trace

    rng = random.Random(args.seed) if args.seed is not None else None
    gen = ProfileGenerator(rng=rng)
    profiles: list[dict] | None = None
    if args.profiles_file:
        profiles = _load_profiles(args.profiles_file)
        if num > len(profiles):
            print(
                f"  [!] num_dialogs={num}, но в profiles_file только {len(profiles)} профилей; "
                f"будет сгенерировано {len(profiles)} диалогов.",
                flush=True,
            )
            num = len(profiles)
    if args.shuffle_profiles and not args.profiles_file:
        print(
            "  [!] --shuffle_profiles без --profiles_file игнорируется.",
            flush=True,
        )
    # Профили подготавливаем заранее (детерминированность и отсутствие shared RNG в потоках).
    if profiles is not None:
        if args.shuffle_profiles:
            pick_rng = random.Random(args.seed) if args.seed is not None else random.Random()
            profiles_for_run = pick_rng.sample(profiles, k=num)
        else:
            profiles_for_run = [profiles[i] for i in range(num)]
    else:
        profiles_for_run = [gen.generate() for _ in range(num)]
    display_client_model = _resolve_llm_model(client_model)
    display_cashier_model = _resolve_llm_model(cashier_model)
    print(
        f"=== Генерация {num} диалогов "
        f"({mode_label}, client_model={display_client_model}, cashier_model={display_cashier_model}) ==="
    )
    print(
        f"    output_dir={out_dir}  max_turns={args.max_turns}"
        f"  workers={workers}"
        f"  rag_trace={collect_rag}  llm_trace={collect_llm}"
        f"  print_trace={args.print_trace}  trace_verbose={args.trace_verbose}"
        f"  seed={args.seed}"
        f"  profiles_file={'yes' if args.profiles_file else 'no'}"
        f"  shuffle_profiles={bool(args.profiles_file and args.shuffle_profiles)}"
        f"  realistic_cashier={args.realistic_cashier}"
        f"  retry_on_loop={max(0, args.retry_on_loop)}"
    )
    if args.print_trace or args.trace_verbose:
        rt = get_llm_runtime_config()
        rw = os.environ.get("REWRITE_MODEL") or display_client_model
        print(
            f"    metrics: provider={rt.get('provider')} base_url={rt.get('base_url')!r} "
            f"dialog_model={rt.get('model')!r} rewrite_model={rw!r} rag_top_k={rag_top_k}",
            flush=True,
        )
    print()

    # Pre-warm Chroma + embedding model in the main thread before spawning workers.
    # Without this, all workers race to call get_menu_collection() on their first
    # search, causing all but one to block on _CHROMA_LOCK while the model loads.
    if rag_top_k > 0:
        from mcd_voice.menu.chroma import get_menu_collection
        print("  Прогрев Chroma / embedding-модели...", flush=True)
        get_menu_collection()
        print("  Chroma готова.\n", flush=True)

    start_id = _find_next_id(out_dir)
    if start_id > 1:
        print(f"    В «{out_dir}» уже есть файлы; нумерация с {start_id}.")

    ok = 0
    errors = 0
    t0 = time.time()
    done = 0
    print_lock = threading.Lock()
    ex: ThreadPoolExecutor | None = None
    future_to_meta: dict[Any, tuple[int, int]] = {}

    try:
        ex = ThreadPoolExecutor(max_workers=workers)
        future_to_meta = {
            ex.submit(
                _run_one_dialog,
                idx=i,
                dialog_id=start_id + i,
                total=num,
                profile=profiles_for_run[i],
                out_dir=out_dir,
                max_turns=args.max_turns,
                client_model=client_model,
                cashier_model=cashier_model,
                rag_top_k=rag_top_k,
                collect_rag=collect_rag,
                collect_llm=collect_llm,
                print_trace=args.print_trace,
                trace_verbose=args.trace_verbose,
                print_lock=print_lock,
                realistic_cashier=args.realistic_cashier,
                retry_on_loop=max(0, args.retry_on_loop),
            ): (i, start_id + i)
            for i in range(num)
        }
        for fut in as_completed(future_to_meta):
            i, dialog_id = future_to_meta[fut]
            done += 1
            try:
                fut.result()
                ok += 1
            except Exception:
                errors += 1
                with print_lock:
                    print(f"  [!] Ошибка в диалоге {dialog_id} (#{i+1}/{num}):")
                    traceback.print_exc(limit=2, file=sys.stdout)

            if done % 10 == 0 or done == num:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                rate_s = f"{rate:.4f}" if rate < 0.01 else f"{rate:.2f}"
                with print_lock:
                    print(
                        f"  [{done}/{num}]  ok={ok}  errors={errors}  "
                        f"elapsed={elapsed:.1f}s  rate={rate_s} d/s"
                    )

    except KeyboardInterrupt:
        pending = 0
        if ex is not None:
            for fut in future_to_meta:
                if not fut.done():
                    if fut.cancel():
                        pending += 1
            ex.shutdown(wait=False, cancel_futures=True)
        print(f"\n  Прервано пользователем. Отменено ожидающих задач: {pending}.")
    finally:
        if ex is not None:
            ex.shutdown(wait=False, cancel_futures=True)

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
    loops = sum(1 for s in summaries if s.get("loop_detected"))
    stalls = sum(1 for s in summaries if s.get("stall_detected"))
    turns_list = [s.get("turns", 0) for s in summaries]
    avg_turns = sum(turns_list) / total

    print(f"\n=== Статистика ({total} диалогов) ===")
    print(f"  allergen_violation: {allergen_v} ({allergen_v/total:.1%})")
    print(f"  calorie_warning:    {calorie_w} ({calorie_w/total:.1%})")
    print(f"  empty_order:        {empty_o} ({empty_o/total:.1%})")
    print(f"  loop_detected:      {loops} ({loops/total:.1%})")
    print(f"  stall_detected:     {stalls} ({stalls/total:.1%})")
    print(f"  avg turns/dialog:   {avg_turns:.1f}")


if __name__ == "__main__":
    main()
