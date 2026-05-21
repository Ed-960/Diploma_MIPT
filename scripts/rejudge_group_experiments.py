#!/usr/bin/env python3
"""
Пересчёт judge + metrics по сохранённым group-диалогам (без кассира/клиента).

Читает question_*.json в norag_group / vec_rag_group, подставляет expected_item из банка,
пересчитывает эвристики (с фиксами CV), вызывает DialogJudge на history (без старого judge),
обновляет metrics через build_metrics_from_judge.

Запуск из корня репозитория:
  python scripts/rejudge_group_experiments.py
  python scripts/rejudge_group_experiments.py --limit 5
  make rejudge-group
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import _bootstrap

_bootstrap.ensure_src()

from mcd_voice.dialog.question_experiment import (  # noqa: E402
    DialogJudge,
    build_judge_comparison,
    build_menu_index,
    build_metrics_from_judge,
    detect_need_to_specify,
    evaluate_dialog_audit,
    evaluate_single_turn_metrics,
    extract_mentioned_menu_items,
    is_empty_response,
    _load_menu_json_text,
)
from mcd_voice.llm import ensure_llm_credentials, get_llm_runtime_config  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DEFAULT_BANK = REPO / "questions" / "group_questions.json"
DEFAULT_RUN_DIRS = (
    REPO / "experiments" / "no-rag" / "norag_group",
    REPO / "experiments" / "rag" / "vec_rag_group",
)


def _load_bank(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for row in data:
        if not isinstance(row, dict):
            continue
        q = str(row.get("question") or "").strip()
        if q:
            out[q] = row
    return out


def _dialogue_history(data: dict[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for turn in data.get("history") or []:
        if not isinstance(turn, dict):
            continue
        sp = str(turn.get("speaker") or "").strip().lower()
        if sp in {"client", "cashier"}:
            out.append({"speaker": sp, "text": str(turn.get("text") or "")})
    return out


def _rejudge_file(
    path: Path,
    bank: dict[str, dict[str, Any]],
    menu_names: list[str],
    menu_by_name: dict[str, Any],
    menu_json_text: str,
    judge: DialogJudge,
    *,
    dry_run: bool,
) -> dict[str, int]:
    stats = {"ok": 0, "skip": 0, "error": 0}
    data = json.loads(path.read_text(encoding="utf-8"))
    q = str(data.get("question") or "").strip()
    if not q or q not in bank:
        stats["skip"] += 1
        return stats

    bank_row = bank[q]
    row = {
        "question": q,
        "category": bank_row.get("category") or data.get("category") or "group",
        "expected_item": bank_row.get("expected_item"),
        "expected_constraints": bank_row.get("expected_constraints")
        or data.get("expected_constraints")
        or [],
    }
    history = _dialogue_history(data)
    if not history or not any(t["speaker"] == "cashier" for t in history):
        stats["skip"] += 1
        return stats

    cashier_text = str(data.get("final_cashier_response") or "")
    for t in reversed(history):
        if t["speaker"] == "cashier":
            cashier_text = t["text"]
            break
    mentions = list(data.get("mentioned_items") or [])
    if not mentions and cashier_text:
        mentions = extract_mentioned_menu_items(cashier_text, menu_names)
    need_to_specify = detect_need_to_specify(cashier_text, mentions)
    empty = is_empty_response(cashier_text)
    heur = evaluate_single_turn_metrics(
        question_row=row,
        response_text=cashier_text,
        mentioned_items=mentions,
        menu_by_name=menu_by_name,
        all_menu_names=menu_names,
    )
    audit = evaluate_dialog_audit(
        question_row=row,
        response_text=cashier_text,
        mentioned_items=mentions,
        menu_by_name=menu_by_name,
        heuristic_metrics=heur,
        need_to_specify=need_to_specify,
    )

    if dry_run:
        stats["ok"] += 1
        return stats

    judge_raw, judge_json = judge.evaluate(
        menu_json=menu_json_text,
        question_row=row,
        history=history,
        heuristic_metrics=heur,
    )
    final_metrics, metric_sources = build_metrics_from_judge(
        judge_parsed=judge_json,
        heuristic_metrics=heur,
    )
    judge_vs = build_judge_comparison(
        final_metrics=final_metrics,
        heuristic_metrics=heur,
        audit=audit,
    )
    history_out = list(history)
    history_out.append({"speaker": "judge", "text": judge_raw})

    data["expected_item"] = row["expected_item"]
    data["expected_constraints"] = row["expected_constraints"]
    data["mentioned_items"] = mentions
    data["need_to_specify"] = need_to_specify
    data["empty_response"] = empty
    data["heuristic_metrics"] = heur
    data["audit"] = audit
    data["metrics"] = final_metrics
    data["metric_sources"] = metric_sources
    data["judge"] = {
        "raw_response": judge_raw,
        "parsed": judge_json,
        "vs_heuristic": judge_vs,
        "rejudge_at_unix": int(time.time()),
    }
    data["metrics_rejudge_note"] = (
        "judge re-run on saved client/cashier history; expected_item from group_questions.json; "
        "metrics = build_metrics_from_judge (no offline group_rescore)."
    )
    if "metrics_rescore_note" in data:
        del data["metrics_rescore_note"]
    data["history"] = history_out

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stats["ok"] += 1
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Re-judge saved group question JSON (no dialog regen).")
    ap.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    ap.add_argument("--run-dir", type=Path, action="append", default=[])
    ap.add_argument("--judge-model", type=str, default=None)
    ap.add_argument("--limit", type=int, default=0, help="Max files per run dir (0 = all)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ensure_llm_credentials()
    cfg = get_llm_runtime_config()
    print(
        f"[rejudge-group] LLM provider={cfg['provider']} model={cfg['model']}",
        flush=True,
    )

    bank = _load_bank(args.bank.resolve())
    run_dirs = [p.resolve() for p in args.run_dir] if args.run_dir else list(DEFAULT_RUN_DIRS)
    menu_names, menu_items = build_menu_index()
    menu_by_name = {x.name: x for x in menu_items}
    menu_json_text = _load_menu_json_text()
    judge = DialogJudge(model=args.judge_model)

    totals = {"ok": 0, "skip": 0, "error": 0}
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            print(f"SKIP: {run_dir}", file=sys.stderr)
            continue
        paths = sorted(run_dir.glob("question_*.json"))
        if args.limit > 0:
            paths = paths[: args.limit]
        print(f"\n=== {run_dir.relative_to(REPO)} ({len(paths)} files) ===", flush=True)
        for i, path in enumerate(paths, start=1):
            print(f"  [{i}/{len(paths)}] {path.name} …", flush=True)
            try:
                s = _rejudge_file(
                    path,
                    bank,
                    menu_names,
                    menu_by_name,
                    menu_json_text,
                    judge,
                    dry_run=args.dry_run,
                )
            except Exception as exc:
                print(f"  ERROR {path.name}: {exc}", file=sys.stderr, flush=True)
                s = {"ok": 0, "skip": 0, "error": 1}
            for k in totals:
                totals[k] += s.get(k, 0)
            if s.get("ok"):
                print(f"  [{i}/{len(paths)}] {path.name} ok", flush=True)

    print(
        f"\nDone: ok={totals['ok']} skip={totals['skip']} error={totals['error']}"
        + (" (dry-run)" if args.dry_run else ""),
        flush=True,
    )
    if not args.dry_run and totals["ok"]:
        print("Пересоберите отчёт: make analyze-question-experiments", flush=True)
    return 1 if totals["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
