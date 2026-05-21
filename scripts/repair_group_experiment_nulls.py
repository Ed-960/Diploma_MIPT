#!/usr/bin/env python3
"""
Backfill expected_item in saved group question experiments from questions/group_questions.json.

Does not re-run LLM / dialog generation. Updates question_*.json in:
  experiments/no-rag/norag_group
  experiments/rag/vec_rag_group

Also refreshes heuristic_metrics and audit (offline). Optionally rebuilds metrics
from existing judge JSON + new heuristics (--update-metrics, default on).

--rescore-group: offline fix for saved group runs (no dialog regen):
  - constraint_violation / reasons from heuristic (incl. Milk false-positive fix)
  - success_at_*: keep judge pass, or pass if group_completeness>=1 without heur. CV/hallucination
  (success@1 по одному expected_item для group завышает промахи — см. отчёт)

Usage (repo root):
  python scripts/repair_group_experiment_nulls.py
  python scripts/repair_group_experiment_nulls.py --rescore-group
  python scripts/repair_group_experiment_nulls.py --refresh-from-judge
  python scripts/repair_group_experiment_nulls.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import _bootstrap

_bootstrap.ensure_src()

from mcd_voice.dialog.question_experiment import (  # noqa: E402
    build_menu_index,
    build_metrics_from_judge,
    evaluate_dialog_audit,
    evaluate_single_turn_metrics,
    extract_mentioned_menu_items,
)

REPO = Path(__file__).resolve().parent.parent
DEFAULT_BANK = REPO / "questions" / "group_questions.json"
DEFAULT_RUN_DIRS = (
    REPO / "experiments" / "no-rag" / "norag_group",
    REPO / "experiments" / "rag" / "vec_rag_group",
)


def _load_bank(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array: {path}")
    out: dict[str, dict[str, Any]] = {}
    for row in data:
        if not isinstance(row, dict):
            continue
        q = str(row.get("question") or "").strip()
        if not q:
            continue
        if q in out:
            raise ValueError(f"Duplicate question in bank: {q[:80]!r}")
        out[q] = row
    return out


def _group_rescore_success(
    *,
    judge_parsed: dict[str, Any],
    heuristic_metrics: dict[str, Any],
) -> bool:
    """
    Group pass: keep judge success; else full group_completeness without heuristic CV.

    Heuristic item-match (success@1 по одному expected_item) для group не используем —
    no-rag с question-grounding иначе несопоставим с vector RAG.
    """
    if judge_parsed.get("success_at_1"):
        return True
    gc = heuristic_metrics.get("group_completeness")
    if gc is None:
        gc = judge_parsed.get("group_completeness")
    if gc is None:
        return False
    return (
        float(gc) >= 1.0
        and not heuristic_metrics.get("constraint_violation")
        and not bool(judge_parsed.get("hallucination"))
    )


def _apply_group_rescore(data: dict[str, Any]) -> None:
    judge_block = data.get("judge") or {}
    judge_parsed = judge_block.get("parsed") if isinstance(judge_block, dict) else {}
    if not isinstance(judge_parsed, dict):
        judge_parsed = {}
    heur = dict(data.get("heuristic_metrics") or {})
    ok = _group_rescore_success(judge_parsed=judge_parsed, heuristic_metrics=heur)
    out = dict(heur)
    out["constraint_violation"] = bool(heur.get("constraint_violation"))
    if heur.get("constraint_violation_reasons"):
        out["constraint_violation_reasons"] = list(heur["constraint_violation_reasons"])
    elif "constraint_violation_reasons" in out:
        out.pop("constraint_violation_reasons", None)
    out["hallucination"] = bool(judge_parsed.get("hallucination"))
    out["need_to_specify"] = bool(judge_parsed.get("need_to_specify"))
    out["empty_response"] = bool(judge_parsed.get("empty_response"))
    out["success_at_1"] = ok
    out["success_at_3"] = ok
    out["success_at_5"] = ok
    gc = heur.get("group_completeness")
    if gc is None:
        gc = judge_parsed.get("group_completeness")
    if gc is not None:
        out["group_completeness"] = gc
    data["metrics"] = out
    data["metric_sources"] = {
        "success_at_1": "group_offline_rescore",
        "success_at_3": "group_offline_rescore",
        "success_at_5": "group_offline_rescore",
        "constraint_violation": "heuristic_rescore",
        "hallucination": "judge",
        "need_to_specify": "judge",
        "empty_response": "judge",
        "group_completeness": "heuristic_rescore"
        if heur.get("group_completeness") is not None
        else "judge",
    }
    data["metrics_rescore_note"] = (
        "group offline rescore: CV heuristic skips per-person constraints (for=…); "
        "success = judge pass OR (group_completeness>=1 and no global CV and no judge hallucination). "
        "no-rag runs used question-bank grounding (constraint_fit_candidates); RAG used vector retrieval only."
    )


def _repair_file(
    path: Path,
    bank: dict[str, dict[str, Any]],
    menu_names: list[str],
    menu_by_name: dict[str, Any],
    *,
    dry_run: bool,
    update_metrics: bool,
    rescore_group: bool,
    refresh_from_judge: bool,
) -> dict[str, int]:
    stats = {
        "patched": 0,
        "skipped": 0,
        "missing_bank": 0,
        "metrics_updated": 0,
        "rescore_group": 0,
    }
    data = json.loads(path.read_text(encoding="utf-8"))
    q = str(data.get("question") or "").strip()
    if not q:
        stats["skipped"] += 1
        return stats
    bank_row = bank.get(q)
    if not bank_row:
        stats["missing_bank"] += 1
        return stats

    exp = bank_row.get("expected_item")
    if (
        data.get("expected_item") == exp
        and not update_metrics
        and not rescore_group
        and not refresh_from_judge
    ):
        stats["skipped"] += 1
        return stats

    row_for_eval = dict(data)
    row_for_eval["expected_item"] = exp
    if bank_row.get("expected_constraints") is not None:
        row_for_eval["expected_constraints"] = bank_row.get("expected_constraints")

    cashier_text = str(data.get("final_cashier_response") or "")
    mentions = list(data.get("mentioned_items") or [])
    if not mentions and cashier_text:
        mentions = extract_mentioned_menu_items(cashier_text, menu_names)

    need_to_specify = bool(data.get("need_to_specify"))
    heur = evaluate_single_turn_metrics(
        question_row=row_for_eval,
        response_text=cashier_text,
        mentioned_items=mentions,
        menu_by_name=menu_by_name,
        all_menu_names=menu_names,
    )
    audit = evaluate_dialog_audit(
        question_row=row_for_eval,
        response_text=cashier_text,
        mentioned_items=mentions,
        menu_by_name=menu_by_name,
        heuristic_metrics=heur,
        need_to_specify=need_to_specify,
    )

    if dry_run:
        if data.get("expected_item") is None and exp is not None:
            stats["patched"] += 1
        if rescore_group:
            stats["rescore_group"] += 1
        return stats

    data["expected_item"] = exp
    data["heuristic_metrics"] = heur
    data["audit"] = audit

    if rescore_group:
        _apply_group_rescore(data)
        stats["rescore_group"] += 1
    elif refresh_from_judge:
        judge_block = data.get("judge") or {}
        parsed = judge_block.get("parsed") if isinstance(judge_block, dict) else {}
        if isinstance(parsed, dict) and not parsed.get("parse_error"):
            final_metrics, metric_sources = build_metrics_from_judge(
                judge_parsed=parsed,
                heuristic_metrics=heur,
            )
            data["metrics"] = final_metrics
            data["metric_sources"] = metric_sources
            data["metrics_refresh_note"] = (
                "offline refresh: heuristics + build_metrics_from_judge(stored judge.parsed); no new LLM."
            )
            stats["metrics_updated"] += 1
    elif update_metrics:
        judge_block = data.get("judge") or {}
        parsed = judge_block.get("parsed") if isinstance(judge_block, dict) else {}
        if isinstance(parsed, dict) and not parsed.get("parse_error"):
            final_metrics, metric_sources = build_metrics_from_judge(
                judge_parsed=parsed,
                heuristic_metrics=heur,
            )
            data["metrics"] = final_metrics
            data["metric_sources"] = metric_sources
            stats["metrics_updated"] += 1

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stats["patched"] += 1
    return stats


def _repair_run_dir(
    run_dir: Path,
    bank: dict[str, dict[str, Any]],
    menu_names: list[str],
    menu_by_name: dict[str, Any],
    *,
    dry_run: bool,
    update_metrics: bool,
    rescore_group: bool,
    refresh_from_judge: bool,
) -> dict[str, int]:
    totals = {
        "files": 0,
        "patched": 0,
        "skipped": 0,
        "missing_bank": 0,
        "metrics_updated": 0,
        "rescore_group": 0,
    }
    if not run_dir.is_dir():
        print(f"SKIP (not a dir): {run_dir}", file=sys.stderr)
        return totals
    paths = sorted(run_dir.glob("question_*.json"))
    totals["files"] = len(paths)
    for path in paths:
        s = _repair_file(
            path,
            bank,
            menu_names,
            menu_by_name,
            dry_run=dry_run,
            update_metrics=update_metrics,
            rescore_group=rescore_group,
            refresh_from_judge=refresh_from_judge,
        )
        for k in totals:
            if k != "files":
                totals[k] += s.get(k, 0)
    return totals


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill group expected_item from question bank.")
    ap.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    ap.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=[],
        help="Repeatable; default: norag_group + vec_rag_group",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--no-update-metrics",
        action="store_true",
        help="Only fix expected_item / heuristic_metrics / audit; keep metrics as-is",
    )
    ap.add_argument(
        "--rescore-group",
        action="store_true",
        help=(
            "Replace metrics with group offline rescore (heuristic CV + group_completeness pass). "
            "Implies refresh of heuristics/audit; do not use together with default --update-metrics."
        ),
    )
    ap.add_argument(
        "--refresh-from-judge",
        action="store_true",
        help=(
            "Без LLM: пересчитать heuristics/audit и metrics = build_metrics_from_judge(старый judge.parsed, новые heuristics)."
        ),
    )
    args = ap.parse_args()

    bank_path = args.bank.resolve()
    if not bank_path.is_file():
        print(f"Bank not found: {bank_path}", file=sys.stderr)
        return 1
    bank = _load_bank(bank_path)
    print(f"Bank: {len(bank)} questions from {bank_path.relative_to(REPO)}")

    run_dirs = [p.resolve() for p in args.run_dir] if args.run_dir else list(DEFAULT_RUN_DIRS)
    menu_names, menu_items = build_menu_index()
    menu_by_name = {x.name: x for x in menu_items}
    rescore_group = bool(args.rescore_group)
    refresh_from_judge = bool(args.refresh_from_judge)
    update_metrics = (
        not args.no_update_metrics and not rescore_group and not refresh_from_judge
    )
    if rescore_group and args.no_update_metrics:
        print("Note: --rescore-group still refreshes heuristics and rewrites metrics.", flush=True)

    rc = 0
    for run_dir in run_dirs:
        print(f"\n=== {run_dir.relative_to(REPO)} ===", flush=True)
        totals = _repair_run_dir(
            run_dir,
            bank,
            menu_names,
            menu_by_name,
            dry_run=args.dry_run,
            update_metrics=update_metrics,
            rescore_group=rescore_group,
            refresh_from_judge=refresh_from_judge,
        )
        extra = ""
        if rescore_group:
            extra = f" rescore_group={totals['rescore_group']}"
        elif refresh_from_judge:
            extra = f" refresh_from_judge={totals['metrics_updated']}"
        print(
            f"  files={totals['files']} patched={totals['patched']} "
            f"skipped={totals['skipped']} missing_bank={totals['missing_bank']} "
            f"metrics_updated={totals['metrics_updated']}{extra}"
            + (" (dry-run)" if args.dry_run else ""),
            flush=True,
        )
        if totals["missing_bank"]:
            rc = 1

    if not args.dry_run and (update_metrics or rescore_group or refresh_from_judge):
        print("\nПересоберите отчёт: make analyze-question-experiments  или  make reports", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
