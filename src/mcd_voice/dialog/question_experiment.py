"""Question-bank experiments: fixed JSON questions only (no REG persona sampling).

Cashier turns are grounded by compact facts derived from the question row and
``mcd.json``. The raw full menu is reserved for offline checks/judge facts, not
dumped into the cashier prompt for every turn.

``neutral_drive_through_profile()`` is only a minimal struct so ``CashierAgent`` /
``build_initial_order_state`` match the dialog pipeline API; with ``realistic_cashier=True``
the cashier system prompt does not inject psychotype or hidden customer traits from it.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Literal, Sequence

from mcd_voice.dialog.pipeline import build_initial_order_state
from mcd_voice.config import MCD_JSON_PATH
from mcd_voice.llm import CashierAgent
from mcd_voice.llm.agent import (
    _build_openai_client,
    _call_llm,
    _resolve_model,
)
from mcd_voice.menu.dataset import load_menu_from_json
from mcd_voice.menu.chroma import get_menu_collection
from mcd_voice.profile import neutral_drive_through_profile


_EMPTY_RESPONSE_RE = re.compile(
    r"^\s*(?:i\s+did(?:n't| not)\s+understand|i\s+did(?:n't| not)\s+catch|"
    r"sorry[,.]?\s*(?:i\s+did(?:n't| not)\s+understand|could\s+you\s+repeat)|"
    r"can\s+you\s+repeat|what\?|huh\?)\s*$",
    re.IGNORECASE,
)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_MEATY_RE = re.compile(
    r"\b(beef|chicken|fish|bacon|sausage|ham|meat|pork|turkey)\b",
    re.IGNORECASE,
)

_DIET_THRESHOLDS: dict[str, tuple[str, float, str]] = {
    "low carb": ("carbs", 20.0, "max"),
    "low calorie": ("energy", 400.0, "max"),
    "high protein": ("protein", 15.0, "min"),
    "low sugar": ("total_sugar", 10.0, "max"),
    "low sodium": ("sodium", 500.0, "max"),
}
_ALLERGEN_ALIASES: dict[str, str] = {
    "milk": "milk",
    "dairy": "milk",
    "nut": "nuts",
    "nuts": "nuts",
    "peanut": "nuts",
    "fish": "fish",
    "egg": "egg",
    "eggs": "egg",
    "gluten": "cereal containing gluten",
    "sesame": "sesame",
    "soya": "soya",
    "soy": "soya",
}
_UNAVAILABLE_PATTERNS = (
    r"(?:don't|do not|cannot|can't)\s+(?:have|offer|serve|find|see|confirm)",
    r"(?:not|isn't|is not)\s+(?:available|on(?:\s+(?:the|our))?\s+menu|in(?:\s+(?:the|our))?\s+menu)",
    r"(?:no|none)\s+(?:of\s+)?(?:that|this|such)?\s*(?:item|option)?",
)
_FOREIGN_BRAND_RE = re.compile(
    r"\b(burger\s*king|kfc|wendy'?s|subway|starbucks|taco\s*bell|domino'?s|pizza\s*hut)\b",
    re.IGNORECASE,
)
_ALLERGEN_TOKEN_ALIASES: dict[str, str] = {
    "milk": "milk",
    "dairy": "milk",
    "egg": "egg",
    "eggs": "egg",
    "fish": "fish",
    "sesame": "sesame",
    "soy": "soya",
    "soya": "soya",
    "gluten": "cereal containing gluten",
    "wheat": "cereal containing gluten",
    "nut": "nuts",
    "nuts": "nuts",
    "peanut": "nuts",
}


@dataclass(frozen=True)
class MenuItem:
    name: str
    norm_name: str
    ingredients: str
    allergens: set[str]
    restriction_map: dict[str, bool]
    nutrients: dict[str, float]


@dataclass(frozen=True)
class DialogTurn:
    speaker: str
    text: str


def load_question_banks(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Question file must be JSON array: {path}")
        for i, row in enumerate(data, start=1):
            if not isinstance(row, dict):
                continue
            question = str(row.get("question") or "").strip()
            if not question:
                continue
            item = dict(row)
            item["_source_file"] = path
            item["_source_index"] = i
            rows.append(item)
    return rows


def parse_category_filter(value: str | None) -> list[str] | None:
    """
    CLI helper: comma- or whitespace-separated category labels from question JSON.

    Examples: ``"simple"``, ``"simple, allergy"``.
    Empty / whitespace-only → no filter (``None``).
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    parts = re.split(r"[\s,]+", s)
    out = [p.strip() for p in parts if p.strip()]
    return out or None


def filter_questions_by_categories(
    rows: list[dict[str, Any]],
    categories: Sequence[str] | None,
) -> list[dict[str, Any]]:
    """Keep only rows whose ``category`` field is in ``categories``. Identity match after strip."""
    if not categories:
        return rows
    allow = {str(c).strip() for c in categories if str(c).strip()}
    if not allow:
        return rows
    return [
        r
        for r in rows
        if str(r.get("category") or "").strip() in allow
    ]


def evaluate_retrieval_probe_for_row(
    question_row: dict[str, Any],
    rag_trace: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Offline retrieval metrics from a stored ``rag_trace``: rank of ``expected_item`` in
    ``candidates`` of the last ``event == 'rag'`` record.
    """
    expected = str(question_row.get("expected_item") or "").strip()
    probe: dict[str, Any] = {}
    empty_metrics = {
        "success_at_1": False,
        "success_at_3": False,
        "success_at_5": False,
    }
    if not rag_trace:
        probe["error"] = "no_rag_event"
        return dict(empty_metrics), probe

    rag_ev = None
    for ev in rag_trace:
        if isinstance(ev, dict) and ev.get("event") == "rag":
            rag_ev = ev
    if rag_ev is None:
        probe["error"] = "no_rag_event"
        return dict(empty_metrics), probe

    cands_raw = rag_ev.get("candidates") or []
    names: list[str] = []
    for c in cands_raw:
        if isinstance(c, dict) and str(c.get("name") or "").strip():
            names.append(str(c["name"]).strip())

    if not expected:
        probe["error"] = "no_expected_item"
        return dict(empty_metrics), probe

    rank: int | None = None
    for i, n in enumerate(names, start=1):
        if n == expected:
            rank = i
            break
    probe["expected_rank"] = rank
    probe["candidate_count"] = len(names)

    inj = rag_ev.get("injected_hits") or []
    inj_names = [
        str(x.get("name") or "").strip()
        for x in inj
        if isinstance(x, dict) and str(x.get("name") or "").strip()
    ]
    probe["expected_injected_hits"] = expected in inj_names

    metrics = {
        "success_at_1": rank is not None and rank <= 1,
        "success_at_3": rank is not None and rank <= 3,
        "success_at_5": rank is not None and rank <= 5,
    }
    return metrics, probe


def build_menu_index() -> tuple[list[str], list[MenuItem]]:
    _, _, metas = load_menu_from_json()
    by_name: dict[str, MenuItem] = {}
    ordered_names: list[str] = []
    for meta in metas:
        name = str(meta.get("name") or "").strip()
        if not name:
            continue
        norm = normalize_name(name)
        allergens = {str(a).strip().lower() for a in (meta.get("allergens") or []) if str(a).strip()}
        ingredients = str(meta.get("ingredients") or "")
        if name not in by_name:
            ordered_names.append(name)
            by_name[name] = MenuItem(
                name=name,
                norm_name=norm,
                ingredients=ingredients,
                allergens=allergens,
                restriction_map=_restriction_map_for_meta(meta, allergens, ingredients),
                nutrients={
                    "energy": _as_float(meta.get("energy")),
                    "protein": _as_float(meta.get("protein")),
                    "carbs": _as_float(meta.get("carbs")),
                    "total_sugar": _as_float(meta.get("total_sugar")),
                    "sodium": _as_float(meta.get("sodium")),
                },
            )
            continue
        prev = by_name[name]
        merged_allergens = set(prev.allergens)
        merged_allergens.update(allergens)
        merged_ingredients = "\n".join(x for x in [prev.ingredients, ingredients] if x)
        merged_restrictions = dict(prev.restriction_map)
        current_restrictions = _restriction_map_for_meta(meta, allergens, ingredients)
        for k, v in current_restrictions.items():
            merged_restrictions[k] = bool(merged_restrictions.get(k) or v)
        merged_nutrients = dict(prev.nutrients)
        for key in merged_nutrients:
            merged_nutrients[key] = (merged_nutrients[key] + _as_float(meta.get(key))) / 2.0
        by_name[name] = MenuItem(
            name=name,
            norm_name=norm,
            ingredients=merged_ingredients,
            allergens=merged_allergens,
            restriction_map=merged_restrictions,
            nutrients=merged_nutrients,
        )
    return ordered_names, [by_name[name] for name in ordered_names]


def build_question_grounding_context(
    question_row: dict[str, Any],
    menu_by_name: dict[str, MenuItem],
    *,
    candidate_limit: int = 8,
) -> str:
    """
    Machine-built grounding for question-bank runs.

    The full menu can be long; this compact block pins the row(s) implied by the
    prepared JSON question before the raw catalog, without using prose patterns
    from the cashier response.
    """
    expected_item = str(question_row.get("expected_item") or "").strip()
    constraints = question_row.get("expected_constraints", [])
    payload: dict[str, Any] = {
        "question": str(question_row.get("question") or "").strip(),
        "expected_item": expected_item or None,
        "expected_constraints": constraints if isinstance(constraints, list) else [],
    }
    if expected_item and expected_item in menu_by_name:
        payload["expected_item_exact_menu_match"] = _menu_item_payload(menu_by_name[expected_item])
    elif not expected_item:
        candidates: list[dict[str, Any]] = []
        for item in menu_by_name.values():
            violates, _ = detect_constraint_violation(
                expected_constraints=constraints,
                mentioned_items=[item.name],
                menu_by_name=menu_by_name,
            )
            if not violates:
                candidates.append(_menu_item_payload(item))
            if len(candidates) >= candidate_limit:
                break
        if candidates:
            payload["constraint_fit_candidates"] = candidates
    return (
        "Question-bank structured grounding (built from question JSON and mcd.json; "
        "use this before scanning the full raw menu):\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _menu_item_payload(item: MenuItem) -> dict[str, Any]:
    return {
        "name": item.name,
        "ingredients": item.ingredients,
        "allergens": sorted(item.allergens),
        "nutrients": item.nutrients,
    }


def save_incremental_question_row(output_dir: str | Path, row: dict[str, Any]) -> None:
    """
    Persist one question-dialog result as soon as it is finished.

    Writes:
    - ``incremental/question_XXXX.json`` — full row (same shape as in ``rows.json`` later);
    - appends one JSON line to ``incremental/rows_partial.jsonl`` (crash-safe progress log).
    """
    base = Path(output_dir)
    inc = base / "incremental"
    inc.mkdir(parents=True, exist_ok=True)
    qid = int(row.get("question_id") or 0)
    path = inc / f"question_{qid:04d}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=2)
    jsonl = inc / "rows_partial.jsonl"
    with open(jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def run_question_dialog_experiment(
    questions: list[dict[str, Any]],
    *,
    cashier_model: str | None = None,
    client_model: str | None = None,
    judge_model: str | None = None,
    retrieval_mode: Literal["none", "vector"] = "none",
    use_question_grounding: bool | None = None,
    max_questions: int = 0,
    max_dialog_turns: int = 4,
    trace_verbose: bool = False,
    incremental_save_dir: str | Path | None = None,
    client_nudge_on_miss: bool = False,
) -> list[dict[str, Any]]:
    # Not a "customer profile" experiment: stub dict only for pipeline API + empty order_state.
    # Client LLM (ClarifyingClient) runs only when the cashier utterance looks like a clarification
    # request (detect_need_to_specify: "?" and no menu names extracted). Scripted "nudge" follow-ups
    # when the expected item is not acknowledged happen only if client_nudge_on_miss=True.
    pipeline_stub = neutral_drive_through_profile()
    menu_names, menu_items = build_menu_index()
    menu_by_name = {x.name: x for x in menu_items}
    menu_json_text = _load_menu_json_text()
    retrieval_mode_effective = retrieval_mode.strip().lower()
    if retrieval_mode_effective not in {"none", "vector"}:
        raise ValueError(f"Unsupported retrieval_mode: {retrieval_mode!r}")
    # Default: grounding only for pure no-RAG mode.
    grounding_enabled = (
        (retrieval_mode_effective == "none")
        if use_question_grounding is None
        else bool(use_question_grounding)
    )
    if retrieval_mode_effective == "vector":
        try:
            # Fail fast with a clear action item instead of crashing mid-run.
            get_menu_collection()
        except Exception as exc:
            raise RuntimeError(
                "Vector RAG mode requires populated Chroma collection 'menu'. "
                "Run: python scripts/load_chroma.py"
            ) from exc
        # Keep standard cashier retrieval path used in full dialog generation:
        # vector RAG + mini-LLM rewrite, no full menu JSON in prompt.
        cashier = CashierAgent(
            model=cashier_model,
            full_menu_context=False,
            realistic_cashier=True,
            trace_verbose=trace_verbose,
        )
    else:
        cashier = CashierAgent(
            model=cashier_model,
            rag_top_k=0,
            full_menu_context=False,
            realistic_cashier=True,
            trace_verbose=trace_verbose,
        )
    clarifier = ClarifyingClient(model=client_model)
    judge = DialogJudge(model=judge_model)

    rows: list[dict[str, Any]] = []
    total = len(questions) if max_questions <= 0 else min(len(questions), max_questions)
    print(
        f"[question-experiment] starting {total} dialogs "
        f"(retrieval_mode={retrieval_mode_effective}, grounding={grounding_enabled}, "
        f"max_dialog_turns={max_dialog_turns}, client_nudge_on_miss={client_nudge_on_miss})...",
        file=sys.stderr,
        flush=True,
    )
    for idx, row in enumerate(questions[:total], start=1):
        cat = row.get("category")
        cat_s = repr(cat) if cat is not None else "?"
        print(
            f"[question-experiment] {idx}/{total} category={cat_s}",
            file=sys.stderr,
            flush=True,
        )
        question = str(row.get("question") or "").strip()
        order_state = build_initial_order_state(pipeline_stub)
        history: list[DialogTurn] = [DialogTurn(speaker="client", text=question)]
        grounding_context = (
            build_question_grounding_context(row, menu_by_name)
            if grounding_enabled
            else ""
        )
        rag_trace: list[dict[str, Any]] = []
        llm_trace: list[dict[str, Any]] = []
        turn_timings_ms: list[dict[str, float]] = []
        finished_reason = "cashier_answered"

        for step in range(1, max(1, max_dialog_turns) + 1):
            t0 = time.perf_counter()
            cashier_text = cashier.generate_response(
                pipeline_stub,
                [{"speaker": t.speaker, "text": t.text} for t in history],
                order_state,
                extra_grounding_context=grounding_context,
                rag_trace=rag_trace,
                rag_meta={"call": "question_dialog", "question_idx": idx, "step": step},
                llm_trace=llm_trace,
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0
            turn_timings_ms.append({"cashier_step": float(step), "duration_ms": round(dt_ms, 2)})
            history.append(DialogTurn(speaker="cashier", text=cashier_text))

            mentions = extract_mentioned_menu_items(cashier_text, menu_names)
            continue_mode = _question_dialog_continue_after_cashier(
                row=row,
                cashier_text=cashier_text,
                mentions=mentions,
                menu_by_name=menu_by_name,
            )
            if continue_mode == "stop":
                finished_reason = "cashier_direct_answer"
                break
            if step >= max_dialog_turns:
                finished_reason = "max_dialog_turns_reached"
                break

            if continue_mode == "nudge" and not client_nudge_on_miss:
                finished_reason = "cashier_without_client_nudge"
                break

            t1 = time.perf_counter()
            if continue_mode == "clarify":
                client_text = clarifier.answer(
                    original_question=question,
                    expected_constraints=row.get("expected_constraints", []),
                    dialogue_history=[{"speaker": t.speaker, "text": t.text} for t in history],
                    cashier_question=cashier_text,
                )
                finished_reason = "cashier_requested_clarification"
            else:
                exp_raw = row.get("expected_item")
                exp_s = str(exp_raw).strip() if exp_raw is not None else ""
                client_text = _client_nudge_confirm_expected_item(exp_s)
                finished_reason = "client_reinforced_order"
            dt_client = (time.perf_counter() - t1) * 1000.0
            turn_timings_ms.append({"client_step": float(step), "duration_ms": round(dt_client, 2)})
            history.append(DialogTurn(speaker="client", text=client_text))

        final_cashier_text = _last_text(history, "cashier")
        mentioned_items = extract_mentioned_menu_items(final_cashier_text, menu_names)
        need_to_specify = detect_need_to_specify(final_cashier_text, mentioned_items)
        empty = is_empty_response(final_cashier_text)
        metrics = evaluate_single_turn_metrics(
            question_row=row,
            response_text=final_cashier_text,
            mentioned_items=mentioned_items,
            menu_by_name=menu_by_name,
            all_menu_names=menu_names,
        )
        audit = evaluate_dialog_audit(
            question_row=row,
            response_text=final_cashier_text,
            mentioned_items=mentioned_items,
            menu_by_name=menu_by_name,
            heuristic_metrics=metrics,
            need_to_specify=need_to_specify,
        )
        judge_raw, judge_json = judge.evaluate(
            menu_json=menu_json_text,
            question_row=row,
            history=[{"speaker": t.speaker, "text": t.text} for t in history],
            heuristic_metrics=metrics,
        )
        final_metrics, metric_sources = build_metrics_from_judge(
            judge_parsed=judge_json,
            heuristic_metrics=metrics,
        )
        judge_vs_heuristic = build_judge_comparison(
            final_metrics=final_metrics,
            heuristic_metrics=metrics,
            audit=audit,
        )
        history_with_judge = [
            {"speaker": t.speaker, "text": t.text}
            for t in history
        ]
        history_with_judge.append({"speaker": "judge", "text": judge_raw})

        row_out = {
            "question_id": idx,
            "category": row.get("category"),
            "question": question,
            "expected_item": row.get("expected_item"),
            "expected_constraints": row.get("expected_constraints", []),
            "dialog_finished_reason": finished_reason,
            "dialog_turns": len(history),
            "history": history_with_judge,
            "final_cashier_response": final_cashier_text,
            "mentioned_items": mentioned_items,
            "need_to_specify": need_to_specify,
            "empty_response": empty,
            "response_time_ms": round(sum(x.get("duration_ms", 0.0) for x in turn_timings_ms), 2),
            "turn_timings_ms": turn_timings_ms,
            "rag_trace": rag_trace,
            "llm_trace": llm_trace,
            "metrics": final_metrics,
            "heuristic_metrics": metrics,
            "metric_sources": metric_sources,
            "audit": audit,
            "judge": {
                "raw_response": judge_raw,
                "parsed": judge_json,
                "vs_heuristic": judge_vs_heuristic,
            },
            "source_file": row.get("_source_file"),
            "source_index": row.get("_source_index"),
            "retrieval_mode": retrieval_mode_effective,
            "question_grounding_enabled": grounding_enabled,
            "client_nudge_on_miss": client_nudge_on_miss,
        }
        rows.append(row_out)
        if incremental_save_dir is not None:
            save_incremental_question_row(incremental_save_dir, row_out)
    return rows


def run_single_turn_experiment(
    questions: list[dict[str, Any]],
    *,
    model: str | None = None,
    retrieval_mode: Literal["none", "vector"] = "none",
    use_question_grounding: bool | None = None,
    max_questions: int = 0,
    trace_verbose: bool = False,
) -> list[dict[str, Any]]:
    pipeline_stub = neutral_drive_through_profile()
    order_state = build_initial_order_state(pipeline_stub)
    mode = retrieval_mode.strip().lower()
    if mode not in {"none", "vector"}:
        raise ValueError(f"Unsupported retrieval_mode: {retrieval_mode!r}")
    grounding_enabled = (mode == "none") if use_question_grounding is None else bool(use_question_grounding)
    if mode == "vector":
        try:
            get_menu_collection()
        except Exception as exc:
            raise RuntimeError(
                "Vector RAG mode requires populated Chroma collection 'menu'. "
                "Run: python scripts/load_chroma.py"
            ) from exc
        cashier = CashierAgent(
            model=model,
            full_menu_context=False,
            realistic_cashier=True,
            trace_verbose=trace_verbose,
        )
    else:
        cashier = CashierAgent(
            model=model,
            rag_top_k=0,
            full_menu_context=False,
            realistic_cashier=True,
            trace_verbose=trace_verbose,
        )
    menu_names, menu_items = build_menu_index()
    menu_by_name = {x.name: x for x in menu_items}

    rows: list[dict[str, Any]] = []
    total = len(questions) if max_questions <= 0 else min(len(questions), max_questions)
    print(
        f"[single-turn-experiment] starting {total} questions "
        f"(retrieval_mode={mode}, grounding={grounding_enabled})...",
        file=sys.stderr,
        flush=True,
    )
    for idx, row in enumerate(questions[:total], start=1):
        cat = row.get("category")
        cat_s = repr(cat) if cat is not None else "?"
        print(
            f"[single-turn-experiment] {idx}/{total} category={cat_s}",
            file=sys.stderr,
            flush=True,
        )
        question = str(row.get("question") or "").strip()
        history = [{"speaker": "client", "text": question}]
        grounding_context = (
            build_question_grounding_context(row, menu_by_name)
            if grounding_enabled
            else ""
        )
        rag_trace: list[dict[str, Any]] = []
        llm_trace: list[dict[str, Any]] = []
        t0 = time.perf_counter()
        response = cashier.generate_response(
            pipeline_stub,
            history,
            order_state,
            extra_grounding_context=grounding_context,
            rag_trace=rag_trace,
            rag_meta={"call": "single_turn", "question_idx": idx},
            llm_trace=llm_trace,
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        mentions = extract_mentioned_menu_items(response, menu_names)
        need_to_specify = detect_need_to_specify(response, mentions)
        empty = is_empty_response(response)
        metrics = evaluate_single_turn_metrics(
            question_row=row,
            response_text=response,
            mentioned_items=mentions,
            menu_by_name=menu_by_name,
            all_menu_names=menu_names,
        )
        rows.append(
            {
                "question_id": idx,
                "category": row.get("category"),
                "question": question,
                "expected_item": row.get("expected_item"),
                "expected_constraints": row.get("expected_constraints", []),
                "response": response,
                "mentioned_items": mentions,
                "need_to_specify": need_to_specify,
                "empty_response": empty,
                "response_time_ms": round(dt_ms, 2),
                "rag_trace": rag_trace,
                "llm_trace": llm_trace,
                "metrics": metrics,
                "source_file": row.get("_source_file"),
                "source_index": row.get("_source_index"),
                "retrieval_mode": mode,
                "question_grounding_enabled": grounding_enabled,
            }
        )
    return rows


def aggregate_single_turn_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"total": 0}
    n = len(rows)
    times = sorted(float(r.get("response_time_ms") or 0.0) for r in rows)
    categories = sorted({str(r.get("category") or "") for r in rows if r.get("category")})
    by_category: dict[str, dict[str, Any]] = {}
    for cat in categories:
        bucket = [r for r in rows if r.get("category") == cat]
        by_category[cat] = _aggregate_bucket(bucket)
    out = _aggregate_bucket(rows)
    out["total"] = n
    out["categories"] = by_category
    out["latency_ms"] = {
        "median": _percentile(times, 50),
        "p95": _percentile(times, 95),
    }
    return out


def save_dialogs_by_category(
    rows: list[dict[str, Any]],
    *,
    output_dir: str,
) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    by_cat_root = root / "by_category"
    by_cat_root.mkdir(parents=True, exist_ok=True)

    all_rows_path = root / "rows.json"
    with open(all_rows_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        cat = str(row.get("category") or "unknown")
        by_category.setdefault(cat, []).append(row)

    category_artifacts: dict[str, Any] = {}
    for cat, items in by_category.items():
        cat_dir = by_cat_root / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        rows_path = cat_dir / "rows.json"
        with open(rows_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        dialogs_dir = cat_dir / "dialogs"
        dialogs_dir.mkdir(parents=True, exist_ok=True)
        for i, row in enumerate(items, start=1):
            fp = dialogs_dir / f"dialog_{i:04d}.json"
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(row, f, ensure_ascii=False, indent=2)
        cat_summary = aggregate_single_turn_metrics(items)
        summary_path = cat_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(cat_summary, f, ensure_ascii=False, indent=2)
        category_artifacts[cat] = {
            "n": len(items),
            "rows_path": str(rows_path),
            "dialogs_dir": str(dialogs_dir),
            "summary_path": str(summary_path),
        }

    global_summary = aggregate_single_turn_metrics(rows)
    global_summary["storage"] = {
        "all_rows_path": str(all_rows_path),
        "by_category_root": str(by_cat_root),
        "categories": category_artifacts,
    }
    global_summary_path = root / "summary.json"
    with open(global_summary_path, "w", encoding="utf-8") as f:
        json.dump(global_summary, f, ensure_ascii=False, indent=2)
    return {
        "summary": global_summary,
        "summary_path": str(global_summary_path),
        "rows_path": str(all_rows_path),
        "by_category_root": str(by_cat_root),
    }


def evaluate_single_turn_metrics(
    *,
    question_row: dict[str, Any],
    response_text: str,
    mentioned_items: list[str],
    menu_by_name: dict[str, MenuItem],
    all_menu_names: list[str],
) -> dict[str, Any]:
    expected_item = str(question_row.get("expected_item") or "").strip()
    lower_mentions = [x.lower() for x in mentioned_items]
    success_1 = bool(expected_item) and expected_item.lower() in lower_mentions[:1]
    success_3 = bool(expected_item) and expected_item.lower() in lower_mentions[:3]
    success_5 = bool(expected_item) and expected_item.lower() in lower_mentions[:5]
    hall = detect_hallucination(
        response_text=response_text,
        mentioned_items=mentioned_items,
        all_menu_names=all_menu_names,
    )
    violation, reasons = detect_constraint_violation(
        expected_constraints=question_row.get("expected_constraints", []),
        mentioned_items=mentioned_items,
        menu_by_name=menu_by_name,
    )
    group_completeness = compute_group_completeness(
        category=str(question_row.get("category") or ""),
        expected_constraints=question_row.get("expected_constraints", []),
        mentioned_items=mentioned_items,
    )
    return {
        "success_at_1": success_1,
        "success_at_3": success_3,
        "success_at_5": success_5,
        "hallucination": hall,
        "constraint_violation": violation,
        "constraint_violation_reasons": reasons,
        "group_completeness": group_completeness,
    }


def extract_mentioned_menu_items(response_text: str, menu_names: list[str]) -> list[str]:
    text = f" {response_text.lower()} "
    found: list[tuple[int, int, str]] = []
    for name in menu_names:
        pattern = _name_pattern(name)
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            found.append((m.start(), m.end(), name))
            break
    found.sort(key=lambda x: x[0])
    return [x[2] for x in found]


def detect_need_to_specify(response_text: str, mentioned_items: list[str]) -> bool:
    return ("?" in response_text) and not mentioned_items


def cashier_named_expected_item(
    expected_item: Any,
    *,
    response_text: str,
    mentioned_items: list[str],
) -> bool:
    """True if there is no expected menu item to verify, or cashier text names it (by menu row name)."""
    if expected_item is None:
        return True
    exp = str(expected_item).strip()
    if not exp:
        return True
    if exp in mentioned_items:
        return True
    blob = f" {response_text.lower()} "
    return bool(re.search(_name_pattern(exp), blob, flags=re.IGNORECASE))


def _client_nudge_confirm_expected_item(expected_menu_name: str) -> str:
    item = expected_menu_name.strip()
    return (
        f"I'm ordering {item} from the McDonald's menu in this chat. "
        "Please confirm it's available and add it to my order."
    )


def _question_dialog_continue_after_cashier(
    *,
    row: dict[str, Any],
    cashier_text: str,
    mentions: list[str],
    menu_by_name: dict[str, MenuItem],
) -> Literal["stop", "clarify", "nudge"]:
    """After a cashier turn: stop, ask clarifier to answer ?, or nudge when item not acknowledged."""
    if detect_need_to_specify(cashier_text, mentions):
        return "clarify"
    raw = row.get("expected_item")
    exp = str(raw).strip() if raw is not None else ""
    if not exp or exp not in menu_by_name:
        return "stop"
    if _claims_item_unavailable(cashier_text, exp):
        return "nudge"
    if _FOREIGN_BRAND_RE.search(cashier_text):
        return "nudge"
    if cashier_named_expected_item(raw, response_text=cashier_text, mentioned_items=mentions):
        return "stop"
    return "nudge"


def is_empty_response(response_text: str) -> bool:
    txt = (response_text or "").strip()
    if not txt:
        return True
    return bool(_EMPTY_RESPONSE_RE.match(txt))


def detect_hallucination(
    *,
    response_text: str,
    mentioned_items: list[str],
    all_menu_names: list[str],
) -> bool:
    if mentioned_items:
        return False
    text = (response_text or "").strip()
    if not text:
        return False
    if "?" in text and len(text) < 180:
        return False
    if re.search(r"\b(not|no)\b.*\b(menu|item|option)\b", text, re.IGNORECASE):
        return False
    menu_hints = re.findall(
        r"(?:recommend|suggest|try|have|take)\s+([A-Za-z][A-Za-z0-9\s\-()]{3,60})",
        text,
        flags=re.IGNORECASE,
    )
    if not menu_hints:
        return False
    known_norm = {normalize_name(x) for x in all_menu_names}
    for hint in menu_hints:
        clean = normalize_name(hint)
        if not clean:
            continue
        if clean in known_norm:
            continue
        if len(clean.split()) >= 2:
            return True
    return False


def evaluate_dialog_audit(
    *,
    question_row: dict[str, Any],
    response_text: str,
    mentioned_items: list[str],
    menu_by_name: dict[str, MenuItem],
    heuristic_metrics: dict[str, Any],
    need_to_specify: bool,
) -> dict[str, Any]:
    expected_item = str(question_row.get("expected_item") or "").strip()
    expected_exists = expected_item in menu_by_name if expected_item else False
    denied_expected = False
    if expected_item and expected_exists:
        denied_expected = _claims_item_unavailable(response_text, expected_item)

    fact_conflicts: list[str] = []
    target_item = None
    if expected_item and expected_exists and expected_item in mentioned_items:
        target_item = expected_item
    elif len(mentioned_items) == 1 and mentioned_items[0] in menu_by_name:
        target_item = mentioned_items[0]
    if target_item:
        fact_conflicts = _detect_item_fact_conflicts(response_text, menu_by_name[target_item], target_item)

    context_ignorance = bool(expected_item and expected_exists and denied_expected)
    critical_error = context_ignorance or bool(fact_conflicts)
    expected_item_used = bool(expected_item and expected_item in mentioned_items and not denied_expected)

    return {
        "expected_item": expected_item or None,
        "expected_item_in_menu": expected_exists,
        "expected_item_used": expected_item_used,
        "context_ignorance": context_ignorance,
        "factual_conflicts": fact_conflicts,
        "critical_error": critical_error,
        "need_to_specify": bool(need_to_specify),
        "heuristic_hallucination": bool(heuristic_metrics.get("hallucination")),
    }


def build_metrics_from_judge(
    *,
    judge_parsed: dict[str, Any],
    heuristic_metrics: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    """
    Build final metrics with judge-first priority.

    Heuristics remain a fallback when judge output is missing/invalid for a key.
    """
    judge_values = _flat_judge_metrics(judge_parsed)
    out = dict(heuristic_metrics)
    sources: dict[str, str] = {}
    keys = (
        "success_at_1",
        "success_at_3",
        "success_at_5",
        "hallucination",
        "constraint_violation",
        "need_to_specify",
        "empty_response",
    )
    for key in keys:
        if key in judge_values:
            out[key] = _coerce_bool(judge_values.get(key))
            sources[key] = "judge"
        else:
            out[key] = _coerce_bool(heuristic_metrics.get(key))
            sources[key] = "heuristic_fallback"

    if "group_completeness" in judge_values and judge_values.get("group_completeness") is not None:
        out["group_completeness"] = _as_float(judge_values.get("group_completeness"))
        sources["group_completeness"] = "judge"
    elif heuristic_metrics.get("group_completeness") is not None:
        out["group_completeness"] = heuristic_metrics.get("group_completeness")
        sources["group_completeness"] = "heuristic_fallback"
    else:
        out["group_completeness"] = None
        sources["group_completeness"] = "none"
    return out, sources


def _flat_judge_metrics(judge_parsed: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(judge_parsed, dict):
        return {}
    nested = judge_parsed.get("metrics")
    if isinstance(nested, dict):
        flat = dict(nested)
        for key, value in judge_parsed.items():
            if key != "metrics" and key not in flat:
                flat[key] = value
        return flat
    return judge_parsed


def build_judge_comparison(
    *,
    final_metrics: dict[str, Any],
    heuristic_metrics: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    compared_keys = (
        "success_at_1",
        "success_at_3",
        "success_at_5",
        "hallucination",
        "constraint_violation",
        "need_to_specify",
        "empty_response",
    )
    disagreements: list[str] = []
    for key in compared_keys:
        if _coerce_bool(final_metrics.get(key)) != _coerce_bool(heuristic_metrics.get(key)):
            disagreements.append(key)
    context_ignorance = bool(audit.get("context_ignorance"))
    factual_conflicts = [str(x) for x in (audit.get("factual_conflicts") or []) if str(x).strip()]
    judge_hallucination = _coerce_bool(final_metrics.get("hallucination"))
    supports_audit = (not context_ignorance and not factual_conflicts) or judge_hallucination
    return {
        "disagreement_count": len(disagreements),
        "disagreements": disagreements,
        "supports_audit_signals": supports_audit,
        "audit_signals": {
            "context_ignorance": context_ignorance,
            "factual_conflicts": factual_conflicts,
            "critical_error": bool(audit.get("critical_error")),
        },
    }


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "yes", "1"}:
            return True
        if v in {"false", "no", "0", ""}:
            return False
    return False


def _claims_item_unavailable(text: str, item_name: str) -> bool:
    txt = (text or "").strip()
    if not txt:
        return False
    item_pat = _name_pattern(item_name)
    for unavailable_pat in _UNAVAILABLE_PATTERNS:
        pattern = rf"{unavailable_pat}[\s,:-]{{0,30}}(?:the\s+)?{item_pat}|{item_pat}[\s,:-]{{0,30}}{unavailable_pat}"
        if re.search(pattern, txt, flags=re.IGNORECASE):
            return True
    return False


def _detect_item_fact_conflicts(response_text: str, item: MenuItem, item_name: str) -> list[str]:
    txt = (response_text or "").strip()
    if not txt:
        return []
    conflicts: list[str] = []
    lower = txt.lower()
    if normalize_name(item_name) not in normalize_name(lower):
        return conflicts

    stated_allergens = _extract_stated_allergens(lower)
    if stated_allergens:
        expected = {_normalize_allergen_token(x) for x in item.allergens if _normalize_allergen_token(x)}
        extra = sorted(x for x in stated_allergens if x not in expected)
        missing = sorted(x for x in expected if x not in stated_allergens)
        if extra:
            conflicts.append(
                f"Allergen list mismatch for {item_name}: extra={', '.join(extra)}."
            )
        if missing and len(expected) <= 6:
            conflicts.append(
                f"Allergen list mismatch for {item_name}: missing={', '.join(missing)}."
            )

    for nutrient, pattern, tol in (
        ("energy", r"(?:calories?|kcal|energy)\s*[:\-]?\s*(\d+(?:\.\d+)?)", 25.0),
        ("protein", r"protein\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*g", 4.0),
        ("carbs", r"carb(?:s|ohydrates?)?\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*g", 6.0),
        ("total_sugar", r"(?:sugar|total sugar)\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*g", 4.0),
        ("sodium", r"sodium\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*mg", 100.0),
    ):
        m = re.search(pattern, lower, flags=re.IGNORECASE)
        if not m:
            continue
        stated = _as_float(m.group(1))
        expected_val = _as_float(item.nutrients.get(nutrient))
        if expected_val <= 0:
            continue
        if abs(stated - expected_val) > tol:
            conflicts.append(
                f"{item_name} {nutrient} mismatch: stated={stated:.1f}, menu={expected_val:.1f}."
            )
    return conflicts


def _extract_stated_allergens(lower_text: str) -> set[str]:
    m = re.search(r"allerg(?:en|ies)\s*[:\-]?\s*([^\n.!?]{1,120})", lower_text, flags=re.IGNORECASE)
    if not m:
        return set()
    chunk = m.group(1)
    tokens = re.findall(r"[a-z][a-z\s]{1,24}", chunk)
    out: set[str] = set()
    for token in tokens:
        key = _normalize_allergen_token(token)
        if key:
            out.add(key)
    return out


def _normalize_allergen_token(token: str) -> str:
    raw = normalize_name(token)
    if not raw:
        return ""
    if raw in _ALLERGEN_TOKEN_ALIASES:
        return _ALLERGEN_TOKEN_ALIASES[raw]
    if raw.endswith("s") and raw[:-1] in _ALLERGEN_TOKEN_ALIASES:
        return _ALLERGEN_TOKEN_ALIASES[raw[:-1]]
    return raw


def detect_constraint_violation(
    *,
    expected_constraints: Any,
    mentioned_items: list[str],
    menu_by_name: dict[str, MenuItem],
) -> tuple[bool, list[str]]:
    if not expected_constraints or not mentioned_items:
        return False, []
    reasons: list[str] = []
    for raw in expected_constraints:
        if not isinstance(raw, dict):
            continue
        ctype = str(raw.get("type") or "").strip().lower()
        value = str(raw.get("value") or "").strip().lower()
        if ctype == "allergen":
            alias = _ALLERGEN_ALIASES.get(value, value)
            for name in mentioned_items:
                item = menu_by_name.get(name)
                if not item:
                    continue
                if alias and alias in item.allergens:
                    reasons.append(f"{name} contains allergen '{value}'")
        elif ctype == "exclude":
            if not value:
                continue
            for name in mentioned_items:
                item = menu_by_name.get(name)
                if not item:
                    continue
                if re.search(rf"\b{re.escape(value)}\b", item.ingredients, flags=re.IGNORECASE):
                    reasons.append(f"{name} includes excluded ingredient '{value}'")
        elif ctype == "diet":
            if value in ("vegan", "vegetarian"):
                for name in mentioned_items:
                    item = menu_by_name.get(name)
                    if not item:
                        continue
                    if value == "vegan" and item.restriction_map.get("isVegan", False):
                        reasons.append(f"{name} is not vegan")
                    if value == "vegetarian":
                        if _MEATY_RE.search(item.ingredients or ""):
                            reasons.append(f"{name} is not vegetarian")
            elif value in ("gluten-free", "gluten free"):
                for name in mentioned_items:
                    item = menu_by_name.get(name)
                    if item and item.restriction_map.get("noGluten", False):
                        reasons.append(f"{name} contains gluten")
            else:
                threshold = _DIET_THRESHOLDS.get(value)
                if threshold:
                    nutrient, limit, mode = threshold
                    for name in mentioned_items:
                        item = menu_by_name.get(name)
                        if not item:
                            continue
                        val = float(item.nutrients.get(nutrient) or 0.0)
                        if mode == "max" and val > limit:
                            reasons.append(f"{name} violates '{value}' ({nutrient}={val:.1f})")
                        if mode == "min" and val < limit:
                            reasons.append(f"{name} violates '{value}' ({nutrient}={val:.1f})")
    return bool(reasons), reasons


def compute_group_completeness(
    *,
    category: str,
    expected_constraints: Any,
    mentioned_items: list[str],
) -> float | None:
    if category != "group":
        return None
    required = 1
    for raw in expected_constraints or []:
        if isinstance(raw, dict) and str(raw.get("type") or "").lower() == "group":
            members = raw.get("members")
            if isinstance(members, list) and members:
                required = len(members)
            elif members:
                required = 1
            break
    if required <= 0:
        return 0.0
    return round(min(len(set(mentioned_items)) / float(required), 1.0), 4)


def _aggregate_bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    metric_rows = [r.get("metrics", {}) for r in rows]
    return {
        "n": n,
        "success_at_1": sum(1 for m in metric_rows if m.get("success_at_1")) / n,
        "success_at_3": sum(1 for m in metric_rows if m.get("success_at_3")) / n,
        "success_at_5": sum(1 for m in metric_rows if m.get("success_at_5")) / n,
        "hallucination_rate": sum(1 for m in metric_rows if m.get("hallucination")) / n,
        "constraint_violation_rate": sum(1 for m in metric_rows if m.get("constraint_violation")) / n,
        "need_to_specify_rate": sum(1 for r in rows if r.get("need_to_specify")) / n,
        "empty_response_rate": sum(1 for r in rows if r.get("empty_response")) / n,
    }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return float(values[0])
    if q >= 100:
        return float(values[-1])
    k = (len(values) - 1) * (q / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(values) - 1)
    if lo == hi:
        return float(values[lo])
    frac = k - lo
    return float(values[lo] * (1.0 - frac) + values[hi] * frac)


def normalize_name(name: str) -> str:
    return _NON_ALNUM_RE.sub(" ", name.lower()).strip()


def _name_pattern(name: str) -> str:
    parts = [re.escape(p) for p in normalize_name(name).split() if p]
    if not parts:
        return r"$^"
    return r"\b" + r"\s+".join(parts) + r"\b"


def _restriction_map_for_meta(
    meta: dict[str, Any],
    allergens: set[str],
    ingredients: str,
) -> dict[str, bool]:
    blob = " ".join(
        [
            str(meta.get("name") or ""),
            str(meta.get("description") or ""),
            str(meta.get("tag") or ""),
            ingredients or "",
        ]
    ).lower()
    has_milk = "milk" in allergens or bool(
        re.search(r"\b(milk|cheese|paneer|butter|latte|shake|sundae|mcflurry)\b", blob)
    )
    has_fish = "fish" in allergens or bool(
        re.search(r"\b(fish|filet[\s-]*o[\s-]*fish|tuna|salmon|shrimp)\b", blob)
    )
    has_eggs = "egg" in allergens or bool(re.search(r"\b(egg|mayo|mayonnaise)\b", blob))
    has_nuts = "nuts" in allergens or bool(
        re.search(r"\b(nut|nuts|peanut|almond|hazelnut|cashew|walnut)\b", blob)
    )
    has_gluten = "cereal containing gluten" in allergens or bool(
        re.search(r"\b(bun|bread|wrap|tortilla|muffin|biscuit|wheat|gluten)\b", blob)
    )
    has_beef = bool(
        re.search(r"\b(beef|hamburger|cheeseburger|big mac|mcdouble|quarter pounder)\b", blob)
    )
    has_animal = has_milk or has_fish or has_eggs or has_beef or bool(
        re.search(r"\b(chicken|beef|fish|egg|bacon|sausage|ham)\b", blob)
    )
    return {
        "noMilk": has_milk,
        "noFish": has_fish,
        "noNuts": has_nuts,
        "noEggs": has_eggs,
        "noGluten": has_gluten,
        "noBeef": has_beef,
        "isVegan": has_animal,
    }


class ClarifyingClient:
    def __init__(self, model: str | None = None, timeout: float = 60.0) -> None:
        self.model = _resolve_model(model)
        self._client = _build_openai_client(timeout)

    def answer(
        self,
        *,
        original_question: str,
        expected_constraints: Any,
        dialogue_history: list[dict[str, str]],
        cashier_question: str,
    ) -> str:
        system = (
            "You are the SAME customer as in the initial request. "
            "Answer the cashier clarification question briefly and concretely. "
            "Keep all original constraints and intent. "
            "Do not add random new preferences. "
            "Reply in 1 sentence, natural spoken style, no markdown."
        )
        prompt = {
            "original_question": original_question,
            "expected_constraints": expected_constraints,
            "dialogue_history": dialogue_history[-8:],
            "cashier_question": cashier_question,
            "task": "Provide only the customer's clarification reply.",
        }
        try:
            return _call_llm(
                self._client,
                self.model,
                system,
                [{"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}],
                temperature=0.2,
            ).strip()
        except Exception:
            return "I mean the same constraints as before, please suggest the safest matching options."


class DialogJudge:
    def __init__(self, model: str | None = None, timeout: float = 90.0) -> None:
        self.model = _resolve_model(model)
        self._client = _build_openai_client(timeout)

    def evaluate(
        self,
        *,
        menu_json: str,
        question_row: dict[str, Any],
        history: list[dict[str, str]],
        heuristic_metrics: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        system = (
            "You are an evaluator for McDonald's assistant experiments. "
            "Use the compact menu facts and full dialogue. "
            "Return EXACTLY one flat JSON object (no nesting, no wrappers) with keys: "
            "success_at_1, success_at_3, success_at_5, hallucination, "
            "constraint_violation, need_to_specify, empty_response, "
            "group_completeness, short_analysis, risks, final_label. "
            "Do NOT output keys like heuristic_metrics, result, scores, or data. "
            "All success/violation fields must be booleans. "
            "Include risks as an array of short strings. "
            "final_label must be 'pass' or 'fail'. "
            "CRITICAL: If cashier says an item is unavailable while that item is present in menu_json, "
            "set hallucination=true and all success_at_* to false. "
            "CRITICAL: If cashier provides item facts (allergens/nutrition) that conflict with menu_json, "
            "set hallucination=true and final_label='fail'. "
            "Do not reward retrieval presence; score only what cashier actually said."
        )
        payload = {
            "question_row": question_row,
            "dialogue": history,
            "menu_facts": _compact_menu_facts(menu_json, question_row),
            "heuristic_metrics": heuristic_metrics,
            "instruction": (
                "Re-evaluate metrics with your own judgment. "
                "If uncertain, keep conservative values and explain in short_analysis."
            ),
        }
        raw = _call_llm(
            self._client,
            self.model,
            system,
            [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            temperature=0.0,
            response_format={"type": "json_object"},
        ).strip()
        parsed = _parse_json_object(raw)
        if not parsed.get("parse_error"):
            return raw, parsed

        # Retry once with explicit repair instructions when provider/model ignores JSON mode.
        repair_system = (
            "Convert the input into one strict JSON object. "
            "No markdown, no explanations, no code fences, no wrapper keys."
        )
        repair_payload = {
            "required_keys": [
                "success_at_1",
                "success_at_3",
                "success_at_5",
                "hallucination",
                "constraint_violation",
                "need_to_specify",
                "empty_response",
                "group_completeness",
                "short_analysis",
                "risks",
                "final_label",
            ],
            "bool_keys": [
                "success_at_1",
                "success_at_3",
                "success_at_5",
                "hallucination",
                "constraint_violation",
                "need_to_specify",
                "empty_response",
            ],
            "notes": [
                "risks must be a JSON array of strings",
                "group_completeness can be number or null",
                "final_label should be 'pass' or 'fail'",
                "output must be a flat object with exactly required_keys",
                "do not include keys like heuristic_metrics/result/scores/data",
            ],
            "source_text": raw,
        }
        repaired_raw = _call_llm(
            self._client,
            self.model,
            repair_system,
            [{"role": "user", "content": json.dumps(repair_payload, ensure_ascii=False)}],
            temperature=0.0,
            response_format={"type": "json_object"},
        ).strip()
        repaired = _parse_json_object(repaired_raw)
        if not repaired.get("parse_error"):
            return repaired_raw, repaired
        return raw, parsed


def _compact_menu_facts(menu_json: str, question_row: dict[str, Any]) -> dict[str, Any]:
    expected_item = str(question_row.get("expected_item") or "").strip()
    facts: dict[str, Any] = {
        "expected_item": expected_item or None,
        "expected_item_in_menu": False,
        "expected_item_row": None,
        "known_menu_names": [],
    }
    try:
        rows = json.loads(menu_json)
    except Exception:
        return facts
    if not isinstance(rows, list):
        return facts

    names: list[str] = []
    expected_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        if name not in names:
            names.append(name)
        if expected_item and name == expected_item:
            expected_rows.append(row)

    facts["known_menu_names"] = names
    if expected_rows:
        facts["expected_item_in_menu"] = True
        facts["expected_item_row"] = expected_rows[0]
        if len(expected_rows) > 1:
            facts["expected_item_variants"] = expected_rows
    return facts


def _parse_json_object(text: str) -> dict[str, Any]:
    txt = (text or "").strip()
    m = re.search(r"\{[\s\S]*\}", txt)
    if m:
        txt = m.group(0)
    try:
        data = json.loads(txt)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {"parse_error": True, "raw_preview": txt[:1500]}


def _last_text(history: list[DialogTurn], speaker: str) -> str:
    for turn in reversed(history):
        if turn.speaker == speaker:
            return turn.text
    return ""


def _load_menu_json_text() -> str:
    with open(MCD_JSON_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _as_float(value: Any) -> float:
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0
