"""Single-turn experiments over prepared question banks."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import sys
import time
from typing import Any

from mcd_voice.dialog.pipeline import build_initial_order_state
from mcd_voice.config import MCD_JSON_PATH
from mcd_voice.llm import CashierAgent
from mcd_voice.llm.agent import (
    _build_openai_client,
    _call_llm,
    _resolve_model,
)
from mcd_voice.menu.dataset import load_menu_from_json
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


def run_question_dialog_experiment(
    questions: list[dict[str, Any]],
    *,
    cashier_model: str | None = None,
    client_model: str | None = None,
    judge_model: str | None = None,
    max_questions: int = 0,
    max_dialog_turns: int = 4,
    trace_verbose: bool = False,
) -> list[dict[str, Any]]:
    profile = neutral_drive_through_profile()
    menu_names, menu_items = build_menu_index()
    menu_by_name = {x.name: x for x in menu_items}
    menu_json_text = _load_menu_json_text()
    cashier = CashierAgent(
        model=cashier_model,
        rag_top_k=0,
        full_menu_context=True,
        realistic_cashier=True,
        trace_verbose=trace_verbose,
    )
    clarifier = ClarifyingClient(model=client_model)
    judge = DialogJudge(model=judge_model)

    rows: list[dict[str, Any]] = []
    total = len(questions) if max_questions <= 0 else min(len(questions), max_questions)
    print(
        f"[question-experiment] starting {total} dialogs (max_dialog_turns={max_dialog_turns})...",
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
        order_state = build_initial_order_state(profile)
        history: list[DialogTurn] = [DialogTurn(speaker="client", text=question)]
        rag_trace: list[dict[str, Any]] = []
        llm_trace: list[dict[str, Any]] = []
        turn_timings_ms: list[dict[str, float]] = []
        finished_reason = "cashier_answered"

        for step in range(1, max(1, max_dialog_turns) + 1):
            t0 = time.perf_counter()
            cashier_text = cashier.generate_response(
                profile,
                [{"speaker": t.speaker, "text": t.text} for t in history],
                order_state,
                rag_trace=rag_trace,
                rag_meta={"call": "question_dialog", "question_idx": idx, "step": step},
                llm_trace=llm_trace,
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0
            turn_timings_ms.append({"cashier_step": float(step), "duration_ms": round(dt_ms, 2)})
            history.append(DialogTurn(speaker="cashier", text=cashier_text))

            mentions = extract_mentioned_menu_items(cashier_text, menu_names)
            if not detect_need_to_specify(cashier_text, mentions):
                finished_reason = "cashier_direct_answer"
                break
            if step >= max_dialog_turns:
                finished_reason = "max_dialog_turns_reached"
                break

            t1 = time.perf_counter()
            client_text = clarifier.answer(
                original_question=question,
                expected_constraints=row.get("expected_constraints", []),
                dialogue_history=[{"speaker": t.speaker, "text": t.text} for t in history],
                cashier_question=cashier_text,
            )
            dt_client = (time.perf_counter() - t1) * 1000.0
            turn_timings_ms.append({"client_step": float(step), "duration_ms": round(dt_client, 2)})
            history.append(DialogTurn(speaker="client", text=client_text))
            finished_reason = "cashier_requested_clarification"

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
        judge_raw, judge_json = judge.evaluate(
            menu_json=menu_json_text,
            question_row=row,
            history=[{"speaker": t.speaker, "text": t.text} for t in history],
            heuristic_metrics=metrics,
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
            "metrics": metrics,
            "judge": {
                "raw_response": judge_raw,
                "parsed": judge_json,
            },
            "source_file": row.get("_source_file"),
            "source_index": row.get("_source_index"),
        }
        rows.append(row_out)
    return rows


def run_single_turn_experiment(
    questions: list[dict[str, Any]],
    *,
    model: str | None = None,
    max_questions: int = 0,
    trace_verbose: bool = False,
) -> list[dict[str, Any]]:
    profile = neutral_drive_through_profile()
    order_state = build_initial_order_state(profile)
    cashier = CashierAgent(
        model=model,
        rag_top_k=0,
        full_menu_context=True,
        realistic_cashier=True,
        trace_verbose=trace_verbose,
    )
    menu_names, menu_items = build_menu_index()
    menu_by_name = {x.name: x for x in menu_items}

    rows: list[dict[str, Any]] = []
    total = len(questions) if max_questions <= 0 else min(len(questions), max_questions)
    print(
        f"[single-turn-experiment] starting {total} questions...",
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
        rag_trace: list[dict[str, Any]] = []
        llm_trace: list[dict[str, Any]] = []
        t0 = time.perf_counter()
        response = cashier.generate_response(
            profile,
            history,
            order_state,
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
    from pathlib import Path

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
            "Use the full menu JSON and full dialogue. "
            "Return strict JSON only with keys: "
            "success_at_1, success_at_3, success_at_5, hallucination, "
            "constraint_violation, need_to_specify, empty_response, "
            "group_completeness, short_analysis, risks, final_label. "
            "Include risks as an array of short strings."
        )
        payload = {
            "question_row": question_row,
            "dialogue": history,
            "menu_json": menu_json,
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
        ).strip()
        parsed = _parse_json_object(raw)
        return raw, parsed


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
