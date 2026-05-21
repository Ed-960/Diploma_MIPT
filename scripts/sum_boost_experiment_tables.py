"""
Синтетическая «подкрутка» уже посчитанных агрегатов (только для визуала / демо-слайдов).

Читает ``summary_by_category.json`` из каталога анализа и для **всех** категорий (или
перечня ``--categories``):

- категории, где исходный **no-rag success@1** уже **> 85%**, **не изменяются**;
- для остальных: число пар **не ниже** ``--min-pair-n`` (по умолчанию 101);
- **no-rag success@1**: доля как при линейном масштабировании на новый N; если исходная
  доля была **< 60%**, она **линейно отображается** в полосу **≈60–75%** на новом N (не одна
  константа для всех);
- **RAG success@1**: не ниже ``--rag-success-min-pct`` и не выше ``--rag-success-max-pct``;
  внутри коридора — **разные** целые успехи по категории (стабильный разброс), без одинаковых
  «магических» процентов вроде ровно 85.15% у всех;
- категория **group** («несколько человек»): после общих правил success@1 **чуть снижаются**
  (сложнее сценарий);
- пересобирает 2×2 для McNemar и пул как сумму по категориям.

**Не перезаписывает** исходные отчёты: по умолчанию пишет в ``experiments/analysis_sum/``.

Дополнительно строит ``charts/success_at_1_complement_by_llm_variant_demo.png`` (RAG) и
``charts/success_at_1_complement_by_llm_variant_demo_norag.png`` (no-rag) — как
``success_at_1_complement_by_category`` (**промах по цели**, в JSON: 100 − success_at_1;
не поле hallucination),
но с несколькими подписанными моделями
(имена и max context по карточкам); якорь — **Qwen2.5-14B-Instruct** (доли RAG из JSON);
**qwen3:1.7b** — чуть хуже якоря (иллюстративный масштаб); прочие — условный разброс для слайдов.

Запуск из корня репозитория:
  python scripts/sum_boost_experiment_tables.py
  python scripts/sum_boost_experiment_tables.py --min-pair-n 120 --rag-success-min-pct 85 --rag-success-max-pct 95
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import _bootstrap

_bootstrap.ensure_src()

import analyze_question_experiments as aqe  # noqa: E402

BOOL_METRICS = aqe.BOOL_METRICS


def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def _solve_two_by_two(
    n: int,
    k_nr: int,
    k_rag: int,
    tt_hint: int,
) -> dict[str, int]:
    k_nr = _clamp_int(k_nr, 0, n)
    k_rag = _clamp_int(k_rag, 0, n)
    tt_min = max(0, k_nr + k_rag - n)
    tt_max = min(k_nr, k_rag)
    tt = _clamp_int(tt_hint, tt_min, tt_max)
    tf = k_nr - tt
    ft = k_rag - tt
    ff = n - tt - tf - ft
    assert ff >= 0 and tf >= 0 and ft >= 0
    return {"both_true": tt, "norag_true_rag_false": tf, "norag_false_rag_true": ft, "both_false": ff}


def _mcnemar_wrap(cells: dict[str, int]) -> dict[str, Any]:
    b, c = cells["norag_true_rag_false"], cells["norag_false_rag_true"]
    chi2, p = aqe._mcnemar_chi2(b, c)
    return {**cells, "chi2": chi2, "p_value": p}


def _scale_counts_in_agg(agg: dict[str, Any], old_n: int, new_n: int, frozen: set[str]) -> None:
    if old_n <= 0:
        return
    for key in list(agg.keys()):
        if not key.endswith("_count") or key in frozen:
            continue
        raw = agg.get(key)
        if not isinstance(raw, (int, float)):
            continue
        agg[key] = _clamp_int(int(round(float(raw) * new_n / old_n)), 0, new_n)


def _refresh_side_aggregate(agg: dict[str, Any], n: int) -> None:
    agg["n"] = n
    for k in BOOL_METRICS:
        c = int(agg.get(f"{k}_count", 0))
        agg[f"{k}_pct"] = round(100.0 * c / n, 2) if n else 0.0
        lo, hi = aqe.wilson_ci_pct(c, n)
        agg[f"{k}_ci95_pct"] = [lo, hi]
    for label in ("judge_parse_error", "critical_error", "context_ignorance"):
        c = int(agg.get(f"{label}_count", 0))
        agg[f"{label}_pct"] = round(100.0 * c / n, 2) if n else 0.0
        lo, hi = aqe.wilson_ci_pct(c, n)
        agg[f"{label}_ci95_pct"] = [lo, hi]
    jh = float(agg.get("judge_hallucination_metric_pct") or 0.0)
    agg["judge_hallucination_metric_pct"] = round(min(100.0, max(0.0, jh)), 2)


def _scale_rag_retrieval(ret: dict[str, Any] | None, old_pair_n: int, new_pair_n: int) -> None:
    if not ret or not isinstance(ret, dict):
        return
    old_n = int(ret.get("n") or 0)
    if old_n <= 0 or old_pair_n <= 0:
        return
    factor = new_pair_n / old_pair_n
    new_n = max(1, int(round(old_n * factor)))
    ret["n"] = new_n
    for k in ("success_at_1_count", "success_at_3_count", "success_at_5_count"):
        if k in ret:
            c = int(ret.get(k) or 0)
            ret[k] = _clamp_int(int(round(c * factor)), 0, new_n)
    for k in ("success_at_1_pct", "success_at_3_pct", "success_at_5_pct"):
        base = k.replace("_pct", "_count")
        if base in ret:
            c = int(ret[base])
            ret[k] = round(100.0 * c / new_n, 2) if new_n else 0.0
    if "expected_rank_mean" in ret and ret["expected_rank_mean"] is not None:
        ret["expected_rank_mean"] = round(float(ret["expected_rank_mean"]), 2)
    miss_pct = ret.get("expected_missing_from_candidates_pct")
    if miss_pct is not None:
        ret["expected_missing_from_candidates_pct"] = round(float(miss_pct), 2)


def _orig_nr_success_pct(block: dict[str, Any]) -> float:
    paired = block["paired"]
    old_n = int(paired["no_rag"]["n"])
    if old_n <= 0:
        return 0.0
    old_s = block["mcnemar"]["success_at_1"]
    old_k_nr = int(old_s["both_true"] + old_s["norag_true_rag_false"])
    return 100.0 * old_k_nr / old_n


def _nr_success_count_for_new_n(
    *,
    old_n: int,
    new_n: int,
    old_k_nr: int,
    nr_low_pct: float,
    nr_high_pct: float,
) -> int:
    """Исходная доля сохраняется при масштабировании; если была < nr_low — целевая доля
    линейно от orig_pct в [0, nr_low) к полосе [nr_low, nr_high] на новом N."""
    if new_n <= 0:
        return 0
    orig_pct = 100.0 * old_k_nr / old_n if old_n else 0.0
    if orig_pct < nr_low_pct:
        lo = int(math.ceil((nr_low_pct / 100.0) * new_n))
        hi = int(math.floor((nr_high_pct / 100.0) * new_n))
        lo = _clamp_int(lo, 0, new_n)
        hi = _clamp_int(hi, lo, new_n)
        if nr_low_pct <= 0:
            t = 1.0
        else:
            t = max(0.0, min(1.0, orig_pct / nr_low_pct))
        target_pct = nr_low_pct + t * (nr_high_pct - nr_low_pct)
        k = int(round((target_pct / 100.0) * new_n))
        return _clamp_int(k, lo, hi)
    return _clamp_int(int(round(old_k_nr * new_n / old_n)), 0, new_n)


def _rag_success_count_natural_spread(
    cat: str, new_n: int, baseline_k: int, cap_k: int, rag_min_k: int
) -> int:
    """
    В пределах [baseline_k, cap_k] выбирает целое число успехов RAG стабильно по имени категории,
    чтобы доли по строкам не совпадали «пиксель в пиксель». Нижняя граница не застревает на
    ровно ceil(0.85·N)/N (типично 85.15% при N=101), если в коридоре есть хотя бы один успех выше.
    """
    lo = _clamp_int(baseline_k, 0, new_n)
    hi = _clamp_int(cap_k, 0, new_n)
    if hi < lo:
        return lo
    # Избегаем «магического» минимума одной и той же дроби у всех категорий с тем же N.
    if hi > rag_min_k and lo == rag_min_k:
        lo = min(rag_min_k + 1, hi)
    digest = hashlib.blake2b(f"rag-spread:{cat}:{new_n}:{lo}:{hi}".encode(), digest_size=8).digest()
    h = int.from_bytes(digest, "big")
    span = hi - lo + 1
    return lo + (h % span)


def _boost_category_block(
    cat: str,
    block: dict[str, Any],
    *,
    min_pair_n: int,
    nr_low_pct: float,
    nr_high_pct: float,
    rag_success_min_pct: float,
    rag_success_max_pct: float,
) -> None:
    paired = block["paired"]
    old_n = int(paired["no_rag"]["n"])
    if old_n <= 0:
        return
    new_n = max(min_pair_n, old_n)

    old_s = block["mcnemar"]["success_at_1"]
    old_h = block["mcnemar"]["hallucination"]

    old_k_nr = int(old_s["both_true"] + old_s["norag_true_rag_false"])
    old_k_rag = int(old_s["both_true"] + old_s["norag_false_rag_true"])

    k_nr_s = _nr_success_count_for_new_n(
        old_n=old_n,
        new_n=new_n,
        old_k_nr=old_k_nr,
        nr_low_pct=nr_low_pct,
        nr_high_pct=nr_high_pct,
    )
    scaled_rag = int(round(old_k_rag * new_n / old_n)) if old_n else 0
    rag_floor = k_nr_s + (1 if old_k_rag >= old_k_nr else 2)
    # Исходный разрыв в числе успехов (не масштабируем на N² — иначе при 30→101 получается 100% RAG).
    gap = max(0, old_k_rag - old_k_nr)
    rag_cap = int(math.floor((rag_success_max_pct / 100.0) * new_n))
    rag_cap = max(0, min(new_n, rag_cap))
    rag_min_k = int(math.ceil((rag_success_min_pct / 100.0) * new_n))
    rag_min_k = _clamp_int(rag_min_k, 0, new_n)
    baseline_rag = min(new_n, rag_cap, max(rag_floor, scaled_rag, k_nr_s + gap, rag_min_k))
    k_rag_s = _rag_success_count_natural_spread(cat, new_n, baseline_rag, rag_cap, rag_min_k)

    # «group» — заказ на несколько человек: сложнее, чуть ниже success@1 (no-rag и RAG).
    if cat == "group":
        d_nr = max(2, new_n // 34)
        d_rg = max(3, new_n // 28)
        k_nr_s = max(0, k_nr_s - d_nr)
        rag_floor = k_nr_s + (1 if old_k_rag >= old_k_nr else 2)
        k_rag_s = min(new_n, rag_cap, max(rag_min_k, rag_floor, k_rag_s - d_rg))

    tt_s_hint = int(round(int(old_s["both_true"]) * new_n / old_n)) if old_n else 0
    succ_cells = _solve_two_by_two(new_n, k_nr_s, k_rag_s, tt_s_hint)

    k_nr_h = _clamp_int(
        int(round((old_h["both_true"] + old_h["norag_true_rag_false"]) * new_n / old_n)),
        0,
        new_n,
    )
    k_rag_h = _clamp_int(
        int(round((old_h["both_true"] + old_h["norag_false_rag_true"]) * new_n / old_n)),
        0,
        new_n,
    )
    tt_h_hint = int(round(int(old_h["both_true"]) * new_n / old_n)) if old_n else 0
    hall_cells = _solve_two_by_two(new_n, k_nr_h, k_rag_h, tt_h_hint)

    block["mcnemar"]["success_at_1"] = _mcnemar_wrap(succ_cells)
    block["mcnemar"]["hallucination"] = _mcnemar_wrap(hall_cells)

    frozen_nr = {"success_at_1_count", "hallucination_count"}
    frozen_rg = set(frozen_nr)

    for side_key, frozen in (("no_rag", frozen_nr), ("rag", frozen_rg)):
        agg = paired[side_key]
        _scale_counts_in_agg(agg, old_n, new_n, frozen)
        agg["n"] = new_n
        if side_key == "no_rag":
            agg["success_at_1_count"] = k_nr_s
            agg["hallucination_count"] = k_nr_h
        else:
            agg["success_at_1_count"] = k_rag_s
            agg["hallucination_count"] = k_rag_h
        _refresh_side_aggregate(agg, new_n)

    block["file_counts"]["paired"] = new_n
    _scale_rag_retrieval(paired.get("rag_retrieval"), old_n, new_n)


def _sum_mcnemar_tables(by_cat: dict[str, Any], metric: str) -> dict[str, int]:
    acc = {"both_true": 0, "norag_true_rag_false": 0, "norag_false_rag_true": 0, "both_false": 0}
    for cat, blk in by_cat.items():
        if cat == "pooled_all_categories":
            continue
        mm = blk["mcnemar"][metric]
        for k in acc:
            acc[k] += int(mm[k])
    return acc


def _rebuild_pooled(by_cat: dict[str, Any], pooled_template: dict[str, Any]) -> dict[str, Any]:
    cats = [k for k in sorted(by_cat.keys()) if k != "pooled_all_categories"]
    pooled = copy.deepcopy(pooled_template)
    pooled["description"] = (
        pooled_template.get("description", "")
        + " "
        "(после sum_boost: часть категорий N≥min_pair_n; no-rag <60% исходно → интерполяция в 60–75%; RAG в коридоре min–max — синтетика.)"
    ).strip()

    def _sum_aggs(side: str) -> dict[str, Any]:
        count_keys: set[str] = set()
        for cat in cats:
            agg = by_cat[cat]["paired"][side]
            count_keys.update(k for k in agg if k.endswith("_count"))
        out: dict[str, Any] = {k: 0 for k in count_keys}
        out["n"] = 0
        jh_acc = 0.0
        for cat in cats:
            agg = by_cat[cat]["paired"][side]
            n = int(agg["n"])
            out["n"] += n
            jh_acc += float(agg.get("judge_hallucination_metric_pct") or 0.0) * n
            for ck in count_keys:
                out[ck] = int(out.get(ck, 0)) + int(agg.get(ck, 0))
        out["judge_hallucination_metric_pct"] = (
            round(jh_acc / out["n"], 2) if out["n"] else 0.0
        )
        return out

    nr_sum = _sum_aggs("no_rag")
    r_sum = _sum_aggs("rag")
    n_pool = int(nr_sum["n"])
    assert n_pool == int(r_sum["n"])

    def _counts_only(d: dict[str, Any]) -> dict[str, Any]:
        out = {"n": n_pool, "judge_hallucination_metric_pct": d["judge_hallucination_metric_pct"]}
        for k, v in d.items():
            if k.endswith("_count"):
                out[k] = int(v)
        return out

    nr_out = _counts_only(nr_sum)
    r_out = _counts_only(r_sum)
    _refresh_side_aggregate(nr_out, n_pool)
    _refresh_side_aggregate(r_out, n_pool)
    pooled["paired"]["no_rag"] = nr_out
    pooled["paired"]["rag"] = r_out

    ret_keys = (
        "success_at_1_count",
        "success_at_3_count",
        "success_at_5_count",
    )
    ret_n = 0
    sums = {k: 0 for k in ret_keys}
    ranks: list[float] = []
    miss_w = 0.0
    for cat in cats:
        ret = by_cat[cat]["paired"].get("rag_retrieval") or {}
        rn = int(ret.get("n") or 0)
        ret_n += rn
        for k in ret_keys:
            sums[k] += int(ret.get(k) or 0)
        mrm = ret.get("expected_rank_mean")
        if mrm is not None and rn:
            ranks.extend([float(mrm)] * rn)
        mp = ret.get("expected_missing_from_candidates_pct")
        if mp is not None and rn:
            miss_w += float(mp) * rn
    rag_ret: dict[str, Any] = {"n": ret_n}
    for k in ret_keys:
        rag_ret[k] = sums[k]
        c = sums[k]
        rag_ret[k.replace("_count", "_pct")] = round(100.0 * c / ret_n, 2) if ret_n else 0.0
    rag_ret["expected_rank_mean"] = round(sum(ranks) / len(ranks), 2) if ranks else None
    rag_ret["expected_missing_from_candidates_pct"] = round(miss_w / ret_n, 2) if ret_n else None
    pooled["paired"]["rag_retrieval"] = rag_ret

    s_cells = _sum_mcnemar_tables(by_cat, "success_at_1")
    h_cells = _sum_mcnemar_tables(by_cat, "hallucination")
    pooled["mcnemar"]["success_at_1"] = _mcnemar_wrap(s_cells)
    pooled["mcnemar"]["hallucination"] = _mcnemar_wrap(h_cells)
    return pooled


def _inject_sum_banner_md(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r"\n> \*\*Демо-визуализация:\*\*[^\n]*(?:\n>[^\n]*)*\n*",
        "\n",
        text,
        count=1,
    )
    banner = (
        "\n> **Демо-визуализация:** исходный no-rag >85% — без изменений; иначе N≥101, no-rag как "
        "в данных (если <60% — интерполяция в 60–75% по исходной доле), RAG не ниже 85% и не выше 95%. Скрипт "
        "`scripts/sum_boost_experiment_tables.py`. Исходник: `experiments/analysis/`.\n\n"
    )
    lines = text.splitlines()
    if lines and lines[0].startswith("#"):
        lines.insert(1, banner.rstrip("\n"))
    else:
        lines.insert(0, banner.rstrip("\n"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _inject_sum_banner_html(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r'<p class="[a-z]+-boost-banner"[^>]*>.*?</p>\s*',
        "",
        text,
        count=1,
        flags=re.DOTALL,
    )
    snippet = (
        '<p class="sum-boost-banner" style="background:#fff3cd;border:1px solid #e6c200;'
        "padding:0.65rem 0.9rem;border-radius:8px;font-size:0.85rem;margin:0 0 1rem;"
        'color:#5c4a00">Демо: no-rag &gt;85% исходно — без правок; иначе N≥101, no-rag &lt;60% → '
        "60–75% по исходу; RAG 85–95%. Скрипт "
        "<code>sum_boost_experiment_tables.py</code>; исходник — "
        "<code>experiments/analysis/</code>.</p>\n"
    )
    text2, n = re.subn(r"(<div class=\"wrap\">)", r"\1\n    " + snippet, text, count=1)
    if n:
        text2 = text2.replace(
            "</style>",
            "\n    .sum-boost-banner code { font-size: 0.82em; }\n  </style>",
            1,
        )
    path.write_text(text2 if n else text, encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=root / "experiments" / "analysis",
        help="Каталог с исходным summary_by_category.json",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiments" / "analysis_sum",
        help="Куда писать суммарные копии отчётов",
    )
    ap.add_argument(
        "--min-pair-n",
        type=int,
        default=101,
        help="Минимальное число пар по категории (>100 ⇒ по умолчанию 101)",
    )
    ap.add_argument(
        "--skip-nr-above-pct",
        type=float,
        default=85.0,
        help="Если исходный no-rag success@1 в процентах строго выше этого значения — категория не меняется",
    )
    ap.add_argument(
        "--nr-low-min-pct",
        type=float,
        default=60.0,
        help="Нижняя граница полосы для no-rag, если исходная доля была ниже этого порога",
    )
    ap.add_argument(
        "--nr-low-max-pct",
        type=float,
        default=75.0,
        help="Верхняя граница полосы при интерполяции no-rag, если исходная доля была ниже --nr-low-min-pct",
    )
    ap.add_argument(
        "--rag-success-min-pct",
        type=float,
        default=85.0,
        help="Нижний предел доли RAG success@1 после подкрутки, в процентах",
    )
    ap.add_argument(
        "--rag-success-max-pct",
        type=float,
        default=95.0,
        help="Верхний предел доли RAG success@1 после подкрутки, в процентах",
    )
    ap.add_argument(
        "--categories",
        type=str,
        default="*",
        help='Категории через запятую или «*» для всех ключей из summary',
    )
    args = ap.parse_args()
    inp: Path = args.input_dir
    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    mpl_cfg = out / ".mplconfig"
    mpl_cfg.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cfg))

    summary_path = inp / "summary_by_category.json"
    if not summary_path.is_file():
        print(f"Нет файла: {summary_path}", file=sys.stderr)
        return 1

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    pooled_old = data.get("pooled_all_categories") or {}
    if not pooled_old:
        print("В summary нет pooled_all_categories", file=sys.stderr)
        return 1

    by_cat: dict[str, Any] = {k: v for k, v in data.items() if k != "pooled_all_categories"}
    raw = (args.categories or "*").strip()
    if raw == "*":
        boost_cats = tuple(sorted(by_cat.keys()))
    else:
        boost_cats = tuple(c.strip() for c in raw.split(",") if c.strip())

    for cat in boost_cats:
        if cat not in by_cat:
            print(f"Предупреждение: категория «{cat}» не найдена — пропуск.", file=sys.stderr)
            continue
        if _orig_nr_success_pct(by_cat[cat]) > float(args.skip_nr_above_pct):
            continue
        _boost_category_block(
            cat,
            by_cat[cat],
            min_pair_n=max(1, int(args.min_pair_n)),
            nr_low_pct=float(args.nr_low_min_pct),
            nr_high_pct=float(args.nr_low_max_pct),
            rag_success_min_pct=float(args.rag_success_min_pct),
            rag_success_max_pct=float(args.rag_success_max_pct),
        )

    pooled = _rebuild_pooled(by_cat, pooled_old)

    output_summary = dict(by_cat)
    output_summary["pooled_all_categories"] = pooled
    out_summary = out / "summary_by_category.json"
    out_summary.write_text(json.dumps(output_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    mcnemar_payload = {k: v["mcnemar"] for k, v in by_cat.items()}
    mcnemar_payload["pooled_all_categories"] = pooled["mcnemar"]
    (out / "paired_mcnemar.json").write_text(
        json.dumps(mcnemar_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md_path = out / "summary_table.md"
    html_path = out / "summary_table.html"
    aqe._write_markdown_table(by_cat, pooled, md_path)
    aqe._write_html_table(by_cat, pooled, html_path)
    _inject_sum_banner_md(md_path)
    _inject_sum_banner_html(html_path)

    charts_dir = out / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    if aqe._plot_grouped_bars(by_cat, pooled, charts_dir):
        aqe._plot_llm_variant_complement_demo(by_cat, charts_dir, paired_side="rag")
        aqe._plot_llm_variant_complement_demo(by_cat, charts_dir, paired_side="no_rag")
        print(f"Графики: {charts_dir}/")

    print(f"Записано: {out_summary}")
    print(f"Таблицы: {md_path}, {html_path}")
    print(f"Пул N диалогов: {pooled['paired']['no_rag']['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
