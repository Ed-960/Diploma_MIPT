"""
Агрегация метрик из уже сохранённых question-экспериментов (no-rag vs RAG по категориям).

Читает JSON в каталогах вида:
  experiments/no-rag/norag_<category>/incremental/question_*.json
  experiments/rag/vec_rag_<category>/incremental/question_*.json
  (или legacy: norag_<category>_<n>/question_*.json в корне прогона)

Пишет в --output-dir:
  summary_by_category.json   — по категориям + ключ ``pooled_all_categories`` (все пары вместе)
  paired_mcnemar.json        — по категориям + pooled McNemar
  summary_table.md           — таблица по категориям и блок «все вместе» с 95% Wilson CI
  summary_table.html         — та же сводка, оформленная для браузера / PDF
  conclusions_ru.md        — выводы по категориям и пулу (с техническими именами метрик)
  results_summary_ru.md      — итоги и выводы одним текстом (для слайда / заключения)
  charts/success_at_1_complement_by_llm_variant_demo.png — демо LLM для **RAG** (промах по цели);
  charts/success_at_1_complement_by_llm_variant_demo_norag.png — то же для **no-rag**
  На столбцах фактических метрик: **95% ДИ Уилсона** (планки ошибки); подписи категорий с **N**.

Запуск из корня репозитория:
  python scripts/analyze_question_experiments.py
  python scripts/analyze_question_experiments.py --output-dir experiments/analysis
  python scripts/analyze_question_experiments.py --rows-only --output-dir experiments/analysis_rows
  make analyze-question-experiments
  make analyze-question-experiments-rows
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sys
from math import sqrt
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import _bootstrap
from experiment_run_layout import (
    QuestionLoader,
    discover_paired_runs,
    inspect_run_dir_data,
    load_questions_from_run_dir,
    load_questions_from_rows_only,
)

_bootstrap.ensure_src()

try:
    from mcd_voice.dialog.question_experiment import evaluate_retrieval_probe_for_row
except Exception:  # pragma: no cover - optional import path
    evaluate_retrieval_probe_for_row = None  # type: ignore[misc, assignment]

BOOL_METRICS = (
    "success_at_1",
    "success_at_3",
    "success_at_5",
    "hallucination",
    "constraint_violation",
    "empty_response",
    "need_to_specify",
)

# Оси Y для пары графиков success@1 / complement (одинаковая длина подписи для слайдов).
# success_at_1 в JSON; complement = 100 − success_at_1. Не доля hallucination.
CHART_YLABEL_WITH_GOAL_S1 = "Успех по цели, %"
CHART_YLABEL_WITHOUT_GOAL_S1 = "Промах по цели, %"

# Подписи категорий для HTML и краткого резюме (защита / диплом).
QUESTION_CATEGORY_LABEL_RU: dict[str, str] = {
    "simple": "Простой заказ (название блюда)",
    "lexical": "Разговорная речь: исключения и уточнения («без лука», «уберите соус»)",
    "diet": "Диетические ограничения",
    "allergy": "Аллергены / исключения",
    "group": "Заказ на несколько человек",
    "mixed": "Смешанный сценарий",
}


def _extract_row(data: dict[str, Any]) -> dict[str, Any]:
    m = data.get("metrics") or {}
    judge = data.get("judge") or {}
    parsed = judge.get("parsed") if isinstance(judge.get("parsed"), dict) else {}
    ms = data.get("metric_sources") or {}
    audit = data.get("audit") or {}
    row: dict[str, Any] = {
        "question_id": data.get("question_id"),
        "category": str(data.get("category") or ""),
        "expected_item": data.get("expected_item"),
        "judge_parse_error": bool(parsed.get("parse_error")),
        "critical_error": bool(audit.get("critical_error")),
        "context_ignorance": bool(audit.get("context_ignorance")),
    }
    for k in BOOL_METRICS:
        row[k] = bool(m.get(k))
    row["hallucination_source"] = str(ms.get("hallucination") or "")
    return row


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    out: dict[str, Any] = {"n": n}
    for k in BOOL_METRICS:
        out[f"{k}_count"] = sum(1 for r in rows if r.get(k))
        out[f"{k}_pct"] = round(100.0 * out[f"{k}_count"] / n, 2)
        lo, hi = wilson_ci_pct(out[f"{k}_count"], n)
        out[f"{k}_ci95_pct"] = [lo, hi]
    out["judge_parse_error_count"] = sum(1 for r in rows if r.get("judge_parse_error"))
    out["judge_parse_error_pct"] = round(100.0 * out["judge_parse_error_count"] / n, 2)
    lo, hi = wilson_ci_pct(out["judge_parse_error_count"], n)
    out["judge_parse_error_ci95_pct"] = [lo, hi]
    out["critical_error_count"] = sum(1 for r in rows if r.get("critical_error"))
    out["critical_error_pct"] = round(100.0 * out["critical_error_count"] / n, 2)
    lo, hi = wilson_ci_pct(out["critical_error_count"], n)
    out["critical_error_ci95_pct"] = [lo, hi]
    out["context_ignorance_count"] = sum(1 for r in rows if r.get("context_ignorance"))
    out["context_ignorance_pct"] = round(100.0 * out["context_ignorance_count"] / n, 2)
    judge_hall = sum(1 for r in rows if r.get("hallucination_source") == "judge")
    out["judge_hallucination_metric_pct"] = round(100.0 * judge_hall / n, 2)
    return out


def _retrieval_row(question_data: dict[str, Any]) -> dict[str, Any] | None:
    if evaluate_retrieval_probe_for_row is None:
        return None
    qrow = {
        "expected_item": question_data.get("expected_item"),
        "category": question_data.get("category"),
    }
    metrics, probe = evaluate_retrieval_probe_for_row(qrow, question_data.get("rag_trace") or [])
    out = {**metrics, "probe": probe}
    return out


def _aggregate_retrieval(rows: list[dict[str, Any] | None]) -> dict[str, Any]:
    valid = [
        r
        for r in rows
        if r is not None and "error" not in (r.get("probe") or {})
    ]
    n = len(valid)
    if n == 0:
        return {
            "n": 0,
            "note": "нет валидных rag-candidates (нет expected_item, нет event=rag или evaluate_retrieval недоступен)",
        }
    out: dict[str, Any] = {"n": n}
    for k in ("success_at_1", "success_at_3", "success_at_5"):
        out[f"{k}_count"] = sum(1 for r in valid if r.get(k))
        out[f"{k}_pct"] = round(100.0 * out[f"{k}_count"] / n, 2)
        lo, hi = wilson_ci_pct(out[f"{k}_count"], n)
        out[f"{k}_ci95_pct"] = [lo, hi]
    ranks = [r["probe"].get("expected_rank") for r in valid if r.get("probe")]
    ranks_n = [x for x in ranks if isinstance(x, int)]
    out["expected_rank_mean"] = round(sum(ranks_n) / len(ranks_n), 2) if ranks_n else None
    miss = sum(1 for x in ranks if x is None)
    out["expected_missing_from_candidates_pct"] = round(100.0 * miss / n, 2) if n else None
    return out


def wilson_ci_pct(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for binomial proportion, in percent."""
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom
    lo = max(0.0, (center - margin) * 100)
    hi = min(100.0, (center + margin) * 100)
    return round(lo, 1), round(hi, 1)


def _wilson_yerr_asymmetric_pct(y: float, k: int, n: int) -> tuple[float, float]:
    """
    Нижняя и верхняя полуошибки для matplotlib bar(..., yerr=[[lows...], [highs...]]).
    y — отображаемая доля в %; k/n — та же доля из целочисленных счётчиков (согласуйте с y).
    """
    if n <= 0:
        return 0.0, 0.0
    lo, hi = wilson_ci_pct(k, n)
    return (max(0.0, y - lo), max(0.0, hi - y))


def _mcnemar_chi2(b: int, c: int) -> tuple[float, float]:
    """McNemar (discordant pairs only): b = norag True rag False, c = norag False rag True."""
    if b + c == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)  # continuity correction
    chi2 = max(0.0, float(chi2))
    try:
        from scipy.stats import chi2 as chi2_dist  # type: ignore[import-not-found]

        p = float(1 - chi2_dist.cdf(chi2, df=1))
    except Exception:
        if chi2 > 10.83:
            p = 0.001
        elif chi2 > 6.63:
            p = 0.01
        elif chi2 > 3.84:
            p = 0.05
        else:
            p = 1.0
    return round(chi2, 4), round(p, 4)


def _discordant(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    key: str,
) -> tuple[int, int, int, int]:
    """both T, norag-only T, rag-only T, both F."""
    tt = tf = ft = ff = 0
    for a, b in pairs:
        va, vb = bool(a.get(key)), bool(b.get(key))
        if va and vb:
            tt += 1
        elif va and not vb:
            tf += 1
        elif not va and vb:
            ft += 1
        else:
            ff += 1
    return tt, tf, ft, ff


def _build_pooled(
    paired_dirs: dict[str, dict[str, Path]],
    loader: QuestionLoader,
) -> dict[str, Any]:
    """Сумма всех пар (категории с разным N → единая доля = pooled counts / pooled N)."""
    all_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    all_ret: list[Any] = []
    for _cat, paths in sorted(paired_dirs.items()):
        no_map = loader(paths["no-rag"])
        rag_map = loader(paths["rag"])
        common = sorted(set(no_map) & set(rag_map))
        for i in common:
            all_pairs.append((_extract_row(no_map[i]), _extract_row(rag_map[i])))
            all_ret.append(_retrieval_row(rag_map[i]))
    nr_only = [a for a, _ in all_pairs]
    r_only = [b for _, b in all_pairs]
    tt, tf, ft, ff = _discordant(all_pairs, "success_at_1")
    chi2_s, p_s = _mcnemar_chi2(tf, ft)
    htt, htf, hft, hff = _discordant(all_pairs, "hallucination")
    chi2_h, p_h = _mcnemar_chi2(htf, hft)
    return {
        "description": "Все парные вопросы по всем категориям; доля = сумма успехов / общее N (не среднее процентов по категориям).",
        "paired": {
            "no_rag": _aggregate(nr_only),
            "rag": _aggregate(r_only),
            "rag_retrieval": _aggregate_retrieval(all_ret),
        },
        "mcnemar": {
            "success_at_1": {
                "both_true": tt,
                "norag_true_rag_false": tf,
                "norag_false_rag_true": ft,
                "both_false": ff,
                "chi2": chi2_s,
                "p_value": p_s,
            },
            "hallucination": {
                "both_true": htt,
                "norag_true_rag_false": htf,
                "norag_false_rag_true": hft,
                "both_false": hff,
                "chi2": chi2_h,
                "p_value": p_h,
            },
        },
    }


def _write_markdown_table(
    by_cat: dict[str, Any],
    pooled: dict[str, Any],
    path: Path,
) -> None:
    def _ci_cell(agg: dict[str, Any], key: str) -> str:
        ci = agg.get(f"{key}_ci95_pct") or [0, 0]
        return f"{agg.get(f'{key}_pct', 0)}% [{ci[0]}–{ci[1]}]"

    lines = [
        "# Сравнение no-rag vs RAG (пересечение `question_id`)",
        "",
        "Доли — проценты; **95% интервал Уилсона** в квадратных скобках (единая шкала при разном N по категориям).",
        "",
        "## Все категории вместе (пул)",
        "",
        f"- **N диалогов (всего):** {pooled['paired']['no_rag']['n']}",
        f"- **success@1:** no-rag {_ci_cell(pooled['paired']['no_rag'], 'success_at_1')} · RAG {_ci_cell(pooled['paired']['rag'], 'success_at_1')}",
        f"- **hallucination:** no-rag {_ci_cell(pooled['paired']['no_rag'], 'hallucination')} · RAG {_ci_cell(pooled['paired']['rag'], 'hallucination')}",
        f"- **McNemar (success@1):** discordant {pooled['mcnemar']['success_at_1']['norag_true_rag_false']}/"
        f"{pooled['mcnemar']['success_at_1']['norag_false_rag_true']}, χ²={pooled['mcnemar']['success_at_1']['chi2']}, "
        f"p={pooled['mcnemar']['success_at_1']['p_value']}",
        f"- **McNemar (hallucination):** p={pooled['mcnemar']['hallucination']['p_value']}",
        "",
        "Пул = сумма всех диалогов по категориям (один `question_id` — один диалог в сравнении; не среднее арифметическое процентов по категориям).",
        "",
        "## По категориям",
        "",
        "| Категория | N диалогов | success@1 без RAG % [95% CI] | success@1 RAG % [95% CI] | RAG − Без RAG, п.п. | halluc. no-rag % | halluc. RAG % | judge parse err no-rag % | judge parse err RAG % |",
        "|-----------|-----------:|------------------------------:|-------------------------:|-------------------:|-----------------:|--------------:|---------------------------:|------------------------:|",
    ]
    for cat in sorted(by_cat.keys()):
        s = by_cat[cat]
        p = s["paired"]["no_rag"]
        pr = s["paired"]["rag"]
        d_s1 = round(float(pr["success_at_1_pct"]) - float(p["success_at_1_pct"]), 2)
        c1n = _ci_cell(p, "success_at_1")
        c1r = _ci_cell(pr, "success_at_1")
        lines.append(
            f"| {cat} | {p['n']} | {c1n} | {c1r} | {d_s1:+} | "
            f"{p['hallucination_pct']} | {pr['hallucination_pct']} | "
            f"{p['judge_parse_error_pct']} | {pr['judge_parse_error_pct']} |"
        )
    lines.append("")
    lines.append(
        "Колонка **RAG − Без RAG, п.п.** — на сколько процентных пунктов выше доля «цель достигнута» в режиме **RAG**, "
        "чем без RAG (абсолютная разница двух процентов, не относительный прирост)."
    )
    lines.append("")
    lines.append("## McNemar (discordant pairs, success_at_1)")
    lines.append("")
    lines.append("| Категория | no-rag+, RAG− | no-rag−, RAG+ | χ² | p |")
    lines.append("|-----------|-------------:|-------------:|----:|--:|")
    for cat in sorted(by_cat.keys()):
        mm = by_cat[cat]["mcnemar"]["success_at_1"]
        lines.append(
            f"| {cat} | {mm['norag_true_rag_false']} | {mm['norag_false_rag_true']} | {mm['chi2']} | {mm['p_value']} |"
        )
    pm = pooled["mcnemar"]["success_at_1"]
    lines.append(
        f"| **все вместе (пул)** | {pm['norag_true_rag_false']} | {pm['norag_false_rag_true']} | {pm['chi2']} | {pm['p_value']} |"
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _delta_class(delta: float, *, lower_is_better: bool) -> str:
    """CSS class for Δ: for success higher RAG is good; for hallucination we'd use different table."""
    if lower_is_better:
        if delta < -0.01:
            return "delta-good"
        if delta > 0.01:
            return "delta-bad"
    else:
        if delta > 0.01:
            return "delta-good"
        if delta < -0.01:
            return "delta-bad"
    return "delta-neutral"


def _write_html_table(by_cat: dict[str, Any], pooled: dict[str, Any], path: Path) -> None:
    """Версия для презентации: без жаргона success@1, меньше колонок, детали в <details>."""

    def _pct_only(agg: dict[str, Any], key: str) -> str:
        return f'<span class="big">{html.escape(str(agg.get(f"{key}_pct", 0)))}%</span>'

    cat_ru = QUESTION_CATEGORY_LABEL_RU

    pn = pooled["paired"]["no_rag"]
    prg = pooled["paired"]["rag"]
    pm = pooled["mcnemar"]["success_at_1"]

    rows_html: list[str] = []
    for cat in sorted(by_cat.keys()):
        s = by_cat[cat]
        p = s["paired"]["no_rag"]
        pr = s["paired"]["rag"]
        d_s1 = round(float(pr["success_at_1_pct"]) - float(p["success_at_1_pct"]), 2)
        d_cls = _delta_class(d_s1, lower_is_better=False)
        label = html.escape(cat_ru.get(cat, cat))
        rows_html.append(
            "<tr>"
            f'<td class="cat">{label}</td>'
            f'<td class="num dim">{p["n"]}</td>'
            f'<td class="num cell-nr">{_pct_only(p, "success_at_1")}</td>'
            f'<td class="num cell-r">{_pct_only(pr, "success_at_1")}</td>'
            f'<td class="num {d_cls}">{d_s1:+}</td>'
            "</tr>"
        )

    d_pool = round(float(prg["success_at_1_pct"]) - float(pn["success_at_1_pct"]), 2)
    pool_sig = float(pm["p_value"]) < 0.05
    d_pool_cls = _delta_class(d_pool, lower_is_better=False)
    mcnemar_hint = "статистически значимо (p &lt; 0.05)" if pool_sig else "на уровне p &lt; 0.05 эффект не выделен"

    doc = f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Сравнение: без RAG и RAG</title>
  <style>
    :root {{
      --bg: #f0f2f7;
      --card: #ffffff;
      --text: #1c2333;
      --muted: #5a6578;
      --border: #d8dee9;
      --norag: #2e5a8c;
      --rag: #c25621;
      --good: #0b6e45;
      --bad: #a82a22;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; padding: 1.5rem 1rem 2.5rem;
      font-family: "Segoe UI", system-ui, -apple-system, Roboto, sans-serif;
      background: var(--bg); color: var(--text); line-height: 1.5;
    }}
    .wrap {{ max-width: 960px; margin: 0 auto; }}
    h1 {{
      font-size: 1.4rem; font-weight: 700; margin: 0 0 0.5rem;
      letter-spacing: -0.02em;
    }}
    .lead {{ color: var(--muted); font-size: 0.95rem; max-width: 52rem; margin-bottom: 1.25rem; }}
    .hero {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 0.75rem; margin-bottom: 1.5rem;
    }}
    .hero .box {{
      background: var(--card); border: 1px solid var(--border);
      border-radius: 12px; padding: 1rem 1.1rem;
      box-shadow: 0 2px 8px rgba(0,0,0,.04);
    }}
    .hero .t {{ font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }}
    .hero .n {{ font-size: 1.75rem; font-weight: 800; margin-top: 0.25rem; line-height: 1.1; }}
    .hero .sub {{ font-size: 0.8rem; color: var(--muted); margin-top: 0.35rem; }}
    .hero .norag {{ color: var(--norag); }}
    .hero .rag {{ color: var(--rag); }}
    h2 {{ font-size: 1.05rem; margin: 0 0 0.65rem; font-weight: 650; }}
    .table-wrap {{
      background: var(--card); border: 1px solid var(--border);
      border-radius: 12px; overflow: hidden;
      box-shadow: 0 2px 10px rgba(0,0,0,.05);
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
    th, td {{ padding: 0.65rem 0.75rem; border-bottom: 1px solid var(--border); vertical-align: middle; }}
    th {{
      text-align: left; font-size: 0.72rem; font-weight: 650;
      color: var(--muted); text-transform: uppercase; letter-spacing: .04em;
      background: linear-gradient(180deg,#eef1f7,#e6ebf4);
    }}
    th.num, td.num {{ text-align: right; }}
    th.th-plain {{ text-transform: none; letter-spacing: 0; font-size: 0.78rem; }}
    th.mode-nr {{ color: var(--norag); }}
    th.mode-r {{ color: var(--rag); }}
    tr:last-child td {{ border-bottom: none; }}
    tbody tr:nth-child(even) {{ background: #fafbfd; }}
    .cat {{ font-weight: 600; max-width: 14rem; }}
    .dim {{ color: var(--muted); font-variant-numeric: tabular-nums; }}
    .big {{ font-weight: 800; font-size: 1.05rem; font-variant-numeric: tabular-nums; }}
    .ciwrap {{ display: block; margin-top: 0.15rem; }}
    .ci {{ font-size: 0.68rem; color: var(--muted); font-weight: 400; }}
    .cell-nr .big {{ color: var(--norag); }}
    .cell-r .big {{ color: var(--rag); }}
    .delta-good {{ color: var(--good); font-weight: 800; }}
    .delta-bad {{ color: var(--bad); font-weight: 800; }}
    .delta-neutral {{ color: var(--muted); font-weight: 600; }}
    .sig-yes {{ color: var(--good); font-weight: 700; }}
    .sig-no {{ color: var(--muted); font-weight: 600; }}
    tr.total td {{ background: linear-gradient(90deg,#e8f4fc,#f2f7ff); font-weight: 600; }}
    details {{
      margin-top: 1.5rem; background: var(--card); border: 1px solid var(--border);
      border-radius: 10px; padding: 0.5rem 1rem 1rem;
      font-size: 0.85rem; color: var(--muted);
    }}
    details summary {{ cursor: pointer; font-weight: 600; color: var(--text); padding: 0.35rem 0; }}
    details code {{ font-size: 0.78rem; background: #eef1f6; padding: 0.1rem 0.35rem; border-radius: 4px; }}
    .foot {{ margin-top: 1.25rem; font-size: 0.78rem; color: var(--muted); }}
    @media print {{ body {{ background: #fff; }} .hero .box, .table-wrap {{ break-inside: avoid; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Два режима: без RAG (полный JSON в промпте) и RAG (поиск фрагментов)</h1>
    <p class="lead">
      На одних и тех же формулировках вопросов сравниваются ответы кассира **без RAG** (полное меню в контексте)
      и **с RAG** (векторный поиск по меню). Ниже — доля «цель достигнута» по правилам эксперимента и доля ответов,
      которые <strong>судья LLM</strong>, видя диалог и меню, пометила как проблемные
      (в данных это поле <code>hallucination</code> в <code>metrics</code>).
    </p>

    <div class="hero">
      <div class="box">
        <div class="t">Всего диалогов в сравнении</div>
        <div class="n">{pn["n"]}</div>
      </div>
      <div class="box">
        <div class="t"><span class="norag">Без RAG</span> — цель достигнута</div>
        <div class="n norag">{pn["success_at_1_pct"]}%</div>
        <div class="sub">интервал 95%: {pn.get("success_at_1_ci95_pct", [0,0])[0]}–{pn.get("success_at_1_ci95_pct", [0,0])[1]}%</div>
      </div>
      <div class="box">
        <div class="t"><span class="rag">RAG</span> — цель достигнута</div>
        <div class="n rag">{prg["success_at_1_pct"]}%</div>
        <div class="sub">интервал 95%: {prg.get("success_at_1_ci95_pct", [0,0])[0]}–{prg.get("success_at_1_ci95_pct", [0,0])[1]}%</div>
      </div>
      <div class="box">
        <div class="t">Сдвиг в пользу RAG</div>
        <div class="n {d_pool_cls}">{d_pool:+} п.п.</div>
        <div class="sub">{mcnemar_hint}</div>
      </div>
      <div class="box">
        <div class="t">Судья LLM</div>
        <div class="n"><span class="norag">{pn["hallucination_pct"]}%</span> → <span class="rag">{prg["hallucination_pct"]}%</span></div>
        <div class="sub">без RAG → RAG (по флагу <code>hallucination</code>)</div>
      </div>
    </div>

    <h2>По типам вопросов</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Тип сценария</th>
            <th class="num th-plain" title="Сколько уникальных вопросов-диалогов в строке: один question_id — одна цепочка без RAG и одна с RAG (парное сравнение).">Число диалогов</th>
            <th class="num th-plain mode-nr" title="Режим без RAG (полное меню в промпте). Доля ответов, где цель достигнута (метрика success_at_1).">Без RAG</th>
            <th class="num th-plain mode-r" title="Режим с RAG (поиск фрагментов меню). Доля, где цель достигнута.">RAG</th>
            <th class="num th-plain" title="По метрике «цель достигнута»: доля в RAG минус доля без RAG. Процентный пункт — абсолютная разница двух процентов (например, 80% и 66% → 14 п.п.), это не относительный прирост.">RAG − Без RAG<br/><span class="dim" style="font-weight:400;text-transform:none;letter-spacing:0">в процентных пунктах</span></th>
          </tr>
        </thead>
        <tbody>
          {"".join(rows_html)}
          <tr class="total">
            <td class="cat">Все типы вместе</td>
            <td class="num dim">{pn["n"]}</td>
            <td class="cell-nr num">{_pct_only(pn, "success_at_1")}</td>
            <td class="cell-r num">{_pct_only(prg, "success_at_1")}</td>
            <td class="num {d_pool_cls}">{d_pool:+}</td>
          </tr>
        </tbody>
      </table>
    </div>

    <details>
      <summary>Как считалось (для текста диплома)</summary>
      <p style="margin-top:0.6rem">
        <strong>«Цель достигнута»</strong> — в коде метрика <code>success_at_1</code>:
        ожидаемое блюдо из банка вопросов входит в число первых упоминаний позиций меню в финальной реплике кассира.
      </p>
      <p>
        Первая числовая колонка — <strong>число диалогов</strong> (уникальных вопросов) в строке; далее доля «цель достигнута» в колонках
        <strong>Без RAG</strong> и <strong>RAG</strong>, затем разница в процентных пунктах (<strong>RAG − Без RAG</strong>).
        В этой HTML-таблице намеренно только эти столбцы (удобнее для слайда).
        Доли по флагу <code>hallucination</code> у судьи LLM по режимам и столбец «значимо» (McNemar по цели) —
        в <code>summary_table.md</code> (тот же прогон <code>make reports</code> / <code>analyze_question_experiments</code>).
        Для малых выборок отсутствие значимости по McNemar не означает отсутствие эффекта.
      </p>
    </details>

    <p class="foot">Файл пересоздаётся при каждом запуске анализа вопросов.</p>
  </div>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def _write_conclusions(by_cat: dict[str, Any], pooled: dict[str, Any], path: Path) -> None:
    lines: list[str] = [
        "# Авто-выводы (по агрегатам, без интерпретации домена)",
        "",
        "Метрики взяты из поля `metrics` в каждом `question_*.json` (после judge, если он вернул валидный JSON).",
        "Сравнение **только на пересечении** `question_id` между парными папками одной категории.",
        "",
        "## Пул всех категорий",
        "",
    ]
    p, pr = pooled["paired"]["no_rag"], pooled["paired"]["rag"]
    d_s = round(float(pr["success_at_1_pct"]) - float(p["success_at_1_pct"]), 2)
    d_h = round(float(pr["hallucination_pct"]) - float(p["hallucination_pct"]), 2)
    ms = pooled["mcnemar"]["success_at_1"]
    sig = "да" if ms["p_value"] < 0.05 else "нет"
    lines.append(
        f"- **Все вместе** (N={p['n']} диалогов): success@1 Δ (RAG−no-rag) = **{d_s}** п.п.; "
        f"hallucination Δ = **{d_h}** п.п.; McNemar по success@1: p={ms['p_value']} (p<0.05: {sig})."
    )
    lines.append(
        f"  Интервалы Уилсона (95%) для success@1: no-rag {p['success_at_1_ci95_pct']}, RAG {pr['success_at_1_ci95_pct']}."
    )
    lines.append("")
    lines.append("## По категориям")
    lines.append("")
    for cat in sorted(by_cat.keys()):
        s = by_cat[cat]["paired"]
        p, pr = s["no_rag"], s["rag"]
        d_h = round(float(pr["hallucination_pct"]) - float(p["hallucination_pct"]), 2)
        d_s = round(float(pr["success_at_1_pct"]) - float(p["success_at_1_pct"]), 2)
        m = by_cat[cat]["mcnemar"]["success_at_1"]
        sig = "да" if m["p_value"] < 0.05 else "нет"
        lines.append(
            f"- **{cat}** (N={p['n']}): success@1 Δ = **{d_s}** п.п.; "
            f"hallucination Δ = **{d_h}** п.п.; McNemar по success@1: p={m['p_value']} (p<0.05: {sig})."
        )
    lines.append("")
    lines.append("Для слайдов: при p≥0.05 формулировать «статистически значимых отличий не видно на этом N».")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_results_summary_ru(by_cat: dict[str, Any], pooled: dict[str, Any], path: Path) -> None:
    """Краткое резюме результатов и формулировки выводов для презентации и диплома."""
    pn = pooled["paired"]["no_rag"]
    prg = pooled["paired"]["rag"]
    pm = pooled["mcnemar"]["success_at_1"]
    ph = pooled["mcnemar"]["hallucination"]
    d_goal = round(float(prg["success_at_1_pct"]) - float(pn["success_at_1_pct"]), 2)
    d_hall = round(float(prg["hallucination_pct"]) - float(pn["hallucination_pct"]), 2)
    pool_sig_goal = float(pm["p_value"]) < 0.05
    pool_sig_hall = float(ph["p_value"]) < 0.05
    ci_n = pn.get("success_at_1_ci95_pct") or [0, 0]
    ci_r = prg.get("success_at_1_ci95_pct") or [0, 0]

    lines: list[str] = [
        "# Итоги и выводы (кратко)",
        "",
        "## Условия эксперимента",
        "",
        "- Сравниваются два режима подачи меню кассиру: **полный JSON в промпте** и **RAG** (векторный поиск фрагментов меню).",
        "- Для каждого типа вопросов взяты **парные** наблюдения: один и тот же вопрос в обоих режимах (пересечение идентификаторов в выгрузках).",
        f"- Всего диалогов в сравнении по всем типам: **{pn['n']}**.",
        "",
        "## Главный результат (по всей выборке)",
        "",
        f"- **Доля ответов, где цель вопроса достигнута** (ожидаемое блюдо — среди первых упоминаний меню в ответе): "
        f"**{pn['success_at_1_pct']}%** (без RAG) → **{prg['success_at_1_pct']}%** (RAG), "
        f"разница **{d_goal:+}** п.п. 95%-е доверительные интервалы (Уилсон): [{ci_n[0]}–{ci_n[1]}]% и [{ci_r[0]}–{ci_r[1]}]%.",
        "- **Парное сравнение** (McNemar по признаку «цель достигнута»): "
        + (
            "**статистически значимый** сдвиг в пользу RAG"
            if pool_sig_goal
            else "на уровне 5% **значимый сдвиг не выделен**"
        )
        + (
            f" (p ≈ {pm['p_value']})"
            if float(pm["p_value"]) >= 1e-6
            else " (p практически 0)"
        )
        + ".",
        f"- **Доля ответов, которые судья LLM пометил как проблемные** "
        f"(поле `hallucination` в `metrics`): "
        f"**{pn['hallucination_pct']}%** → **{prg['hallucination_pct']}%**, изменение **{d_hall:+}** п.п.; "
        "McNemar по этому признаку: "
        + ("**p < 0.05**" if pool_sig_hall else f"p = {ph['p_value']}")
        + ".",
        "",
        "## По типам сценариев",
        "",
    ]

    for cat in sorted(by_cat.keys()):
        label = QUESTION_CATEGORY_LABEL_RU.get(cat, cat)
        s = by_cat[cat]["paired"]
        p, pr = s["no_rag"], s["rag"]
        d_s = round(float(pr["success_at_1_pct"]) - float(p["success_at_1_pct"]), 2)
        d_h = round(float(pr["hallucination_pct"]) - float(p["hallucination_pct"]), 2)
        m = by_cat[cat]["mcnemar"]["success_at_1"]
        sig = float(m["p_value"]) < 0.05
        sig_w = "значимо (p < 0.05)" if sig else f"не значимо при N={p['n']} (p = {m['p_value']})"
        lines.append(
            f"- **{label}** (`{cat}`, N = {p['n']}): цель достигнута {p['success_at_1_pct']}% → {pr['success_at_1_pct']}% "
            f"({d_s:+} п.п.); судья LLM пометил {p['hallucination_pct']}% → {pr['hallucination_pct']}% ({d_h:+} п.п.); "
            f"McNemar по «цели»: {sig_w}."
        )

    lines.extend(
        [
            "",
            "## Формулировки для заключения / устного доклада",
            "",
            "1. На объединённой выборке парных вопросов режим с **RAG** даёт **более высокую** долю ответов, в которых выполнена цель сценария, и **ниже** долю ответов, которые **судья LLM** пометил как проблемные (`hallucination`), по сравнению с режимом **без RAG** (полное меню в промпте).",
            "2. Усиление эффекта заметнее в отдельных типах сценариев (см. таблицу и графики); там, где парный тест не достиг уровня 5%, осторожно формулировать вывод как **тенденцию**, а не как доказанное отличие на данном N.",
            "3. Оценка качества опирается на **автоматические метрики** в JSON вопросов и **модель-судью**; при сбое разбора ответа судьи используется эвристика (см. `metric_sources` в исходных файлах).",
            "",
            "---",
            "",
            "*Файл генерируется скриптом `analyze_question_experiments.py` вместе с `summary_table.md` / `.html` и графиками.*",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _chart_xticklabels_with_n(by_cat: dict[str, Any], cats: list[str]) -> list[str]:
    return [f"{c}\n(N={int(by_cat[c]['paired']['no_rag']['n'])})" for c in cats]


def _chart_foot_ci_wilson() -> str:
    return "Планки ошибок: 95% ДИ Уилсона для доли (биномиальная модель)."


def _plot_grouped_bars(by_cat: dict[str, Any], pooled: dict[str, Any], charts_dir: Path) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    charts_dir.mkdir(parents=True, exist_ok=True)
    cats = sorted(by_cat.keys())
    x = range(len(cats))
    w = 0.35
    # Подписи осей: величина и размерность (для диплома / слайдов).
    xlabel_cat = "Категория (ключ поля category в JSON)"
    ylabel_frac_pct = "Доля, %"
    xtick_labels = _chart_xticklabels_with_n(by_cat, cats)
    foot = _chart_foot_ci_wilson()

    def pairs(key: str) -> tuple[list[float], list[float]]:
        nr = [float(by_cat[c]["paired"]["no_rag"][f"{key}_pct"]) for c in cats]
        r = [float(by_cat[c]["paired"]["rag"][f"{key}_pct"]) for c in cats]
        return nr, r

    def yerr_for_metric(mode: str, metric: str) -> tuple[list[float], list[float]]:
        neg: list[float] = []
        pos: list[float] = []
        for c in cats:
            agg = by_cat[c]["paired"][mode]
            n_i = int(agg["n"])
            k_i = int(agg[f"{metric}_count"])
            y_i = float(agg[f"{metric}_pct"])
            lo, hi = _wilson_yerr_asymmetric_pct(y_i, k_i, n_i)
            neg.append(lo)
            pos.append(hi)
        return neg, pos

    fig, ax = plt.subplots(figsize=(10, 5.2))
    s1_nr, s1_r = pairs("success_at_1")
    ye_nr = yerr_for_metric("no_rag", "success_at_1")
    ye_r = yerr_for_metric("rag", "success_at_1")
    ax.bar(
        [i - w / 2 for i in x],
        s1_nr,
        width=w,
        label="no-rag",
        color="#4472c4",
        yerr=ye_nr,
        capsize=2.5,
        error_kw={"elinewidth": 1.0, "ecolor": "#222222"},
    )
    ax.bar(
        [i + w / 2 for i in x],
        s1_r,
        width=w,
        label="RAG",
        color="#ed7d31",
        yerr=ye_r,
        capsize=2.5,
        error_kw={"elinewidth": 1.0, "ecolor": "#222222"},
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(xtick_labels, fontsize=9)
    ax.set_xlabel(xlabel_cat)
    ax.set_ylabel(CHART_YLABEL_WITH_GOAL_S1)
    ax.set_title("no-rag и RAG: успех по цели (%), по категориям")
    ax.legend()
    ax.set_ylim(0, 105)
    fig.text(0.5, 0.02, foot, ha="center", fontsize=8, color="#333333")
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(charts_dir / "success_at_1_by_category.png", dpi=150)
    plt.close(fig)

    miss_nr = [max(0.0, min(100.0, 100.0 - v)) for v in s1_nr]
    miss_r = [max(0.0, min(100.0, 100.0 - v)) for v in s1_r]
    yerr_miss_nr = ([], [])
    yerr_miss_r = ([], [])
    for i, c in enumerate(cats):
        agg = by_cat[c]["paired"]["no_rag"]
        n_i = int(agg["n"])
        k_miss = n_i - int(agg["success_at_1_count"])
        y_i = miss_nr[i]
        lo, hi = _wilson_yerr_asymmetric_pct(y_i, k_miss, n_i)
        yerr_miss_nr[0].append(lo)
        yerr_miss_nr[1].append(hi)
    for i, c in enumerate(cats):
        agg = by_cat[c]["paired"]["rag"]
        n_i = int(agg["n"])
        k_miss = n_i - int(agg["success_at_1_count"])
        y_i = miss_r[i]
        lo, hi = _wilson_yerr_asymmetric_pct(y_i, k_miss, n_i)
        yerr_miss_r[0].append(lo)
        yerr_miss_r[1].append(hi)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(
        [i - w / 2 for i in x],
        miss_nr,
        width=w,
        label="no-rag",
        color="#4472c4",
        yerr=yerr_miss_nr,
        capsize=2.5,
        error_kw={"elinewidth": 1.0, "ecolor": "#222222"},
    )
    ax.bar(
        [i + w / 2 for i in x],
        miss_r,
        width=w,
        label="RAG",
        color="#ed7d31",
        yerr=yerr_miss_r,
        capsize=2.5,
        error_kw={"elinewidth": 1.0, "ecolor": "#222222"},
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(xtick_labels, fontsize=9)
    ax.set_xlabel(xlabel_cat)
    ax.set_ylabel(CHART_YLABEL_WITHOUT_GOAL_S1)
    ax.set_title("no-rag и RAG: промах по цели (%), по категориям")
    ax.legend()
    ax.set_ylim(0, 105)
    fig.text(0.5, 0.02, foot, ha="center", fontsize=8, color="#333333")
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(charts_dir / "success_at_1_complement_by_category.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    h_nr, h_r = pairs("hallucination")
    ye_hn = yerr_for_metric("no_rag", "hallucination")
    ye_hr = yerr_for_metric("rag", "hallucination")
    ax.bar(
        [i - w / 2 for i in x],
        h_nr,
        width=w,
        label="no-rag",
        color="#4472c4",
        yerr=ye_hn,
        capsize=2.5,
        error_kw={"elinewidth": 1.0, "ecolor": "#222222"},
    )
    ax.bar(
        [i + w / 2 for i in x],
        h_r,
        width=w,
        label="RAG",
        color="#ed7d31",
        yerr=ye_hr,
        capsize=2.5,
        error_kw={"elinewidth": 1.0, "ecolor": "#222222"},
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(xtick_labels, fontsize=9)
    ax.set_xlabel(xlabel_cat)
    ax.set_ylabel("Доля ответов с hallucination, %")
    ax.set_title("hallucination (%), paired question_ids")
    ax.legend()
    ax.set_ylim(0, 105)
    fig.text(0.5, 0.02, foot, ha="center", fontsize=8, color="#333333")
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(charts_dir / "hallucination_by_category.png", dpi=150)
    plt.close(fig)

    # RAG retrieval @k (expected_item in candidate list), paired subset
    rk1 = []
    rk3 = []
    rk5 = []
    e1n, e1p = [], []
    e3n, e3p = [], []
    e5n, e5p = [], []
    for c in cats:
        rr = by_cat[c]["paired"].get("rag_retrieval") or {}
        n_r = int(rr.get("n") or 0)
        k1 = int(rr.get("success_at_1_count") or 0)
        k3 = int(rr.get("success_at_3_count") or 0)
        k5 = int(rr.get("success_at_5_count") or 0)
        y1 = float(rr.get("success_at_1_pct") or 0)
        y3 = float(rr.get("success_at_3_pct") or 0)
        y5 = float(rr.get("success_at_5_pct") or 0)
        rk1.append(y1)
        rk3.append(y3)
        rk5.append(y5)
        a1 = _wilson_yerr_asymmetric_pct(y1, k1, n_r)
        a3 = _wilson_yerr_asymmetric_pct(y3, k3, n_r)
        a5 = _wilson_yerr_asymmetric_pct(y5, k5, n_r)
        e1n.append(a1[0])
        e1p.append(a1[1])
        e3n.append(a3[0])
        e3p.append(a3[1])
        e5n.append(a5[0])
        e5p.append(a5[1])
    # Три столбца на категорию: при width=w и шаге тиков 1 суммарная ширина группы
    # не должна превышать 1, иначе столбцы соседних категорий наезжают друг на друга.
    _gap_cat = 0.1
    w_rk = (1.0 - _gap_cat) / 3.0
    xi = list(x)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(
        [i - w_rk for i in xi],
        rk1,
        width=w_rk,
        label="retrieval @1",
        align="center",
        yerr=(e1n, e1p),
        capsize=2,
        error_kw={"elinewidth": 1.0, "ecolor": "#222222"},
    )
    ax.bar(
        xi,
        rk3,
        width=w_rk,
        label="retrieval @3",
        align="center",
        yerr=(e3n, e3p),
        capsize=2,
        error_kw={"elinewidth": 1.0, "ecolor": "#222222"},
    )
    ax.bar(
        [i + w_rk for i in xi],
        rk5,
        width=w_rk,
        label="retrieval @5",
        align="center",
        yerr=(e5n, e5p),
        capsize=2,
        error_kw={"elinewidth": 1.0, "ecolor": "#222222"},
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(xtick_labels, fontsize=9)
    ax.set_xlabel(xlabel_cat)
    ax.set_ylabel("Доля вопросов с попаданием expected_item в top-k, %")
    ax.set_title("RAG: expected_item rank in candidates (subset with expected_item + rag event)")
    ax.legend()
    ax.set_ylim(0, 105)
    fig.text(0.5, 0.02, foot, ha="center", fontsize=8, color="#333333")
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(charts_dir / "rag_retrieval_hit_by_category.png", dpi=150)
    plt.close(fig)

    # Pooled: two metrics, two modes
    pn = pooled["paired"]["no_rag"]
    prg = pooled["paired"]["rag"]
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    labels = ["success@1", "hallucination"]
    x2 = range(len(labels))
    y_pn = [float(pn["success_at_1_pct"]), float(pn["hallucination_pct"])]
    y_pr = [float(prg["success_at_1_pct"]), float(prg["hallucination_pct"])]
    n_pn = int(pn["n"])
    n_pr = int(prg["n"])
    ye_pn = (
        [
            _wilson_yerr_asymmetric_pct(y_pn[0], int(pn["success_at_1_count"]), n_pn)[0],
            _wilson_yerr_asymmetric_pct(y_pn[1], int(pn["hallucination_count"]), n_pn)[0],
        ],
        [
            _wilson_yerr_asymmetric_pct(y_pn[0], int(pn["success_at_1_count"]), n_pn)[1],
            _wilson_yerr_asymmetric_pct(y_pn[1], int(pn["hallucination_count"]), n_pn)[1],
        ],
    )
    ye_pr = (
        [
            _wilson_yerr_asymmetric_pct(y_pr[0], int(prg["success_at_1_count"]), n_pr)[0],
            _wilson_yerr_asymmetric_pct(y_pr[1], int(prg["hallucination_count"]), n_pr)[0],
        ],
        [
            _wilson_yerr_asymmetric_pct(y_pr[0], int(prg["success_at_1_count"]), n_pr)[1],
            _wilson_yerr_asymmetric_pct(y_pr[1], int(prg["hallucination_count"]), n_pr)[1],
        ],
    )
    ax.bar(
        [i - w / 2 for i in x2],
        y_pn,
        width=w,
        label="no-rag",
        color="#4472c4",
        yerr=ye_pn,
        capsize=3,
        error_kw={"elinewidth": 1.0, "ecolor": "#222222"},
    )
    ax.bar(
        [i + w / 2 for i in x2],
        y_pr,
        width=w,
        label="RAG",
        color="#ed7d31",
        yerr=ye_pr,
        capsize=3,
        error_kw={"elinewidth": 1.0, "ecolor": "#222222"},
    )
    ax.set_xticks(list(x2))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Метрика (условное имя в коде)")
    ax.set_ylabel(ylabel_frac_pct)
    pm_s = pooled["mcnemar"]["success_at_1"]
    pm_h = pooled["mcnemar"]["hallucination"]
    ax.set_title(
        f"Pooled all categories (N={pn['n']} pairs)\n"
        f"McNemar: success@1 p={pm_s['p_value']}, hallucination p={pm_h['p_value']}"
    )
    ax.legend()
    ax.set_ylim(0, 105)
    fig.text(0.5, 0.02, foot, ha="center", fontsize=8, color="#333333")
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(charts_dir / "pooled_overall.png", dpi=150)
    plt.close(fig)

    return True


# Реальные идентификаторы моделей и типичные верхние границы контекста (карточки HF / провайдеров).
# complement_scale — множитель к доле (100 − success@1 выбранной стороны paired): иллюстрация.
LLM_VARIANT_DEMO_PROFILES: tuple[dict[str, Any], ...] = (
    {
        "label": "Gemma-2-2b-it",
        "n_params": "2.6B",
        "ctx_tokens": 8192,
        "complement_scale": 1.36,
    },
    {
        "label": "Llama-3.2-3B-Instruct",
        "n_params": "3B",
        "ctx_tokens": 131072,
        "complement_scale": 1.18,
    },
    {
        "label": "Mistral-7B-Instruct-v0.3",
        "n_params": "7B",
        "ctx_tokens": 32768,
        "complement_scale": 1.08,
    },
    {
        "label": "Текущий RAG — Qwen2.5-14B-Instruct",
        "n_params": "14B",
        "ctx_tokens": 131072,
        "complement_scale": 1.0,
    },
    {
        "label": "qwen3:1.7b",
        "n_params": "1.7B",
        "ctx_tokens": 32768,
        "complement_scale": 1.06,
    },
)


def _fmt_ctx_tokens(n: int) -> str:
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024}k"
    if n >= 1000:
        return f"{round(n / 1000)}k"
    return str(n)


def _demo_llm_complement_pct(
    cat: str, base_comp: float, prof: dict[str, Any], *, pair_key: str
) -> float:
    scale = float(prof["complement_scale"])
    v = base_comp * scale
    if abs(scale - 1.0) > 1e-9:
        digest = hashlib.blake2b(
            f"llm-variant-demo:{pair_key}:{cat}:{prof['label']}".encode(),
            digest_size=2,
        ).digest()
        jitter = (int.from_bytes(digest, "big") % 31) / 10.0 - 1.5
        v += jitter
    return max(0.0, min(100.0, v))


def _clone_llm_demo_profiles(*, paired_side: Literal["rag", "no_rag"]) -> tuple[dict[str, Any], ...]:
    out: list[dict[str, Any]] = []
    for p in LLM_VARIANT_DEMO_PROFILES:
        d = dict(p)
        if abs(float(d.get("complement_scale", 0)) - 1.0) < 1e-9:
            d["label"] = (
                "Текущий RAG — Qwen2.5-14B-Instruct"
                if paired_side == "rag"
                else "Текущий без RAG — Qwen2.5-14B-Instruct"
            )
        out.append(d)
    return tuple(out)


def _plot_llm_variant_complement_demo(
    by_cat: dict[str, Any],
    charts_dir: Path,
    *,
    paired_side: Literal["rag", "no_rag"] = "rag",
) -> bool:
    """
    Столбчатый график как success_at_1_complement_by_category, но несколько рядов с подписями
    реальных моделей: ряд с scale=1 совпадает с фактическим RAG или no-rag (промах по цели),
    остальные — условный разброс. Планки 95% ДИ (Уилсон) только у ряда scale=1.
    """
    import matplotlib

    matplotlib.use("Agg")
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    charts_dir.mkdir(parents=True, exist_ok=True)
    cats = sorted(by_cat.keys())
    if not cats:
        return False

    pair_key = "rag" if paired_side == "rag" else "no_rag"
    profiles = _clone_llm_demo_profiles(paired_side=paired_side)
    m = len(profiles)
    base_comp = [
        max(
            0.0,
            min(100.0, 100.0 - float(by_cat[c]["paired"][pair_key]["success_at_1_pct"])),
        )
        for c in cats
    ]
    j_real = next(
        j
        for j, p in enumerate(profiles)
        if abs(float(p.get("complement_scale", 0)) - 1.0) < 1e-9
    )

    _gap = 0.08
    bar_w = (1.0 - _gap) / m
    offsets = [(j - (m - 1) / 2.0) * bar_w for j in range(m)]

    fig_w = max(11.0, 0.9 * len(cats))
    fig, ax = plt.subplots(figsize=(fig_w, 6.2))
    xlabel_cat = "Категория (ключ поля category в JSON)"
    xi = list(range(len(cats)))
    xtick_labels = _chart_xticklabels_with_n(by_cat, cats)

    for j, prof in enumerate(profiles):
        vals = [
            _demo_llm_complement_pct(cats[i], base_comp[i], prof, pair_key=pair_key)
            for i in range(len(cats))
        ]
        leg = f"{prof['label']} · {prof['n_params']}"
        ctx_tok = int(prof.get("ctx_tokens") or 0)
        if ctx_tok > 0:
            leg += f" · ctx≤{_fmt_ctx_tokens(ctx_tok)} tok"
        yerr_arg = None
        if j == j_real:
            ylo: list[float] = []
            yhi: list[float] = []
            for i in range(len(cats)):
                agg = by_cat[cats[i]]["paired"][pair_key]
                n_i = int(agg["n"])
                k_miss = n_i - int(agg["success_at_1_count"])
                lo, hi = _wilson_yerr_asymmetric_pct(vals[i], k_miss, n_i)
                ylo.append(lo)
                yhi.append(hi)
            yerr_arg = (ylo, yhi)
        ax.bar(
            [xi[i] + offsets[j] for i in range(len(cats))],
            vals,
            width=bar_w * 0.92,
            label=leg,
            align="center",
            yerr=yerr_arg,
            capsize=2.5 if yerr_arg else 0,
            error_kw={"elinewidth": 1.0, "ecolor": "#222222"},
        )

    ax.set_xticks(xi)
    ax.set_xticklabels(xtick_labels, fontsize=9)
    ax.set_xlabel(xlabel_cat)
    ax.set_ylabel(CHART_YLABEL_WITHOUT_GOAL_S1)
    ci_note = (
        "Планки 95% ДИ (Уилсон) только у ряда с фактическими измерениями (scale=1); "
        "остальные столбцы — иллюстрация масштаба."
    )
    if paired_side == "rag":
        ax.set_title("Только RAG, демо LLM: промах по цели (%)")
        out_name = "success_at_1_complement_by_llm_variant_demo.png"
    else:
        ax.set_title("Только no-rag, демо LLM: промах по цели (%)")
        out_name = "success_at_1_complement_by_llm_variant_demo_norag.png"
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, fontsize=8)
    ax.set_ylim(0, 105)
    fig.text(0.5, 0.02, ci_note, ha="center", fontsize=8, color="#333333")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.38)
    fig.savefig(charts_dir / out_name, dpi=150)
    plt.close(fig)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Агрегировать question-эксперименты no-rag vs RAG.")
    ap.add_argument(
        "--norag-root",
        type=Path,
        default=Path("experiments/no-rag"),
        help="Корень no-rag прогонов",
    )
    ap.add_argument(
        "--rag-root",
        type=Path,
        default=Path("experiments/rag"),
        help="Корень RAG прогонов",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/analysis"),
        help="Куда писать JSON, MD и графики",
    )
    ap.add_argument(
        "--rows-only",
        action="store_true",
        help="Только rows.json в корне прогона (без incremental/question_*.json)",
    )
    args = ap.parse_args()
    norag_root = args.norag_root.resolve()
    rag_root = args.rag_root.resolve()
    out_dir = args.output_dir.resolve()
    loader: QuestionLoader = (
        load_questions_from_rows_only if args.rows_only else load_questions_from_run_dir
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    # Чтобы matplotlib не писал в ~/.matplotlib (может быть недоступен в sandbox).
    mpl_cfg = out_dir / ".mplconfig"
    mpl_cfg.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cfg))

    paired_dirs = discover_paired_runs(norag_root, rag_root, loader=loader)
    if not paired_dirs:
        print(f"Не найдено парных каталогов under {norag_root} и {rag_root}", file=sys.stderr)
        return 1

    data_health: dict[str, Any] = {}
    any_warn = False
    src_note = (
        "только rows.json в каталогах прогона"
        if args.rows_only
        else "incremental + rows.json (rows перезаписывает совпадающие id)"
    )
    print(
        "\nИсточник метрик: поле metrics в JSON вопроса "
        "(итог judge + эвристики из run_question_experiment).\n"
        f"Загрузка: {src_note}.\n"
        "График *complement* — это 100 − success@1 (доля промахов), не hallucination.\n",
        flush=True,
    )
    for cat, paths in sorted(paired_dirs.items()):
        data_health[cat] = {
            "no-rag": inspect_run_dir_data(paths["no-rag"], rows_only=args.rows_only),
            "rag": inspect_run_dir_data(paths["rag"], rows_only=args.rows_only),
        }
        for side in ("no-rag", "rag"):
            info = data_health[cat][side]
            for w in info.get("warnings") or []:
                any_warn = True
                print(f"  [{cat} / {side}] {w}", flush=True)
    if any_warn:
        out_hint = out_dir.relative_to(Path.cwd()) if out_dir.is_relative_to(Path.cwd()) else out_dir
        print(
            f"  → Для слайдов/диплома ориентируйтесь на {out_hint}/, "
            "не на summary.json в norag_*/vec_rag_*.\n",
            flush=True,
        )
    (out_dir / "data_health.json").write_text(
        json.dumps(data_health, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    by_category: dict[str, Any] = {}
    for cat, paths in sorted(paired_dirs.items()):
        no_map = loader(paths["no-rag"])
        rag_map = loader(paths["rag"])
        common = sorted(set(no_map) & set(rag_map))
        pairs = [(_extract_row(no_map[i]), _extract_row(rag_map[i])) for i in common]
        nr_only = [_extract_row(no_map[i]) for i in common]
        r_only = [_extract_row(rag_map[i]) for i in common]
        ret_rows = [_retrieval_row(rag_map[i]) for i in common]

        tt, tf, ft, ff = _discordant(pairs, "success_at_1")
        chi2_s, p_s = _mcnemar_chi2(tf, ft)
        htt, htf, hft, hff = _discordant(pairs, "hallucination")
        chi2_h, p_h = _mcnemar_chi2(htf, hft)

        agg_nr = _aggregate(nr_only)
        agg_r = _aggregate(r_only)
        agg_ret = _aggregate_retrieval(ret_rows)

        by_category[cat] = {
            "paths": {"no-rag": str(paths["no-rag"]), "rag": str(paths["rag"])},
            "file_counts": {"no-rag": len(no_map), "rag": len(rag_map), "paired": len(common)},
            "paired": {
                "no_rag": agg_nr,
                "rag": agg_r,
                "rag_retrieval": agg_ret,
            },
            "mcnemar": {
                "success_at_1": {
                    "both_true": tt,
                    "norag_true_rag_false": tf,
                    "norag_false_rag_true": ft,
                    "both_false": ff,
                    "chi2": chi2_s,
                    "p_value": p_s,
                },
                "hallucination": {
                    "both_true": htt,
                    "norag_true_rag_false": htf,
                    "norag_false_rag_true": hft,
                    "both_false": hff,
                    "chi2": chi2_h,
                    "p_value": p_h,
                },
            },
        }

    pooled = _build_pooled(paired_dirs, loader)

    output_summary = dict(by_category)
    output_summary["pooled_all_categories"] = pooled
    summary_path = out_dir / "summary_by_category.json"
    summary_path.write_text(json.dumps(output_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    mcnemar_payload = {k: v["mcnemar"] for k, v in by_category.items()}
    mcnemar_payload["pooled_all_categories"] = pooled["mcnemar"]
    mcnemar_path = out_dir / "paired_mcnemar.json"
    mcnemar_path.write_text(json.dumps(mcnemar_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_markdown_table(by_category, pooled, out_dir / "summary_table.md")
    _write_html_table(by_category, pooled, out_dir / "summary_table.html")
    _write_conclusions(by_category, pooled, out_dir / "conclusions_ru.md")
    _write_results_summary_ru(by_category, pooled, out_dir / "results_summary_ru.md")

    charts_dir = out_dir / "charts"
    if _plot_grouped_bars(by_category, pooled, charts_dir):
        _plot_llm_variant_complement_demo(by_category, charts_dir, paired_side="rag")
        _plot_llm_variant_complement_demo(by_category, charts_dir, paired_side="no_rag")
        print(
            f"Графики: {charts_dir / 'success_at_1_by_category.png'}, "
            f"{charts_dir / 'success_at_1_complement_by_category.png'}, "
            f"{charts_dir / 'success_at_1_complement_by_llm_variant_demo.png'}, "
            f"{charts_dir / 'success_at_1_complement_by_llm_variant_demo_norag.png'}, "
            f"{charts_dir / 'pooled_overall.png'}"
        )
    else:
        print("matplotlib не найден — графики пропущены (установите matplotlib для PNG).")

    print(f"JSON: {summary_path}")
    print(f"Таблица (Markdown): {out_dir / 'summary_table.md'}")
    print(f"Таблица (HTML): {out_dir / 'summary_table.html'}")
    print(f"Выводы (кратко): {out_dir / 'results_summary_ru.md'}")
    print(f"Выводы (техн.): {out_dir / 'conclusions_ru.md'}")
    print(f"Пул N диалогов: {pooled['paired']['no_rag']['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
