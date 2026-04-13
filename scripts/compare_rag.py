"""
Сравнение RAG vs non-RAG: анализ ошибок и качества диалогов.

Запуск (из корня репозитория):
  python scripts/compare_rag.py --rag_dir dialogs_rag --norag_dir dialogs_norag
  python scripts/compare_rag.py --rag_dir dialogs_rag  # только одна группа

Результат: таблица в консоль + report.json + (опционально) rag_comparison.png.
"""

from __future__ import annotations

import argparse
import json
from math import sqrt
import sys
from pathlib import Path
from typing import Any

import _bootstrap

_bootstrap.ensure_src()


# ── Загрузка JSON-сводки ─────────────────────────────────────────────

def load_summary(directory: str | Path) -> list[dict[str, Any]]:
    """Загружает summary.json из каталога."""
    path = Path(directory) / "summary.json"
    if not path.exists():
        print(f"  [!] Файл не найден: {path}", file=sys.stderr)
        print(f"      Запустите: python scripts/generate_dataset.py --output_dir {directory}",
              file=sys.stderr)
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Статистика ────────────────────────────────────────────────────────

def compute_stats(rows: list[dict[str, Any]], label: str = "") -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"label": label, "total": 0}

    allergen_v = sum(1 for r in rows if r.get("allergen_violation"))
    calorie_w = sum(1 for r in rows if r.get("calorie_warning"))
    empty_o = sum(1 for r in rows if r.get("empty_order"))
    halluc = sum(1 for r in rows if r.get("hallucination"))
    incomplete = sum(1 for r in rows if r.get("incomplete_order"))
    turns = [r.get("turns", 0) for r in rows]
    items = [r.get("total_items", 0) for r in rows]

    return {
        "label": label,
        "total": n,
        "allergen_violation_%": round(allergen_v / n * 100, 1),
        "calorie_warning_%": round(calorie_w / n * 100, 1),
        "empty_order_%": round(empty_o / n * 100, 1),
        "hallucination_%": round(halluc / n * 100, 1),
        "incomplete_order_%": round(incomplete / n * 100, 1),
        "avg_turns": round(sum(turns) / n, 1),
        "avg_items": round(sum(items) / n, 2),
    }


def compute_group_stats(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Группировка ошибок по характеристикам клиентов."""
    groups: dict[str, list[dict[str, Any]]] = {
        "noMilk=True": [],
        "noMilk=False": [],
        "with_children": [],
        "no_children": [],
        "group_size>1": [],
        "solo": [],
    }
    psycho_groups: dict[str, list[dict[str, Any]]] = {}

    for r in rows:
        if r.get("noMilk"):
            groups["noMilk=True"].append(r)
        else:
            groups["noMilk=False"].append(r)

        cq = r.get("childQuant", 0)
        if cq > 0:
            groups["with_children"].append(r)
        else:
            groups["no_children"].append(r)

        gs = r.get("group_size", 1)
        if gs > 1:
            groups["group_size>1"].append(r)
        else:
            groups["solo"].append(r)

        ps = r.get("psycho", "unknown")
        psycho_groups.setdefault(ps, []).append(r)

    result: dict[str, dict[str, Any]] = {}
    for key, subset in groups.items():
        result[key] = compute_stats(subset, label=key)
    for key, subset in sorted(psycho_groups.items()):
        result[f"psycho={key}"] = compute_stats(subset, label=f"psycho={key}")
    return result


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson confidence interval for proportion, in percent."""
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom
    lo = max(0.0, (center - margin) * 100)
    hi = min(100.0, (center + margin) * 100)
    return round(lo, 1), round(hi, 1)


def chi_square_2x2(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """Chi-square test for a 2x2 contingency table with Yates correction."""
    n = a + b + c + d
    if n <= 0:
        return 0.0, 1.0
    row1 = a + b
    row2 = c + d
    col1 = a + c
    col2 = b + d
    denom = row1 * row2 * col1 * col2
    if denom == 0:
        return 0.0, 1.0
    yates_adj = n / 2
    corrected = max(0.0, abs(a * d - b * c) - yates_adj)
    chi2 = (n * corrected**2) / denom
    chi2 = max(0.0, chi2)

    try:
        from scipy.stats import chi2 as chi2_dist  # type: ignore[import-not-found]

        p = 1 - chi2_dist.cdf(chi2, df=1)
    except Exception:
        if chi2 > 10.83:
            p = 0.001
        elif chi2 > 6.63:
            p = 0.01
        elif chi2 > 3.84:
            p = 0.05
        else:
            p = 1.0
    return round(chi2, 3), round(float(p), 4)


def compare_with_stats(rag: dict[str, Any], norag: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Statistical comparison of key outcomes between RAG and non-RAG.
    Returns rows with confidence intervals and p-values.
    """
    n_rag = int(rag.get("total", 0) or 0)
    n_norag = int(norag.get("total", 0) or 0)
    metrics = [
        ("allergen_violation_%", "Allergen violation"),
        ("calorie_warning_%", "Calorie warning"),
        ("empty_order_%", "Empty order"),
        ("hallucination_%", "Hallucination"),
        ("incomplete_order_%", "Incomplete order"),
    ]
    rows: list[dict[str, Any]] = []
    for key, label in metrics:
        pct_rag = float(rag.get(key, 0.0) or 0.0)
        pct_norag = float(norag.get(key, 0.0) or 0.0)
        k_rag = round(pct_rag / 100 * n_rag)
        k_norag = round(pct_norag / 100 * n_norag)
        ci_rag = wilson_ci(k_rag, n_rag)
        ci_norag = wilson_ci(k_norag, n_norag)
        chi2, p_value = chi_square_2x2(
            k_rag,
            max(0, n_rag - k_rag),
            k_norag,
            max(0, n_norag - k_norag),
        )
        rows.append(
            {
                "metric": label,
                "rag_%": round(pct_rag, 1),
                "rag_ci": f"[{ci_rag[0]}–{ci_rag[1]}]",
                "norag_%": round(pct_norag, 1),
                "norag_ci": f"[{ci_norag[0]}–{ci_norag[1]}]",
                "delta_pp": round(pct_norag - pct_rag, 1),
                "chi2": chi2,
                "p_value": p_value,
                "significant": p_value < 0.05,
            }
        )
    return rows


# ── Форматирование ────────────────────────────────────────────────────

_STAT_KEYS = [
    ("total", "Диалогов"),
    ("allergen_violation_%", "Allergen violation %"),
    ("calorie_warning_%", "Calorie warning %"),
    ("empty_order_%", "Empty order %"),
    ("hallucination_%", "Hallucination %"),
    ("incomplete_order_%", "Incomplete order %"),
    ("avg_turns", "Avg turns"),
    ("avg_items", "Avg items"),
]


def format_comparison_table(stats_list: list[dict[str, Any]]) -> str:
    col_w = 25
    label_w = 24
    lines: list[str] = []

    header = f"{'Метрика':<{label_w}}"
    for s in stats_list:
        header += f"  {s.get('label', '?'):>{col_w}}"
    lines.append(header)
    lines.append("-" * len(header))

    for key, display_name in _STAT_KEYS:
        row = f"{display_name:<{label_w}}"
        for s in stats_list:
            val = s.get(key, "—")
            row += f"  {str(val):>{col_w}}"
        lines.append(row)
    return "\n".join(lines)


def format_group_table(group_stats: dict[str, dict[str, Any]]) -> str:
    lines: list[str] = []
    hdr = (
        f"{'Группа':<24}  {'N':>5}  {'AllerViol%':>10}  {'CalWarn%':>8}  "
        f"{'Empty%':>7}  {'Halluc%':>8}  {'Incompl%':>8}  {'AvgTurns':>8}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for key, st in group_stats.items():
        n = st.get("total", 0)
        if n == 0:
            continue
        lines.append(
            f"{key:<24}  {n:>5}  "
            f"{st.get('allergen_violation_%', 0):>10.1f}  "
            f"{st.get('calorie_warning_%', 0):>8.1f}  "
            f"{st.get('empty_order_%', 0):>7.1f}  "
            f"{st.get('hallucination_%', 0):>8.1f}  "
            f"{st.get('incomplete_order_%', 0):>8.1f}  "
            f"{st.get('avg_turns', 0):>8.1f}"
        )
    return "\n".join(lines)


def format_significance_table(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    hdr = (
        f"{'Metric':<22}  {'RAG % [95% CI]':>21}  {'non-RAG % [95% CI]':>24}  "
        f"{'Delta p.p.':>10}  {'chi2':>7}  {'p-value':>8}  {'Sig<0.05':>9}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in rows:
        rag_cell = f"{r['rag_%']:.1f} {r['rag_ci']}"
        norag_cell = f"{r['norag_%']:.1f} {r['norag_ci']}"
        lines.append(
            f"{r['metric']:<22}  {rag_cell:>21}  {norag_cell:>24}  "
            f"{r['delta_pp']:>10.1f}  {r['chi2']:>7.3f}  {r['p_value']:>8.4f}  "
            f"{str(r['significant']):>9}"
        )
    return "\n".join(lines)


# ── Диаграмма (matplotlib, опционально) ───────────────────────────────

def try_plot(
    stats_list: list[dict[str, Any]],
    output_path: str | Path = "rag_comparison.png",
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    metrics = [
        "allergen_violation_%",
        "calorie_warning_%",
        "empty_order_%",
        "hallucination_%",
        "incomplete_order_%",
    ]
    labels = [s.get("label", "?") for s in stats_list]
    x_labels = [
        "Allergen viol.",
        "Calorie warn.",
        "Empty order",
        "Hallucination",
        "Incomplete",
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_w = 0.35
    positions = list(range(len(metrics)))

    for i, s in enumerate(stats_list):
        vals = [s.get(m, 0) for m in metrics]
        offsets = [p + i * bar_w for p in positions]
        ax.bar(offsets, vals, bar_w, label=labels[i])

    ax.set_ylabel("% диалогов")
    ax.set_title("RAG vs non-RAG: доля ошибок")
    ax.set_xticks([p + bar_w / 2 for p in positions])
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    return True


# ── main ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Анализ ошибок и сравнение RAG vs non-RAG (диплом).",
    )
    parser.add_argument("--rag_dir", default="dialogs_rag", help="Каталог RAG-диалогов.")
    parser.add_argument("--norag_dir", default="dialogs_norag", help="Каталог non-RAG-диалогов.")
    parser.add_argument("--output", default="rag_comparison.json", help="JSON-отчёт.")
    parser.add_argument("--plot", default="rag_comparison.png", help="Путь к PNG-диаграмме.")
    args = parser.parse_args()

    rag_rows = load_summary(args.rag_dir)
    norag_rows = load_summary(args.norag_dir)

    stats_list: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []

    if rag_rows:
        st = compute_stats(rag_rows, label="RAG")
        stats_list.append(st)
        all_rows.extend(rag_rows)
    if norag_rows:
        st = compute_stats(norag_rows, label="non-RAG")
        stats_list.append(st)
        all_rows.extend(norag_rows)

    if not stats_list:
        print("Нет данных. Сначала запустите generate_dataset.py.", file=sys.stderr)
        raise SystemExit(1)

    table = format_comparison_table(stats_list)
    print("\n=== Сравнение RAG vs non-RAG ===\n")
    print(table)

    group_stats = compute_group_stats(all_rows)
    group_table = format_group_table(group_stats)
    print("\n=== Ошибки по группам клиентов ===\n")
    print(group_table)

    interpretation = ""
    significance_rows: list[dict[str, Any]] = []
    if len(stats_list) == 2:
        interpretation = _interpret(stats_list[0], stats_list[1])
        significance_rows = compare_with_stats(stats_list[0], stats_list[1])
        print(f"\n=== Интерпретация ===\n{interpretation}")
        print("\n=== Статистическая значимость ===\n")
        print(format_significance_table(significance_rows))

    # JSON-отчёт
    report: dict[str, Any] = {
        "comparison": stats_list,
        "groups": {k: v for k, v in group_stats.items() if v.get("total", 0) > 0},
    }
    if interpretation:
        report["interpretation"] = interpretation
    if significance_rows:
        report["significance"] = significance_rows
    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"\nОтчёт сохранён: {args.output}")

    if len(stats_list) >= 2:
        if try_plot(stats_list, args.plot):
            print(f"Диаграмма: {args.plot}")
        else:
            print("matplotlib не установлен — диаграмма пропущена.")


def _interpret(rag: dict[str, Any], norag: dict[str, Any]) -> str:
    lines: list[str] = []

    av_rag = rag.get("allergen_violation_%", 0)
    av_norag = norag.get("allergen_violation_%", 0)
    if av_norag > av_rag:
        lines.append(
            f"RAG снижает долю аллергенных нарушений с {av_norag}% до {av_rag}% "
            f"(разница {av_norag - av_rag:.1f} п.п.)."
        )
    elif av_rag > av_norag:
        lines.append(
            f"Неожиданно: RAG-версия показывает больше аллергенных нарушений "
            f"({av_rag}% vs {av_norag}%). Возможно, RAG-контекст провоцирует "
            f"кассира упоминать аллергенные блюда."
        )
    else:
        lines.append(f"Доля аллергенных нарушений одинакова ({av_rag}%).")

    eo_rag = rag.get("empty_order_%", 0)
    eo_norag = norag.get("empty_order_%", 0)
    if eo_norag > eo_rag:
        lines.append(
            f"Доля пустых заказов (вероятные галлюцинации) снижается с "
            f"{eo_norag}% (non-RAG) до {eo_rag}% (RAG)."
        )
    elif eo_rag > eo_norag:
        lines.append(
            f"RAG-версия чаще даёт пустые заказы ({eo_rag}% vs {eo_norag}%)."
        )

    hl_rag = rag.get("hallucination_%", 0)
    hl_norag = norag.get("hallucination_%", 0)
    if hl_norag > hl_rag:
        lines.append(
            f"RAG снижает галлюцинации позиций с {hl_norag}% до {hl_rag}%."
        )
    elif hl_rag > hl_norag:
        lines.append(
            f"RAG-версия чаще содержит галлюцинации ({hl_rag}% vs {hl_norag}%)."
        )

    io_rag = rag.get("incomplete_order_%", 0)
    io_norag = norag.get("incomplete_order_%", 0)
    if io_norag > io_rag:
        lines.append(
            f"RAG снижает долю неполных заказов для групп ({io_norag}% → {io_rag}%)."
        )
    elif io_rag > io_norag:
        lines.append(
            f"RAG повышает долю неполных заказов ({io_rag}% vs {io_norag}%)."
        )

    cw_rag = rag.get("calorie_warning_%", 0)
    cw_norag = norag.get("calorie_warning_%", 0)
    diff = abs(cw_rag - cw_norag)
    if diff < 2:
        lines.append(f"Разница по calorie_warning минимальна ({cw_rag}% vs {cw_norag}%).")
    elif cw_norag > cw_rag:
        lines.append(f"RAG снижает calorie_warning с {cw_norag}% до {cw_rag}%.")
    else:
        lines.append(f"RAG повышает calorie_warning с {cw_norag}% до {cw_rag}%.")

    at_rag = rag.get("avg_turns", 0)
    at_norag = norag.get("avg_turns", 0)
    lines.append(
        f"Средняя длина диалога: RAG={at_rag} ходов, non-RAG={at_norag} ходов."
    )

    return "\n".join(lines)


if __name__ == "__main__":
    main()
