"""
Один запуск отчётов по уже собранным данным (без LLM, без generate_dataset).

Сейчас делает:
  1) Агрегация question-экспериментов (experiments/no-rag/norag_*, experiments/rag/vec_rag_*) → experiments/analysis/
  2) compare_rag.py — только если в каталогах есть summary.json (полные диалоги)
  3) pytest tests/test_question_experiment.py — быстрая проверка, что анализ не сломан

Пишет experiments/analysis/REPORTS_INDEX.md — список артефактов.

Режим --rows-only: только rows.json в прогонах → experiments/analysis_rows/ (make reports-rows).

Запуск из корня репозитория:
  python scripts/run_all_reports.py
  python scripts/run_all_reports.py --rows-only
  make reports
  make reports-rows
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], *, check: bool = True) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, check=check)


def main() -> int:
    ap = argparse.ArgumentParser(description="Все отчёты по уже собранным данным (без LLM).")
    ap.add_argument(
        "--rows-only",
        action="store_true",
        help="Только rows.json из experiments/no-rag и experiments/rag",
    )
    args = ap.parse_args()

    out = REPO / ("experiments/analysis_rows" if args.rows_only else "experiments/analysis")
    charts = out / "charts"
    out.mkdir(parents=True, exist_ok=True)
    charts.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".mplconfig"))
    (out / ".mplconfig").mkdir(parents=True, exist_ok=True)

    step1 = "rows.json" if args.rows_only else "question_*.json / incremental"
    print(f"\n=== 1) Анализ ({step1}, no-rag vs RAG) → {out} ===\n", flush=True)
    analyze_cmd = [
        sys.executable,
        str(REPO / "scripts" / "analyze_question_experiments.py"),
        "--output-dir",
        str(out),
    ]
    if args.rows_only:
        analyze_cmd.append("--rows-only")
    _run(analyze_cmd)

    rag_summary = REPO / "dialogs_rag" / "summary.json"
    norag_summary = REPO / "dialogs_norag" / "summary.json"
    print("\n=== 2) Сравнение полных диалогов (compare_rag) ===\n", flush=True)
    if rag_summary.is_file() and norag_summary.is_file():
        _run(
            [
                sys.executable,
                str(REPO / "scripts" / "compare_rag.py"),
                "--rag_dir",
                str(REPO / "dialogs_rag"),
                "--norag_dir",
                str(REPO / "dialogs_norag"),
                "--output",
                str(out / "rag_dataset_comparison.json"),
                "--plot",
                str(charts / "rag_dataset_comparison.png"),
            ]
        )
    else:
        print(
            "Пропуск: нет обоих файлов summary.json\n"
            f"  ожидалось: {rag_summary}\n"
            f"  ожидалось: {norag_summary}\n"
            "  (сгенерируйте диалоги через generate_dataset.py в dialogs_rag / dialogs_norag.)",
            flush=True,
        )

    print("\n=== 3) Тесты агрегатора вопросов ===\n", flush=True)
    rc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(REPO / "tests" / "test_question_experiment.py"),
            "-q",
            "--tb=line",
        ],
        cwd=REPO,
    ).returncode
    if rc != 0:
        print(f"pytest завершился с кодом {rc} (см. выше).", flush=True)

    make_cmd = "make reports-rows" if args.rows_only else "make reports"
    try:
        out_rel = out.relative_to(REPO)
    except ValueError:
        out_rel = out
    lines = [
        f"# Индекс отчётов (`{make_cmd}` / `run_all_reports.py`)",
        "",
        f"Источник: {'только `rows.json` в каталогах прогона' if args.rows_only else 'incremental + rows.json'}.",
        "",
        f"## Question-эксперименты (`{out_rel}/`)",
        "",
        "| Файл | Назначение |",
        "|------|------------|",
        "| `summary_by_category.json` | Метрики по категориям + `pooled_all_categories` |",
        "| `paired_mcnemar.json` | McNemar по категориям и пулу |",
        "| `summary_table.md` | Таблица для презентации / диплома (Markdown) |",
        "| `summary_table.html` | Та же сводка — **кратко для слайдов** (понятные подписи; техника в «Как считалось») |",
        "| `conclusions_ru.md` | Авто-выводы по категориям (с именами метрик в коде) |",
        "| `results_summary_ru.md` | **Итоги и выводы одним текстом** — для слайда / заключения |",
        "| `charts/success_at_1_by_category.png` | Столбцы success@1 |",
        "| `charts/success_at_1_complement_by_category.png` | 100 − success@1 (доля без цели @1) |",
        "| `charts/hallucination_by_category.png` | Столбцы hallucination |",
        "| `charts/rag_retrieval_hit_by_category.png` | RAG retrieval @k |",
        "| `charts/pooled_overall.png` | Пул всех категорий |",
        "",
    ]
    if (out / "rag_dataset_comparison.json").is_file():
        lines += [
            "## Полные диалоги (если был compare_rag)",
            "",
            "| Файл | Назначение |",
            "|------|------------|",
            "| `rag_dataset_comparison.json` | Сводка RAG vs non-RAG по summary.json |",
            "| `charts/rag_dataset_comparison.png` | Диаграмма ошибок |",
            "",
        ]
    lines.append(f"Обновление: перезапустите `{make_cmd}`.")
    (out / "REPORTS_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nИндекс: {out / 'REPORTS_INDEX.md'}", flush=True)
    return 0 if rc == 0 else rc


if __name__ == "__main__":
    raise SystemExit(main())
