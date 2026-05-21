#!/usr/bin/env python3
"""
RAGAS: оценка faithfulness и answer_relevancy по уже сохранённым question_*.json.

Ожидается английский текст вопроса/ответа (как в банке вопросов). Судья RAGAS — отдельные
вызовы LLM (см. LLM_BASE_URL / API_MODEL в .env, как для основного пайплайна).

Установка (опциональная группа зависимостей):
  pip install -e ".[ragas]"

Запуск (пример):
  python scripts/run_ragas_question_metrics.py --run-root experiments/rag \\
      --output experiments/analysis/ragas_scores_rag.json --max-rows 50

Makefile:
  make ragas-question-metrics
  make ragas-question-metrics-sum   # то же → experiments/analysis_sum/ragas_scores_rag.json
  make ragas-reports-sum             # только отчёты из уже посчитанного JSON в analysis_sum

После расчёта (или при --reports-only) пишет в каталог **родителя** `--output` (рядом с JSON):
  ragas_summary.md, ragas_summary.html
  charts/ragas_faithfulness_by_category.png, charts/ragas_answer_relevancy_by_category.png
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import _bootstrap
from experiment_run_layout import iter_question_json_paths

_bootstrap.ensure_src()

# Подписи категорий (как в analyze_question_experiments) — для таблицы/графиков.
QUESTION_CATEGORY_LABEL_RU: dict[str, str] = {
    "simple": "Простой заказ (название блюда)",
    "lexical": "Разговорная речь: исключения и уточнения",
    "diet": "Диетические ограничения",
    "allergy": "Аллергены / исключения",
    "group": "Заказ на несколько человек",
    "mixed": "Смешанный сценарий",
}


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


def _truncate(s: str, max_len: int) -> str:
    s = s.strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _contexts_from_question_json(data: dict[str, Any], *, max_chunks: int, max_chars: int) -> list[str]:
    """Список строк контекста для RAGAS: top-k чанков из Chroma или grounding preview без RAG."""
    rag_trace = data.get("rag_trace") or []
    for ev in rag_trace:
        if ev.get("event") == "chroma_response" and isinstance(ev.get("documents"), list):
            docs = ev["documents"]
            out: list[str] = []
            for d in docs[:max_chunks]:
                if isinstance(d, str) and d.strip():
                    out.append(_truncate(d, max_chars))
            return out if out else ["(empty retrieval documents)"]
    for ev in reversed(rag_trace):
        if ev.get("event") == "extra_grounding_context":
            prev = ev.get("grounding_preview") or ev.get("context_preview") or ""
            if isinstance(prev, str) and prev.strip():
                return [_truncate(prev, max_chars * max_chunks)]
    prev = data.get("context_preview")
    if isinstance(prev, str) and prev.strip():
        return [_truncate(prev, max_chars * max_chunks)]
    return ["(no retrieval context in trace)"]


def _iter_question_files(run_root: Path) -> list[Path]:
    return iter_question_json_paths(run_root)


def _json_safe(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    try:
        import numpy as np  # type: ignore[import-not-found]

        if isinstance(x, (np.floating, np.integer)):
            return float(x) if isinstance(x, np.floating) else int(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
    except Exception:
        pass
    return x


def _mean_std(vals: list[float]) -> tuple[float, float | None]:
    if not vals:
        return 0.0, None
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, None
    var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return m, math.sqrt(var) if var > 0 else 0.0


def _fmt_mean_std_cell(vals: list[float]) -> str:
    if not vals:
        return "—"
    mf, sf = _mean_std(vals)
    if sf is not None:
        return f"{mf:.3f} ± {sf:.3f}"
    return f"{mf:.3f}"


def _aggregate_scores_by_category(agg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """category -> {n_rows, faithfulness[], answer_relevancy[]}. n_rows — все строки категории."""
    n_by_cat: dict[str, int] = defaultdict(int)
    buckets: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"faithfulness": [], "answer_relevancy": []}
    )
    for row in agg.get("rows") or []:
        cat = str(row.get("category") or "unknown").strip() or "unknown"
        n_by_cat[cat] += 1
        sc = row.get("scores") or {}
        for key in ("faithfulness", "answer_relevancy"):
            v = sc.get(key)
            if v is None:
                continue
            try:
                x = float(v)
                if not math.isnan(x):
                    buckets[cat][key].append(x)
            except (TypeError, ValueError):
                continue
    all_cats = sorted(set(n_by_cat) | set(buckets))
    out: dict[str, dict[str, Any]] = {}
    for cat in all_cats:
        d = buckets[cat]
        out[cat] = {
            "n": n_by_cat.get(cat, 0),
            "faithfulness": d["faithfulness"],
            "answer_relevancy": d["answer_relevancy"],
        }
    return out


def _write_ragas_markdown(
    agg: dict[str, Any],
    by_cat: dict[str, dict[str, Any]],
    path: Path,
) -> None:
    lines = [
        "# RAGAS: faithfulness и answer relevancy",
        "",
        f"- **Корень прогона:** `{agg.get('run_root', '')}`",
        f"- **Строк оценено:** {agg.get('n_rows', 0)}",
        f"- **LLM для RAGAS:** `{agg.get('judge_model', '')}`",
        f"- **Эмбеддинги:** `{agg.get('embedding_model', '')}`",
        "",
        "## Среднее по всей выборке",
        "",
    ]
    for name, ru in (
        ("faithfulness", "Согласованность ответа с поданным контекстом (0–1)"),
        ("answer_relevancy", "Релевантность ответа вопросу (0–1)"),
    ):
        m = (agg.get("metrics") or {}).get(name) or {}
        if m:
            lines.append(f"- **{name}** ({ru}): среднее **{m.get('mean')}** (n={m.get('n_scored')})")
        else:
            lines.append(f"- **{name}**: нет данных")
    lines.extend(["", "## По категориям (среднее ± выборочное std)", "", "| Категория | N | faithfulness | answer_relevancy |", "|-----------|---:|---:|---:|"])
    for cat in sorted(by_cat.keys()):
        label = QUESTION_CATEGORY_LABEL_RU.get(cat, cat)
        d = by_cat[cat]
        n = int(d.get("n", 0) or 0)
        fs = _fmt_mean_std_cell(d.get("faithfulness") or [])
        ars = _fmt_mean_std_cell(d.get("answer_relevancy") or [])
        lines.append(f"| `{cat}` — {label} | {n} | {fs} | {ars} |")
    lines.extend(["", "*faithfulness — опора утверждений ответа на переданный в RAGAS контекст; answer_relevancy — близость ответа к формулировке вопроса (метрики RAGAS).*", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_ragas_html(
    agg: dict[str, Any],
    by_cat: dict[str, dict[str, Any]],
    path: Path,
    *,
    show_chart_hint: bool = True,
) -> None:
    rows_html = []
    for cat in sorted(by_cat.keys()):
        label = html_mod.escape(QUESTION_CATEGORY_LABEL_RU.get(cat, cat))
        d = by_cat[cat]
        n = int(d.get("n", 0) or 0)
        fs = html_mod.escape(_fmt_mean_std_cell(d.get("faithfulness") or []))
        ars = html_mod.escape(_fmt_mean_std_cell(d.get("answer_relevancy") or []))
        rows_html.append(
            f"<tr><td class='cat'><code>{html_mod.escape(cat)}</code><br/><span class='dim'>{label}</span></td>"
            f"<td class='num'>{n}</td><td class='num'>{fs}</td><td class='num'>{ars}</td></tr>"
        )
    pool_f = (agg.get("metrics") or {}).get("faithfulness") or {}
    pool_a = (agg.get("metrics") or {}).get("answer_relevancy") or {}
    body = f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RAGAS — сводка</title>
  <style>
    body {{ margin:0; padding:1.25rem; font-family: system-ui, sans-serif; background:#f0f2f7; color:#1c2333; }}
    .wrap {{ max-width: 920px; margin: 0 auto; }}
    h1 {{ font-size: 1.35rem; margin: 0 0 0.5rem; }}
    .lead {{ color:#5a6578; font-size:0.92rem; margin-bottom:1rem; }}
    .hero {{ display:flex; gap:0.75rem; flex-wrap:wrap; margin-bottom:1.25rem; }}
    .box {{ background:#fff; border:1px solid #d8dee9; border-radius:10px; padding:0.85rem 1rem; min-width:140px; }}
    .box .t {{ font-size:0.72rem; color:#5a6578; text-transform:uppercase; }}
    .box .n {{ font-size:1.4rem; font-weight:800; margin-top:0.2rem; }}
    table {{ width:100%; border-collapse:collapse; font-size:0.88rem; background:#fff; border:1px solid #d8dee9; border-radius:10px; overflow:hidden; }}
    th, td {{ padding:0.55rem 0.65rem; border-bottom:1px solid #e8ecf2; text-align:left; }}
    th {{ background:#e6ebf4; font-size:0.72rem; color:#5a6578; }}
    th.num, td.num {{ text-align:right; }}
    .cat {{ max-width:16rem; }}
    .dim {{ color:#5a6578; font-size:0.82rem; }}
    tr:last-child td {{ border-bottom:none; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>RAGAS: faithfulness и answer relevancy</h1>
    <p class="lead">Корень: <code>{html_mod.escape(str(agg.get("run_root", "")))}</code> · строк: {agg.get("n_rows", 0)} · LLM: <code>{html_mod.escape(str(agg.get("judge_model", "")))}</code></p>
    <div class="hero">
      <div class="box"><div class="t">faithfulness (среднее)</div><div class="n">{pool_f.get("mean", "—")}</div><div class="dim">n={pool_f.get("n_scored", "—")}</div></div>
      <div class="box"><div class="t">answer relevancy (среднее)</div><div class="n">{pool_a.get("mean", "—")}</div><div class="dim">n={pool_a.get("n_scored", "—")}</div></div>
    </div>
    <h2 style="font-size:1.05rem;">По категориям</h2>
    <table>
      <thead><tr><th>Категория</th><th class="num">N</th><th class="num">faithfulness</th><th class="num">answer relevancy</th></tr></thead>
      <tbody>{"".join(rows_html)}</tbody>
    </table>
    {"<p class='lead' style='margin-top:1rem;'>Графики: <code>charts/ragas_*_by_category.png</code></p>" if show_chart_hint else ""}
  </div>
</body>
</html>"""
    path.write_text(body, encoding="utf-8")


def _plot_ragas_charts(by_cat: dict[str, dict[str, Any]], charts_dir: Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    charts_dir.mkdir(parents=True, exist_ok=True)
    cats = sorted(by_cat.keys())
    if not cats:
        return []

    created: list[Path] = []

    def one_chart(metric: str, fname: str, title: str, ylabel: str) -> None:
        cats_m = [c for c in cats if by_cat[c].get(metric)]
        if not cats_m:
            return
        means = []
        errs = []
        labels_m = [
            QUESTION_CATEGORY_LABEL_RU.get(c, c)[:28]
            + ("…" if len(QUESTION_CATEGORY_LABEL_RU.get(c, c)) > 28 else "")
            for c in cats_m
        ]
        for c in cats_m:
            vals = [float(x) for x in (by_cat[c].get(metric) or [])]
            m, s = _mean_std(vals)
            means.append(m)
            errs.append(0.0 if s is None else s)
        x = range(len(cats_m))
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.bar(x, means, yerr=errs, capsize=3, color="#4472c4", ecolor="#333", error_kw={"elinewidth": 1})
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels_m, rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.tight_layout()
        out_p = charts_dir / fname
        fig.savefig(out_p, dpi=150)
        plt.close(fig)
        created.append(out_p)

    one_chart(
        "faithfulness",
        "ragas_faithfulness_by_category.png",
        "RAGAS faithfulness по категориям (среднее ± std)",
        "faithfulness, 0–1",
    )
    one_chart(
        "answer_relevancy",
        "ragas_answer_relevancy_by_category.png",
        "RAGAS answer relevancy по категориям (среднее ± std)",
        "answer relevancy, 0–1",
    )
    return created


def render_ragas_reports_from_agg(
    agg: dict[str, Any],
    analysis_dir: Path,
    *,
    include_charts: bool = True,
) -> list[str]:
    """Пишет MD, HTML и при include_charts — PNG в analysis_dir/charts. Возвращает созданные пути."""
    by_cat = _aggregate_scores_by_category(agg)
    written: list[str] = []
    md_path = analysis_dir / "ragas_summary.md"
    html_path = analysis_dir / "ragas_summary.html"
    _write_ragas_markdown(agg, by_cat, md_path)
    written.append(str(md_path))
    _write_ragas_html(agg, by_cat, html_path, show_chart_hint=include_charts)
    written.append(str(html_path))
    charts_dir = analysis_dir / "charts"
    if include_charts:
        for p in _plot_ragas_charts(by_cat, charts_dir):
            written.append(str(p))
    return written


def main() -> int:
    ap = argparse.ArgumentParser(
        description="RAGAS faithfulness + answer_relevancy по question_*.json (один корень прогона)."
    )
    ap.add_argument(
        "--run-root",
        type=Path,
        default=Path("experiments/rag"),
        help="Каталог с vec_rag_<category>/incremental/question_*.json (или один прогон)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/analysis/ragas_scores.json"),
        help="Куда записать JSON со средними и построчными оценками",
    )
    ap.add_argument("--max-rows", type=int, default=0, help="Лимит строк (0 = все найденные файлы)")
    ap.add_argument("--max-chunks", type=int, default=10, help="Сколько документов Chroma брать в контекст")
    ap.add_argument("--max-chars", type=int, default=1800, help="Обрезка одного чанка, символов")
    ap.add_argument(
        "--relevancy-strictness",
        type=int,
        default=1,
        help="Число перефраз-вопросов для answer_relevancy (меньше = дешевле)",
    )
    ap.add_argument("--timeout", type=int, default=600, help="Таймаут одной LLM-операции RAGAS, сек")
    ap.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Параллельные задачи RAGAS (меньше — стабильнее для локального Ollama)",
    )
    ap.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Модель LangChain HuggingFaceEmbeddings для answer_relevancy",
    )
    ap.add_argument(
        "--reports-only",
        action="store_true",
        help="Только пересобрать ragas_summary.* и графики из существующего --output JSON (без LLM)",
    )
    ap.add_argument(
        "--no-charts",
        action="store_true",
        help="Не строить PNG (только JSON и таблицы md/html)",
    )
    args = ap.parse_args()

    out_path = args.output.resolve()
    analysis_dir = out_path.parent

    if args.reports_only:
        if not out_path.is_file():
            print(f"Нет файла для --reports-only: {out_path}", file=sys.stderr)
            return 1
        agg = json.loads(out_path.read_text(encoding="utf-8"))
        paths = render_ragas_reports_from_agg(agg, analysis_dir, include_charts=not args.no_charts)
        for p in paths:
            print(f"Report: {p}")
        return 0

    try:
        from datasets import Dataset  # type: ignore[import-not-found]
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore[import-not-found]
        from openai import AsyncOpenAI  # type: ignore[import-not-found]
        from ragas import evaluate  # type: ignore[import-not-found]
        from ragas.llms.base import llm_factory  # type: ignore[import-not-found]
        from ragas.metrics._answer_relevance import AnswerRelevancy  # type: ignore[import-not-found]
        from ragas.metrics._faithfulness import Faithfulness  # type: ignore[import-not-found]
        from ragas.run_config import RunConfig  # type: ignore[import-not-found]
    except ImportError as e:
        print(
            "Не установлены зависимости RAGAS. Выполните: pip install -e \".[ragas]\"\n"
            f"Импорт: {e}",
            file=sys.stderr,
        )
        return 1

    repo = Path(__file__).resolve().parents[1]
    _load_env_file(repo / ".env")

    run_root = args.run_root.resolve()
    if not run_root.is_dir():
        print(f"Нет каталога: {run_root}", file=sys.stderr)
        return 1

    files = _iter_question_files(run_root)
    if args.max_rows > 0:
        files = files[: args.max_rows]

    rows_q: list[str] = []
    rows_a: list[str] = []
    rows_c: list[list[str]] = []
    manifest: list[dict[str, Any]] = []

    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        q = str(data.get("question") or "").strip()
        a = str(data.get("final_cashier_response") or "").strip()
        if not q or not a:
            continue
        ctx = _contexts_from_question_json(
            data, max_chunks=args.max_chunks, max_chars=args.max_chars
        )
        rows_q.append(q)
        rows_a.append(a)
        rows_c.append(ctx)
        try:
            rel = str(path.relative_to(repo))
        except ValueError:
            rel = str(path)
        manifest.append(
            {
                "path": rel,
                "question_id": data.get("question_id"),
                "category": data.get("category"),
            }
        )

    if not rows_q:
        print(f"Нет валидных строк в {run_root}", file=sys.stderr)
        return 1

    base_url = os.environ.get("LLM_BASE_URL") or "https://api.openai.com/v1"
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "ollama"
    model = os.environ.get("RAGAS_JUDGE_MODEL") or os.environ.get("API_MODEL") or "gpt-4o-mini"

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    llm = llm_factory(model, client=client)
    embeddings = HuggingFaceEmbeddings(model_name=args.embedding_model)

    ds = Dataset.from_dict({"question": rows_q, "answer": rows_a, "contexts": rows_c})
    metrics = [
        Faithfulness(),
        AnswerRelevancy(strictness=max(1, args.relevancy_strictness)),
    ]

    run_cfg = RunConfig(timeout=args.timeout, max_workers=max(1, args.max_workers))

    result = evaluate(
        ds,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=run_cfg,
        show_progress=True,
        raise_exceptions=False,
    )

    scores_list = getattr(result, "scores", None) or []
    agg: dict[str, Any] = {
        "run_root": str(run_root),
        "n_rows": len(rows_q),
        "judge_model": model,
        "llm_base_url": base_url,
        "embedding_model": args.embedding_model,
        "metrics": {},
        "rows": [],
    }

    for name in ("faithfulness", "answer_relevancy"):
        vals = []
        for s in scores_list:
            if isinstance(s, dict) and name in s and s[name] is not None:
                try:
                    v = float(s[name])
                    if not math.isnan(v):
                        vals.append(v)
                except (TypeError, ValueError):
                    pass
        if vals:
            agg["metrics"][name] = {
                "mean": round(sum(vals) / len(vals), 4),
                "n_scored": len(vals),
            }

    for i, m in enumerate(manifest):
        row_scores = scores_list[i] if i < len(scores_list) else {}
        agg["rows"].append({**m, "scores": _json_safe(row_scores)})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_json_safe(agg), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({agg['n_rows']} rows). Means: {agg.get('metrics', {})}")

    for p in render_ragas_reports_from_agg(agg, analysis_dir, include_charts=not args.no_charts):
        print(f"Report: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
