"""
Общая раскладка каталогов question-экспериментов (no-rag / vector RAG).

Стандартный прогон (run_question_experiment.py):
  experiments/no-rag/norag_<category>/
  experiments/rag/vec_rag_<category>/
    incremental/question_XXXX.json
    rows.json, summary.json
    by_category/<category>/...

Старый формат (по-прежнему поддерживается):
  norag_<category>_<N>/question_*.json в корне прогона.

Для отчётов: берём все question_id из incremental/, затем rows.json
перезаписывает совпадающие id (финальная выгрузка прогона). summary.json
в каталоге прогона может быть короче (только последний resume-батч).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

QuestionLoader = Callable[[Path], dict[int, dict[str, Any]]]

RUN_DIR_RE = re.compile(
    r"^(?P<prefix>norag|vec_rag)_(?P<cat>simple|allergy|diet|lexical|mixed|group)(?:_(?P<num>\d+))?$"
)


def parse_run_dir(name: str) -> tuple[str, str] | None:
    """
    norag_simple / norag_simple_228 -> ('no-rag', 'simple')
    vec_rag_diet / vec_rag_diet_30 -> ('rag', 'diet')
    """
    m = RUN_DIR_RE.match(name)
    if not m:
        return None
    prefix, cat = m.group("prefix"), m.group("cat")
    mode = "no-rag" if prefix == "norag" else "rag"
    return mode, cat


def _ingest_question_file(
    out: dict[int, dict[str, Any]],
    path: Path,
    *,
    overwrite: bool = False,
) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(data, dict):
        return
    qid = data.get("question_id")
    if isinstance(qid, int) and (overwrite or qid not in out):
        out[qid] = data


def _load_rows_list(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    return [r for r in data if isinstance(r, dict)]


def _success_at_1_pct(rows: dict[int, dict[str, Any]]) -> float | None:
    if not rows:
        return None
    hits = sum(1 for r in rows.values() if (r.get("metrics") or {}).get("success_at_1"))
    return round(100.0 * hits / len(rows), 2)


def load_questions_from_rows_only(run_dir: Path) -> dict[int, dict[str, Any]]:
    """Только ``rows.json`` в корне прогона (без incremental и legacy question_*.json)."""
    out: dict[int, dict[str, Any]] = {}
    for row in _load_rows_list(run_dir / "rows.json"):
        qid = row.get("question_id")
        if isinstance(qid, int):
            out[qid] = row
    return out


def load_questions_from_run_dir(run_dir: Path) -> dict[int, dict[str, Any]]:
    """
    Строки по question_id для агрегации отчётов.

    Порядок (поздний источник перезаписывает metrics для того же id):
      1) incremental/question_*.json — полный лог прогона;
      2) question_*.json в корне (legacy);
      3) rows.json — финальная выгрузка save_dialogs_by_category.
    """
    out: dict[int, dict[str, Any]] = {}
    inc = run_dir / "incremental"
    if inc.is_dir():
        for path in sorted(inc.glob("question_*.json")):
            _ingest_question_file(out, path)
    for path in sorted(run_dir.glob("question_*.json")):
        _ingest_question_file(out, path)
    for row in _load_rows_list(run_dir / "rows.json"):
        qid = row.get("question_id")
        if isinstance(qid, int):
            out[qid] = row
    return out


def inspect_run_dir_data(run_dir: Path, *, rows_only: bool = False) -> dict[str, Any]:
    """Сравнение incremental, rows.json и summary.json (для предупреждений в отчётах)."""
    loader: QuestionLoader = (
        load_questions_from_rows_only if rows_only else load_questions_from_run_dir
    )
    inc_dir = run_dir / "incremental"
    inc_files = len(list(inc_dir.glob("question_*.json"))) if inc_dir.is_dir() else 0
    rows_list = _load_rows_list(run_dir / "rows.json")
    loaded = loader(run_dir)
    summary_n: int | None = None
    summary_s1: float | None = None
    sp = run_dir / "summary.json"
    if sp.is_file():
        try:
            summary = json.loads(sp.read_text(encoding="utf-8"))
            summary_n = int(summary.get("n") or summary.get("total") or 0)
            if summary_n > 0 and summary.get("success_at_1") is not None:
                summary_s1 = round(100.0 * float(summary["success_at_1"]), 2)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    loaded_s1 = _success_at_1_pct(loaded)
    warnings: list[str] = []
    if rows_only and not rows_list:
        warnings.append("rows.json отсутствует или пуст — отчёт по этому прогону будет пустым.")
    if rows_only and rows_list and inc_files > len(rows_list):
        warnings.append(
            f"режим rows-only: incremental={inc_files}, rows.json={len(rows_list)} — "
            "в отчёт попадут только строки из rows.json."
        )
    if not rows_only and rows_list and inc_files > len(rows_list):
        warnings.append(
            f"incremental={inc_files} файлов, rows.json={len(rows_list)} — "
            "в каталоге прогона rows/summary часто только последний resume-батч; "
            "отчёты используют полный incremental (+ rows для совпадающих id)."
        )
    if summary_n is not None and summary_n != len(loaded):
        warnings.append(
            f"summary.json n={summary_n}, загружено для отчёта n={len(loaded)} — "
            "смотрите experiments/analysis/, не summary.json в каталоге прогона."
        )
    if summary_s1 is not None and loaded_s1 is not None and abs(summary_s1 - loaded_s1) >= 1.0:
        warnings.append(
            f"success@1: summary.json {summary_s1}% vs отчёт {loaded_s1}% "
            "(разные объёмы выборки или только последний батч в summary)."
        )
    only_inc = inc_files - len(rows_list) if rows_list else 0
    if only_inc > 0 and rows_list:
        row_ids = {r.get("question_id") for r in rows_list if isinstance(r.get("question_id"), int)}
        inc_ids = set()
        if inc_dir.is_dir():
            for path in inc_dir.glob("question_*.json"):
                try:
                    d = json.loads(path.read_text(encoding="utf-8"))
                    qid = d.get("question_id")
                    if isinstance(qid, int):
                        inc_ids.add(qid)
                except (OSError, json.JSONDecodeError):
                    pass
        extra = sorted(inc_ids - row_ids)
        if extra:
            warnings.append(
                f"только в incremental (нет в rows.json): {len(extra)} вопросов "
                f"(id {extra[0]}…{extra[-1]})."
            )
    return {
        "run_dir": str(run_dir),
        "incremental_files": inc_files,
        "rows_json_rows": len(rows_list),
        "loaded_rows": len(loaded),
        "summary_n": summary_n,
        "summary_success_at_1_pct": summary_s1,
        "loaded_success_at_1_pct": loaded_s1,
        "warnings": warnings,
    }


def iter_run_dirs(parent: Path, *, expected_mode: str | None = None) -> list[Path]:
    """Подкаталоги parent, имя которых — norag_* или vec_rag_*."""
    if not parent.is_dir():
        return []
    found: list[Path] = []
    for child in sorted(parent.iterdir()):
        if not child.is_dir():
            continue
        parsed = parse_run_dir(child.name)
        if not parsed:
            continue
        mode, _cat = parsed
        if expected_mode is not None and mode != expected_mode:
            continue
        found.append(child)
    return found


def _pick_best_run_dir(candidates: list[Path], *, loader: QuestionLoader) -> Path:
    if len(candidates) == 1:
        return candidates[0]
    return max(candidates, key=lambda p: len(loader(p)))


def discover_paired_runs(
    norag_root: Path,
    rag_root: Path,
    *,
    loader: QuestionLoader | None = None,
) -> dict[str, dict[str, Path]]:
    """category -> {'no-rag': Path, 'rag': Path} (лучший прогон при нескольких каталогах на категорию)."""
    load_fn = loader or load_questions_from_run_dir
    by_cat_mode: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for base, mode_key in ((norag_root, "no-rag"), (rag_root, "rag")):
        for child in iter_run_dirs(base, expected_mode=mode_key):
            parsed = parse_run_dir(child.name)
            if not parsed:
                continue
            _mode, cat = parsed
            by_cat_mode[cat][mode_key].append(child)
    paired: dict[str, dict[str, Path]] = {}
    for cat, modes in sorted(by_cat_mode.items()):
        if "no-rag" not in modes or "rag" not in modes:
            continue
        paired[cat] = {
            "no-rag": _pick_best_run_dir(modes["no-rag"], loader=load_fn),
            "rag": _pick_best_run_dir(modes["rag"], loader=load_fn),
        }
    return paired


def iter_question_json_paths(run_root: Path) -> list[Path]:
    """
    По одному JSON на question_id (тот же merge, что load_questions_from_run_dir).
    """
    run_dirs = iter_run_dirs(run_root)
    if not run_dirs and parse_run_dir(run_root.name):
        run_dirs = [run_root]
    paths: list[Path] = []
    for run_dir in run_dirs:
        loaded = load_questions_from_run_dir(run_dir)
        for qid in sorted(loaded):
            row = loaded[qid]
            inc_path = run_dir / "incremental" / f"question_{qid:04d}.json"
            if inc_path.is_file():
                paths.append(inc_path)
            else:
                legacy = run_dir / f"question_{qid:04d}.json"
                if legacy.is_file():
                    paths.append(legacy)
    return paths
