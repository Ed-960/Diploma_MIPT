from __future__ import annotations

import json
from pathlib import Path

from experiment_run_layout import (
    discover_paired_runs,
    inspect_run_dir_data,
    iter_question_json_paths,
    load_questions_from_run_dir,
    load_questions_from_rows_only,
    parse_run_dir,
)


def test_parse_run_dir_standard_and_legacy() -> None:
    assert parse_run_dir("norag_simple") == ("no-rag", "simple")
    assert parse_run_dir("vec_rag_allergy") == ("rag", "allergy")
    assert parse_run_dir("norag_diet_30") == ("no-rag", "diet")
    assert parse_run_dir("vec_rag_lexical_101") == ("rag", "lexical")
    assert parse_run_dir("reports_smoke") is None
    assert parse_run_dir("norag_unknown") is None


def test_load_questions_rows_only_ignores_incremental(tmp_path: Path) -> None:
    run = tmp_path / "norag_simple"
    inc = run / "incremental"
    inc.mkdir(parents=True)
    (inc / "question_0001.json").write_text(
        json.dumps({"question_id": 1, "category": "simple", "metrics": {"success_at_1": False}}),
        encoding="utf-8",
    )
    (inc / "question_0002.json").write_text(
        json.dumps({"question_id": 2, "category": "simple", "metrics": {"success_at_1": True}}),
        encoding="utf-8",
    )
    (run / "rows.json").write_text(
        json.dumps([{"question_id": 1, "category": "simple", "metrics": {"success_at_1": True}}]),
        encoding="utf-8",
    )
    loaded = load_questions_from_rows_only(run)
    assert set(loaded) == {1}
    assert loaded[1]["metrics"]["success_at_1"] is True


def test_load_questions_rows_json_overwrites_incremental(tmp_path: Path) -> None:
    run = tmp_path / "norag_simple"
    inc = run / "incremental"
    inc.mkdir(parents=True)
    (inc / "question_0001.json").write_text(
        json.dumps({"question_id": 1, "category": "simple", "metrics": {"success_at_1": False}}),
        encoding="utf-8",
    )
    (run / "rows.json").write_text(
        json.dumps([{"question_id": 1, "category": "simple", "metrics": {"success_at_1": True}}]),
        encoding="utf-8",
    )
    loaded = load_questions_from_run_dir(run)
    assert loaded[1]["metrics"]["success_at_1"] is True


def test_inspect_run_dir_data_warns_on_partial_rows(tmp_path: Path) -> None:
    run = tmp_path / "norag_allergy"
    inc = run / "incremental"
    inc.mkdir(parents=True)
    for qid in (1, 2):
        (inc / f"question_{qid:04d}.json").write_text(
            json.dumps({"question_id": qid, "metrics": {"success_at_1": False}}),
            encoding="utf-8",
        )
    (run / "rows.json").write_text(
        json.dumps([{"question_id": 2, "metrics": {"success_at_1": True}}]),
        encoding="utf-8",
    )
    (run / "summary.json").write_text(
        json.dumps({"n": 1, "success_at_1": 1.0}),
        encoding="utf-8",
    )
    info = inspect_run_dir_data(run)
    assert info["loaded_rows"] == 2
    assert info["rows_json_rows"] == 1
    assert any("incremental=" in w for w in info["warnings"])


def test_discover_paired_runs_and_iter_paths(tmp_path: Path) -> None:
    norag_root = tmp_path / "no-rag"
    rag_root = tmp_path / "rag"
    for mode, cat in (("norag", "simple"), ("vec_rag", "simple")):
        run = (norag_root if mode == "norag" else rag_root) / f"{mode}_{cat}"
        inc = run / "incremental"
        inc.mkdir(parents=True)
        (inc / "question_0001.json").write_text(
            json.dumps({"question_id": 1, "category": "simple"}),
            encoding="utf-8",
        )
    (rag_root / "reports_smoke").mkdir()
    (rag_root / "reports_smoke" / "question_9999.json").write_text("{}", encoding="utf-8")

    paired = discover_paired_runs(norag_root, rag_root)
    assert set(paired) == {"simple"}
    paths = iter_question_json_paths(rag_root)
    assert len(paths) == 1
    assert paths[0].name == "question_0001.json"
