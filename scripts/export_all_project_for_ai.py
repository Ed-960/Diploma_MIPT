"""
Собирает один текстовый файл со всем кодом проекта для ИИ / ревью.

Запуск (из корня репозитория):
  make export-ai
  make export-ai AI_EXPORT=my_dump.txt
  python scripts/export_all_project_for_ai.py
  python scripts/export_all_project_for_ai.py --output my_dump.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Порядок секций — как в allProject_forAI_Test.txt (scripts → src → tests → корень).
FILE_ORDER: tuple[str, ...] = (
    "scripts/_bootstrap.py",
    "scripts/agents_demo.py",
    "scripts/compare_rag.py",
    "scripts/dialog_demo.py",
    "scripts/export_diploma_docx.py",
    "scripts/generate_dataset.py",
    "scripts/generate_profiles.py",
    "scripts/load_chroma.py",
    "scripts/menu_search_demo.py",
    "scripts/profile_demo.py",
    "scripts/test_menu_search.py",
    "scripts/run_experiment.sh",
    "scripts/visualize_profile_graph.py",
    "scripts/visualize_menu_graph.py",
    "scripts/apply_llm_mode.py",
    "scripts/voice_browser_server.py",
    "scripts/static/voice_browser_demo.html",
    "scripts/export_all_project_for_ai.py",
    "src/mcd_voice/__init__.py",
    "src/mcd_voice/config.py",
    "src/mcd_voice/dialog/__init__.py",
    "src/mcd_voice/dialog/allergens.py",
    "src/mcd_voice/dialog/catalog.py",
    "src/mcd_voice/dialog/pipeline.py",
    "src/mcd_voice/dialog/save_dialog.py",
    "src/mcd_voice/dialog/trace_format.py",
    "src/mcd_voice/dialog/human_voice_session.py",
    "src/mcd_voice/llm/__init__.py",
    "src/mcd_voice/llm/agent.py",
    "src/mcd_voice/llm/prompts.py",
    "src/mcd_voice/menu/__init__.py",
    "src/mcd_voice/menu/chroma.py",
    "src/mcd_voice/menu/dataset.py",
    "src/mcd_voice/menu/parsing.py",
    "src/mcd_voice/menu/search.py",
    "src/mcd_voice/menu/search_checks.py",
    "src/mcd_voice/menu/graph_rag.py",
    "src/mcd_voice/profile/__init__.py",
    "src/mcd_voice/profile/decision_graph.py",
    "src/mcd_voice/profile/generator.py",
    "tests/test_llm_config.py",
    "tests/test_pipeline_helpers.py",
    "tests/test_human_voice_session.py",
    "tests/test_profile_generator.py",
    "tests/test_profile_decision_graph.py",
    "tests/test_rag_trace.py",
    "tests/test_realistic_cashier.py",
    "tests/test_save_dialog.py",
    "tests/test_search.py",
    "pyproject.toml",
    "mcd.json",
    "PROJECT_CONTEXT.md",
    "docs/ARCHITECTURE.md",
    "docs/menu_graph_rag.html",
    "README.md",
    "Makefile",
    "start.md",
    "requirements.txt",
    "doploma.txt",
)

SEP = "=" * 88


def build_concat(root: Path) -> str:
    missing = [p for p in FILE_ORDER if not (root / p).exists()]
    if missing:
        raise FileNotFoundError(
            "Отсутствуют файлы:\n" + "\n".join(f"  - {m}" for m in missing),
        )

    py_count = sum(1 for p in FILE_ORDER if p.endswith(".py"))
    lines: list[str] = [
        "# CONCATENATED PROJECT CODE FOR AI ANALYSIS",
        f"# Root: {root.resolve()}",
        "# Python: "
        f"{py_count} files (src/mcd_voice, scripts, tests) + pyproject.toml + mcd.json + "
        "docs/README/Makefile/start/requirements + doploma.txt + shell helper",
        "",
    ]

    for rel in FILE_ORDER:
        p = root / rel
        content = p.read_text(encoding="utf-8")
        if not content.endswith("\n"):
            content += "\n"
        lines.extend(
            [
                SEP,
                "",
                SEP,
                f"# FILE: {rel}",
                SEP,
                "",
                content.rstrip("\n"),
                "",
                SEP,
                "",
                SEP,
                "",
            ],
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Собрать один файл со всем кодом проекта для ИИ.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("allProject_forAI_Test.txt"),
        help="Путь к выходному файлу (по умолчанию allProject_forAI_Test.txt в корне).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    text = build_concat(root)
    out = args.output
    if not out.is_absolute():
        out = root / out
    out.write_text(text, encoding="utf-8")
    n_files = len(FILE_ORDER)
    print(f"OK: {out} ({out.stat().st_size} bytes, {n_files} files)")


if __name__ == "__main__":
    main()
