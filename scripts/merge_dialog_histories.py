#!/usr/bin/env python3
"""Склеивает поле history из всех dialog_*.json в один файл (без LLM)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_repo_path(p: Path) -> Path:
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Объединить history из dialog_*.json в один JSON или текстовый файл."
    )
    ap.add_argument("--dialogs_dir", type=Path, default=PROJECT_ROOT / "dialogs_rag")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Файл результата (по умолчанию: <dialogs_dir>/merged_histories.json)",
    )
    ap.add_argument(
        "--format",
        choices=("json", "txt"),
        default="json",
        help="json — один массив объектов {dialog_id, history}; txt — транскрипты с разделителями",
    )
    args = ap.parse_args()

    dialogs_dir = _resolve_repo_path(args.dialogs_dir)
    if not dialogs_dir.is_dir():
        print("Каталог не найден:", dialogs_dir, file=sys.stderr)
        return 1

    files = sorted(dialogs_dir.glob("dialog_*.json"))
    if not files:
        print("Нет dialog_*.json в", dialogs_dir, file=sys.stderr)
        return 1

    out_path = _resolve_repo_path(args.out) if args.out else dialogs_dir / "merged_histories.json"
    if args.format == "txt":
        out_path = out_path.with_suffix(".txt")

    merged: list[dict] = []
    text_blocks: list[str] = []

    for fp in files:
        data = json.loads(fp.read_text(encoding="utf-8"))
        did = data.get("dialog_id")
        hist = data.get("history") or []
        merged.append({"dialog_id": did, "history": hist})
        lines = ["=== dialog_id=%s ===" % did]
        for i, t in enumerate(hist, 1):
            sp = (t.get("speaker") or "?").upper()
            tx = (t.get("text") or "").strip()
            lines.append("%d. %s: %s" % (i, sp, tx))
        text_blocks.append("\n".join(lines))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "json":
        out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        out_path.write_text("\n\n".join(text_blocks) + "\n", encoding="utf-8")

    print("written:", out_path, "dialogs:", len(merged))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
