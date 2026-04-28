#!/usr/bin/env python3
"""
Извлекает четыре диаграммы из docs/architecture_rag_visual.html в отдельные HTML
и при наличии Google Chrome / Chromium — в PDF (один файл на диаграмму).

Запуск из корня репозитория:
  python scripts/export_architecture_diagram_pdfs.py
  python scripts/export_architecture_diagram_pdfs.py --html-only

PDF: docs/diagrams/export/*.pdf
HTML: docs/diagrams/export/*.html
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "docs" / "architecture_rag_visual.html"
OUT_DIR = REPO / "docs" / "diagrams" / "export"

# slug, title, @page (высота под пропорции SVG — меньше пустого поля в PDF)
PAGES = (
    ("diagram_01_basic_llm", "Базовая LLM (без RAG)", "size: 297mm 78mm; margin: 2mm;"),
    ("diagram_02_vector_rag", "Vector RAG + мини-LLM", "size: 297mm 158mm; margin: 3mm;"),
    ("diagram_03_graph_rag", "Graph RAG", "size: 297mm 158mm; margin: 3mm;"),
    (
        "diagram_04_dialog_generation",
        "Генерация диалогов и итерация",
        "size: 297mm 172mm; margin: 3mm;",
    ),
)


def _read_style(html: str) -> str:
    m = re.search(r"<style>(.*?)</style>", html, re.DOTALL)
    if not m:
        raise SystemExit("Не найден блок <style> в architecture_rag_visual.html")
    return m.group(1).strip()


def _extract_sections(html: str) -> list[str]:
    pat = re.compile(r'<section\s+class="panel">.*?</section>', re.DOTALL)
    found = pat.findall(html)
    if len(found) != len(PAGES):
        raise SystemExit(
            f"Ожидалось {len(PAGES)} секций class=panel, найдено {len(found)}. Проверьте разметку."
        )
    return found


def _wrap_page(title: str, style: str, section_html: str, page_rule: str) -> str:
    extra = f"""
    @page {{ {page_rule} }}
    html, body {{
      margin: 0 !important;
      padding: 0 !important;
      background: #fff !important;
      min-height: 0 !important;
      height: auto !important;
    }}
    .panel {{
      max-width: none !important;
      margin: 0 !important;
      border-radius: 6px;
      box-shadow: none !important;
      page-break-inside: avoid;
    }}
    .panel-body {{
      padding: 2px 0 4px !important;
      background: #fff !important;
    }}
    .diagram {{ display: block; width: 100%; height: auto; }}
    @media print {{
      body {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
    }}
    """
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
{style}
{extra}
  </style>
</head>
<body>
{section_html}
</body>
</html>
"""


def _chrome_candidates() -> list[Path]:
    env = __import__("os").environ.get("CHROME_PATH", "").strip()
    paths: list[Path] = []
    if env:
        paths.append(Path(env))
    paths.extend(
        [
            Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
            Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
            Path("/usr/bin/google-chrome-stable"),
            Path("/usr/bin/google-chrome"),
            Path("/usr/bin/chromium"),
            Path("/usr/bin/chromium-browser"),
        ]
    )
    which = shutil.which("google-chrome") or shutil.which("chromium")
    if which:
        paths.append(Path(which))
    return paths


def _print_to_pdf(chrome: Path, html_path: Path, pdf_path: Path) -> None:
    url = html_path.resolve().as_uri()
    cmd = [
        str(chrome),
        "--headless=new",
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--no-first-run",
        "--no-default-browser-check",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path.resolve()}",
        url,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Экспорт диаграмм архитектуры в HTML/PDF.")
    ap.add_argument(
        "--html-only",
        action="store_true",
        help="Только HTML-файлы, без вызова Chrome.",
    )
    args = ap.parse_args()

    if not SOURCE.is_file():
        print(f"Нет файла: {SOURCE}", file=sys.stderr)
        return 1

    html = SOURCE.read_text(encoding="utf-8")
    style = _read_style(html)
    sections = _extract_sections(html)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    chrome: Path | None = None
    if not args.html_only:
        for c in _chrome_candidates():
            if c.is_file():
                chrome = c
                break

    for (slug, title, page_rule), section in zip(PAGES, sections, strict=True):
        out_html = OUT_DIR / f"{slug}.html"
        out_pdf = OUT_DIR / f"{slug}.pdf"
        page = _wrap_page(title, style, section.strip(), page_rule)
        out_html.write_text(page, encoding="utf-8")
        print(f"OK  {out_html.relative_to(REPO)}")

        if chrome and not args.html_only:
            try:
                _print_to_pdf(chrome, out_html, out_pdf)
                print(f"OK  {out_pdf.relative_to(REPO)}")
            except (subprocess.CalledProcessError, OSError) as e:
                print(f"[!] PDF {slug}: {e}", file=sys.stderr)

    if not args.html_only and chrome is None:
        print(
            "[i] Chrome/Chromium не найден — созданы только HTML. "
            "Задайте CHROME_PATH или установите Chrome; либо откройте .html → Печать → PDF.",
            file=sys.stderr,
        )
    elif args.html_only:
        print("[i] Режим --html-only: откройте HTML в браузере и сохраните как PDF.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
