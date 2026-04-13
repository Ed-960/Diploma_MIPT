"""
Экспорт графа меню для graph-RAG в Mermaid + HTML (открыть в браузере).

Usage:
  python scripts/visualize_menu_graph.py
  python scripts/visualize_menu_graph.py --open
  python scripts/visualize_menu_graph.py --max-edges 120 --output docs/menu_graph_rag.mmd
"""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

import _bootstrap

_bootstrap.ensure_src()

from mcd_voice.menu.graph_rag import _build_menu_graph, menu_graph_to_mermaid

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Menu graph-RAG</title>
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
</head>
<body>
  <pre class="mermaid">
{MERMAID}
  </pre>
  <script>mermaid.initialize({ startOnLoad: true, theme: "neutral" });</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render menu co-occurrence graph (graph-RAG) as Mermaid + HTML.",
    )
    parser.add_argument(
        "--output",
        default="docs/menu_graph_rag.mmd",
        help="Path to Mermaid output (.mmd).",
    )
    parser.add_argument(
        "--html-output",
        default="docs/menu_graph_rag.html",
        help="Standalone HTML (Mermaid CDN) for browser preview.",
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=220,
        help="Cap on undirected edges drawn (strongest first).",
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.25,
        help="Minimum edge weight to include.",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Only write .mmd, skip HTML.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the HTML file in the default browser (no-op if --no-html).",
    )
    args = parser.parse_args()

    mermaid = menu_graph_to_mermaid(max_edges=args.max_edges, min_weight=args.min_weight)
    graph = _build_menu_graph()
    n_nodes = len(graph["nodes"])
    n_undirected = sum(len(graph["edges"][k]) for k in graph["edges"]) // 2

    out_mmd = Path(args.output)
    out_mmd.parent.mkdir(parents=True, exist_ok=True)
    out_mmd.write_text(mermaid, encoding="utf-8")
    print(f"Mermaid: {out_mmd}  (nodes={n_nodes}, undirected edges≈{n_undirected})")

    if not args.no_html:
        out_html = Path(args.html_output)
        out_html.parent.mkdir(parents=True, exist_ok=True)
        body = mermaid.rstrip("\n")
        html = _HTML_TEMPLATE.replace("{MERMAID}", body)
        out_html.write_text(html, encoding="utf-8")
        print(f"HTML:    {out_html}")
        if args.open:
            webbrowser.open(out_html.resolve().as_uri())


if __name__ == "__main__":
    main()
