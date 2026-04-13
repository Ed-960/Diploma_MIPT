"""
Exports the profile decision graph to Mermaid/JSON for diploma artifacts.

Usage:
  python scripts/visualize_profile_graph.py
  python scripts/visualize_profile_graph.py --output docs/profile_decision_graph.mmd
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap

_bootstrap.ensure_src()

from mcd_voice.profile import PROFILE_DECISION_GRAPH, to_mermaid, walk_graph


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render profile decision graph for documentation.",
    )
    parser.add_argument(
        "--output",
        default="docs/profile_decision_graph.mmd",
        help="Path to Mermaid output file.",
    )
    parser.add_argument(
        "--rows-output",
        default="docs/profile_decision_graph_rows.json",
        help="Path to flattened JSON rows.",
    )
    args = parser.parse_args()

    mermaid = to_mermaid(PROFILE_DECISION_GRAPH)
    rows = walk_graph(PROFILE_DECISION_GRAPH)

    out_mmd = Path(args.output)
    out_rows = Path(args.rows_output)
    out_mmd.parent.mkdir(parents=True, exist_ok=True)
    out_rows.parent.mkdir(parents=True, exist_ok=True)

    out_mmd.write_text(mermaid + "\n", encoding="utf-8")
    out_rows.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Mermaid graph: {out_mmd}")
    print(f"Node rows JSON: {out_rows}")
    print(f"Nodes exported: {len(rows)}")


if __name__ == "__main__":
    main()
