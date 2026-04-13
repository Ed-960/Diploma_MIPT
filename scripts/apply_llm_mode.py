"""
Принудительно выставляет API_PROVIDER перед запуском дочернего процесса.

Нужен для make-целей *-api / *-ollama (voice-browser, demo-dialog, demo-agents,
dataset-*, …) на любой ОС
(переменные вида VAR=value перед командой в Makefile не везде работают).

Пример:
  python scripts/apply_llm_mode.py api .venv/Scripts/python.exe scripts/voice_browser_server.py
  python scripts/apply_llm_mode.py ollama .venv/Scripts/python.exe scripts/dialog_demo.py --max_turns 10
"""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> None:
    if len(sys.argv) < 3:
        print(
            "Usage: apply_llm_mode.py <api|ollama> <executable> [arg ...]",
            file=sys.stderr,
        )
        sys.exit(2)
    mode = sys.argv[1].strip().lower()
    if mode == "api":
        os.environ["API_PROVIDER"] = "openai"
    elif mode == "ollama":
        os.environ["API_PROVIDER"] = "ollama"
    else:
        print("First argument must be 'api' or 'ollama'.", file=sys.stderr)
        sys.exit(2)
    cmd = sys.argv[2:]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
