#!/usr/bin/env python3
"""LLM-судья только по history; без импорта mcd_voice (только openai + env)."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import _bootstrap

_bootstrap.load_dotenv(PROJECT_ROOT / ".env")

from openai import APITimeoutError, OpenAI, OpenAIError


def _normalize_base_url(raw: str | None) -> str:
    if not raw:
        return ""
    u = raw.strip().rstrip("/")
    if not u.endswith("/v1"):
        u = u + "/v1"
    return u


def build_client(timeout: float = 120.0) -> OpenAI:
    key = (os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        env_path = PROJECT_ROOT / ".env"
        raise RuntimeError(
            "Нет ключа API: задайте LLM_API_KEY или OPENAI_API_KEY "
            "(файл %(env)s или переменные окружения в терминале)."
            % {"env": env_path}
        )
    base = (
        os.environ.get("LLM_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("XAI_BASE_URL")
        or ""
    ).strip()
    kw: dict = {"api_key": key, "timeout": timeout}
    if base:
        kw["base_url"] = _normalize_base_url(base)
    return OpenAI(**kw)


def judge_model() -> str:
    return (
        os.environ.get("JUDGE_MODEL", "").strip()
        or os.environ.get("REWRITE_MODEL", "").strip()
        or os.environ.get("API_MODEL", "").strip()
        or "gpt-4o-mini"
    )


def openrouter_extra_body() -> dict | None:
    raw = (os.environ.get("OPENROUTER_PROVIDER_IGNORE") or "").strip().lower()
    if raw in ("", "none", "false", "0", "off", "-"):
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return None
    return {"provider": {"ignore": parts}}


JUDGE_SYSTEM = """You are an expert evaluator of fast-food drive-through dialog transcripts in English.
You ONLY see the conversation turns. You do NOT know any menu, customer profile, or final order JSON.

Return a SINGLE JSON object (no markdown fences):
{
  "realism_drive_through_1_5": <int 1-5>,
  "coherence_1_5": <int 1-5>,
  "cashier_helpful_clear_1_5": <int 1-5>,
  "client_natural_1_5": <int 1-5>,
  "severe_defect": <bool>,
  "defect_codes": [<string>],
  "one_line_summary": <string>,
  "evidence_quotes": [<string, max 3 short quotes>]
}

defect_codes from: loop_or_stall, ignores_customer_request, contradictory_turns, meta_or_cot_leak,
overly_robotic_cashier, overly_verbose_ingredient_dump, nonsensical_response,
wrong_channel_tone, off_brand_competitor_request, other

severe_defect: true if a real customer would be frustrated or basic coherence fails.
Be strict about repeated exchanges and cashier ignoring the customer."""


def history_to_transcript(history: list, max_chars: int = 28000) -> str:
    lines = []
    for i, t in enumerate(history, 1):
        sp = t.get("speaker") or "?"
        tx = (t.get("text") or "").strip()
        lines.append("%d. %s: %s" % (i, sp.upper(), tx))
    full = "\n".join(lines)
    if len(full) <= max_chars:
        return full
    h, t = max_chars // 2 - 40, max_chars // 2 - 40
    return full[:h] + "\n\n[... middle omitted ...]\n\n" + full[-t:]


def call_judge(client: OpenAI, model: str, transcript: str) -> str:
    payload = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": "TRANSCRIPT:\n" + transcript + "\n\nRespond with JSON only."},
    ]
    kwargs = {"model": model, "messages": payload, "temperature": 0.15}
    xb = openrouter_extra_body()
    if xb is not None:
        kwargs["extra_body"] = xb
    last_err = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(**kwargs)
            c = resp.choices[0].message.content
            if not c:
                raise RuntimeError("empty response")
            return c.strip()
        except (APITimeoutError, OpenAIError) as exc:
            last_err = exc
            time.sleep(1.2 * (attempt + 1))
    raise RuntimeError("judge failed: %s" % last_err)


def _resolve_repo_path(p: Path) -> Path:
    """Относительные пути — от корня репозитория (удобно при любом cwd)."""
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def parse_judge_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}\s*$", text)
    if m:
        text = m.group(0)
    return json.loads(text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogs_dir", type=Path, default=PROJECT_ROOT / "dialogs_rag")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.35)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    dialogs_dir = _resolve_repo_path(args.dialogs_dir)
    if not dialogs_dir.is_dir():
        print(
            "Каталог с диалогами не найден: %s\n"
            "(ожидался dialog_*.json; при относительном пути он ищется от %s)"
            % (dialogs_dir, PROJECT_ROOT),
            file=sys.stderr,
        )
        return 1

    files = sorted(dialogs_dir.glob("dialog_*.json"))
    if args.limit:
        files = files[: args.limit]
    out_path = _resolve_repo_path(args.out) if args.out else (dialogs_dir / "history_judge.jsonl")

    if args.dry_run:
        print("dry_run:", len(files), "dialogs_dir=", dialogs_dir, "->", out_path)
        return 0

    if not files:
        print(
            "Нет файлов dialog_*.json в %s" % dialogs_dir,
            file=sys.stderr,
        )
        return 1

    try:
        client = build_client()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    model = judge_model()
    print("judge model:", model, file=sys.stderr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(int(json.loads(line)["dialog_id"]))
                except Exception:
                    pass

    print(
        "[history-judge] files=%d output=%s ids_already_done=%d"
        % (len(files), out_path, len(done)),
        file=sys.stderr,
        flush=True,
    )

    n = 0
    with out_path.open("a", encoding="utf-8") as out:
        for fi, f in enumerate(files, start=1):
            data = json.loads(f.read_text(encoding="utf-8"))
            did = int(data["dialog_id"])
            if did in done:
                continue
            print(
                "[history-judge] %d/%d dialog_id=%d"
                % (fi, len(files), did),
                file=sys.stderr,
                flush=True,
            )
            hist = data.get("history") or []
            tr = history_to_transcript(hist)
            try:
                raw = call_judge(client, model, tr)
                verdict = parse_judge_json(raw)
            except Exception as exc:
                out.write(json.dumps({"dialog_id": did, "error": str(exc)}, ensure_ascii=False) + "\n")
                out.flush()
                print("ERR", did, exc, file=sys.stderr)
            else:
                out.write(
                    json.dumps(
                        {
                            "dialog_id": did,
                            "transcript_turns": len(hist),
                            "transcript_chars": len(tr),
                            "judge_model": model,
                            "verdict": verdict,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                out.flush()
                n += 1
            if args.sleep:
                time.sleep(args.sleep)
    print("appended rows:", n, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
