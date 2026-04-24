"""
Локальный HTTP-сервер: статическая страница + JSON API для голосового демо.

Запуск из корня репозитория:
  python scripts/voice_browser_server.py

Откройте http://127.0.0.1:8765/  (нужны API-ключ / Chroma, как для dialog_demo).

Поведение: поочерёдные реплики (не одновременный full-duplex разговор).
Браузер распознаёт речь и озвучивает ответы кассира (Web Speech API).
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlparse

import _bootstrap

_bootstrap.ensure_src()

from mcd_voice.dialog.human_voice_session import HumanDriveThroughSession
from mcd_voice.dialog.save_dialog import save_dialog


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VOICE_DIALOGS_DIR = REPO_ROOT / "ALL_DIALOGS" / "voice-dialogs"

_voice_dialog_id_lock = threading.Lock()


def _save_voice_dialog_snapshot(
    out_dir: Path,
    profile: dict[str, Any],
    history: list[dict[str, str]],
    order_state: dict[str, Any],
    flags: dict[str, Any],
) -> Path:
    """Следующий dialog_NNNN.json и запись — под одной блокировкой (без гонок по id)."""
    with _voice_dialog_id_lock:
        out_dir.mkdir(parents=True, exist_ok=True)
        best = 0
        for p in out_dir.glob("dialog_*.json"):
            try:
                n = int(p.stem.split("_", maxsplit=1)[1])
            except (ValueError, IndexError):
                continue
            best = max(best, n)
        dialog_id = best + 1
        return save_dialog(
            dialog_id,
            profile,
            history,
            order_state,
            flags,
            output_dir=out_dir,
        )


def _prewarm_menu_rag() -> None:
    """Первый RAG тянет Chroma + эмбеддинги; делаем до HTTP, чтобы UI не «молчал» минуту."""
    print("Prewarm: loading Chroma + sentence-transformers (first time can be slow)…", flush=True)
    t0 = time.perf_counter()
    try:
        from mcd_voice.menu.search import search_menu

        search_menu("Big Mac", top_k=1)
    except Exception as exc:
        print(f"Prewarm vector RAG failed (Start session may still trigger load): {exc}", flush=True)
        return
    mode = (os.environ.get("RAG_MODE") or "vector").strip().lower()
    if mode == "graph":
        try:
            from mcd_voice.menu.graph_rag import search_menu_graph

            search_menu_graph("burger", top_k=1)
        except Exception as exc:
            print(f"Prewarm graph RAG failed: {exc}", flush=True)
    print(f"Prewarm done in {time.perf_counter() - t0:.1f}s", flush=True)


STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_PATH = STATIC_DIR / "voice_browser_demo.html"

_sessions: dict[str, HumanDriveThroughSession] = {}
_sessions_lock = threading.Lock()


def _json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


class VoiceBrowserHandler(BaseHTTPRequestHandler):
    server_version = "McdVoiceBrowser/0.1"
    _log_requests: ClassVar[bool] = False

    def log_message(self, fmt: str, *args: Any) -> None:
        if VoiceBrowserHandler._log_requests:
            super().log_message(fmt, *args)

    def _send(
        self,
        code: int,
        body: bytes,
        *,
        content_type: str,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            if not INDEX_PATH.is_file():
                self._send(
                    HTTPStatus.NOT_FOUND,
                    b"Missing voice_browser_demo.html",
                    content_type="text/plain; charset=utf-8",
                )
                return
            html = INDEX_PATH.read_bytes()
            self._send(HTTPStatus.OK, html, content_type="text/html; charset=utf-8")
            return
        self._send(HTTPStatus.NOT_FOUND, b"Not found", content_type="text/plain; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._send(
                HTTPStatus.BAD_REQUEST,
                _json_bytes({"error": "invalid_json"}),
                content_type="application/json; charset=utf-8",
            )
            return

        cfg: argparse.Namespace = self.server.cfg  # type: ignore[attr-defined]

        if parsed.path == "/api/session/status":
            sid = str(data.get("session_id") or "")
            if not sid:
                self._send(
                    HTTPStatus.BAD_REQUEST,
                    _json_bytes({"error": "missing_session_id"}),
                    content_type="application/json; charset=utf-8",
                )
                return
            with _sessions_lock:
                alive = sid in _sessions
            self._send(
                HTTPStatus.OK,
                _json_bytes({"alive": alive, "session_id": sid}),
                content_type="application/json; charset=utf-8",
            )
            return

        if parsed.path == "/api/session/abandon":
            sid = str(data.get("session_id") or "")
            if not sid:
                self._send(
                    HTTPStatus.BAD_REQUEST,
                    _json_bytes({"error": "missing_session_id"}),
                    content_type="application/json; charset=utf-8",
                )
                return
            with _sessions_lock:
                _sessions.pop(sid, None)
            self._send(
                HTTPStatus.OK,
                _json_bytes({"ok": True, "session_id": sid}),
                content_type="application/json; charset=utf-8",
            )
            return

        if parsed.path == "/api/session/start":
            print("[api] POST /api/session/start …", flush=True)
            sid = uuid.uuid4().hex
            session = HumanDriveThroughSession(
                max_turns=cfg.max_turns,
                model=cfg.model,
                realistic_cashier=cfg.realistic_cashier,
                trace_verbose=cfg.trace_verbose,
            )
            try:
                start_payload = session.start()
            except Exception as exc:
                self._send(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    _json_bytes({"error": str(exc)}),
                    content_type="application/json; charset=utf-8",
                )
                return
            with _sessions_lock:
                _sessions[sid] = session
            out = {
                "session_id": sid,
                "greeting": start_payload["greeting"],
                "profile": start_payload["profile"],
                "validation": start_payload["validation"],
            }
            print("[api] POST /api/session/start OK", flush=True)
            self._send(
                HTTPStatus.OK,
                _json_bytes(out),
                content_type="application/json; charset=utf-8",
            )
            return

        if parsed.path == "/api/session/message":
            sid = str(data.get("session_id") or "")
            text = str(data.get("text") or "")
            with _sessions_lock:
                session = _sessions.get(sid)
            if session is None:
                self._send(
                    HTTPStatus.NOT_FOUND,
                    _json_bytes({"error": "unknown_session"}),
                    content_type="application/json; charset=utf-8",
                )
                return
            try:
                out = session.step(text)
            except ValueError as exc:
                self._send(
                    HTTPStatus.BAD_REQUEST,
                    _json_bytes({"error": str(exc)}),
                    content_type="application/json; charset=utf-8",
                )
                return
            except Exception as exc:
                self._send(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    _json_bytes({"error": str(exc)}),
                    content_type="application/json; charset=utf-8",
                )
                return
            snap = None
            if out.get("dialog_ended") and cfg.save_voice_dialogs:
                snap = session.snapshot_for_save()
            if out.get("dialog_ended"):
                with _sessions_lock:
                    _sessions.pop(sid, None)
            if snap is not None:
                profile, history, order_state, flags = snap
                out_dir = Path(cfg.voice_output_dir)
                try:
                    saved_path = _save_voice_dialog_snapshot(
                        out_dir, profile, history, order_state, flags,
                    )
                    out["saved_dialog_path"] = str(saved_path)
                    out["saved_dialog_id"] = int(saved_path.stem.split("_", maxsplit=1)[1])
                    print(f"[api] Saved voice dialog → {saved_path}", flush=True)
                except Exception as exc:
                    print(f"[api] Voice dialog save failed: {exc}", flush=True)
                    out["saved_dialog_error"] = str(exc)
            self._send(
                HTTPStatus.OK,
                _json_bytes(out),
                content_type="application/json; charset=utf-8",
            )
            return

        self._send(
            HTTPStatus.NOT_FOUND,
            _json_bytes({"error": "not_found"}),
            content_type="application/json; charset=utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Browser STT/TTS + AI cashier (local demo).")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8765, help="TCP port (default 8765).")
    parser.add_argument("--max-turns", type=int, default=20, help="Max client lines per session.")
    parser.add_argument(
        "--realistic-cashier",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cashier without hidden profile (default: true; good for free-form speech).",
    )
    parser.add_argument("--model", default=None, help="Override API_MODEL for this process.")
    parser.add_argument("--trace-verbose", action="store_true", help="Verbose LLM traces in agents.")
    parser.add_argument("--verbose-http", action="store_true", help="Log each HTTP request.")
    parser.add_argument(
        "--no-prewarm",
        action="store_true",
        help="Skip Chroma/embeddings preload (faster server boot; first Start waits longer).",
    )
    parser.add_argument(
        "--voice-output-dir",
        default=str(DEFAULT_VOICE_DIALOGS_DIR),
        help=f"Каталог для dialog_*.json при завершении сессии (по умолчанию: {DEFAULT_VOICE_DIALOGS_DIR}).",
    )
    parser.add_argument(
        "--save-voice-dialogs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Сохранять диалог в JSON при dialog_ended (как generate_dataset; по умолчанию: да).",
    )
    args = parser.parse_args()

    VoiceBrowserHandler._log_requests = args.verbose_http
    if not args.no_prewarm:
        _prewarm_menu_rag()
    httpd = ThreadingHTTPServer((args.host, args.port), VoiceBrowserHandler)
    httpd.cfg = args  # type: ignore[attr-defined]
    print(f"Open http://{args.host}:{args.port}/  (Ctrl+C to stop)", flush=True)
    if args.save_voice_dialogs:
        print(
            f"On session end, dialogs are saved like generate_dataset → {args.voice_output_dir}/",
            flush=True,
        )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)


if __name__ == "__main__":
    main()
