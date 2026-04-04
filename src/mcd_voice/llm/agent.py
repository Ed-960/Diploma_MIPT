"""
Агенты ClientAgent и CashierAgent поверх OpenAI-compatible Chat Completions.

Конфигурация через переменные окружения:
  API_PROVIDER   — "openai" (по умолчанию) или "ollama".
  API_MODEL      — основная модель для диалога (приоритетнее DEFAULT_MODEL).
  REWRITE_MODEL  — маленькая/быстрая модель для query rewriting перед RAG.
                   Используется из .env как основной источник для mini-LLM.
                   Если не задана, используется API_MODEL.
  OPENAI_API_KEY — для облачного OpenAI.
  OLLAMA_URL     — base URL для Ollama (http://localhost:11434/v1).
  RAG_DISTANCE_THRESHOLD  — жёсткий порог уверенного попадания (косинусное расстояние, 0..∞).
  RAG_SOFT_DISTANCE_MAX   — мягкий порог: если лучший хит хуже жёсткого, но не хуже этого,
                             позиции всё равно подставляются в контекст с предупреждением.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from openai import APITimeoutError, OpenAI, OpenAIError

from mcd_voice.llm.prompts import get_cashier_system_prompt, get_client_system_prompt
from mcd_voice.menu.search import search_menu
from mcd_voice.profile.generator import get_group_allergen_blacklist

# ── Константы ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "gpt-4o"

RAG_DISTANCE_THRESHOLD: float = 0.60
try:
    RAG_SOFT_DISTANCE_MAX: float = float(
        os.environ.get("RAG_SOFT_DISTANCE_MAX") or "1.15"
    )
except (TypeError, ValueError):
    RAG_SOFT_DISTANCE_MAX = 1.15

HistoryEntry = dict[str, str]


# ── LLM-клиент ────────────────────────────────────────────────────────────────

def _normalize_base_url(url: str) -> str:
    base = url.strip().rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    return base


def _resolve_model(explicit: str | None) -> str:
    """Явный аргумент > API_MODEL из env > DEFAULT_MODEL."""
    return explicit or os.environ.get("API_MODEL") or DEFAULT_MODEL


def _build_openai_client(timeout: float = 60.0) -> OpenAI:
    provider = (os.environ.get("API_PROVIDER") or "openai").strip().lower()

    if provider == "ollama":
        raw_url = os.environ.get("OLLAMA_URL") or os.environ.get("OPENAI_BASE_URL")
        if not raw_url:
            raise RuntimeError(
                "Для API_PROVIDER=ollama задайте OLLAMA_URL "
                "(например http://localhost:11434/v1)."
            )
        return OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "ollama"),
            base_url=_normalize_base_url(raw_url),
            timeout=timeout,
        )

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Задайте OPENAI_API_KEY для облачного OpenAI или "
            "используйте API_PROVIDER=ollama с OLLAMA_URL."
        )
    kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout}
    raw_base = os.environ.get("OPENAI_BASE_URL")
    if raw_base:
        kwargs["base_url"] = _normalize_base_url(raw_base)
    return OpenAI(**kwargs)


def get_llm_runtime_config() -> dict[str, str]:
    """Текущая runtime-конфигурация LLM из env (для логов и отладки)."""
    provider = (os.environ.get("API_PROVIDER") or "openai").strip().lower()
    model = _resolve_model(None)
    raw_url = os.environ.get("OLLAMA_URL") or os.environ.get("OPENAI_BASE_URL") or ""
    return {
        "provider": provider,
        "model": model,
        "base_url": _normalize_base_url(raw_url) if raw_url else "",
    }


# ── Низкоуровневые helpers ────────────────────────────────────────────────────

def _call_llm(
    client: OpenAI,
    model: str,
    system: str,
    messages: list[dict[str, str]],
    temperature: float = 0.8,
) -> str:
    """Один Chat Completions вызов; бросает RuntimeError при пустом ответе."""
    payload = [{"role": "system", "content": system}, *messages]
    timeout_err: APITimeoutError | None = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=payload,
                temperature=temperature,
            )
            content = resp.choices[0].message.content
            if not content:
                raise RuntimeError("Пустой ответ от модели.")
            return content.strip()
        except APITimeoutError as exc:
            timeout_err = exc
            if attempt < 2:
                # brief backoff for transient local Ollama stalls
                time.sleep(0.7 * (attempt + 1))
                continue
            raise RuntimeError(f"Таймаут OpenAI API: {exc}") from exc
        except OpenAIError as exc:
            raise RuntimeError(f"Ошибка OpenAI API: {exc}") from exc
    # unreachable, but keeps type checkers happy
    raise RuntimeError(f"Таймаут OpenAI API: {timeout_err}")


def _history_to_messages(
    history: list[HistoryEntry],
    *,
    my_role: str,
) -> list[dict[str, str]]:
    """
    Преобразует внутреннюю историю в формат OpenAI messages.
    Реплики говорящего агента → assistant, остальные → user.
    """
    return [
        {
            "role": "assistant" if entry.get("speaker") == my_role else "user",
            "content": entry.get("text") or "",
        }
        for entry in history
        if entry.get("text") is not None
    ]


# ── ClientAgent ───────────────────────────────────────────────────────────────

class ClientAgent:
    """Агент-клиент: генерирует реплики на основе профиля и истории диалога."""

    def __init__(
        self,
        model: str | None = None,
        timeout: float = 60.0,
        *,
        trace_verbose: bool = False,
    ) -> None:
        self.model = _resolve_model(model)
        self._client = _build_openai_client(timeout)
        self._trace_verbose = trace_verbose

    def generate_response(
        self,
        profile: dict[str, Any],
        history: list[HistoryEntry],
        *,
        llm_trace: list[dict[str, Any]] | None = None,
    ) -> str:
        system = get_client_system_prompt(profile)
        if not history:
            seed = (
                "You just pulled up to the drive-through speaker. "
                "Say one short opening line as the customer."
            )
            messages: list[dict[str, str]] = [{"role": "user", "content": seed}]
        else:
            messages = _history_to_messages(history, my_role="client")
        try:
            t0 = time.perf_counter()
            response = _call_llm(self._client, self.model, system, messages)
            dt_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as exc:
            _trace(
                llm_trace,
                _llm_error_payload(
                    agent="client",
                    model=self.model,
                    messages=messages,
                    system=system,
                    error=str(exc),
                    verbose=self._trace_verbose,
                ),
            )
            raise
        _trace(
            llm_trace,
            _llm_call_payload(
                agent="client",
                model=self.model,
                system=system,
                messages=messages,
                response=response,
                duration_ms=dt_ms,
                verbose=self._trace_verbose,
            ),
        )
        return response


# ── CashierAgent ──────────────────────────────────────────────────────────────

class CashierAgent:
    """
    Агент-кассир: формирует ответы с учётом профиля клиента, истории,
    текущего заказа и семантического поиска по меню (RAG).
    """

    _FALLBACK_QUERY = "popular menu items burger chicken fries"

    def __init__(
        self,
        model: str | None = None,
        timeout: float = 60.0,
        rag_top_k: int = 3,
        distance_threshold: float = RAG_DISTANCE_THRESHOLD,
        rewrite_model: str | None = None,
        *,
        trace_verbose: bool = False,
    ) -> None:
        self.model = _resolve_model(model)
        self.rag_top_k = rag_top_k
        self.distance_threshold = distance_threshold
        self._client = _build_openai_client(timeout)
        self._trace_verbose = trace_verbose
        # Mini-LLM для query rewriting: используем REWRITE_MODEL из .env.
        # Параметр rewrite_model оставлен для обратной совместимости, но не применяется.
        _ = rewrite_model
        self._rewrite_model = os.environ.get("REWRITE_MODEL") or self.model

    # ── Публичный интерфейс ───────────────────────────────────────────────────

    def generate_response(
        self,
        profile: dict[str, Any],
        history: list[HistoryEntry],
        order_state: dict[str, Any],
        query: str | None = None,
        *,
        rag_trace: list[dict[str, Any]] | None = None,
        rag_meta: dict[str, Any] | None = None,
        llm_trace: list[dict[str, Any]] | None = None,
    ) -> str:
        client_text = query or _last_client_text(history)
        rag_context = self._resolve_rag_context(
            client_text,
            profile,
            rag_trace=rag_trace,
            rag_meta=rag_meta,
            llm_trace=llm_trace,
        )
        system = self._build_system(profile, order_state, rag_context)
        messages = _history_to_messages(history, my_role="cashier")
        try:
            t0 = time.perf_counter()
            response = _call_llm(self._client, self.model, system, messages)
            dt_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as exc:
            _trace(
                llm_trace,
                _llm_error_payload(
                    agent="cashier",
                    model=self.model,
                    messages=messages,
                    system=system,
                    error=str(exc),
                    verbose=self._trace_verbose,
                ),
            )
            raise
        _trace(
            llm_trace,
            _llm_call_payload(
                agent="cashier",
                model=self.model,
                system=system,
                messages=messages,
                response=response,
                duration_ms=dt_ms,
                verbose=self._trace_verbose,
            ),
        )
        return response

    # ── RAG ───────────────────────────────────────────────────────────────────

    def _resolve_rag_context(
        self,
        client_text: str,
        profile: dict[str, Any],
        *,
        rag_trace: list[dict[str, Any]] | None,
        rag_meta: dict[str, Any] | None,
        llm_trace: list[dict[str, Any]] | None,
    ) -> str:
        """
        Запускает семантический поиск по меню и возвращает контекст для промпта.

        Нет keyword-фильтрации: запрос всегда идёт в Chroma.
        Если семантическое расстояние слишком большое — контекст пустой
        (модель не засоряется нерелевантным меню).
        Если клиент ещё не говорил (приветствие) — используется fallback-запрос.
        """
        base_trace = {**(rag_meta or {}), "client_query": client_text[:800]}

        if self.rag_top_k <= 0:
            _trace(rag_trace, {**base_trace, "event": "rag_disabled", "rag_top_k": 0})
            return ""

        if client_text:
            search_query = _rewrite_query(
                client_text,
                self._client,
                self._rewrite_model,
                llm_trace=llm_trace,
                trace_meta=rag_meta,
                trace_verbose=self._trace_verbose,
            )
            is_fallback = False
        else:
            search_query = self._FALLBACK_QUERY
            is_fallback = True

        context, info = self._do_rag(
            search_query, profile, rag_trace=rag_trace,
        )
        _trace(rag_trace, {
            **base_trace,
            "event": "rag",
            "search_query": search_query,
            "rewrite_model": self._rewrite_model,
            **({"fallback": True} if is_fallback else {}),
            **info,
            "context_preview": _preview(context),
        })
        return context

    def _do_rag(
        self,
        text: str,
        profile: dict[str, Any],
        *,
        rag_trace: list[dict[str, Any]] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Семантический поиск в Chroma с фильтрацией по аллергенам группы.
        Возвращает (текст для system prompt, метаданные для rag_trace).
        Применяет два порога: жёсткий (уверенный hit) и мягкий (fallback с предупреждением).
        """
        blacklist = get_group_allergen_blacklist(profile)
        chroma_buf: list[dict[str, Any]] = []
        rows = search_menu(
            text,
            allergens_blacklist=blacklist or None,
            top_k=self.rag_top_k,
            chroma_trace=chroma_buf if rag_trace is not None else None,
        )
        for ev in chroma_buf:
            _trace(rag_trace, ev)
        soft_max = RAG_SOFT_DISTANCE_MAX
        info: dict[str, Any] = {
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "metric": "cosine_distance",
            "top_k": self.rag_top_k,
            "distance_threshold": self.distance_threshold,
            "soft_distance_max": soft_max,
            "allergen_blacklist_tokens": list(blacklist),
            "candidates": [
                {"name": r["name"], "distance": r["distance"], "energy": r["energy"]}
                for r in rows
            ],
        }

        if not rows:
            info["outcome"] = "no_chroma_hits"
            return "(no matching menu items found for this request)", info

        best = rows[0]["distance"]
        info["best_distance"] = best

        if best <= self.distance_threshold:
            lines, used = _render_rows(rows, max_dist=self.distance_threshold)
            info.update(outcome="injected", injected_hits=used)
            return "\n".join(lines), info

        if best <= soft_max:
            lines, used = _render_rows(rows, max_dist=soft_max)
            if lines:
                info.update(outcome="injected_soft", injected_hits=used)
                return _SOFT_PREFIX + "\n".join(lines), info

        info["outcome"] = "above_threshold"
        return "(no matching menu items found — closest match too distant)", info

    @staticmethod
    def _build_system(
        profile: dict[str, Any],
        order_state: dict[str, Any],
        rag_context: str,
    ) -> str:
        """Собирает системный промпт: базовый + опциональные блоки RAG и заказа."""
        base = get_cashier_system_prompt(profile)
        extras: list[str] = []
        if rag_context:
            extras.append(
                "Menu data slice for this turn (each line: product name, kcal estimate, "
                "declared-allergen tags only — not ingredients, sugar, or full nutrition):\n"
                f"{rag_context}"
            )
        if any(p.get("items") for p in order_state.get("persons", [])):
            extras.append(
                "Current order state:\n"
                + json.dumps(order_state, ensure_ascii=False)
            )
        if not extras:
            return base
        return base + "\n\n--- Context ---\n" + "\n\n".join(extras)


# ── Приватные helpers ─────────────────────────────────────────────────────────

def _last_client_text(history: list[HistoryEntry]) -> str:
    for entry in reversed(history):
        if entry["speaker"] == "client":
            return entry["text"]
    return ""


def _render_rows(
    rows: list[dict[str, Any]],
    *,
    max_dist: float,
) -> tuple[list[str], list[dict[str, Any]]]:
    """
    Форматирует строки меню для промпта и список использованных хитов.

    mcd.json содержит несколько вариантов одного блюда (разные размеры порций).
    Дедупликация по имени объединяет их в одну строку с диапазоном калорий,
    чтобы не занимать лишние слоты top_k одинаковыми позициями.
    """
    # Собираем квалифицированные строки (в пределах порога)
    by_name: dict[str, dict[str, Any]] = {}
    name_energies: dict[str, list[float]] = {}
    name_added_sugar: dict[str, list[float]] = {}
    name_total_sugar: dict[str, list[float]] = {}
    order: list[str] = []  # сохраняем порядок первого вхождения
    for r in rows:
        if r["distance"] > max_dist:
            continue
        name = r["name"]
        if name not in by_name:
            by_name[name] = r
            name_energies[name] = []
            name_added_sugar[name] = []
            name_total_sugar[name] = []
            order.append(name)
        name_energies[name].append(float(r["energy"]))
        if r.get("added_sugar") is not None:
            name_added_sugar[name].append(float(r["added_sugar"]))
        if r.get("total_sugar") is not None:
            name_total_sugar[name].append(float(r["total_sugar"]))

    lines: list[str] = []
    used: list[dict[str, Any]] = []
    for name in order:
        r = by_name[name]
        allergens = ", ".join(r["allergens"]) if r["allergens"] else "none listed"
        energies = name_energies[name]
        if len(energies) > 1:
            lo, hi = round(min(energies), 1), round(max(energies), 1)
            energy_str = f"~{lo}–{hi}"
        else:
            energy_str = f"~{energies[0]}"
        sugar_parts: list[str] = []
        added_vals = name_added_sugar[name]
        total_vals = name_total_sugar[name]
        if added_vals:
            if len(added_vals) > 1:
                sugar_parts.append(
                    f"added sugar: ~{round(min(added_vals), 1)}–{round(max(added_vals), 1)} g"
                )
            else:
                sugar_parts.append(f"added sugar: ~{round(added_vals[0], 1)} g")
        else:
            sugar_parts.append("added sugar: unknown")
        if total_vals:
            if len(total_vals) > 1:
                sugar_parts.append(
                    f"total sugar: ~{round(min(total_vals), 1)}–{round(max(total_vals), 1)} g"
                )
            else:
                sugar_parts.append(f"total sugar: ~{round(total_vals[0], 1)} g")
        else:
            sugar_parts.append("total sugar: unknown")
        lines.append(
            f"- {name} ({energy_str} kcal, allergens: {allergens}; {'; '.join(sugar_parts)})"
        )
        used.append({"name": name, "distance": r["distance"]})
    return lines, used


_SOFT_PREFIX = (
    "Semantic match is weak; below are real menu rows. "
    "Use exact names and calories; do NOT claim the menu is empty or has no "
    "suitable items if the list is non-empty.\n\n"
)


def _rewrite_query(
    client_text: str,
    client: OpenAI,
    model: str,
    *,
    llm_trace: list[dict[str, Any]] | None = None,
    trace_meta: dict[str, Any] | None = None,
    trace_verbose: bool = False,
) -> str:
    """
    Rewrites a conversational customer message into a concise menu search query.

    Example: "I'm lactose intolerant, what can my 6-year-old have?"
             → "children's meal without dairy"

    Falls back to the original text if the LLM call fails.
    """
    system = (
        "You extract a short food search query (3–8 words, English) "
        "from a customer's drive-through message. "
        "Focus on food type, category, or dietary need. "
        "Reply with ONLY the query — no explanation, no punctuation at the end. "
        "If there is no food intent, reply: general menu items"
    )
    messages = [{"role": "user", "content": client_text}]
    try:
        t0 = time.perf_counter()
        rewritten = _call_llm(
            client, model, system,
            messages,
            temperature=0.0,
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        _trace(
            llm_trace,
            {
                "event": "llm_rewrite",
                **(trace_meta or {}),
                "model": model,
                "kind": "mini_llm_menu_query_rewrite",
                "rewrite_duration_ms": round(dt_ms, 2),
                **(
                    {
                        "rewrite_system": system,
                        "rewrite_input": client_text,
                        "rewrite_messages": messages,
                        "rewrite_output": rewritten,
                    }
                    if trace_verbose
                    else {
                        "rewrite_input": _preview(client_text),
                        "messages_preview": _messages_preview(messages),
                        "rewrite_output": _preview(rewritten),
                    }
                ),
            },
        )
        return rewritten
    except Exception:
        _trace(
            llm_trace,
            {
                "event": "llm_rewrite_fallback",
                **(trace_meta or {}),
                "model": model,
                "kind": "mini_llm_menu_query_rewrite",
                **(
                    {
                        "rewrite_system": system,
                        "rewrite_input": client_text,
                        "rewrite_messages": messages,
                        "rewrite_output": client_text,
                    }
                    if trace_verbose
                    else {
                        "rewrite_input": _preview(client_text),
                        "messages_preview": _messages_preview(messages),
                        "rewrite_output": _preview(client_text),
                    }
                ),
            },
        )
        return client_text


def _llm_call_payload(
    *,
    agent: str,
    model: str,
    system: str,
    messages: list[dict[str, str]],
    response: str,
    duration_ms: float,
    verbose: bool,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "event": "llm_call",
        "agent": agent,
        "model": model,
        "kind": "dialog_llm",
        "messages_count": len(messages),
        "duration_ms": round(duration_ms, 2),
    }
    if verbose:
        base["system"] = system
        base["messages"] = [{"role": m["role"], "content": m["content"]} for m in messages]
        base["response"] = response
    else:
        base["system_preview"] = _preview(system)
        base["messages_preview"] = _messages_preview(messages)
        base["response_preview"] = _preview(response)
    return base


def _llm_error_payload(
    *,
    agent: str,
    model: str,
    messages: list[dict[str, str]],
    system: str,
    error: str,
    verbose: bool,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "event": "llm_error",
        "agent": agent,
        "model": model,
        "messages_count": len(messages),
        "error": error,
    }
    if verbose:
        base["system"] = system
        base["messages"] = [{"role": m["role"], "content": m["content"]} for m in messages]
    else:
        base["system_preview"] = _preview(system)
        base["messages_preview"] = _messages_preview(messages)
    return base


def _preview(text: str | None, limit: int = 1200) -> str:
    if not text:
        return ""
    return text[:limit] + "…" if len(text) > limit else text


def _messages_preview(
    messages: list[dict[str, str]],
    *,
    max_items: int = 3,
) -> list[dict[str, str]]:
    tail = messages[-max_items:]
    return [{"role": m["role"], "content": _preview(m["content"], 280)} for m in tail]


def _trace(
    rag_trace: list[dict[str, Any]] | None,
    event: dict[str, Any],
) -> None:
    if rag_trace is not None:
        rag_trace.append(event)
