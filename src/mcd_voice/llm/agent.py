"""
Агенты ClientAgent и CashierAgent поверх OpenAI-compatible Chat Completions.

Конфигурация через переменные окружения:
  API_PROVIDER   — "openai" (по умолчанию) или "ollama".
  API_MODEL      — основная модель для диалога (приоритетнее DEFAULT_MODEL).
  REWRITE_MODEL  — маленькая/быстрая модель для query rewriting перед RAG.
                   Используется из .env как основной источник для mini-LLM.
                   Если не задана, используется API_MODEL.
  LLM_API_KEY    — ключ (любой OpenAI-compatible провайдер: xAI, OpenAI, Groq, …).
  LLM_BASE_URL   — base URL …/v1 (облако или локальный Ollama).
  Дубликаты не нужны: OPENAI_API_KEY, XAI_API_KEY, OPENAI_BASE_URL, XAI_BASE_URL, OLLAMA_URL
  поддерживаются только для старых .env и не рекомендуются к новым проектам.
  RAG_DISTANCE_THRESHOLD  — жёсткий порог уверенного попадания (косинусное расстояние, 0..∞).
  RAG_SOFT_DISTANCE_MAX   — мягкий порог: если лучший хит хуже жёсткого, но не хуже этого,
                             позиции всё равно подставляются в контекст с предупреждением.
  RAG_MAX_PROMPT_LINES    — максимум строк меню в system prompt после порога по distance
                             (по умолчанию 15; порядок = релевантность Chroma). 0 или отрицательное
                             — без ограничения (старое поведение).
  OPENROUTER_PROVIDER_IGNORE — только для LLM_BASE_URL с openrouter.ai: список slug через запятую
                             для provider.ignore (см. OpenRouter routing). Не задано → по умолчанию
                             Venice/venice (частые 403 «Venice.ai is at capacity» на free-маршрутах).
                             Значение none|false|0|off|- отключает передачу ignore.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Sequence
from typing import Any

from openai import APITimeoutError, OpenAI, OpenAIError

from mcd_voice.llm.prompts import get_cashier_system_prompt, get_client_system_prompt
from mcd_voice.menu.graph_rag import search_menu_graph
from mcd_voice.menu.rag_constraints import merge_rag_allergen_blacklist
from mcd_voice.menu.rag_structured import get_rag_json_system_prompt, parse_rag_json_response
from mcd_voice.menu.search import search_menu
from mcd_voice.profile.generator import get_group_allergen_blacklist
from mcd_voice.text_normalization import normalize_item_text as _normalize_item_text

# ── Константы ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "gpt-4o"
# Full-catalog retrieval: mcd.json contains exactly 42 menu positions.
# We intentionally retrieve all rows, then filter by distance/allergens.
RAG_CATALOG_TOP_K = 42
# Experimental narrow setup for A/B checks (not used by default).
RAG_EXPERIMENT_TOP_K = 5
# Backward-compatible alias used across scripts/tests.
RAG_FULL_TOP_K = RAG_CATALOG_TOP_K

RAG_DISTANCE_THRESHOLD: float = 0.60
try:
    RAG_SOFT_DISTANCE_MAX: float = float(
        os.environ.get("RAG_SOFT_DISTANCE_MAX") or "1.15"
    )
except (TypeError, ValueError):
    RAG_SOFT_DISTANCE_MAX = 1.15


def _parse_rag_max_prompt_lines() -> int | None:
    """Сколько уникальных позиций меню включать в промпт; None = без лимита."""
    raw = (os.environ.get("RAG_MAX_PROMPT_LINES") or "15").strip()
    try:
        n = int(raw)
    except ValueError:
        return 15
    if n <= 0:
        return None
    return n


RAG_MAX_PROMPT_LINES: int | None = _parse_rag_max_prompt_lines()
RAG_MODE_VECTOR = "vector"
RAG_MODE_GRAPH = "graph"

HistoryEntry = dict[str, str]


def _use_lexical_exclusions() -> bool:
    """
    Post-filter by lexical terms is optional and disabled by default.
    Default vector-only behavior: rely on embeddings + metadata filters.
    """
    v = (os.environ.get("RAG_USE_LEXICAL_EXCLUDE") or "0").strip().lower()
    return v in ("1", "true", "yes", "on")

def _was_name_already_discussed(name: str, history: list[HistoryEntry]) -> bool:
    n = _normalize_item_text(name)
    if not n:
        return False
    for h in history[-24:]:
        t = _normalize_item_text(h.get("text") or "")
        if not t:
            continue
        if n in t:
            return True
    return False


def _rag_intent(spec: dict[str, Any] | None) -> str:
    raw = str((spec or {}).get("intent") or "lookup").strip().lower()
    if raw in {"lookup", "alternatives", "details", "calorie_tune", "compare"}:
        return raw
    return "lookup"


_NON_FOOD_CLIENT_UTTERANCE_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(yes|yeah|yep|correct|right|okay|ok)\s*[.!?]*\s*$", re.I),
    re.compile(r"^\s*(okay|ok),?\s*(that'?s|thats)\s+all(?:,?\s*thanks?)?\s*[.!?]*\s*$", re.I),
    re.compile(r"^\s*(okay|ok|yes|yeah|yep),?\s*thanks?(?:\s+you)?\s*[.!?]*\s*$", re.I),
    re.compile(r"^\s*(yes|yeah|yep),?\s*(that'?s|thats)\s+right\s*[.!?]*\s*$", re.I),
    re.compile(r"^\s*(that'?s|thats)\s+(right|correct|fine)\s*[.!?]*\s*$", re.I),
    re.compile(r"^\s*(that'?s|thats)\s+all(?:,?\s*thanks?)?\s*[.!?]*\s*$", re.I),
    re.compile(r"^\s*all\s+set(?:,?\s*thanks?)?\s*[.!?]*\s*$", re.I),
    re.compile(r"^\s*thanks?(?:\s+you)?\s*[.!?]*\s*$", re.I),
)


_ORDER_INTENT_RE = re.compile(
    r"\b("
    r"order|add|take|want|need|get|have|choose|pick|recommend|suggest|"
    r"burger|sandwich|nuggets?|fries|salad|drink|coffee|tea|cola|coke|sprite|fanta|"
    r"meal|combo|side|dessert|kcal|calories?|nutrition|protein|fat|carbs|sugar|sodium"
    r")\b",
    re.I,
)


def _is_non_food_client_utterance(text: str) -> bool:
    """True for short confirmation/closure phrases where RAG should be skipped."""
    t = (text or "").strip()
    if not t:
        return False
    # Generic order/menu/nutrition intent should keep RAG enabled.
    if _ORDER_INTENT_RE.search(t):
        return False
    if any(rx.match(t) for rx in _NON_FOOD_CLIENT_UTTERANCE_RES):
        return True
    # Support combined confirmations: "Yeah, that's right. That's all, thanks."
    parts = [p.strip() for p in re.split(r"[.!?]+", t) if p.strip()]
    if not parts:
        return False
    return all(any(rx.match(p) for rx in _NON_FOOD_CLIENT_UTTERANCE_RES) for p in parts)


def _extract_names_from_rag_context(rag_context: str) -> list[str]:
    """Extract menu names from '- Name (...)' lines in RAG context."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in (rag_context or "").splitlines():
        line = raw.strip()
        if not line.startswith("- "):
            continue
        name = line[2:].split("(", maxsplit=1)[0].strip()
        if not name:
            continue
        low = name.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(name)
    return out


def _matches_menu_name_in_text(name: str, text: str) -> bool:
    n = _normalize_item_text(name)
    t = _normalize_item_text(text)
    if not n or not t:
        return False
    return n in t


def _allergen_hits(
    allergens: Sequence[str] | None,
    blacklist: Sequence[str] | None,
) -> list[str]:
    if not allergens or not blacklist:
        return []
    bl = {str(x).strip().lower() for x in blacklist if str(x).strip()}
    if not bl:
        return []
    hits: list[str] = []
    for a in allergens:
        token = str(a).strip()
        if token and token.lower() in bl and token not in hits:
            hits.append(token)
    return hits


def _collect_allergen_excluded_candidates(
    *,
    query: str,
    shown_rows: Sequence[dict[str, Any]],
    blacklist: Sequence[str],
    top_k: int,
    max_energy: float | None,
    min_energy: float | None,
    excluded_lexical: Sequence[str] | None,
) -> list[dict[str, Any]]:
    if not blacklist:
        return []
    try:
        unfiltered_rows = search_menu(
            query,
            allergens_blacklist=None,
            top_k=max(top_k, RAG_CATALOG_TOP_K),
            max_energy=max_energy,
            min_energy=min_energy,
            excluded_lexical=excluded_lexical,
        )
    except Exception:
        return []

    shown_names = {str(r.get("name") or "") for r in shown_rows}
    by_name: dict[str, dict[str, Any]] = {}
    for row in unfiltered_rows:
        name = str(row.get("name") or "")
        if not name or name in shown_names:
            continue
        hits = _allergen_hits(row.get("allergens"), blacklist)
        if not hits:
            continue
        existing = by_name.get(name)
        if existing is None:
            by_name[name] = {
                "name": name,
                "distance": float(row.get("distance") or 0.0),
                "allergens": hits,
                "mentioned_in_query": _matches_menu_name_in_text(name, query),
            }
            continue
        existing["distance"] = min(existing["distance"], float(row.get("distance") or 0.0))
        merged = list(existing.get("allergens") or [])
        for h in hits:
            if h not in merged:
                merged.append(h)
        existing["allergens"] = merged
        existing["mentioned_in_query"] = bool(existing["mentioned_in_query"]) or _matches_menu_name_in_text(name, query)

    out = sorted(
        by_name.values(),
        key=lambda r: (
            0 if bool(r.get("mentioned_in_query")) else 1,
            float(r.get("distance") or 0.0),
            str(r.get("name") or ""),
        ),
    )
    return out[:3]


def _render_excluded_constraints_block(
    excluded_rows: Sequence[dict[str, Any]],
) -> str:
    if not excluded_rows:
        return ""
    lines = [
        "Items excluded by dietary constraints this turn (they may exist on menu but are not suitable):",
    ]
    for row in excluded_rows:
        name = str(row.get("name") or "").strip()
        allergens = ", ".join(str(a) for a in (row.get("allergens") or []) if str(a).strip())
        if not name:
            continue
        if allergens:
            lines.append(f"* {name} (contains: {allergens})")
        else:
            lines.append(f"* {name}")
    return "\n".join(lines) if len(lines) > 1 else ""


def _normalize_rewrite_output(text: str) -> str:
    """
    Normalize mini-LLM rewrite output:
    - strip trailing punctuation,
    - collapse whitespace,
    - guard against meaningless one-word outputs like 'all'.
    """
    cleaned = (text or "").strip().replace(",", " ")
    # Remove label-like punctuation from mini-LLM outputs (e.g. "food type: burgers").
    cleaned = re.sub(r"[:|/\\]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"[.!?,;:]+$", "", cleaned).strip()
    if not cleaned:
        return "general menu items"
    low = cleaned.lower()
    if low in {
        "all", "ok", "okay", "yes", "yeah", "thanks", "thank you",
        "ok thanks", "okay thanks", "ok thank you", "okay thank you",
        "thats all", "that's all", "thats all thanks", "that's all thanks",
    }:
        return "general menu items"
    return cleaned


def _should_skip_rag(client_text: str, search_query: str) -> bool:
    """Decide whether current turn has no food intent and RAG should be skipped."""
    if _is_non_food_client_utterance(client_text):
        return True
    q = (search_query or "").strip().lower()
    if not q:
        return True
    if q == "general menu items":
        return True
    courtesy_tokens = {
        "ok", "okay", "thanks", "thank", "you", "yes", "yeah", "yep",
        "right", "correct", "all", "thats", "that's",
    }
    q_tokens = [tok for tok in re.split(r"\s+", q) if tok]
    if q_tokens and all(tok in courtesy_tokens for tok in q_tokens):
        return True
    return False


def _sanitize_cashier_response(text: str, *, allow_calories: bool = True) -> str:
    """Remove formatting/emoji artifacts forbidden by cashier prompt."""
    cleaned = (text or "").strip()
    if not cleaned:
        return cleaned
    # Markdown-ish emphasis markers.
    cleaned = cleaned.replace("**", "").replace("__", "")
    # Strip common emoji ranges while preserving plain ASCII punctuation/text.
    cleaned = re.sub(
        r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U0001F1E6-\U0001F1FF]",
        "",
        cleaned,
    )
    # Avoid misleading "not in menu" claims when item may exist outside current slice.
    replacements = {
        "not currently in our menu": "not in my current options",
        "not in our menu": "not in my current options",
        "i don't have": "I can't confirm",
        "we don't have": "We can't confirm right now",
        "on the current menu": "in the current options",
    }
    low = cleaned.lower()
    for src, dst in replacements.items():
        if src in low:
            # case-insensitive single replacement preserving surrounding text.
            cleaned = re.sub(re.escape(src), dst, cleaned, flags=re.I)
            low = cleaned.lower()
    cleaned = re.sub(r"\bfoodwould\b", "food would", cleaned, flags=re.I)
    if not allow_calories:
        # Remove unsolicited calories from free-form cashier output.
        cleaned = re.sub(r"\(\s*~?\s*\d[\d.,]*(?:\s*[–-]\s*\d[\d.,]*)?\s*kcal\s*\)", "", cleaned, flags=re.I)
        cleaned = re.sub(r"~?\s*\d[\d.,]*(?:\s*[–-]\s*\d[\d.,]*)?\s*kcal", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# ── LLM-клиент ────────────────────────────────────────────────────────────────

def _normalize_base_url(url: str) -> str:
    base = url.strip().rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    return base


def _first_env(*keys: str) -> str:
    """Первое непустое значение среди переменных окружения."""
    for k in keys:
        v = (os.environ.get(k) or "").strip()
        if v:
            return v
    return ""


def _cloud_api_key() -> str:
    return _first_env("LLM_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY")


def _cloud_base_url() -> str:
    base = _first_env("LLM_BASE_URL", "OPENAI_BASE_URL", "XAI_BASE_URL")
    if base:
        return base
    # Старый сценарий: только XAI_API_KEY без LLM_/OPENAI_ — дефолтный хост xAI.
    if (
        _first_env("XAI_API_KEY")
        and not _first_env("LLM_API_KEY")
        and not _first_env("OPENAI_API_KEY")
    ):
        return "https://api.x.ai/v1"
    return ""


def _base_url_is_openrouter() -> bool:
    """True если облачный base URL указывает на OpenRouter (там есть provider routing)."""
    return "openrouter.ai" in (_cloud_base_url() or "").lower()


def _openrouter_provider_ignore_list() -> list[str] | None:
    """
    Список slug провайдеров OpenRouter для поля provider.ignore.

    ENV:
      OPENROUTER_PROVIDER_IGNORE — через запятую, например: venice,deepinfra
      Пусто (не задано) при LLM_BASE_URL=openrouter → по умолчанию ['venice'],
      чтобы реже ловить 403 «Venice.ai is at capacity» на бесплатных маршрутах.
      none|false|0|off|- — не передавать provider.ignore.
    """
    if not _base_url_is_openrouter():
        return None
    raw = (os.environ.get("OPENROUTER_PROVIDER_IGNORE") or "").strip()
    if raw.lower() in {"none", "false", "0", "off", "-"}:
        return None
    if not raw:
        # slug в метадатах OpenRouter встречается как Venice; в ignore обычно lowercase
        return ["Venice", "venice"]
    out: list[str] = []
    for part in raw.split(","):
        p = part.strip()
        if p and p not in out:
            out.append(p)
    return out or None


def _openrouter_extra_body() -> dict[str, Any] | None:
    """Доп. тело запроса для OpenRouter (Chat Completions)."""
    ign = _openrouter_provider_ignore_list()
    if not ign:
        return None
    return {"provider": {"ignore": ign}}


def _ollama_base_url() -> str:
    return _first_env("LLM_BASE_URL", "OLLAMA_URL", "OPENAI_BASE_URL")


def _resolve_model(explicit: str | None) -> str:
    """Явный аргумент > API_MODEL из env > DEFAULT_MODEL."""
    return explicit or os.environ.get("API_MODEL") or DEFAULT_MODEL


def _build_openai_client(timeout: float = 60.0) -> OpenAI:
    provider = (os.environ.get("API_PROVIDER") or "openai").strip().lower()

    if provider == "ollama":
        raw_url = _ollama_base_url()
        if not raw_url:
            raise RuntimeError(
                "Для API_PROVIDER=ollama задайте LLM_BASE_URL "
                "(например http://127.0.0.1:11434/v1)."
            )
        api_key = _first_env("LLM_API_KEY", "OPENAI_API_KEY") or "ollama"
        return OpenAI(
            api_key=api_key,
            base_url=_normalize_base_url(raw_url),
            timeout=timeout,
        )

    api_key = _cloud_api_key()
    if not api_key:
        raise RuntimeError(
            "Задайте LLM_API_KEY для облака, либо API_PROVIDER=ollama и LLM_BASE_URL. "
            "При необходимости поддерживаются устаревшие OPENAI_* / XAI_* / OLLAMA_URL в .env."
        )
    kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout}
    raw_base = _cloud_base_url()
    if raw_base:
        kwargs["base_url"] = _normalize_base_url(raw_base)
    return OpenAI(**kwargs)


def ensure_llm_credentials() -> None:
    """Проверяет env до параллельной генерации; иначе тот же RuntimeError, что при первом вызове LLM."""
    _build_openai_client()


def get_llm_runtime_config() -> dict[str, str]:
    """Текущая runtime-конфигурация LLM из env (для логов и отладки)."""
    provider = (os.environ.get("API_PROVIDER") or "openai").strip().lower()
    model = _resolve_model(None)
    if provider == "ollama":
        raw_url = _ollama_base_url()
    else:
        raw_url = _cloud_base_url()
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
    openai_err: OpenAIError | None = None
    for attempt in range(3):
        try:
            create_kwargs: dict[str, Any] = {
                "model": model,
                "messages": payload,
                "temperature": temperature,
            }
            xb = _openrouter_extra_body()
            if xb is not None:
                create_kwargs["extra_body"] = xb
            resp = client.chat.completions.create(**create_kwargs)
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
            openai_err = exc
            # Free/cloud routes often emit transient 429/5xx; small retry helps
            # reduce hard dialog failures during batch dataset generation.
            err_text = str(exc).lower()
            is_retryable = (
                "429" in err_text
                or "rate limit" in err_text
                or "timeout" in err_text
                or "temporarily unavailable" in err_text
                or "503" in err_text
                or "502" in err_text
                or "500" in err_text
            )
            if is_retryable and attempt < 2:
                time.sleep(1.0 * (attempt + 1))
                continue
            raise RuntimeError(f"Ошибка OpenAI API: {exc}") from exc
    # unreachable, but keeps type checkers happy
    if timeout_err is not None:
        raise RuntimeError(f"Таймаут OpenAI API: {timeout_err}")
    raise RuntimeError(f"Ошибка OpenAI API: {openai_err}")


def _rewrite_rag_structured_json(
    rag_text: str,
    client: OpenAI,
    model: str,
    *,
    llm_trace: list[dict[str, Any]] | None = None,
    trace_meta: dict[str, Any] | None = None,
    trace_verbose: bool = False,
) -> tuple[dict[str, Any] | None, str | None]:
    """
    Один вызов mini-LLM: JSON с intent, search_query, excluded_allergens,
    excluded_lexical, max/min kcal.
    Успех: (spec, None). Иначе (None, error_message).
    """
    t0 = time.perf_counter()
    try:
        raw = _call_llm(
            client,
            model,
            get_rag_json_system_prompt(),
            [{"role": "user", "content": rag_text}],
            temperature=0.0,
        )
        spec = parse_rag_json_response(raw)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        _trace(
            llm_trace,
            {
                "event": "rag_json_rewrite",
                **(trace_meta or {}),
                "model": model,
                "kind": "rag_structured_json",
                "rag_json_duration_ms": round(dt_ms, 2),
                **(
                    {
                        "raw_response": raw,
                        "spec": spec,
                    }
                    if trace_verbose
                    else {
                        "raw_preview": _preview(raw),
                    }
                ),
            },
        )
        return (spec, None)
    except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        _trace(
            llm_trace,
            {
                "event": "rag_json_parse_error",
                **(trace_meta or {}),
                "model": model,
                "error": str(exc)[:200],
                "rag_json_duration_ms": round(dt_ms, 2),
            },
        )
        return (None, str(exc))
    except (RuntimeError, OSError) as exc:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        _trace(
            llm_trace,
            {
                "event": "rag_json_llm_error",
                **(trace_meta or {}),
                "model": model,
                "error": str(exc)[:200],
                "rag_json_duration_ms": round(dt_ms, 2),
            },
        )
        return (None, str(exc))


def _use_rag_json_rewrite() -> bool:
    v = (os.environ.get("RAG_JSON_REWRITE") or "1").strip().lower()
    return v not in ("0", "false", "no", "off")


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
    Агент-кассир: формирует ответы с учётом истории, заказа и RAG по меню.

    По умолчанию в системный промпт попадают психотип и состав группы из profile
    (удобно для симуляции). При realistic_cashier=True кассир видит только общие
    правила и реплики из диалога; фильтр аллергенов в RAG по профилю отключается.
    """

    def __init__(
        self,
        model: str | None = None,
        timeout: float = 60.0,
        rag_top_k: int = RAG_FULL_TOP_K,
        distance_threshold: float = RAG_DISTANCE_THRESHOLD,
        rewrite_model: str | None = None,
        rag_mode: str = RAG_MODE_VECTOR,
        *,
        trace_verbose: bool = False,
        realistic_cashier: bool = True,
        rag_max_prompt_lines: int | None = None,
    ) -> None:
        self.model = _resolve_model(model)
        # Always query broad vector slice by default; very small top_k leads
        # to artificial menu narrowing (e.g. only fries/coffee).
        if rag_top_k <= 0:
            self.rag_top_k = 0
        else:
            self.rag_top_k = max(int(rag_top_k), RAG_CATALOG_TOP_K)
        self.distance_threshold = distance_threshold
        self.rag_max_prompt_lines = (
            rag_max_prompt_lines
            if rag_max_prompt_lines is not None
            else RAG_MAX_PROMPT_LINES
        )
        mode = (rag_mode or RAG_MODE_VECTOR).strip().lower()
        if mode not in (RAG_MODE_VECTOR, RAG_MODE_GRAPH):
            raise ValueError(f"Unsupported rag_mode: {rag_mode!r}")
        self.rag_mode = mode
        self._client = _build_openai_client(timeout)
        self._trace_verbose = trace_verbose
        self._realistic_cashier = realistic_cashier
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
        rag_context, rag_spec = self._resolve_rag_context(
            client_text,
            profile,
            history,
            rag_trace=rag_trace,
            rag_meta=rag_meta,
            llm_trace=llm_trace,
        )
        intent = _rag_intent(rag_spec)
        allow_calories = (
            intent == "calorie_tune"
            or any(
                str(m.get("field")) == "energy"
                for m in (rag_spec or {}).get("compare_metrics", [])
                if isinstance(m, dict)
            )
            or (rag_spec or {}).get("max_kcal") is not None
            or (rag_spec or {}).get("min_kcal") is not None
        )
        macro_reply = (
            self._deterministic_compare_reply(profile, client_text, rag_spec)
            if intent == "compare"
            else None
        )
        if macro_reply:
            _trace(
                llm_trace,
                {
                    "event": "deterministic_compare_reply",
                    **(rag_meta or {}),
                },
            )
            return macro_reply
        meal_details = (
            self._deterministic_meal_details_reply(client_text, rag_context)
            if intent == "details"
            else None
        )
        if meal_details:
            _trace(
                llm_trace,
                {
                    "event": "deterministic_meal_details_reply",
                    **(rag_meta or {}),
                },
            )
            return meal_details
        tune_reply = (
            self._deterministic_calorie_tuning_reply(profile, order_state, client_text)
            if intent == "calorie_tune"
            else None
        )
        if tune_reply:
            _trace(
                llm_trace,
                {
                    "event": "deterministic_calorie_tuning_reply",
                    **(rag_meta or {}),
                    "target_kcal": profile.get("calApprValue"),
                },
            )
            return tune_reply
        system = self._build_system(
            profile,
            order_state,
            rag_context,
            allow_calories=allow_calories,
        )
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
        response = _sanitize_cashier_response(response, allow_calories=allow_calories)
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

    def _deterministic_meal_details_reply(self, client_text: str, rag_context: str) -> str | None:
        if not (client_text or "").strip():
            return None
        names = _extract_names_from_rag_context(rag_context)
        if not names:
            return None
        target = names[0]
        rows = search_menu(target, top_k=min(max(self.rag_top_k, 8), RAG_CATALOG_TOP_K))
        if not rows:
            return None
        norm_target = re.sub(r"[^a-z0-9]+", " ", target.lower()).strip()
        picked = None
        for r in rows:
            n = re.sub(r"[^a-z0-9]+", " ", str(r.get("name") or "").lower()).strip()
            if n == norm_target:
                picked = r
                break
        if picked is None:
            picked = rows[0]
        name = str(picked.get("name") or target).strip()
        ingredients = str(picked.get("ingredients") or "").strip()
        description = str(picked.get("description") or "").strip()
        allergens = picked.get("allergens") or []
        allergens_text = ", ".join(str(x) for x in allergens if str(x).strip()) or "none listed"
        kcal = float(picked.get("energy") or 0.0)
        parts: list[str] = []
        if description:
            parts.append(f"{name}: {description}")
        else:
            parts.append(f"{name}: this is one of our available menu items.")
        if ingredients:
            parts.append(f"Ingredients: {ingredients}.")
        parts.append(f"Allergens: {allergens_text}.")
        if kcal > 0:
            parts.append(f"Approx. {kcal:.0f} kcal.")
        parts.append("Would you like to add it to your order?")
        return " ".join(parts)

    def _deterministic_calorie_tuning_reply(
        self,
        profile: dict[str, Any],
        order_state: dict[str, Any],
        client_text: str,
    ) -> str | None:
        if not (client_text or "").strip():
            return None
        try:
            target = float(profile.get("calApprValue") or 0.0)
        except (TypeError, ValueError):
            return None
        if target <= 0:
            return None
        persons = order_state.get("persons", []) or []
        current = float(sum(float(p.get("total_energy") or 0.0) for p in persons))
        delta = round(target - current, 1)
        if abs(delta) <= 40:
            return (
                f"You are already very close to your target: about {current:.0f} kcal "
                f"vs target {target:.0f} kcal. We can keep the order as-is."
            )
        base: list[str] = [] if self._realistic_cashier else get_group_allergen_blacklist(profile)
        blacklist, _ = merge_rag_allergen_blacklist(base, [client_text])
        rows = search_menu(
            "light sides drinks snacks simple add-on options",
            allergens_blacklist=blacklist or None,
            top_k=max(self.rag_top_k, RAG_CATALOG_TOP_K),
        )
        if not rows:
            return None
        current_items = {
            str(it.get("name", "")).strip()
            for p in persons
            for it in (p.get("items", []) or [])
            if str(it.get("name", "")).strip()
        }
        dedup: dict[str, dict[str, Any]] = {}
        for r in rows:
            name = str(r.get("name") or "").strip()
            if not name:
                continue
            if name in dedup and float(dedup[name].get("distance", 9.0)) <= float(r.get("distance", 9.0)):
                continue
            dedup[name] = r
        candidates = [
            r for r in dedup.values()
            if float(r.get("energy") or 0.0) > 0.0 and str(r.get("name") or "").strip() not in current_items
        ]
        if not candidates:
            candidates = [r for r in dedup.values() if float(r.get("energy") or 0.0) > 0.0]
        if not candidates:
            return None
        if delta > 0:
            best = min(candidates, key=lambda r: abs(float(r.get("energy") or 0.0) - delta))
            add_kcal = float(best.get("energy") or 0.0)
            return (
                f"To stay near your target with minimal change, add 1 {best['name']} "
                f"(~{add_kcal:.0f} kcal). This moves the order from ~{current:.0f} to "
                f"~{current + add_kcal:.0f} kcal."
            )
        reduce = abs(delta)
        best_low = min(candidates, key=lambda r: abs(float(r.get("energy") or 0.0) - reduce))
        low_kcal = float(best_low.get("energy") or 0.0)
        return (
            f"You are above target by about {reduce:.0f} kcal. Minimal tweak: replace one higher-calorie "
            f"item with {best_low['name']} (~{low_kcal:.0f} kcal) to bring total closer to "
            f"{target:.0f} kcal."
        )

    def _deterministic_compare_reply(
        self,
        profile: dict[str, Any],
        client_text: str,
        rag_spec: dict[str, Any] | None,
    ) -> str | None:
        metrics = [
            m for m in (rag_spec or {}).get("compare_metrics", [])
            if isinstance(m, dict) and m.get("field") and m.get("goal")
        ]
        if not metrics:
            return None
        search_query = str((rag_spec or {}).get("search_query") or "").strip()
        if not search_query:
            return None
        base: list[str] = [] if self._realistic_cashier else get_group_allergen_blacklist(profile)
        blacklist, _ = merge_rag_allergen_blacklist(base, [client_text])
        rows = search_menu(
            search_query,
            allergens_blacklist=blacklist or None,
            max_energy=float((rag_spec or {}).get("max_kcal"))
            if (rag_spec or {}).get("max_kcal") is not None else None,
            min_energy=float((rag_spec or {}).get("min_kcal"))
            if (rag_spec or {}).get("min_kcal") is not None else None,
            top_k=max(self.rag_top_k, RAG_CATALOG_TOP_K),
        )
        if not rows:
            return None
        uniq: dict[str, dict[str, Any]] = {}
        for r in rows:
            name = str(r.get("name") or "").strip()
            if not name:
                continue
            cur = uniq.get(name)
            if cur is None or float(r.get("distance") or 9.0) < float(cur.get("distance") or 9.0):
                uniq[name] = r
        cands = list(uniq.values())
        if not cands:
            return None
        def _metric_val(row: dict[str, Any], field: str) -> float:
            try:
                return float(row.get(field) or 0.0)
            except (TypeError, ValueError):
                return 0.0

        def _sort_key(row: dict[str, Any]) -> tuple[float, ...]:
            vals: list[float] = []
            for m in metrics:
                fld = str(m.get("field"))
                goal = str(m.get("goal"))
                val = _metric_val(row, fld)
                vals.append(-val if goal == "max" else val)
            vals.append(float(row.get("distance") or 9.0))
            return tuple(vals)

        cands.sort(key=_sort_key)
        best = cands[0]
        alt = cands[1] if len(cands) > 1 else None
        field_units = {
            "energy": "kcal",
            "protein": "g",
            "total_fat": "g",
            "sat_fat": "g",
            "trans_fat": "g",
            "chol": "mg",
            "carbs": "g",
            "total_sugar": "g",
            "added_sugar": "g",
            "sodium": "mg",
        }
        field_labels = {
            "energy": "energy",
            "protein": "protein",
            "total_fat": "fat",
            "sat_fat": "saturated fat",
            "trans_fat": "trans fat",
            "chol": "cholesterol",
            "carbs": "carbs",
            "total_sugar": "total sugar",
            "added_sugar": "added sugar",
            "sodium": "sodium",
        }

        def _metrics_text(row: dict[str, Any]) -> str:
            parts: list[str] = []
            for m in metrics:
                fld = str(m.get("field"))
                v = _metric_val(row, fld)
                parts.append(f"{field_labels.get(fld, fld)} ~{v:.0f} {field_units.get(fld, '')}".strip())
            return ", ".join(parts)

        msg = f"Best fit is {best['name']}: {_metrics_text(best)}."
        if alt:
            msg += f" Another option is {alt['name']}: {_metrics_text(alt)}."
        msg += " Would you like to go with the first one?"
        return msg

    # ── RAG ───────────────────────────────────────────────────────────────────

    def _resolve_rag_context(
        self,
        client_text: str,
        profile: dict[str, Any],
        history: list[HistoryEntry],
        *,
        rag_trace: list[dict[str, Any]] | None,
        rag_meta: dict[str, Any] | None,
        llm_trace: list[dict[str, Any]] | None,
    ) -> tuple[str, dict[str, Any] | None]:
        """
        Запускает семантический поиск по меню и возвращает контекст для промпта.

        Нет keyword-фильтрации: запрос всегда идёт в Chroma.
        Если семантическое расстояние слишком большое — контекст пустой
        (модель не засоряется нерелевантным меню).
        Если клиент ещё не говорил (приветствие) — используется fallback-запрос.
        """
        base_trace = {**(rag_meta or {}), "client_query": client_text[:800]}
        rag_text = (client_text or "").strip()

        if self.rag_top_k <= 0:
            _trace(
                rag_trace,
                {
                    **base_trace,
                    "event": "rag_disabled",
                    "rag_top_k": 0,
                    "retrieval_mode": self.rag_mode,
                },
            )
            return ("", None)

        if client_text:
            rag_json_spec: dict[str, Any] | None = None
            if _use_rag_json_rewrite():
                jspec, _jerr = _rewrite_rag_structured_json(
                    rag_text,
                    self._client,
                    self._rewrite_model,
                    llm_trace=llm_trace,
                    trace_meta=rag_meta,
                    trace_verbose=self._trace_verbose,
                )
                if jspec is not None:
                    rag_json_spec = dict(jspec)
            if rag_json_spec is not None:
                search_query = str(rag_json_spec.get("search_query", "")).strip()
                if not search_query:
                    rag_json_spec = None
            if rag_json_spec is None:
                search_query = _normalize_rewrite_output(rag_text)
            is_fallback = False
            if _should_skip_rag(client_text, search_query):
                _trace(
                    rag_trace,
                    {
                        **base_trace,
                        "event": "rag_skipped_non_food",
                        "search_query": search_query,
                        "rewrite_model": self._rewrite_model,
                        "retrieval_mode": self.rag_mode,
                    },
                )
                return ("", rag_json_spec)
        else:
            _trace(
                rag_trace,
                {
                    **base_trace,
                    "event": "rag_skipped_no_client_query",
                    "retrieval_mode": self.rag_mode,
                },
            )
            return ("", None)

        ctexts: tuple[str, ...] = (
            (str(rag_text), str(search_query))
            if (client_text and not is_fallback)
            else (str(search_query),)
        )
        rj = locals().get("rag_json_spec")
        prefer_novel = _rag_intent(rj) == "alternatives"
        context, info = self._do_rag(
            search_query,
            profile,
            rag_constraint_texts=ctexts,
            rag_json_spec=rj if client_text and not is_fallback else None,
            history=history,
            prefer_novel=prefer_novel,
            rag_trace=rag_trace,
        )
        _trace(rag_trace, {
            **base_trace,
            "event": "rag",
            "search_query": search_query,
            "rewrite_model": self._rewrite_model,
            **({"fallback": True} if is_fallback else {}),
            "retrieval_mode": self.rag_mode,
            **(
                {
                    "rag_json_spec": {
                        "intent": rj.get("intent"),
                        "compare_metrics": rj.get("compare_metrics", []),
                        "excluded_allergens": rj.get("excluded_allergens", []),
                        "excluded_lexical": rj.get("excluded_lexical", []),
                        "max_kcal": rj.get("max_kcal"),
                        "min_kcal": rj.get("min_kcal"),
                    }
                }
                if rj
                else {}
            ),
            **info,
            "context_preview": _preview(context),
        })
        return (context, rj if isinstance(rj, dict) else None)

    def _do_rag(
        self,
        text: str,
        profile: dict[str, Any],
        *,
        rag_constraint_texts: Sequence[str] = (),
        rag_json_spec: dict[str, Any] | None = None,
        history: list[HistoryEntry] | None = None,
        prefer_novel: bool = False,
        rag_trace: list[dict[str, Any]] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Поиск по меню с фильтрацией по аллергенам группы.
        Возвращает (текст для system prompt, метаданные для rag_trace).
        """
        if self.rag_mode == RAG_MODE_GRAPH:
            return self._do_graph_rag(
                text,
                profile,
                rag_constraint_texts=rag_constraint_texts,
                rag_json_spec=rag_json_spec,
            )

        base: list[str] = (
            []
            if self._realistic_cashier
            else get_group_allergen_blacklist(profile)
        )
        if rag_json_spec is not None:
            u = set(base) | set(rag_json_spec.get("excluded_allergens", []) or [])
            blacklist = sorted(u)
            cmeta = {
                "utterance_allergen_exclusions": [],
                "rag_json_excluded": list(rag_json_spec.get("excluded_allergens", [])),
                "rag_json_lexical": list(rag_json_spec.get("excluded_lexical", []) or []),
            }
        else:
            blacklist, cmeta = merge_rag_allergen_blacklist(base, rag_constraint_texts)
        max_e = rag_json_spec.get("max_kcal") if rag_json_spec else None
        min_e = rag_json_spec.get("min_kcal") if rag_json_spec else None
        lex_e = (
            list(rag_json_spec.get("excluded_lexical") or [])
            if (rag_json_spec and _use_lexical_exclusions())
            else []
        )
        chroma_buf: list[dict[str, Any]] = []
        rows = search_menu(
            text,
            allergens_blacklist=blacklist or None,
            top_k=self.rag_top_k,
            max_energy=float(max_e) if max_e is not None else None,
            min_energy=float(min_e) if min_e is not None else None,
            excluded_lexical=lex_e or None,
            chroma_trace=chroma_buf if rag_trace is not None else None,
        )
        excluded_by_constraints = _collect_allergen_excluded_candidates(
            query=text,
            shown_rows=rows,
            blacklist=blacklist,
            top_k=self.rag_top_k,
            max_energy=float(max_e) if max_e is not None else None,
            min_energy=float(min_e) if min_e is not None else None,
            excluded_lexical=lex_e or None,
        )
        excluded_block = _render_excluded_constraints_block(excluded_by_constraints)
        if prefer_novel and rows:
            seen: list[dict[str, Any]] = []
            fresh: list[dict[str, Any]] = []
            for r in rows:
                nm = str(r.get("name") or "")
                if _was_name_already_discussed(nm, history or []):
                    seen.append(r)
                else:
                    fresh.append(r)
            if fresh:
                rows = [*fresh, *seen]
        for ev in chroma_buf:
            _trace(rag_trace, ev)
        soft_max = RAG_SOFT_DISTANCE_MAX
        info: dict[str, Any] = {
            "retrieval_mode": RAG_MODE_VECTOR,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "metric": "cosine_distance",
            "top_k": self.rag_top_k,
            "distance_threshold": self.distance_threshold,
            "soft_distance_max": soft_max,
            "allergen_blacklist_tokens": list(blacklist),
            "prefer_novel": bool(prefer_novel),
            **cmeta,
            "candidates": [
                {"name": r["name"], "distance": r["distance"], "energy": r["energy"]}
                for r in rows
            ],
            "excluded_by_constraints": [
                {
                    "name": str(r.get("name") or ""),
                    "distance": float(r.get("distance") or 0.0),
                    "allergens": list(r.get("allergens") or []),
                    "mentioned_in_query": bool(r.get("mentioned_in_query")),
                }
                for r in excluded_by_constraints
            ],
        }

        if not rows:
            info["outcome"] = "no_chroma_hits"
            base_context = "(no matching menu items found for this request)"
            if excluded_block:
                return base_context + "\n\n" + excluded_block, info
            return base_context, info

        best = rows[0]["distance"]
        info["best_distance"] = best

        if best <= self.distance_threshold:
            lines, used = _render_rows(
                rows,
                max_dist=self.distance_threshold,
                max_lines=self.rag_max_prompt_lines,
            )
            if self.rag_max_prompt_lines is not None:
                info["rag_prompt_line_cap"] = self.rag_max_prompt_lines
                info["rag_prompt_lines_included"] = len(lines)
            info.update(outcome="injected", injected_hits=used)
            context = "\n".join(lines)
            if excluded_block:
                context += "\n\n" + excluded_block
            return context, info

        if best <= soft_max:
            lines, used = _render_rows(
                rows,
                max_dist=soft_max,
                max_lines=self.rag_max_prompt_lines,
            )
            if lines:
                if self.rag_max_prompt_lines is not None:
                    info["rag_prompt_line_cap"] = self.rag_max_prompt_lines
                    info["rag_prompt_lines_included"] = len(lines)
                info.update(outcome="injected_soft", injected_hits=used)
                context = _SOFT_PREFIX + "\n".join(lines)
                if excluded_block:
                    context += "\n\n" + excluded_block
                return context, info

        info["outcome"] = "above_threshold"
        base_context = "(no matching menu items found — closest match too distant)"
        if excluded_block:
            return base_context + "\n\n" + excluded_block, info
        return base_context, info

    def _do_graph_rag(
        self,
        text: str,
        profile: dict[str, Any],
        *,
        rag_constraint_texts: Sequence[str] = (),
        rag_json_spec: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        base: list[str] = (
            []
            if self._realistic_cashier
            else get_group_allergen_blacklist(profile)
        )
        if rag_json_spec is not None:
            u = set(base) | set(rag_json_spec.get("excluded_allergens", []) or [])
            blacklist = sorted(u)
            cmeta = {
                "utterance_allergen_exclusions": [],
                "rag_json_excluded": list(rag_json_spec.get("excluded_allergens", [])),
                "rag_json_lexical": list(rag_json_spec.get("excluded_lexical", []) or []),
            }
        else:
            blacklist, cmeta = merge_rag_allergen_blacklist(base, rag_constraint_texts)
        max_e = rag_json_spec.get("max_kcal") if rag_json_spec else None
        min_e = rag_json_spec.get("min_kcal") if rag_json_spec else None
        lex_e = (
            list(rag_json_spec.get("excluded_lexical") or [])
            if rag_json_spec
            else []
        )
        rows, graph_info = search_menu_graph(
            text,
            allergens_blacklist=blacklist or None,
            top_k=self.rag_top_k,
            max_energy=float(max_e) if max_e is not None else None,
            min_energy=float(min_e) if min_e is not None else None,
            excluded_lexical=lex_e or None,
        )
        info: dict[str, Any] = {
            "top_k": self.rag_top_k,
            "allergen_blacklist_tokens": list(blacklist),
            **cmeta,
            "candidates": [
                {"name": r["name"], "distance": r["distance"], "energy": r["energy"]}
                for r in rows
            ],
            **graph_info,
        }
        if not rows:
            info["outcome"] = "no_graph_hits"
            return "(no matching menu items found for this request)", info

        lines, used = _render_rows(
            rows,
            max_dist=1.0,
            max_lines=self.rag_max_prompt_lines,
        )
        if self.rag_max_prompt_lines is not None:
            info["rag_prompt_line_cap"] = self.rag_max_prompt_lines
            info["rag_prompt_lines_included"] = len(lines)
        info.update(outcome="injected_graph", injected_hits=used)
        return "\n".join(lines), info

    def _build_system(
        self,
        profile: dict[str, Any],
        order_state: dict[str, Any],
        rag_context: str,
        *,
        allow_calories: bool,
    ) -> str:
        """Собирает системный промпт: базовый + опциональные блоки RAG и заказа."""
        base = get_cashier_system_prompt(profile, realistic=self._realistic_cashier)
        extras: list[str] = []
        if rag_context:
            context_payload = (
                rag_context if allow_calories else _hide_kcal_in_rag_context(rag_context)
            )
            extras.append(
                "Menu data slice for this turn (each line: product name, kcal estimate, "
                "allergen tags, approximate added/total sugar — not full ingredients or full "
                "nutrition):\n"
                f"{context_payload}"
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
    max_lines: int | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """
    Форматирует строки меню для промпта и список использованных хитов.

    mcd.json содержит несколько вариантов одного блюда (разные размеры порций).
    Дедупликация по имени объединяет их в одну строку с диапазоном калорий,
    чтобы не занимать лишние слоты top_k одинаковыми позициями.

    max_lines: обрезка после порога по distance (порядок = релевантность в rows).
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
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        used = used[:max_lines]
    return lines, used


def _hide_kcal_in_rag_context(rag_context: str) -> str:
    out_lines: list[str] = []
    for raw in (rag_context or "").splitlines():
        line = re.sub(
            r"~?\d[\d.,]*(?:\s*[–-]\s*\d[\d.,]*)?\s*kcal,\s*",
            "",
            raw,
            flags=re.I,
        )
        line = re.sub(
            r"\(\s*~?\d[\d.,]*(?:\s*[–-]\s*\d[\d.,]*)?\s*kcal\s*\)",
            "",
            line,
            flags=re.I,
        )
        out_lines.append(re.sub(r"\s+", " ", line).strip())
    return "\n".join(out_lines)


_SOFT_PREFIX = (
    "Semantic match is weak; below are real menu rows. "
    "Use exact names and calories; do NOT claim the menu is empty or has no "
    "suitable items if the list is non-empty.\n\n"
)


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
