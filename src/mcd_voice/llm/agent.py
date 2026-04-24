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
from typing import Any

from openai import APITimeoutError, OpenAI, OpenAIError

from mcd_voice.llm.prompts import get_cashier_system_prompt, get_client_system_prompt
from mcd_voice.menu.graph_rag import search_menu_graph
from mcd_voice.menu.search import search_menu
from mcd_voice.profile.generator import get_group_allergen_blacklist

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

# RAG: клиент просит «что ещё / другие позиции», а не уточнение одного блюда.
_BROAD_MENU_RAG_QUERY = (
    "popular menu items burgers chicken sandwiches sides drinks dessert coffee"
)

# RAG: клиент просит *другие* варианты кофе — узкий rewrite («coffee») тянет одни и те же хиты.
_COFFEE_BROAD_RAG_QUERY = (
    "McCafe coffee menu black coffee cold coffee iced coffee premium roast "
    "French vanilla latte caramel macchiato mocha frappe caramel frappe "
    "cold coffee McFloat"
)

_COFFEE_VARIETY_INTENT_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(another|different|other)\s+coffee\b", re.I),
    re.compile(r"\bother\s+kinds?\s+of\s+coffee\b", re.I),
    re.compile(r"\bother\s+coffee\b", re.I),
    re.compile(r"\bany\s+other\b.*\bcoffee\b", re.I),
    re.compile(r"\bcoffee\b.*\bany\s+other\b", re.I),
    re.compile(r"\bwhat\s+(other|kinds?|types?|sorts?)\b.*\bcoffee\b", re.I),
    re.compile(r"\bkinds?\s+of\s+coffee\b", re.I),
    re.compile(r"\bbesides\b.*\bcoffee\b|\bcoffee\b.*\bbesides\b", re.I),
    # «coffee other …», «other … coffee», variants
    re.compile(
        r"\bcoffee\b.*\b(other|more|different|variants?|options?)\b|"
        r"\b(other|more|different)\b.*\bcoffee\b|\bvariants?\b.*\bcoffee\b|\bcoffee\b.*\bvariants?\b",
        re.I,
    ),
    # Russian (профиль часто RU)
    re.compile(r"\b(другой|ещё|еще|иной)\s+кофе\b", re.I),
    re.compile(r"\bкак(ой|ие)\s+(еще|ещё)\s+кофе\b", re.I),
    re.compile(r"\bдруг(ие|их)\s+сорт(а|ов)?\s+кофе\b", re.I),
)

# Недавний диалог уже про кофе — короткое «what else» / «other variants» без слова coffee.
_COFFEE_TOPIC_IN_HISTORY_RE = re.compile(
    r"\bcoffee\b|\blatte\b|\bmacchiato\b|\bmocha\b|\bfrappe\b|\bespresso\b|"
    r"\bpremium\s+roast\b|\broast\s+coffee\b|\bcold\s+coffee\b|\bblack\s+coffee\b|\biced\s+coffee\b",
    re.I,
)

_SHORT_TOPIC_CARRY_RE = re.compile(
    r"^\s*(what\s+else|something\s+else|anything\s+else|maybe\s+something\s+else)\b",
    re.I,
)

_SHORT_OTHER_VARIANTS_RE = re.compile(r"^\s*other\s+variants?\s*[.!?,]?\s*$", re.I)

_COFFEE_COUNT_INTENT_RE = re.compile(
    r"\b(how\s+many|count|number\s+of|сколько|кол-?во|количество)\b",
    re.I,
)
_COFFEE_NAME_HINT_RE = re.compile(
    r"\bcoffee\b|\blatte\b|\bmacchiato\b|\bfrappe\b|\bmocha\b|\bmcfloat\b|\bкофе\b",
    re.I,
)

_MENU_MORE_THAN_LISTED_RE = re.compile(
    r"\b(menu|your\s+menu|in\s+the\s+menu)\b.*\b(more|other)\b|"
    r"\b(more|other)\s+variants?\b.*\b(menu|list)\b|"
    r"\bsee\s+.*\b(more|other)\b.*\bvariants?\b",
    re.I,
)


def _enrich_client_text_for_menu_rag(
    history: list[HistoryEntry],
    client_text: str,
) -> str:
    """
    Короткие уточнения («what else», «other variants») без слова coffee — подмешиваем
    тему из недавней истории, иначе RAG уезжает в чай/десерты.
    """
    t = (client_text or "").strip()
    if not t or len(t) > 200:
        return t
    recent = " ".join((h.get("text") or "") for h in history[-16:])
    if not _COFFEE_TOPIC_IN_HISTORY_RE.search(recent.lower()):
        return t
    if _SHORT_TOPIC_CARRY_RE.match(t) or _SHORT_OTHER_VARIANTS_RE.match(t):
        return (
            f"{t} — customer still exploring coffee drink options "
            "latte frappe mocha macchiato premium roast iced coffee"
        )
    if len(t) <= 120 and _MENU_MORE_THAN_LISTED_RE.search(t):
        return (
            f"{t} — still about coffee line more drink names "
            "premium roast black cold iced latte frappe macchiato"
        )
    return t

_BROAD_MENU_INTENT_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bwhat\s+else\b", re.I),
    re.compile(r"\banything\s+else\b", re.I),
    re.compile(r"\bsomething\s+else\b", re.I),
    re.compile(r"\b(other\s+options?|other\s+items?|other\s+choices?)\b", re.I),
    re.compile(r"\bbesides\s+(that|this)\b", re.I),
    re.compile(r"\bother\s+than\b", re.I),
    re.compile(r"\bnot\s+just\b", re.I),
    re.compile(r"\bany\s+other\b", re.I),
    re.compile(r"\belse\s+do\s+you\s+have\b", re.I),
    re.compile(r"\belse\s+on\s+the\s+menu\b", re.I),
    re.compile(r"\bwhat\s+other\b", re.I),
)


def _client_wants_broader_menu_scan(text: str) -> bool:
    """True если реплика про расширение выбора, а не про одно конкретное блюдо."""
    if not (text or "").strip():
        return False
    return any(rx.search(text) for rx in _BROAD_MENU_INTENT_RES)


def _client_wants_coffee_variety_scan(text: str) -> bool:
    """
    True если клиент просит *ещё варианты* кофе, а не одну конкретную позицию.

    Тогда векторный поиск по одному слову «coffee» даёт узкий срез — подменяем запрос
    на разнообразные ключевые слова из линейки напитков.
    """
    if not (text or "").strip():
        return False
    if not re.search(
        r"\bcoffee\b|\blatte\b|\bmacchiato\b|\bfrappe\b|\bmocha\b|\bкофе\b",
        text,
        re.I,
    ):
        return False
    return any(rx.search(text) for rx in _COFFEE_VARIETY_INTENT_RES)


def _client_asks_coffee_variant_count(text: str) -> bool:
    """True when customer asks for coffee variants count in menu."""
    t = (text or "").strip()
    if not t:
        return False
    if not _COFFEE_NAME_HINT_RE.search(t):
        return False
    if _COFFEE_COUNT_INTENT_RE.search(t):
        return True
    return bool(re.search(r"\bvariants?\b.*\bmenu\b|\bmenu\b.*\bvariants?\b", t, re.I))


_NON_FOOD_CLIENT_UTTERANCE_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(yes|yeah|yep|correct|right|okay|ok)\s*[.!?]*\s*$", re.I),
    re.compile(r"^\s*(okay|ok|yes|yeah|yep),?\s*thanks?(?:\s+you)?\s*[.!?]*\s*$", re.I),
    re.compile(r"^\s*(yes|yeah|yep),?\s*(that'?s|thats)\s+right\s*[.!?]*\s*$", re.I),
    re.compile(r"^\s*(that'?s|thats)\s+(right|correct|fine)\s*[.!?]*\s*$", re.I),
    re.compile(r"^\s*(that'?s|thats)\s+all(?:,?\s*thanks?)?\s*[.!?]*\s*$", re.I),
    re.compile(r"^\s*all\s+set(?:,?\s*thanks?)?\s*[.!?]*\s*$", re.I),
    re.compile(r"^\s*thanks?(?:\s+you)?\s*[.!?]*\s*$", re.I),
)


def _is_non_food_client_utterance(text: str) -> bool:
    """True for short confirmation/closure phrases where RAG should be skipped."""
    t = (text or "").strip()
    if not t:
        return False
    t_low = t.lower()
    # If customer mentions concrete food, do not skip RAG.
    if re.search(
        r"\b(big mac|burger|hamburger|fries|nuggets?|mcchicken|coke|diet coke|sprite|"
        r"fanta|dr pepper|coffee|tea|lemonade|salad|happy meal)\b",
        t_low,
    ):
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


def _coffee_count_response_from_rag_context(rag_context: str) -> str | None:
    """
    Deterministic fallback for 'how many coffee variants' questions.
    Uses only current RAG context names to avoid hallucinated products.
    """
    names = [
        n for n in _extract_names_from_rag_context(rag_context)
        if _COFFEE_NAME_HINT_RE.search(n)
    ]
    if not names:
        return None
    return (
        f"I can confirm {len(names)} coffee variants in our menu right now: "
        + ", ".join(names)
        + ". Which one would you like?"
    )


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


def _client_is_specific_order(text: str) -> bool:
    t = (text or "").lower()
    if re.search(r"\b(i('ll| will)?|can i|i would like|i want|i'll take)\b", t):
        return True
    return bool(
        re.search(
            r"\b(big mac|burger|hamburger|fries|nuggets?|mcchicken|coke|diet coke|sprite|"
            r"fanta|dr pepper|coffee|tea|lemonade|salad|happy meal)\b",
            t,
        )
    )


def _fallback_query_from_client_text(text: str) -> str:
    t = (text or "").lower()
    picked: list[str] = []

    def add_if(cond: bool, token: str) -> None:
        if cond and token not in picked:
            picked.append(token)

    add_if("big mac" in t, "big mac")
    add_if("mcchicken" in t, "mcchicken")
    add_if("fries" in t, "fries")
    add_if("diet coke" in t, "diet coke")
    add_if("coke" in t and "diet coke" not in t, "coke")
    add_if("nugget" in t, "nuggets")
    add_if("burger" in t and "big mac" not in t, "burger")
    add_if("sandwich" in t, "sandwich")
    add_if("coffee" in t, "coffee")
    add_if("tea" in t, "tea")
    add_if("lemonade" in t, "lemonade")
    add_if("sprite" in t, "sprite")

    if picked:
        return " ".join(picked[:6])
    return "general menu items"


def _sanitize_cashier_response(text: str) -> str:
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
    Агент-кассир: формирует ответы с учётом истории, заказа и RAG по меню.

    По умолчанию в системный промпт попадают психотип и состав группы из profile
    (удобно для симуляции). При realistic_cashier=True кассир видит только общие
    правила и реплики из диалога; фильтр аллергенов в RAG по профилю отключается.
    """

    _FALLBACK_QUERIES = [
        "popular burgers Big Mac Quarter Pounder cheeseburger",
        "chicken sandwiches nuggets crispy spicy tenders",
        "breakfast items Egg McMuffin hash browns hotcakes biscuit",
        "sides fries apple slices mozzarella sticks",
        "drinks Coca-Cola Sprite Diet Coke Dr Pepper sweet tea lemonade",
        "coffee latte macchiato frappe iced coffee hot chocolate",
        "milkshakes chocolate vanilla strawberry smoothie",
        "salads caesar grilled chicken light options",
        "desserts McFlurry sundae apple pie brownie cookie",
        "value menu hamburger McDouble cheeseburger",
    ]

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
        realistic_cashier: bool = False,
        rag_max_prompt_lines: int | None = None,
    ) -> None:
        self.model = _resolve_model(model)
        self.rag_top_k = rag_top_k
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
        rag_context = self._resolve_rag_context(
            client_text,
            profile,
            history,
            rag_trace=rag_trace,
            rag_meta=rag_meta,
            llm_trace=llm_trace,
        )
        if _client_asks_coffee_variant_count(client_text):
            deterministic = _coffee_count_response_from_rag_context(rag_context)
            if deterministic:
                _trace(
                    llm_trace,
                    {
                        "event": "deterministic_coffee_count_reply",
                        **(rag_meta or {}),
                        "source": "rag_context",
                        "coffee_variants_count": len(
                            [
                                n
                                for n in _extract_names_from_rag_context(rag_context)
                                if _COFFEE_NAME_HINT_RE.search(n)
                            ]
                        ),
                    },
                )
                return deterministic
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
        response = _sanitize_cashier_response(response)
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
        history: list[HistoryEntry],
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
        rag_text = _enrich_client_text_for_menu_rag(history, client_text)
        if rag_text != client_text:
            base_trace["rag_query_enriched"] = True
            base_trace["client_query_for_rag"] = rag_text[:800]

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
            return ""

        if client_text:
            search_query = _rewrite_query(
                rag_text,
                self._client,
                self._rewrite_model,
                llm_trace=llm_trace,
                trace_meta=rag_meta,
                trace_verbose=self._trace_verbose,
            )
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
                return ""
        else:
            import random as _rand
            search_query = _rand.choice(self._FALLBACK_QUERIES)
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
            "retrieval_mode": self.rag_mode,
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
        Поиск по меню с фильтрацией по аллергенам группы.
        Возвращает (текст для system prompt, метаданные для rag_trace).
        """
        if self.rag_mode == RAG_MODE_GRAPH:
            return self._do_graph_rag(text, profile)

        blacklist = (
            []
            if self._realistic_cashier
            else get_group_allergen_blacklist(profile)
        )
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
            "retrieval_mode": RAG_MODE_VECTOR,
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
            lines, used = _render_rows(
                rows,
                max_dist=self.distance_threshold,
                max_lines=self.rag_max_prompt_lines,
            )
            if self.rag_max_prompt_lines is not None:
                info["rag_prompt_line_cap"] = self.rag_max_prompt_lines
                info["rag_prompt_lines_included"] = len(lines)
            info.update(outcome="injected", injected_hits=used)
            return "\n".join(lines), info

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
                return _SOFT_PREFIX + "\n".join(lines), info

        info["outcome"] = "above_threshold"
        return "(no matching menu items found — closest match too distant)", info

    def _do_graph_rag(
        self,
        text: str,
        profile: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        blacklist = (
            []
            if self._realistic_cashier
            else get_group_allergen_blacklist(profile)
        )
        rows, graph_info = search_menu_graph(
            text,
            allergens_blacklist=blacklist or None,
            top_k=self.rag_top_k,
        )
        info: dict[str, Any] = {
            "top_k": self.rag_top_k,
            "allergen_blacklist_tokens": list(blacklist),
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
    ) -> str:
        """Собирает системный промпт: базовый + опциональные блоки RAG и заказа."""
        base = get_cashier_system_prompt(profile, realistic=self._realistic_cashier)
        extras: list[str] = []
        if rag_context:
            extras.append(
                "Menu data slice for this turn (each line: product name, kcal estimate, "
                "allergen tags, approximate added/total sugar — not full ingredients or full "
                "nutrition):\n"
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
        "If the customer asks what ELSE is available, for OTHER options, "
        "anything BESIDES what was already mentioned, or NOT JUST one item, "
        "they want a broad menu slice — output a mix of categories, e.g. "
        "burgers chicken sandwiches sides drinks dessert coffee. "
        "Do NOT reduce the whole message to a single side or ingredient "
        "they only mentioned in passing (e.g. do not output only fries). "
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
        rewritten = _normalize_rewrite_output(rewritten)
        if _client_wants_coffee_variety_scan(client_text):
            _trace(
                llm_trace,
                {
                    "event": "rewrite_coffee_variety_override",
                    **(trace_meta or {}),
                    "model": model,
                    "kind": "mini_llm_menu_query_rewrite",
                    "prior_rewrite": rewritten,
                    "rewrite_output": _COFFEE_BROAD_RAG_QUERY,
                },
            )
            rewritten = _COFFEE_BROAD_RAG_QUERY
        elif _client_is_specific_order(client_text):
            broadish = {
                "burgers",
                "chicken",
                "sandwiches",
                "sides",
                "drinks",
                "dessert",
                "coffee",
            }
            rw_tokens = {t.lower() for t in rewritten.split()}
            if len(rw_tokens & broadish) >= 3:
                rewritten = _fallback_query_from_client_text(client_text)
        if _client_wants_broader_menu_scan(client_text):
            n_rw = len(rewritten.split())
            # «Что-то ещё» + кофе в реплике, а rewrite = «coffee» — даёт узкий RAG; тянем линейку McCafe.
            if re.search(r"\bcoffee\b", client_text, re.I) and n_rw <= 2:
                _trace(
                    llm_trace,
                    {
                        "event": "rewrite_coffee_broad_menu_override",
                        **(trace_meta or {}),
                        "model": model,
                        "kind": "mini_llm_menu_query_rewrite",
                        "narrow_rewrite": rewritten,
                        "rewrite_output": _COFFEE_BROAD_RAG_QUERY,
                    },
                )
                rewritten = _COFFEE_BROAD_RAG_QUERY
            elif n_rw <= 1:
                _trace(
                    llm_trace,
                    {
                        "event": "rewrite_broad_menu_override",
                        **(trace_meta or {}),
                        "model": model,
                        "kind": "mini_llm_menu_query_rewrite",
                        "narrow_rewrite": rewritten,
                        "rewrite_output": _BROAD_MENU_RAG_QUERY,
                    },
                )
                rewritten = _BROAD_MENU_RAG_QUERY
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
