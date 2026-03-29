"""
Агенты ClientAgent и CashierAgent поверх OpenAI-compatible Chat Completions.

Конфигурация:
  - DEFAULT_MODEL — модель по умолчанию (gpt-4o).
  - OPENAI_API_KEY — для облака OpenAI.
  - API_PROVIDER=ollama + API_MODEL + OLLAMA_URL — для локального OpenAI-compatible API.
  - RAG_DISTANCE_THRESHOLD — если ближайший результат дальше порога,
    считаем «нет подходящих позиций».
"""

from __future__ import annotations

import json
import os
from typing import Any

from openai import APITimeoutError, OpenAI, OpenAIError

from mcd_voice.llm.prompts import get_cashier_system_prompt, get_client_system_prompt
from mcd_voice.menu.search import search_menu
from mcd_voice.profile.generator import get_group_allergen_blacklist

DEFAULT_MODEL = "gpt-4o"
RAG_DISTANCE_THRESHOLD = 0.60

HistoryEntry = dict[str, str]


# ── Shared helpers ────────────────────────────────────────────────────

def _normalize_base_url(url: str) -> str:
    """Приводит URL OpenAI-compatible API к base_url."""
    base = url.strip().rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    return base


def _resolve_model(explicit_model: str | None) -> str:
    """Явная модель приоритетнее, иначе API_MODEL, иначе DEFAULT_MODEL."""
    if explicit_model:
        return explicit_model
    return os.environ.get("API_MODEL", DEFAULT_MODEL)


def _get_openai_client(timeout: float = 60.0) -> OpenAI:
    provider = os.environ.get("API_PROVIDER", "openai").strip().lower()

    if provider == "ollama":
        raw_url = os.environ.get("OLLAMA_URL") or os.environ.get("OPENAI_BASE_URL")
        if not raw_url:
            raise RuntimeError(
                "Для API_PROVIDER=ollama задайте OLLAMA_URL "
                "(например http://localhost:11434/v1)."
            )
        base_url = _normalize_base_url(raw_url)
        # Для большинства локальных OpenAI-compatible серверов ключ не обязателен.
        api_key = os.environ.get("OPENAI_API_KEY", "ollama")
        return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not api_key:
        raise RuntimeError(
            "Задайте OPENAI_API_KEY для облачного OpenAI или "
            "используйте API_PROVIDER=ollama с OLLAMA_URL."
        )
    kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout}
    if base_url:
        kwargs["base_url"] = _normalize_base_url(base_url)
    return OpenAI(**kwargs)


def get_llm_runtime_config() -> dict[str, str]:
    """Возвращает итоговую runtime-конфигурацию LLM из env."""
    provider = os.environ.get("API_PROVIDER", "openai").strip().lower() or "openai"
    model = _resolve_model(None)
    if provider == "ollama":
        raw_url = os.environ.get("OLLAMA_URL") or os.environ.get("OPENAI_BASE_URL", "")
        return {
            "provider": provider,
            "model": model,
            "base_url": _normalize_base_url(raw_url) if raw_url else "",
        }
    return {
        "provider": provider,
        "model": model,
        "base_url": _normalize_base_url(os.environ.get("OPENAI_BASE_URL", ""))
        if os.environ.get("OPENAI_BASE_URL")
        else "",
    }


def _call_llm(
    client: OpenAI,
    model: str,
    system: str,
    messages: list[dict[str, str]],
    temperature: float = 0.8,
) -> str:
    api_messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    api_messages.extend(messages)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=api_messages,
            temperature=temperature,
        )
        content = resp.choices[0].message.content
        if not content:
            raise RuntimeError("Пустой ответ от модели.")
        return content.strip()
    except APITimeoutError as e:
        raise RuntimeError(f"Таймаут OpenAI API: {e}") from e
    except OpenAIError as e:
        raise RuntimeError(f"Ошибка OpenAI API: {e}") from e


def _history_to_messages(
    history: list[HistoryEntry],
    *,
    my_role: str,
) -> list[dict[str, str]]:
    """
    Конвертирует внутреннюю историю в формат OpenAI messages.
    Реплики «моей» роли → assistant, чужие → user.
    """
    out: list[dict[str, str]] = []
    for entry in history:
        speaker = entry["speaker"]
        oai_role = "assistant" if speaker == my_role else "user"
        out.append({"role": oai_role, "content": entry["text"]})
    return out


# ── ClientAgent ───────────────────────────────────────────────────────

class ClientAgent:
    """Агент-клиент: генерирует реплики на основе профиля и полной истории."""

    def __init__(
        self,
        model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = _resolve_model(model)
        self._openai = _get_openai_client(timeout)

    def generate_response(
        self,
        profile: dict[str, Any],
        history: list[HistoryEntry],
    ) -> str:
        system = get_client_system_prompt(profile)

        if not history:
            lang = profile.get("language", "EN")
            opener = (
                "You just pulled up to the drive-through speaker. "
                "Say one short opening line as the customer."
                if lang == "EN"
                else "Вы подъехали к окну заказа. "
                "Скажите одну короткую реплику в роли клиента."
            )
            messages = [{"role": "user", "content": opener}]
        else:
            messages = _history_to_messages(history, my_role="client")

        return _call_llm(self._openai, self.model, system, messages)


# ── CashierAgent ──────────────────────────────────────────────────────

_RAG_KEYWORDS = (
    "menu", "recommend", "what do you", "what's good", "suggest",
    "options", "anything spicy", "healthy", "vegetarian", "allerg",
    "what can", "do you have", "what kind", "something",
    "меню", "порекомендуй", "что есть", "что посоветуешь", "какие есть",
    "что-нибудь", "посоветуй",
)


class CashierAgent:
    """Агент-кассир с RAG, адаптацией к психотипу и поддержкой групп."""

    def __init__(
        self,
        model: str | None = None,
        timeout: float = 60.0,
        rag_top_k: int = 3,
        distance_threshold: float = RAG_DISTANCE_THRESHOLD,
    ) -> None:
        self.model = _resolve_model(model)
        self.rag_top_k = rag_top_k
        self.distance_threshold = distance_threshold
        self._openai = _get_openai_client(timeout)

    def generate_response(
        self,
        profile: dict[str, Any],
        history: list[HistoryEntry],
        order_state: dict[str, Any],
        query: str | None = None,
    ) -> str:
        client_text = query or self._last_client_text(history)

        rag_context = ""
        if self.rag_top_k > 0 and client_text and self._needs_rag(client_text):
            rag_context = self._do_rag(client_text, profile)

        system = self._build_system(profile, order_state, rag_context)
        messages = _history_to_messages(history, my_role="cashier")

        return _call_llm(self._openai, self.model, system, messages)

    # ── Внутренние методы ─────────────────────────────────────────────

    @staticmethod
    def _last_client_text(history: list[HistoryEntry]) -> str:
        for entry in reversed(history):
            if entry["speaker"] == "client":
                return entry["text"]
        return ""

    @staticmethod
    def _needs_rag(text: str) -> bool:
        t = text.lower()
        if any(k in t for k in _RAG_KEYWORDS):
            return True
        if "what" in t and "have" in t:
            return True
        if "что" in t and ("есть" in t or "можно" in t):
            return True
        return False

    def _do_rag(self, text: str, profile: dict[str, Any]) -> str:
        blacklist = get_group_allergen_blacklist(profile)
        rows = search_menu(
            text,
            allergens_blacklist=blacklist if blacklist else None,
            top_k=self.rag_top_k,
        )
        if not rows:
            return "(no matching menu items found for this request)"

        # Distance threshold: если ближайший результат слишком далеко,
        # считаем что ничего подходящего нет
        best_dist = rows[0]["distance"]
        if best_dist > self.distance_threshold:
            return "(no matching menu items found — closest match too distant)"

        lines: list[str] = []
        for r in rows:
            if r["distance"] > self.distance_threshold:
                break
            ag = ", ".join(r["allergens"]) if r["allergens"] else "none listed"
            lines.append(f"- {r['name']} (~{r['energy']} kcal, allergens: {ag})")
        return "\n".join(lines)

    @staticmethod
    def _build_system(
        profile: dict[str, Any],
        order_state: dict[str, Any],
        rag_context: str,
    ) -> str:
        base = get_cashier_system_prompt(profile)
        extras: list[str] = []
        if rag_context:
            extras.append(f"Relevant menu items (RAG):\n{rag_context}")
        persons = order_state.get("persons", [])
        has_items = any(p.get("items") for p in persons)
        if has_items:
            extras.append(
                f"Current order state:\n{json.dumps(order_state, ensure_ascii=False)}"
            )
        if extras:
            return base + "\n\n--- Context ---\n" + "\n\n".join(extras)
        return base
