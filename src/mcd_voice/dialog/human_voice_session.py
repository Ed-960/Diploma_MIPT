"""
Интерактивная сессия: живой человек вместо ClientAgent, кассир — CashierAgent.

Поток совпадает с «ходом» в DialogPipeline.run: приветствие кассира, затем
циклы «реплика клиента → кассир → обновление заказа».
"""

from __future__ import annotations

from typing import Any

from mcd_voice.dialog.catalog import MenuCatalog
from mcd_voice.dialog.pipeline import (
    DialogPipeline,
    _cashier_signals_end,
    _client_confirms_end,
    _client_says_farewell,
    _has_cashier_hard_repeat,
    _is_cot_leak,
    _is_looping_tail,
    _is_stalled,
    _is_yes_only,
    build_initial_order_state,
    validate_dialog,
)
from mcd_voice.llm import CashierAgent
from mcd_voice.llm.agent import _resolve_model
from mcd_voice.profile import ProfileGenerator


def _compact_validation(flags: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "allergen_violation",
        "restriction_violation",
        "calorie_warning",
        "empty_order",
        "total_items",
        "total_energy",
        "hallucination",
        "incomplete_order",
        "turns",
    )
    return {k: flags[k] for k in keys if k in flags}


class HumanDriveThroughSession:
    """
    Один диалог: кассир (LLM) + реплики клиента извне (голос → текст в UI).

    Не потоковый «одновременный» разговор: ответ кассира запрашивается после
    того, как пользователь отправил финальный текст реплики.
    """

    def __init__(
        self,
        *,
        max_turns: int = 20,
        model: str | None = None,
        realistic_cashier: bool = False,
        trace_verbose: bool = False,
    ) -> None:
        self.max_turns = max_turns
        self.model = _resolve_model(model)
        self.realistic_cashier = realistic_cashier
        self.trace_verbose = trace_verbose
        self._profiles = ProfileGenerator()
        self._catalog = MenuCatalog()
        self._helper = DialogPipeline(
            max_turns=max_turns,
            model=model,
            realistic_cashier=realistic_cashier,
            trace_verbose=trace_verbose,
        )
        self._started = False
        self._profile: dict[str, Any] | None = None
        self._history: list[dict[str, str]] | None = None
        self._order_state: dict[str, Any] | None = None
        self._menu_names: list[str] | None = None
        self._energy_by_name: dict[str, float] | None = None
        self._allergen_map: dict[str, list[str]] | None = None
        self._cashier: CashierAgent | None = None
        self._cashier_signaled = False
        self._n_client_messages = 0
        self._cot_leak_count = 0

    def start(self, profile: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._started:
            raise RuntimeError("Session already started.")
        self._started = True
        prof = profile if profile is not None else self._profiles.generate()
        self._profile = prof
        menu_names, energy_by_name = self._catalog.load()
        self._menu_names = menu_names
        self._energy_by_name = energy_by_name
        self._allergen_map = self._helper._build_allergen_map()
        self._order_state = build_initial_order_state(prof)
        self._history = []
        self._cashier = CashierAgent(
            model=self.model,
            trace_verbose=self.trace_verbose,
            realistic_cashier=self.realistic_cashier,
        )
        cashier = self._cashier
        assert cashier is not None
        assert self._history is not None
        assert self._order_state is not None
        cashier_kwargs: dict[str, Any] = {
            "rag_trace": None,
            "rag_meta": {"call": "greeting"},
        }
        greeting = cashier.generate_response(
            prof, self._history, self._order_state, **cashier_kwargs
        )
        self._history.append({"speaker": "cashier", "text": greeting})
        flags = validate_dialog(
            prof, self._order_state, self._history, menu_names=menu_names,
        )
        return {
            "greeting": greeting,
            "profile": prof,
            "validation": _compact_validation(flags),
        }

    def step(self, client_text: str) -> dict[str, Any]:
        if not self._started:
            raise RuntimeError("Call start() first.")
        assert self._profile is not None
        assert self._history is not None
        assert self._order_state is not None
        assert self._menu_names is not None
        assert self._energy_by_name is not None
        assert self._allergen_map is not None
        assert self._cashier is not None

        client_msg = (client_text or "").strip()
        if not client_msg:
            raise ValueError("Empty client message.")

        if self._n_client_messages >= self.max_turns:
            return {
                "error": "max_turns_reached",
                "cashier_text": "",
                "order_complete": bool(self._order_state.get("order_complete")),
                "dialog_ended": True,
                "reason": "max_turns_reached",
            }

        self._history.append({"speaker": "client", "text": client_msg})
        self._n_client_messages += 1

        if self._cashier_signaled and (
            _client_confirms_end(client_msg) or _client_says_farewell(client_msg)
        ):
            self._order_state["order_complete"] = True
            flags = validate_dialog(
                self._profile,
                self._order_state,
                self._history,
                menu_names=self._menu_names,
            )
            return {
                "cashier_text": "",
                "order_complete": True,
                "dialog_ended": True,
                "reason": "client_confirmed_end",
                "turn": self._n_client_messages,
                "validation": _compact_validation(flags),
            }

        if _is_cot_leak(client_msg):
            self._cot_leak_count += 1

        cashier_kwargs: dict[str, Any] = {
            "rag_trace": None,
            "rag_meta": {"call": "turn", "turn": self._n_client_messages},
        }
        cashier = self._cashier
        cashier_msg = cashier.generate_response(
            self._profile,
            self._history,
            self._order_state,
            **cashier_kwargs,
        )
        if _is_cot_leak(cashier_msg):
            self._cot_leak_count += 1
        self._history.append({"speaker": "cashier", "text": cashier_msg})

        if _is_stalled(self._history):
            self._order_state["order_complete"] = True
            flags = validate_dialog(
                self._profile,
                self._order_state,
                self._history,
                menu_names=self._menu_names,
            )
            return {
                "cashier_text": cashier_msg,
                "order_complete": True,
                "dialog_ended": True,
                "reason": "stall_detected",
                "turn": self._n_client_messages,
                "validation": _compact_validation(flags),
            }

        if _is_looping_tail(self._history) or _has_cashier_hard_repeat(
            self._history, repeat=3,
        ):
            if validate_dialog(
                self._profile,
                self._order_state,
                self._history,
                menu_names=self._menu_names,
            )["total_items"] > 0:
                self._order_state["order_complete"] = True
            flags = validate_dialog(
                self._profile,
                self._order_state,
                self._history,
                menu_names=self._menu_names,
            )
            return {
                "cashier_text": cashier_msg,
                "order_complete": bool(self._order_state.get("order_complete")),
                "dialog_ended": True,
                "reason": "loop_detected",
                "turn": self._n_client_messages,
                "validation": _compact_validation(flags),
            }

        h = self._helper
        h._update_order(
            client_msg,
            self._menu_names,
            self._order_state,
            self._energy_by_name,
            self._allergen_map,
        )
        h._enforce_restriction_safety(
            self._profile,
            self._order_state,
            self._energy_by_name,
            self._allergen_map,
        )
        h._remove_unavailable_from_order(
            cashier_msg,
            self._menu_names,
            self._order_state,
            self._energy_by_name,
            self._allergen_map,
        )
        h._enforce_restriction_safety(
            self._profile,
            self._order_state,
            self._energy_by_name,
            self._allergen_map,
        )

        self._cashier_signaled = _cashier_signals_end(cashier_msg)
        if self._cashier_signaled:
            h._update_order(
                cashier_msg,
                self._menu_names,
                self._order_state,
                self._energy_by_name,
                self._allergen_map,
            )
            h._enforce_restriction_safety(
                self._profile,
                self._order_state,
                self._energy_by_name,
                self._allergen_map,
            )
            if _is_yes_only(client_msg):
                self._order_state["order_complete"] = True
                flags = validate_dialog(
                    self._profile,
                    self._order_state,
                    self._history,
                    menu_names=self._menu_names,
                )
                return {
                    "cashier_text": cashier_msg,
                    "order_complete": True,
                    "dialog_ended": True,
                    "reason": "cashier_finalized_after_yes",
                    "turn": self._n_client_messages,
                    "validation": _compact_validation(flags),
                }

        flags = validate_dialog(
            self._profile,
            self._order_state,
            self._history,
            menu_names=self._menu_names,
        )
        out: dict[str, Any] = {
            "cashier_text": cashier_msg,
            "order_complete": bool(self._order_state.get("order_complete")),
            "dialog_ended": bool(self._order_state.get("order_complete")),
            "reason": None,
            "turn": self._n_client_messages,
            "validation": _compact_validation(flags),
        }
        if self._cot_leak_count:
            out["cot_leak_count"] = self._cot_leak_count
        return out
