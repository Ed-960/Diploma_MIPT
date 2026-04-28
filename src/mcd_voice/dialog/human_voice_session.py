"""
Интерактивная сессия: живой человек вместо ClientAgent, кассир — CashierAgent.

Поток совпадает с «ходом» в DialogPipeline.run: приветствие кассира, затем
циклы «реплика клиента → кассир → обновление заказа».
"""

from __future__ import annotations

import copy
import json
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
    _order_parser_reason_stats,
    build_initial_order_state,
    validate_dialog,
)
from mcd_voice.dialog.trace_format import (
    format_trace_event_pretty,
    summarize_llm_event,
    summarize_rag_event,
)
from mcd_voice.llm import CashierAgent
from mcd_voice.llm.agent import _resolve_model
from mcd_voice.profile import neutral_drive_through_profile


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

    При вызове start() без аргумента профиль нейтральный (без скрытых ограничений REG);
    можно передать свой dict для тестов или расширенного сценария.
    """

    def __init__(
        self,
        *,
        max_turns: int = 20,
        model: str | None = None,
        realistic_cashier: bool = True,
        trace_verbose: bool = False,
        print_trace: bool = False,
        trace_all: bool = False,
    ) -> None:
        self.max_turns = max_turns
        self.model = _resolve_model(model)
        self.realistic_cashier = realistic_cashier
        self.trace_verbose = trace_verbose
        self.print_trace = bool(print_trace or trace_all)
        self.trace_all = bool(trace_all)
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
        self._restriction_map: dict[str, dict[str, bool]] | None = None
        self._cashier: CashierAgent | None = None
        self._cashier_signaled = False
        self._n_client_messages = 0
        self._cot_leak_count = 0
        self._order_parser_events = []
        self._order_parser_events: list[dict[str, Any]] = []

    def _collect_order_parser_events(
        self,
        llm_trace: list[dict[str, Any]] | None,
    ) -> None:
        for ev in llm_trace or []:
            if not isinstance(ev, dict):
                continue
            kind = str(ev.get("event") or "")
            if kind.startswith("order_json_"):
                self._order_parser_events.append(ev)

    def _order_parser_stats(self) -> dict[str, Any]:
        return _order_parser_reason_stats(self._order_parser_events)

    def _attach_order_parser_summary(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if not payload.get("dialog_ended"):
            return payload
        stats = self._order_parser_stats()
        if stats.get("total_order_parser_events", 0) > 0:
            payload["order_parser_stats"] = stats
            if self.trace_all:
                print(
                    "[order_parser_summary] "
                    + json.dumps(stats, ensure_ascii=False),
                    flush=True,
                )
        return payload

    def _emit_trace(
        self,
        *,
        label: str,
        rag_trace: list[dict[str, Any]] | None,
        llm_trace: list[dict[str, Any]] | None,
    ) -> None:
        if not self.print_trace:
            return
        print(f"[trace] {label}", flush=True)
        for ev in rag_trace or []:
            block = (
                format_trace_event_pretty(ev)
                if (self.trace_all or self.trace_verbose)
                else summarize_rag_event(ev)
            )
            for line in block.splitlines():
                print(f"  [rag] {line}", flush=True)
        for ev in llm_trace or []:
            block = (
                format_trace_event_pretty(ev)
                if (self.trace_all or self.trace_verbose)
                else summarize_llm_event(ev)
            )
            for line in block.splitlines():
                print(f"  [llm] {line}", flush=True)

    def _emit_step_snapshot(
        self,
        *,
        stage: str,
        cashier_text: str | None = None,
    ) -> None:
        if not self.trace_all:
            return
        print(f"[session] stage={stage} turn={self._n_client_messages}", flush=True)
        if cashier_text is not None:
            print(f"[session] cashier_text={cashier_text}", flush=True)
        if self._order_state is not None:
            print(
                "[session] order_state="
                + json.dumps(self._order_state, ensure_ascii=False),
                flush=True,
            )

    def start(self, profile: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._started:
            raise RuntimeError("Session already started.")
        self._started = True
        prof = profile if profile is not None else neutral_drive_through_profile()
        self._profile = prof
        menu_names, energy_by_name, allergen_map, restriction_map = (
            self._catalog.load_runtime_index()
        )
        self._menu_names = menu_names
        self._energy_by_name = energy_by_name
        self._allergen_map = allergen_map
        self._restriction_map = restriction_map
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
        rag_trace: list[dict[str, Any]] | None = [] if self.print_trace else None
        llm_trace: list[dict[str, Any]] | None = [] if self.print_trace else None
        cashier_kwargs: dict[str, Any] = {
            "rag_trace": rag_trace,
            "rag_meta": {"call": "greeting"},
        }
        if llm_trace is not None:
            cashier_kwargs["llm_trace"] = llm_trace
        greeting = cashier.generate_response(
            prof, self._history, self._order_state, **cashier_kwargs
        )
        self._emit_trace(label="greeting", rag_trace=rag_trace, llm_trace=llm_trace)
        self._history.append({"speaker": "cashier", "text": greeting})
        flags = validate_dialog(
            prof,
            self._order_state,
            self._history,
            menu_names=menu_names,
            restriction_map=restriction_map,
        )
        self._emit_step_snapshot(stage="start", cashier_text=greeting)
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
        assert self._restriction_map is not None
        assert self._cashier is not None

        client_msg = (client_text or "").strip()
        if not client_msg:
            raise ValueError("Empty client message.")

        if self._n_client_messages >= self.max_turns:
            return self._attach_order_parser_summary({
                "error": "max_turns_reached",
                "cashier_text": "",
                "order_complete": bool(self._order_state.get("order_complete")),
                "dialog_ended": True,
                "reason": "max_turns_reached",
            })

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
                restriction_map=self._restriction_map,
            )
            return self._attach_order_parser_summary({
                "cashier_text": "",
                "order_complete": True,
                "dialog_ended": True,
                "reason": "client_confirmed_end",
                "turn": self._n_client_messages,
                "validation": _compact_validation(flags),
            })

        if _is_cot_leak(client_msg):
            self._cot_leak_count += 1

        rag_trace: list[dict[str, Any]] | None = [] if self.print_trace else None
        llm_trace: list[dict[str, Any]] | None = [] if self.print_trace else None
        cashier_kwargs: dict[str, Any] = {
            "rag_trace": rag_trace,
            "rag_meta": {"call": "turn", "turn": self._n_client_messages},
        }
        if llm_trace is not None:
            cashier_kwargs["llm_trace"] = llm_trace
        cashier = self._cashier
        cashier_msg = cashier.generate_response(
            self._profile,
            self._history,
            self._order_state,
            **cashier_kwargs,
        )
        self._emit_trace(
            label=f"turn={self._n_client_messages}",
            rag_trace=rag_trace,
            llm_trace=llm_trace,
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
                restriction_map=self._restriction_map,
            )
            return self._attach_order_parser_summary({
                "cashier_text": cashier_msg,
                "order_complete": True,
                "dialog_ended": True,
                "reason": "stall_detected",
                "turn": self._n_client_messages,
                "validation": _compact_validation(flags),
            })

        if _is_looping_tail(self._history) or _has_cashier_hard_repeat(
            self._history, repeat=3,
        ):
            if validate_dialog(
                self._profile,
                self._order_state,
                self._history,
                menu_names=self._menu_names,
                restriction_map=self._restriction_map,
            )["total_items"] > 0:
                self._order_state["order_complete"] = True
            flags = validate_dialog(
                self._profile,
                self._order_state,
                self._history,
                menu_names=self._menu_names,
                restriction_map=self._restriction_map,
            )
            return self._attach_order_parser_summary({
                "cashier_text": cashier_msg,
                "order_complete": bool(self._order_state.get("order_complete")),
                "dialog_ended": True,
                "reason": "loop_detected",
                "turn": self._n_client_messages,
                "validation": _compact_validation(flags),
            })

        h = self._helper
        structured_orders = h._parse_structured_orders(
            client_msg,
            self._menu_names,
            self._order_state.get("persons", []),
            llm_trace=llm_trace,
            trace_meta={"event_scope": "order_parser", "turn": self._n_client_messages},
        )
        h._update_order(
            client_msg,
            self._menu_names,
            self._order_state,
            self._energy_by_name,
            self._allergen_map,
            structured_orders=structured_orders,
            llm_trace=llm_trace,
            trace_meta={"event_scope": "order_parser", "turn": self._n_client_messages},
        )
        h._enforce_restriction_safety(
            self._profile,
            self._order_state,
            self._energy_by_name,
            self._allergen_map,
            self._restriction_map,
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
            self._restriction_map,
        )
        self._collect_order_parser_events(llm_trace)

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
                self._restriction_map,
            )
            if _is_yes_only(client_msg):
                self._order_state["order_complete"] = True
                flags = validate_dialog(
                    self._profile,
                    self._order_state,
                    self._history,
                    menu_names=self._menu_names,
                    restriction_map=self._restriction_map,
                )
                return self._attach_order_parser_summary({
                    "cashier_text": cashier_msg,
                    "order_complete": True,
                    "dialog_ended": True,
                    "reason": "cashier_finalized_after_yes",
                    "turn": self._n_client_messages,
                    "validation": _compact_validation(flags),
                })

        flags = validate_dialog(
            self._profile,
            self._order_state,
            self._history,
            menu_names=self._menu_names,
            restriction_map=self._restriction_map,
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
        self._emit_step_snapshot(stage="turn", cashier_text=cashier_msg)
        return self._attach_order_parser_summary(out)

    def snapshot_for_save(
        self,
    ) -> tuple[dict[str, Any], list[dict[str, str]], dict[str, Any], dict[str, Any]] | None:
        """
        Снимок для save_dialog (как после simulate_dialog): профиль, история,
        order_state и полные validation_flags. None, если сессия не запущена.
        """
        if not self._started:
            return None
        assert self._profile is not None
        assert self._history is not None
        assert self._order_state is not None
        assert self._menu_names is not None
        flags = validate_dialog(
            self._profile,
            self._order_state,
            self._history,
            menu_names=self._menu_names,
            restriction_map=self._restriction_map,
        )
        stats = self._order_parser_stats()
        if stats.get("total_order_parser_events", 0) > 0:
            flags = {**flags, "order_parser_stats": stats}
        return (
            copy.deepcopy(self._profile),
            [dict(h) for h in self._history],
            copy.deepcopy(self._order_state),
            flags,
        )
