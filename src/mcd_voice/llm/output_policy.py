from __future__ import annotations

from typing import Any


class OutputPolicy:
    def sanitize(self, *, raw_response: str, allow_calories: bool, sanitizer: Any) -> str:
        return str(sanitizer(raw_response, allow_calories=allow_calories))

    def emit_llm_error(
        self,
        *,
        llm_trace: list[dict[str, Any]] | None,
        trace_fn: Any,
        payload_builder: Any,
        model: str,
        messages: list[dict[str, str]],
        system: str,
        error: str,
        verbose: bool,
    ) -> None:
        trace_fn(
            llm_trace,
            payload_builder(
                agent="cashier",
                model=model,
                messages=messages,
                system=system,
                error=error,
                verbose=verbose,
            ),
        )

    def emit_llm_success(
        self,
        *,
        llm_trace: list[dict[str, Any]] | None,
        trace_fn: Any,
        payload_builder: Any,
        model: str,
        system: str,
        messages: list[dict[str, str]],
        response: str,
        duration_ms: float,
        verbose: bool,
    ) -> None:
        trace_fn(
            llm_trace,
            payload_builder(
                agent="cashier",
                model=model,
                system=system,
                messages=messages,
                response=response,
                duration_ms=duration_ms,
                verbose=verbose,
            ),
        )

