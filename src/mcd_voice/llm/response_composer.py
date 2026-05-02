from __future__ import annotations

from typing import Any

from mcd_voice.llm.grounding import GroundingFacts


class ResponseComposer:
    """Deterministic-first strategies before generic LLM generation."""

    def compose_deterministic(
        self,
        *,
        agent: Any,
        facts: GroundingFacts,
        profile: dict[str, Any],
        order_state: dict[str, Any],
        client_text: str,
        history: list[dict[str, str]],
        llm_trace: list[dict[str, Any]] | None,
        rag_meta: dict[str, Any] | None,
        trace_fn: Any,
        nutrition_reply_fn: Any,
        wants_menu_item_details_fn: Any,
    ) -> str | None:
        intent = facts.intent
        rag_spec = facts.rag_spec
        rag_context = facts.rag_context

        macro_reply = (
            agent._deterministic_compare_reply(profile, client_text, rag_spec)
            if intent == "compare"
            else None
        )
        if macro_reply:
            trace_fn(llm_trace, {"event": "deterministic_compare_reply", **(rag_meta or {})})
            return macro_reply

        meal_details = (
            agent._deterministic_meal_details_reply(client_text, rag_context)
            if intent == "details" and wants_menu_item_details_fn(client_text)
            else None
        )
        if meal_details:
            trace_fn(llm_trace, {"event": "deterministic_meal_details_reply", **(rag_meta or {})})
            return meal_details

        tune_reply = (
            agent._deterministic_calorie_tuning_reply(profile, order_state, client_text)
            if intent == "calorie_tune" and not agent._realistic_cashier
            else None
        )
        if tune_reply:
            trace_fn(
                llm_trace,
                {
                    "event": "deterministic_calorie_tuning_reply",
                    **(rag_meta or {}),
                    **(
                        {"target_kcal": profile.get("calApprValue")}
                        if not agent._realistic_cashier
                        else {}
                    ),
                },
            )
            return tune_reply

        nutrition_reply = nutrition_reply_fn(client_text, history)
        if nutrition_reply:
            trace_fn(
                llm_trace,
                {"event": "deterministic_full_catalog_nutrition_reply", **(rag_meta or {})},
            )
            return nutrition_reply

        menu_browse_reply = agent._deterministic_menu_browse_reply(
            client_text,
            history,
            rag_context,
            rag_spec,
        )
        if menu_browse_reply:
            trace_fn(
                llm_trace,
                {"event": "deterministic_menu_browse_reply", **(rag_meta or {})},
            )
            return menu_browse_reply
        return None

