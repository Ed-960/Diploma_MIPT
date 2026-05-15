from __future__ import annotations

import time
from typing import Any

from mcd_voice.llm.grounding import GroundingService
from mcd_voice.llm.output_policy import OutputPolicy
from mcd_voice.llm.response_composer import ResponseComposer
from mcd_voice.llm.retrieval import RagRetriever
from mcd_voice.llm.turn_plan import TurnContext, TurnPlan


def run_cashier_turn(
    *,
    agent: Any,
    profile: dict[str, Any],
    history: list[dict[str, str]],
    order_state: dict[str, Any],
    query: str | None,
    extra_grounding_context: str | None,
    rag_trace: list[dict[str, Any]] | None,
    rag_meta: dict[str, Any] | None,
    llm_trace: list[dict[str, Any]] | None,
) -> str:
    # Local imports avoid hard circular dependency at module import time.
    from mcd_voice.llm.agent import (
        _call_llm,
        _deterministic_full_catalog_nutrition_reply,
        _history_to_messages,
        _last_client_text,
        _llm_call_payload,
        _llm_error_payload,
        _rag_intent,
        _sanitize_cashier_response,
        _trace,
        _wants_full_nutrition_context,
        _wants_menu_item_details,
    )

    client_text = query or _last_client_text(history)
    ctx = TurnContext(
        client_text=client_text,
        profile=profile,
        history=history,
        order_state=order_state,
    )

    retrieval = RagRetriever()
    retrieval_result = retrieval.resolve(
        agent=agent,
        client_text=ctx.client_text,
        profile=ctx.profile,
        history=ctx.history,
        rag_trace=rag_trace,
        rag_meta=rag_meta,
        llm_trace=llm_trace,
    )
    ctx.rag_context = retrieval_result.rag_context
    ctx.rag_spec = retrieval_result.rag_spec
    extra_grounding = (extra_grounding_context or "").strip()
    if extra_grounding:
        ctx.rag_context = f"{extra_grounding}\n\n{ctx.rag_context}" if ctx.rag_context else extra_grounding
        _trace(
            rag_trace,
            {
                **(rag_meta or {}),
                "event": "extra_grounding_context",
                "grounding_chars": len(extra_grounding),
                "grounding_preview": extra_grounding[:600],
            },
        )
    ctx.plan = TurnPlan.from_legacy(
        client_text=ctx.client_text,
        spec=ctx.rag_spec,
        intent_resolver=_rag_intent,
        nutrition_resolver=_wants_full_nutrition_context,
    )

    grounding = GroundingService()
    facts = grounding.build(
        intent=ctx.plan.intent,
        rag_spec=ctx.rag_spec,
        rag_context=ctx.rag_context,
        full_nutrition_context=ctx.plan.full_nutrition_context,
    )

    composer = ResponseComposer()
    deterministic = None
    if not getattr(agent, "_disable_deterministic_shortcuts", False):
        deterministic = composer.compose_deterministic(
            agent=agent,
            facts=facts,
            profile=ctx.profile,
            order_state=ctx.order_state,
            client_text=ctx.client_text,
            history=ctx.history,
            llm_trace=llm_trace,
            rag_meta=rag_meta,
            trace_fn=_trace,
            nutrition_reply_fn=_deterministic_full_catalog_nutrition_reply,
            wants_menu_item_details_fn=_wants_menu_item_details,
        )
    if deterministic:
        return deterministic

    system = agent._build_system(
        ctx.profile,
        ctx.order_state,
        ctx.rag_context,
        allow_calories=facts.allow_calories,
        allow_full_nutrition=facts.full_nutrition_context,
        finalize_requested=bool(ctx.plan.finalize),
        override_restriction=bool(ctx.plan.override_restriction),
    )
    messages = _history_to_messages(ctx.history, my_role="cashier")
    output_policy = OutputPolicy()
    try:
        t0 = time.perf_counter()
        raw_response = _call_llm(agent._client, agent.model, system, messages)
        dt_ms = (time.perf_counter() - t0) * 1000.0
    except RuntimeError as exc:
        output_policy.emit_llm_error(
            llm_trace=llm_trace,
            trace_fn=_trace,
            payload_builder=_llm_error_payload,
            model=agent.model,
            messages=messages,
            system=system,
            error=str(exc),
            verbose=agent._trace_verbose,
        )
        return "Sorry, I didn't catch that clearly. Could you repeat your order?"
    except Exception as exc:
        output_policy.emit_llm_error(
            llm_trace=llm_trace,
            trace_fn=_trace,
            payload_builder=_llm_error_payload,
            model=agent.model,
            messages=messages,
            system=system,
            error=str(exc),
            verbose=agent._trace_verbose,
        )
        raise

    response = output_policy.sanitize(
        raw_response=raw_response,
        allow_calories=facts.allow_calories,
        sanitizer=_sanitize_cashier_response,
    )
    output_policy.emit_llm_success(
        llm_trace=llm_trace,
        trace_fn=_trace,
        payload_builder=_llm_call_payload,
        model=agent.model,
        system=system,
        messages=messages,
        response=response,
        duration_ms=dt_ms,
        verbose=agent._trace_verbose,
    )
    return response

