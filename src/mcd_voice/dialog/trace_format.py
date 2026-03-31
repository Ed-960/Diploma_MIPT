"""Краткие строки для консольного вывода rag_trace / llm_trace."""

from __future__ import annotations

import json
from typing import Any

_JSON_CAP_VERBOSE = 48_000


def format_trace_event_pretty(ev: dict[str, Any]) -> str:
    """Полный JSON события для консоли (с усечением очень длинных полей)."""
    s = json.dumps(ev, ensure_ascii=False, indent=2)
    if len(s) <= _JSON_CAP_VERBOSE:
        return s
    return s[:_JSON_CAP_VERBOSE] + "\n… [truncated]"


def summarize_rag_event(ev: dict[str, Any]) -> str:
    if ev.get("event") == "rag_disabled":
        return "RAG: disabled (rag_top_k=0)"
    if ev.get("event") == "chroma_request":
        return (
            f"Chroma REQUEST: n_results={ev.get('n_results')} "
            f"query_texts={ev.get('query_texts')!r} where={ev.get('where')!r}"
        )
    if ev.get("event") == "chroma_response":
        metas = ev.get("metadatas") or []
        names = [m.get("name", "") for m in metas if isinstance(m, dict)]
        dists = ev.get("distances") or []
        qms = ev.get("query_duration_ms")
        return (
            f"Chroma RESPONSE: query_duration_ms={qms} "
            f"hits={len(names)} names={names!r} distances={dists!r}"
        )
    if ev.get("event") != "rag":
        return json.dumps(ev, ensure_ascii=False)[:240]
    sq = ev.get("search_query", "")
    oc = ev.get("outcome", "")
    bd = ev.get("best_distance")
    rm = ev.get("rewrite_model", "")
    cands = ev.get("candidates") or []
    top = ", ".join(str(c.get("name", "")) for c in cands[:3])
    bd_s = f"{bd:.4f}" if isinstance(bd, (int, float)) else str(bd)
    return (
        f"RAG/Chroma: search_query={sq!r} rewrite_model={rm!r} "
        f"outcome={oc!r} best_distance={bd_s} top=[{top}]"
    )


def summarize_llm_event(ev: dict[str, Any]) -> str:
    kind = ev.get("event", "")
    model = ev.get("model", "")
    if kind == "llm_rewrite":
        ms = ev.get("rewrite_duration_ms")
        ms_s = f" {ms}ms" if ms is not None else ""
        return (
            f"mini-LLM rewrite ({model}){ms_s}: "
            f"{ev.get('rewrite_input', '')!r} -> {ev.get('rewrite_output', '')!r}"
        )
    if kind == "llm_rewrite_fallback":
        return (
            f"mini-LLM rewrite FAILED, using raw text ({model}): "
            f"{ev.get('rewrite_input', '')!r}"
        )
    if kind == "llm_call":
        agent = ev.get("agent", "")
        ms = ev.get("duration_ms")
        ms_s = f" {ms}ms" if ms is not None else ""
        if "response" in ev:
            rsp = ev.get("response", "")
            tail = rsp if len(rsp) <= 600 else rsp[:600] + "…"
            return f"LLM {agent} ({model}){ms_s}: response={tail!r}"
        rsp = ev.get("response_preview", "")
        return f"LLM {agent} ({model}){ms_s}: reply_preview={rsp!r}"
    if kind == "llm_error":
        return (
            f"LLM ERROR {ev.get('agent', '')} ({model}): {ev.get('error', '')}"
        )
    return json.dumps(ev, ensure_ascii=False)[:240]
