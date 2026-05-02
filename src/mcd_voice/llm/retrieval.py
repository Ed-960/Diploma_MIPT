from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RetrievalResult:
    rag_context: str
    rag_spec: dict[str, Any] | None


class RagRetriever:
    """Compatibility adapter around legacy CashierAgent retrieval stack."""

    def resolve(
        self,
        *,
        agent: Any,
        client_text: str,
        profile: dict[str, Any],
        history: list[dict[str, str]],
        rag_trace: list[dict[str, Any]] | None,
        rag_meta: dict[str, Any] | None,
        llm_trace: list[dict[str, Any]] | None,
    ) -> RetrievalResult:
        rag_context, rag_spec = agent._resolve_rag_context(
            client_text,
            profile,
            history,
            rag_trace=rag_trace,
            rag_meta=rag_meta,
            llm_trace=llm_trace,
        )
        return RetrievalResult(rag_context=rag_context, rag_spec=rag_spec)

