from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TurnPlan:
    intent: str = "lookup"
    search_query: str = ""
    compare_metrics: list[dict[str, Any]] = field(default_factory=list)
    excluded_lexical: list[str] = field(default_factory=list)
    restrictions: list[str] = field(default_factory=list)
    requested_items: list[str] = field(default_factory=list)
    max_kcal: float | None = None
    min_kcal: float | None = None
    override_restriction: bool = False
    finalize: bool = False
    full_nutrition_context: bool = False

    @classmethod
    def from_legacy(
        cls,
        *,
        client_text: str,
        spec: dict[str, Any] | None,
        intent_resolver: Any,
        nutrition_resolver: Any,
    ) -> "TurnPlan":
        raw = spec or {}
        plan = cls(
            intent=str(intent_resolver(raw)),
            search_query=str(raw.get("search_query") or ""),
            compare_metrics=[
                m for m in (raw.get("compare_metrics") or [])
                if isinstance(m, dict)
            ],
            excluded_lexical=[str(x) for x in (raw.get("excluded_lexical") or []) if str(x).strip()],
            restrictions=[str(x) for x in (raw.get("restrictions") or []) if str(x).strip()],
            requested_items=[str(x) for x in (raw.get("requested_items") or []) if str(x).strip()],
            max_kcal=raw.get("max_kcal"),
            min_kcal=raw.get("min_kcal"),
            override_restriction=bool(raw.get("override_restriction")),
            finalize=bool(raw.get("finalize")),
            full_nutrition_context=bool(nutrition_resolver(client_text, raw)),
        )
        if not plan.search_query:
            plan.search_query = str(client_text or "").strip()
        return plan

    def as_legacy_spec(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "search_query": self.search_query,
            "compare_metrics": list(self.compare_metrics),
            "excluded_lexical": list(self.excluded_lexical),
            "max_kcal": self.max_kcal,
            "min_kcal": self.min_kcal,
            "restrictions": list(self.restrictions),
            "requested_items": list(self.requested_items),
            "override_restriction": self.override_restriction,
            "finalize": self.finalize,
        }


@dataclass(slots=True)
class TurnContext:
    client_text: str
    profile: dict[str, Any]
    history: list[dict[str, str]]
    order_state: dict[str, Any]
    plan: TurnPlan | None = None
    rag_context: str = ""
    rag_spec: dict[str, Any] | None = None

