from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class GroundingFacts:
    intent: str
    allow_calories: bool
    full_nutrition_context: bool
    rag_context: str
    rag_spec: dict[str, Any] | None


class GroundingService:
    """Build structured facts used by response composer."""

    def build(
        self,
        *,
        intent: str,
        rag_spec: dict[str, Any] | None,
        rag_context: str,
        full_nutrition_context: bool,
    ) -> GroundingFacts:
        allow_calories = (
            intent == "calorie_tune"
            or any(
                str(m.get("field")) == "energy"
                for m in (rag_spec or {}).get("compare_metrics", [])
                if isinstance(m, dict)
            )
            or (rag_spec or {}).get("max_kcal") is not None
            or (rag_spec or {}).get("min_kcal") is not None
        )
        return GroundingFacts(
            intent=intent,
            allow_calories=allow_calories,
            full_nutrition_context=full_nutrition_context,
            rag_context=rag_context,
            rag_spec=rag_spec,
        )

