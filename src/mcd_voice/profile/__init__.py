"""Генерация профилей клиентов (REG)."""

from mcd_voice.profile.generator import (
    ProfileGenerator,
    generate_profile,
    generate_text_description,
    get_allergen_blacklist,
    get_group_allergen_blacklist,
    neutral_drive_through_profile,
    profile_to_json,
)
from mcd_voice.profile.decision_graph import (
    DecisionNode,
    PROFILE_DECISION_GRAPH,
    to_mermaid,
    walk_graph,
)

__all__ = [
    "ProfileGenerator",
    "generate_profile",
    "generate_text_description",
    "get_allergen_blacklist",
    "get_group_allergen_blacklist",
    "neutral_drive_through_profile",
    "profile_to_json",
    "DecisionNode",
    "PROFILE_DECISION_GRAPH",
    "walk_graph",
    "to_mermaid",
]
