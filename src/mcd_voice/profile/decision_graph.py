"""
Formal decision graph for synthetic profile generation.

This graph mirrors the stochastic branching used by ProfileGenerator.
It is used for diploma documentation and model validation tooling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DecisionNode:
    """Decision graph node: question and probability-weighted outcomes."""

    name: str
    question: str
    outcomes: dict[str, float]
    children: dict[str, "DecisionNode"] = field(default_factory=dict)


def _adult_flags_node() -> DecisionNode:
    return DecisionNode(
        name="adult_dietary",
        question="Adult dietary restrictions (with vegan branch override)?",
        outcomes={
            "isVegan=True": 0.04,
            "isVegan=False": 0.96,
        },
    )


def _child_flags_node() -> DecisionNode:
    return DecisionNode(
        name="child_restrictions",
        question="Child dietary restrictions sampled independently?",
        outcomes={
            "noMilk": 0.10,
            "noEggs": 0.06,
            "noNuts": 0.05,
            "noGluten": 0.03,
        },
    )


PROFILE_DECISION_GRAPH = DecisionNode(
    name="root",
    question="Sex?",
    outcomes={"male": 0.46, "female": 0.54},
    children={
        "male": DecisionNode(
            name="age_group_male",
            question="Age group?",
            outcomes={"18-30": 0.25, "31-55": 0.50, "56-80": 0.25},
            children={
                "next": DecisionNode(
                    name="psycho_male",
                    question="Psychotype?",
                    outcomes={
                        "regular": 0.30,
                        "friendly": 0.20,
                        "impatient": 0.20,
                        "polite_and_respectful": 0.15,
                        "indecisive": 0.15,
                    },
                    children={
                        "next": DecisionNode(
                            name="language_male",
                            question="Language?",
                            outcomes={"EN": 1.0, "RU": 0.0},
                            children={
                                "next": DecisionNode(
                                    name="adult_calories_male",
                                    question="Calorie target from normal distribution?",
                                    outcomes={"N(2200, 300), clipped [800, 3500]": 1.0},
                                    children={
                                        "next": _adult_flags_node(),
                                    },
                                ),
                            },
                        ),
                    },
                ),
            },
        ),
        "female": DecisionNode(
            name="age_group_female",
            question="Age group?",
            outcomes={"18-30": 0.25, "31-55": 0.50, "56-80": 0.25},
            children={
                "next": DecisionNode(
                    name="psycho_female",
                    question="Psychotype?",
                    outcomes={
                        "regular": 0.30,
                        "friendly": 0.20,
                        "impatient": 0.20,
                        "polite_and_respectful": 0.15,
                        "indecisive": 0.15,
                    },
                    children={
                        "next": DecisionNode(
                            name="language_female",
                            question="Language?",
                            outcomes={"EN": 1.0, "RU": 0.0},
                            children={
                                "next": DecisionNode(
                                    name="adult_calories_female",
                                    question="Calorie target from normal distribution?",
                                    outcomes={"N(1800, 300), clipped [800, 3500]": 1.0},
                                    children={
                                        "next": _adult_flags_node(),
                                    },
                                ),
                            },
                        ),
                    },
                ),
            },
        ),
        "companions": DecisionNode(
            name="companions_count",
            question="How many companions?",
            outcomes={
                "childQuant=0": 0.60,
                "childQuant=1": 0.20,
                "childQuant=2": 0.12,
                "childQuant=3": 0.08,
                "friendsQuant=0": 0.83,
                "friendsQuant=1": 0.12,
                "friendsQuant=2": 0.05,
            },
            children={
                "children": _child_flags_node(),
                "friends": _adult_flags_node(),
            },
        ),
    },
)


def walk_graph(node: DecisionNode, depth: int = 0) -> list[dict[str, Any]]:
    """Depth-first traversal flattened into rows for analysis."""
    rows = [
        {
            "depth": depth,
            "name": node.name,
            "question": node.question,
            "branches": len(node.outcomes),
            "outcomes": dict(node.outcomes),
        }
    ]
    for child in node.children.values():
        rows.extend(walk_graph(child, depth + 1))
    return rows


def to_mermaid(node: DecisionNode) -> str:
    """Renders the decision graph as Mermaid flowchart text."""
    lines = ["flowchart TD"]
    seen: set[str] = set()

    def visit(cur: DecisionNode) -> None:
        if cur.name not in seen:
            label = cur.question.replace('"', '\\"')
            lines.append(f'  {cur.name}["{label}"]')
            seen.add(cur.name)
        for edge_name, child in cur.children.items():
            edge_prob = cur.outcomes.get(edge_name)
            edge_label = (
                f"{edge_name} ({edge_prob:.2f})" if edge_prob is not None else edge_name
            )
            lines.append(f"  {cur.name} -->|{edge_label}| {child.name}")
            visit(child)

    visit(node)
    return "\n".join(lines)
