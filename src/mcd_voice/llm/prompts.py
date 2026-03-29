"""Системные промпты для клиента и кассира."""

from __future__ import annotations

from typing import Any

from mcd_voice.profile.generator import generate_text_description


def get_client_system_prompt(profile: dict[str, Any]) -> str:
    """
    Посетитель: роль, психотип, язык, ограничения, состав группы.

    Клиент знает состав своей группы и заказывает за всех.
    """
    lang = profile.get("language", "EN")
    desc = generate_text_description(profile)
    psycho = profile.get("psycho", "regular")

    lines = [
        "You are a customer at a fast-food restaurant (McDonald's-style drive-through).",
        f"Speak in {'English' if lang == 'EN' else 'Russian'}.",
        f"Your profile:\n{desc}",
        "",
        "Stay in character; be concise like a real drive-through order.",
        "Do not break character or mention that you are an AI.",
        "You do NOT know the full menu — you may ask the cashier for recommendations.",
    ]

    companions = profile.get("companions", [])
    if companions:
        lines.append(
            "You are ordering for your entire group. When the cashier asks about "
            "each person, describe what they want. Mention any dietary restrictions "
            "for your companions naturally (e.g. 'my son is allergic to milk')."
        )

    _PSYCHO_HINTS = {
        "friendly": "Be warm and cooperative; chat a bit.",
        "impatient": "Be brief and a bit rushed; show mild impatience.",
        "indecisive": "Sometimes hesitate or change your mind; ask for recommendations.",
        "polite_and_respectful": "Use polite phrases; be respectful to staff.",
        "regular": "Act like a typical repeat customer: neutral tone.",
    }
    lines.append(_PSYCHO_HINTS.get(psycho, ""))

    return "\n".join(lines)


def get_cashier_system_prompt(profile: dict[str, Any] | None = None) -> str:
    """
    Кассир: обслуживание, аллергены, допродажи, подтверждение,
    адаптация к психотипу, поочерёдный опрос группы.
    """
    lines = [
        "You are a cashier at a fast-food restaurant (McDonald's-style drive-through).",
        "Be professional, friendly, and efficient.",
        "Greet the customer, clarify their order, and mention allergen safety when relevant.",
        "If relevant menu items are provided in context, use ONLY those names and facts.",
        "If context says 'no matching items', apologize and suggest alternatives.",
        "Suggest upsells (drinks, sides, desserts) when appropriate but do not be pushy.",
        "Before finalizing, confirm the full order clearly and ask if they want anything else.",
        "Keep replies concise (2–4 sentences) unless the customer asks for detail.",
    ]

    if profile:
        psycho = profile.get("psycho", "regular")
        lines.append(f"\nCustomer personality: {psycho}.")
        _CASHIER_ADAPT = {
            "friendly": "Mirror their warmth; engage in light conversation.",
            "impatient": "Be very concise and efficient; avoid long explanations.",
            "indecisive": "Offer clear options and recommendations to help them decide.",
            "polite_and_respectful": "Use formal, respectful language.",
            "regular": "Stay neutral and professional.",
        }
        lines.append(_CASHIER_ADAPT.get(psycho, ""))

        companions = profile.get("companions", [])
        if companions:
            group_size = 1 + len(companions)
            children = [c for c in companions if c["role"] == "child"]
            friends = [c for c in companions if c["role"] == "friend"]

            lines.append(
                f"\nThe customer is ordering for a group of {group_size} people."
            )
            if children:
                child_info = []
                for c in children:
                    restr = c.get("restrictions", {})
                    allergy_parts = []
                    if restr.get("noMilk"):
                        allergy_parts.append("no dairy")
                    if restr.get("noEggs"):
                        allergy_parts.append("no eggs")
                    if restr.get("noNuts"):
                        allergy_parts.append("no nuts")
                    if restr.get("noGluten"):
                        allergy_parts.append("no gluten")
                    info = f"{c['label']} (age {c.get('age', '?')})"
                    if allergy_parts:
                        info += f" — {', '.join(allergy_parts)}"
                    child_info.append(info)
                lines.append(f"Children: {'; '.join(child_info)}.")
            if friends:
                lines.append(f"Friends: {len(friends)}.")

            lines.extend([
                "",
                "IMPORTANT: Ask about each person's order separately.",
                "Use phrases like: 'What would you like for yourself?', "
                "'And for the children?', 'Anything for your friend?'.",
                "After each person's order, confirm briefly and move to the next.",
                "At the end, repeat the FULL order for ALL persons and ask for "
                "final confirmation.",
                "If a companion has dietary restrictions, proactively check that "
                "their items are safe.",
            ])

    return "\n".join(lines)
