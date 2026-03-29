"""Генерация профилей клиентов (REG)."""

from mcd_voice.profile.generator import (
    ProfileGenerator,
    generate_profile,
    generate_text_description,
    get_allergen_blacklist,
    get_group_allergen_blacklist,
    profile_to_json,
)

__all__ = [
    "ProfileGenerator",
    "generate_profile",
    "generate_text_description",
    "get_allergen_blacklist",
    "get_group_allergen_blacklist",
    "profile_to_json",
]
