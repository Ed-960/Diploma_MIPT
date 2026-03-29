"""Сопоставление флагов профиля с токенами аллергенов в mcd.json / Chroma.

Делегирует в mcd_voice.profile.generator.get_allergen_blacklist —
единственный источник истины для маппинга.
"""

from __future__ import annotations

from typing import Any

from mcd_voice.profile.generator import get_allergen_blacklist


def profile_to_allergen_blacklist(profile: dict[str, Any]) -> list[str]:
    """Обратная совместимость: алиас для get_allergen_blacklist."""
    return get_allergen_blacklist(profile)
