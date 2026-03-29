"""Вызовы LLM и промпты."""

from mcd_voice.llm.agent import (
    DEFAULT_MODEL,
    CashierAgent,
    ClientAgent,
)
from mcd_voice.llm.prompts import get_cashier_system_prompt, get_client_system_prompt

__all__ = [
    "CashierAgent",
    "ClientAgent",
    "DEFAULT_MODEL",
    "get_cashier_system_prompt",
    "get_client_system_prompt",
]
