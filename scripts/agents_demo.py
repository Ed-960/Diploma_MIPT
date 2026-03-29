"""
Демо агентов ClientAgent и CashierAgent.
Запуск: python scripts/agents_demo.py

Требуется OPENAI_API_KEY и загруженная Chroma (scripts/load_chroma.py).
"""

import _bootstrap

_bootstrap.ensure_src()

from mcd_voice.llm import CashierAgent, ClientAgent, DEFAULT_MODEL
from mcd_voice.profile import generate_profile, generate_text_description, profile_to_json

if __name__ == "__main__":
    profile = generate_profile()
    print("=== Профиль ===")
    print(profile_to_json(profile))
    print(f"\nОписание для LLM: {generate_text_description(profile)}\n")

    try:
        client = ClientAgent(model=DEFAULT_MODEL)
        cashier = CashierAgent(model=DEFAULT_MODEL)
    except RuntimeError as e:
        print(f"Пропуск: {e}")
        raise SystemExit(1)

    history: list[dict[str, str]] = []
    order_state = {"items": [], "total_energy": 0.0, "allergens_in_order": []}

    print("=== Мини-диалог (3 хода) ===\n")
    for turn in range(3):
        client_msg = client.generate_response(profile, history)
        history.append({"speaker": "client", "text": client_msg})
        print(f"Клиент: {client_msg}\n")

        cashier_msg = cashier.generate_response(profile, history, order_state)
        history.append({"speaker": "cashier", "text": cashier_msg})
        print(f"Кассир: {cashier_msg}\n")
