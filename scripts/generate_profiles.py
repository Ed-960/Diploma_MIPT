"""
Генерация пула профилей в отдельный JSON файл.

Запуск:
  python scripts/generate_profiles.py --num_profiles 1000 --output_file profiles_1000.json --seed 42
"""

from __future__ import annotations

import argparse
import json
import random

import _bootstrap

_bootstrap.ensure_src()

from mcd_voice.profile import ProfileGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate profile pool JSON.")
    parser.add_argument(
        "--num_profiles",
        type=int,
        default=1000,
        help="Количество профилей (по умолчанию 1000).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="profiles_1000.json",
        help="Файл для сохранения JSON-массива профилей.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed для воспроизводимой генерации.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed) if args.seed is not None else None
    gen = ProfileGenerator(rng=rng)
    profiles = [gen.generate() for _ in range(args.num_profiles)]

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)

    print(
        f"Saved {len(profiles)} profiles to {args.output_file} "
        f"(seed={args.seed})."
    )


if __name__ == "__main__":
    main()
