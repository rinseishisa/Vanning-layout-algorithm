from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


SIZE_SPECS = {
    "small": (760, 1130, 550),
    "medium": (1490, 2260, 900),
    "large": (2550, 2280, 2355),
}

WEIGHT_RANGES = {
    # Match the requirement definition: every generated item is 1,000kg to 15,000kg.
    "small": (1000, 15000),
    "medium": (1000, 15000),
    "large": (1000, 15000),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate packing input json from small/medium/large counts")
    parser.add_argument("--small", type=int, default=8)
    parser.add_argument("--medium", type=int, default=12)
    parser.add_argument("--large", type=int, default=4)
    parser.add_argument("--destinations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("generated_items.json"))
    return parser.parse_args()


def build_rows(size_type: str, count: int, destination_count: int, rng: random.Random, start_index: int) -> list[dict[str, object]]:
    width, length, height = SIZE_SPECS[size_type]
    min_weight, max_weight = WEIGHT_RANGES[size_type]
    rows = []
    for offset in range(count):
        item_index = start_index + offset
        destination_id = f"DEST_{(offset % destination_count) + 1:02d}"
        rows.append(
            {
                "item_id": f"P{item_index:04d}",
                "size_type": size_type,
                "width": width,
                "length": length,
                "height": height,
                "weight": rng.randint(min_weight, max_weight),
                "destination_id": destination_id,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    rows: list[dict[str, object]] = []
    next_index = 1
    for size_type, count in [("small", args.small), ("medium", args.medium), ("large", args.large)]:
        new_rows = build_rows(size_type, count, args.destinations, rng, next_index)
        rows.extend(new_rows)
        next_index += count

    output_payload = {
        "dataset_info": {
            "dataset_name": "generated_items",
            "seed": args.seed,
            "item_count": len(rows),
        },
        "items": [
            {
                "item_id": row["item_id"],
                "size_type": row["size_type"],
                "dimensions": {
                    "w": row["width"],
                    "l": row["length"],
                    "h": row["height"],
                },
                "weight": row["weight"],
                "destination_id": row["destination_id"],
            }
            for row in rows
        ],
    }
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    print(f"generated_file={args.output.resolve()}")
    print(f"item_count={len(rows)}")
    print(f"destinations={args.destinations}")
    print(f"seed={args.seed}")


if __name__ == "__main__":
    main()
