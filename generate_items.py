import json
import random
from typing import Dict, List

# =========================================================
# 概要
# 1. 積荷データをランダム生成する
# 2. 各積荷にサイズ・重量・目的地IDを付与する
# 3. 生成した入力データを items_input.json に保存する
# 4. 一度生成した入力を固定して、設計・評価を公平に比較できるようにする
# =========================================================

# -----------------------------
# 積荷タイプ定義
# dimensions は JSON仕様に合わせて w, l, h を使う
# -----------------------------
ITEM_TYPES = {
    "small": {
        "w": 760,
        "l": 1130,
        "h": 550,
        "weight_range": (1000, 15000)
    },
    "medium": {
        "w": 1490,
        "l": 2260,
        "h": 900,
        "weight_range": (1000, 15000)
    },
    "large": {
        "w": 2280,
        "l": 2550,
        "h": 2355,
        "weight_range": (1000, 15000)
    }
}

DESTINATIONS = ["DEST_A", "DEST_B", "DEST_C"]


def generate_items(
    n: int = 100,
    seed: int = 42,
    output_path: str = "items_input.json"
) -> Dict:
    """
    積荷データを生成して JSON用の辞書として返す
    """
    random.seed(seed)

    items: List[Dict] = []
    size_types = list(ITEM_TYPES.keys())

    for i in range(1, n + 1):
        size_type = random.choice(size_types)
        spec = ITEM_TYPES[size_type]

        weight = round(random.uniform(*spec["weight_range"]), 2)
        destination_id = random.choice(DESTINATIONS)

        item = {
            "item_id": f"P{i:03d}",
            "size_type": size_type,
            "dimensions": {
                "w": spec["w"],
                "l": spec["l"],
                "h": spec["h"]
            },
            "weight": weight,
            "destination_id": destination_id
        }
        items.append(item)

    data = {
        "dataset_info": {
            "dataset_name": "case_01",
            "seed": seed,
            "item_count": n
        },
        "items": items
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return data


def main():
    data = generate_items(n=100, seed=42, output_path="items_input.json")
    print("items_input.json を出力しました。")
    print(f"dataset_name : {data['dataset_info']['dataset_name']}")
    print(f"seed         : {data['dataset_info']['seed']}")
    print(f"item_count   : {data['dataset_info']['item_count']}")


if __name__ == "__main__":
    main()