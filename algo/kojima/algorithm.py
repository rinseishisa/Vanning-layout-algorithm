import json

CONTAINER_WIDTH = 2300
CONTAINER_LENGTH = 12000
CONTAINER_HEIGHT = 2400


def load_items(path="items_input.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["items"]


def sort_items(items):
    # 重い順 + 体積順
    return sorted(
        items,
        key=lambda x: (
            -x["weight"],
            -(x["dimensions"]["w"] * x["dimensions"]["l"] * x["dimensions"]["h"])
        )
    )


def pack(items):
    placed = []

    x = 0
    y = 0
    z = 0
    row_height = 0

    # 中央からスタート
    x = CONTAINER_WIDTH // 2

    for item in items:
        w = item["dimensions"]["w"]
        l = item["dimensions"]["l"]
        h = item["dimensions"]["h"]

        # 横はみ出たら次の行
        if x + w > CONTAINER_WIDTH:
            x = 0
            y += row_height
            row_height = 0

        # 奥はみ出たら上に積む
        if y + l > CONTAINER_LENGTH:
            y = 0
            z += h

        placed.append({
            "item_id": item["item_id"],
            "x": x,
            "y": y,
            "z": z,
            "w": w,
            "l": l,
            "h": h,
            "weight": item["weight"],
            "destination_id": item["destination_id"]
        })

        x += w
        row_height = max(row_height, l)

    return placed


def save_result(result):
    output = {
        "containers": [
            {
                "container_id": 1,
                "items": result
            }
        ]
    }

    with open("layout_result.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def main():
    items = load_items()
    items = sort_items(items)
    result = pack(items)
    save_result(result)
    print("layout_result.json を出力しました")


if __name__ == "__main__":
    main()
