import json
import time

CONTAINER_WIDTH = 2300
CONTAINER_LENGTH = 12000
CONTAINER_HEIGHT = 2400
MAX_WEIGHT = 24000

CENTER_Y = CONTAINER_LENGTH / 2


def load_items(path="items_input.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["items"]


def item_volume(item):
    d = item["dimensions"]
    return d["w"] * d["l"] * d["h"]


def sort_items(items):
    return sorted(
        items,
        key=lambda x: (
            x["destination_id"],
            -x["weight"],
            -item_volume(x)
        )
    )


def rotate_if_needed(w, l):
    if w > CONTAINER_WIDTH and l <= CONTAINER_WIDTH:
        return l, w, True
    return w, l, False


def create_container(container_id, destination):
    return {
        "container_id": container_id,
        "destination_id": destination,
        "total_weight": 0,
        "items": [],
        "_x": 0,
        "_y": 0,
        "_z": 0,
        "_row_length": 0
    }


def can_fit(container, w, l, h, weight):
    if container["total_weight"] + weight > MAX_WEIGHT:
        return False

    x = container["_x"]
    y = container["_y"]
    z = container["_z"]

    if x + w > CONTAINER_WIDTH:
        x = 0
        y += container["_row_length"]

    if y + l > CONTAINER_LENGTH:
        y = 0
        z += h

    if z + h > CONTAINER_HEIGHT:
        return False

    return True


def place_item(container, item):
    w = item["dimensions"]["w"]
    l = item["dimensions"]["l"]
    h = item["dimensions"]["h"]

    w, l, rotated = rotate_if_needed(w, l)

    if container["_x"] + w > CONTAINER_WIDTH:
        container["_x"] = 0
        container["_y"] += container["_row_length"]
        container["_row_length"] = 0

    if container["_y"] + l > CONTAINER_LENGTH:
        container["_y"] = 0
        container["_z"] += h

    x = container["_x"]
    y = container["_y"]
    z = container["_z"]

    placed = {
        "item_id": item["item_id"],
        "size_type": item["size_type"],
        "dimensions": {
            "w": w,
            "l": l,
            "h": h
        },
        "position": {
            "x": x,
            "y": y,
            "z": z
        },
        "weight": item["weight"],
        "is_rotated": rotated
    }

    container["items"].append(placed)

    container["total_weight"] += item["weight"]

    container["_x"] += w
    container["_row_length"] = max(container["_row_length"], l)


def pack_items(items):
    containers = []

    grouped = {}

    for item in items:
        dest = item["destination_id"]
        grouped.setdefault(dest, []).append(item)

    container_id = 1

    for destination, group_items in grouped.items():

        current = create_container(container_id, destination)

        for item in group_items:

            w = item["dimensions"]["w"]
            l = item["dimensions"]["l"]
            h = item["dimensions"]["h"]

            w, l, _ = rotate_if_needed(w, l)

            if not can_fit(current, w, l, h, item["weight"]):
                containers.append(current)
                container_id += 1
                current = create_container(container_id, destination)

            place_item(current, item)

        containers.append(current)
        container_id += 1

    return containers


def cleanup_containers(containers):
    for c in containers:
        c.pop("_x", None)
        c.pop("_y", None)
        c.pop("_z", None)
        c.pop("_row_length", None)


def save_result(containers, execution_time):

    output = {
        "project_info": {
            "team_name": "kojima",
            "execution_time_ms": execution_time
        },
        "containers": containers
    }

    with open("layout_result.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def main():

    start = time.time()

    items = load_items()

    items = sort_items(items)

    containers = pack_items(items)

    cleanup_containers(containers)

    end = time.time()

    execution_time = int((end - start) * 1000)

    save_result(containers, execution_time)

    print("layout_result.json を出力しました")


if __name__ == "__main__":
    main()
