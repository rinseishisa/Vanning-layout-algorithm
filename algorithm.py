from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd


# Match evaluator-side 40ft constants exactly.
CONTAINER_LENGTH_MM = 12000
CONTAINER_WIDTH_MM = 2300
CONTAINER_HEIGHT_MM = 2400
MAX_CONTAINER_WEIGHT_KG = 24000.0
MIN_FILL_RATE = 0.50
Y_CENTER_MM = CONTAINER_LENGTH_MM / 2.0
# Match the updated Y-axis scoring guide for a 12000 mm container.
Y_DEVIATION_FULL_SCORE_MM = 1200.0
Y_DEVIATION_LIMIT_MM = 3000.0
ROTATIONS = [0, 90]
CONTAINER_VOLUME_MM3 = CONTAINER_LENGTH_MM * CONTAINER_WIDTH_MM * CONTAINER_HEIGHT_MM

SIZE_SPECS = {
    "small": (760, 1130, 550),
    "medium": (1490, 2260, 900),
    "large": (2550, 2280, 2355),
}

REQUIRED_COLUMNS = [
    "item_id",
    "size_type",
    "width",
    "length",
    "height",
    "weight",
    "destination_id",
]


@dataclass(frozen=True)
class Item:
    item_id: str
    size_type: str
    width: int
    length: int
    height: int
    weight: float
    destination_id: str
    volume: int


@dataclass(frozen=True)
class PlacedItem:
    item_id: str
    size_type: str
    width: int
    length: int
    height: int
    x: int
    y: int
    z: int
    weight: float
    destination_id: str
    is_rotated: bool

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.length

    @property
    def z2(self) -> int:
        return self.z + self.height

    @property
    def center_y(self) -> float:
        return self.y + self.length / 2.0


@dataclass
class Container:
    container_id: int
    destination_id: str
    items: List[PlacedItem] = field(default_factory=list)

    @property
    def total_weight(self) -> float:
        return sum(item.weight for item in self.items)

    @property
    def used_volume(self) -> int:
        return sum(item.width * item.length * item.height for item in self.items)

    @property
    def fill_rate(self) -> float:
        return self.used_volume / CONTAINER_VOLUME_MM3


def read_generated_items(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"input json not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Accept either a bare item list or an object with an items array.
    if isinstance(raw_data, dict):
        items_data = raw_data.get("items")
    else:
        items_data = raw_data

    if not isinstance(items_data, list):
        raise ValueError("input json must be a list or contain an 'items' list")

    df = pd.DataFrame(items_data)

    # Accept evaluator-compatible items_input.json shape with nested dimensions.
    if "dimensions" in df.columns:
        dimensions = df["dimensions"].apply(lambda value: value if isinstance(value, dict) else {})
        df = df.assign(
            width=dimensions.apply(lambda value: value.get("w")),
            length=dimensions.apply(lambda value: value.get("l")),
            height=dimensions.apply(lambda value: value.get("h")),
        )

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df = df.dropna(subset=REQUIRED_COLUMNS)

    for col in ["width", "length", "height", "weight"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[["width", "length", "height", "weight"]].isna().any().any():
        bad_rows = df[df[["width", "length", "height", "weight"]].isna().any(axis=1)]
        raise ValueError(f"numeric conversion failed:\n{bad_rows}")

    if (df[["width", "length", "height", "weight"]] <= 0).any().any():
        bad_rows = df[(df[["width", "length", "height", "weight"]] <= 0).any(axis=1)]
        raise ValueError(f"non-positive dimensions or weight:\n{bad_rows}")

    if df["item_id"].duplicated().any():
        duplicated = df.loc[df["item_id"].duplicated(), "item_id"].tolist()
        raise ValueError(f"duplicated item_id detected: {duplicated}")

    for size_type, (width, length, height) in SIZE_SPECS.items():
        mask = df["size_type"].astype(str).str.lower() == size_type
        if not mask.any():
            continue
        inconsistent = df[mask & ((df["width"] != width) | (df["length"] != length) | (df["height"] != height))]
        if not inconsistent.empty:
            raise ValueError(f"inconsistent dimensions for size_type={size_type}:\n{inconsistent}")

    df["item_id"] = df["item_id"].astype(str).str.strip()
    df["size_type"] = df["size_type"].astype(str).str.strip().str.lower()
    df["destination_id"] = df["destination_id"].astype(str).str.strip()
    df["width"] = df["width"].astype(int)
    df["length"] = df["length"].astype(int)
    df["height"] = df["height"].astype(int)
    df["weight"] = df["weight"].astype(float)
    return df.reset_index(drop=True)


def build_items(df: pd.DataFrame) -> List[Item]:
    items: List[Item] = []
    for row in df.itertuples(index=False):
        items.append(
            Item(
                item_id=row.item_id,
                size_type=row.size_type,
                width=row.width,
                length=row.length,
                height=row.height,
                weight=row.weight,
                destination_id=row.destination_id,
                volume=int(row.width * row.length * row.height),
            )
        )
    return sorted(items, key=lambda item: (item.destination_id, -item.weight, -item.volume, item.item_id))


def rotated_dims(item: Item, rotation: int) -> Tuple[int, int, int]:
    if rotation == 0:
        return item.width, item.length, item.height
    if rotation == 90:
        return item.length, item.width, item.height
    raise ValueError(f"unsupported rotation: {rotation}")


def fits_in_container(x: int, y: int, z: int, width: int, length: int, height: int) -> bool:
    return (
        x >= 0
        and y >= 0
        and z >= 0
        and x + width <= CONTAINER_WIDTH_MM
        and y + length <= CONTAINER_LENGTH_MM
        and z + height <= CONTAINER_HEIGHT_MM
    )


def overlaps(a: PlacedItem, b: PlacedItem) -> bool:
    return not (
        a.x2 <= b.x
        or b.x2 <= a.x
        or a.y2 <= b.y
        or b.y2 <= a.y
        or a.z2 <= b.z
        or b.z2 <= a.z
    )


def is_supported(candidate: PlacedItem, existing: Sequence[PlacedItem]) -> bool:
    if candidate.z == 0:
        return True

    for base in existing:
        if (
            base.z2 == candidate.z
            and base.x <= candidate.x
            and candidate.x2 <= base.x2
            and base.y <= candidate.y
            and candidate.y2 <= base.y2
        ):
            return True
    return False


def compute_y_center_of_gravity(items: Sequence[PlacedItem]) -> float:
    if not items:
        return Y_CENTER_MM
    total_weight = sum(item.weight for item in items)
    return sum(item.center_y * item.weight for item in items) / total_weight


def y_deviation(items: Sequence[PlacedItem]) -> float:
    return abs(compute_y_center_of_gravity(items) - Y_CENTER_MM)


def make_placed_item(item: Item, container_id: int, rotation: int, x: int, y: int, z: int) -> PlacedItem:
    width, length, height = rotated_dims(item, rotation)
    return PlacedItem(
        item_id=item.item_id,
        size_type=item.size_type,
        width=width,
        length=length,
        height=height,
        x=int(x),
        y=int(y),
        z=int(z),
        weight=float(item.weight),
        destination_id=item.destination_id,
        is_rotated=(rotation == 90),
    )


def can_place(container: Container, candidate: PlacedItem) -> bool:
    if candidate.destination_id != container.destination_id:
        return False
    if not fits_in_container(candidate.x, candidate.y, candidate.z, candidate.width, candidate.length, candidate.height):
        return False
    if any(overlaps(candidate, placed) for placed in container.items):
        return False
    if not is_supported(candidate, container.items):
        return False
    if container.total_weight + candidate.weight > MAX_CONTAINER_WEIGHT_KG + 1e-9:
        return False
    if y_deviation([*container.items, candidate]) > Y_DEVIATION_LIMIT_MM + 1e-9:
        return False
    return True


def generate_candidate_points(container: Container) -> List[Tuple[int, int, int]]:
    points = {(0, 0, 0)}
    for item in container.items:
        points.add((item.x2, item.y, item.z))
        points.add((item.x, item.y2, item.z))
        points.add((item.x, item.y, item.z2))
        points.add((item.x2, item.y2, item.z))
        points.add((item.x2, item.y, item.z2))
        points.add((item.x, item.y2, item.z2))
    feasible = [
        point
        for point in points
        if point[0] <= CONTAINER_WIDTH_MM and point[1] <= CONTAINER_LENGTH_MM and point[2] <= CONTAINER_HEIGHT_MM
    ]
    return sorted(feasible, key=lambda point: (point[2], point[1], point[0]))


def centered_floor_point(width: int, length: int) -> Tuple[int, int, int]:
    x = max(0, min(CONTAINER_WIDTH_MM - width, int(round((CONTAINER_WIDTH_MM - width) / 2.0))))
    y = max(0, min(CONTAINER_LENGTH_MM - length, int(round((CONTAINER_LENGTH_MM - length) / 2.0))))
    return x, y, 0


def bounding_box_volume(items: Sequence[PlacedItem]) -> int:
    if not items:
        return 0
    max_x = max(item.x2 for item in items)
    max_y = max(item.y2 for item in items)
    max_z = max(item.z2 for item in items)
    return max_x * max_y * max_z


def candidate_score(container: Container, candidate: PlacedItem) -> Tuple[float, float, float, int, int, int]:
    new_items = [*container.items, candidate]
    deviation = y_deviation(new_items)
    dead_space = bounding_box_volume(new_items) - sum(item.width * item.length * item.height for item in new_items)
    # Prefer candidates that stay inside the full-score deviation band before dead-space tie-breaks.
    deviation_penalty = max(0.0, deviation - Y_DEVIATION_FULL_SCORE_MM)
    return (deviation_penalty, deviation, dead_space, candidate.z, candidate.y, candidate.x, int(candidate.is_rotated))


def find_best_placement(container: Container, item: Item) -> Optional[PlacedItem]:
    best_candidate: Optional[PlacedItem] = None
    best_score: Optional[Tuple[float, float, float, int, int, int]] = None
    base_points = generate_candidate_points(container)

    for rotation in ROTATIONS:
        width, length, _ = rotated_dims(item, rotation)
        candidate_points = sorted(
            set([*base_points, centered_floor_point(width, length)]),
            key=lambda point: (point[2], point[1], point[0]),
        )
        for x, y, z in candidate_points:
            candidate = make_placed_item(item, container.container_id, rotation, x, y, z)
            if not can_place(container, candidate):
                continue
            score = candidate_score(container, candidate)
            if best_score is None or score < best_score:
                best_candidate = candidate
                best_score = score

    return best_candidate


def pack_items(items: Sequence[Item]) -> List[Container]:
    containers: List[Container] = []
    for item in items:
        placed = False
        for container in containers:
            if container.destination_id != item.destination_id:
                continue
            candidate = find_best_placement(container, item)
            if candidate is not None:
                container.items.append(candidate)
                placed = True
                break

        if placed:
            continue

        new_container = Container(container_id=len(containers) + 1, destination_id=item.destination_id)
        candidate = find_best_placement(new_container, item)
        if candidate is None:
            raise RuntimeError(f"item {item.item_id} cannot be placed even in an empty container")
        new_container.items.append(candidate)
        containers.append(new_container)

    return containers


def evaluate_solution(containers: Sequence[Container]) -> Dict[str, object]:
    violations: List[str] = []
    low_fill_container_ids: List[int] = []
    summaries: List[Dict[str, object]] = []

    for container in containers:
        destination_set = {item.destination_id for item in container.items}
        if len(destination_set) > 1:
            violations.append(f"mixed destinations in container {container.container_id}: {sorted(destination_set)}")

        for item in container.items:
            if not fits_in_container(item.x, item.y, item.z, item.width, item.length, item.height):
                violations.append(f"out of bounds: {item.item_id} in container {container.container_id}")
            if not is_supported(item, [other for other in container.items if other.item_id != item.item_id]):
                violations.append(f"unsupported item: {item.item_id} in container {container.container_id}")

        for i, a in enumerate(container.items):
            for b in container.items[i + 1:]:
                if overlaps(a, b):
                    violations.append(f"overlap: {a.item_id} vs {b.item_id} in container {container.container_id}")

        if container.total_weight > MAX_CONTAINER_WEIGHT_KG + 1e-9:
            violations.append(f"overweight container {container.container_id}: {container.total_weight}")

        deviation = y_deviation(container.items)
        if deviation > Y_DEVIATION_LIMIT_MM + 1e-9:
            violations.append(f"excessive Y-axis deviation in container {container.container_id}: {deviation:.1f}mm")

        if container.fill_rate < MIN_FILL_RATE:
            low_fill_container_ids.append(container.container_id)

        summaries.append(
            {
                "container_id": container.container_id,
                "destination_id": container.destination_id,
                "item_count": len(container.items),
                "total_weight": round(container.total_weight, 3),
                "fill_rate": round(container.fill_rate, 6),
                "y_center_of_gravity": round(compute_y_center_of_gravity(container.items), 3),
                "y_deviation": round(deviation, 3),
            }
        )

    return {
        "disqualified": len(violations) > 0,
        "violations": violations,
        "low_fill_container_ids": low_fill_container_ids,
        "container_count": len(containers),
        "average_fill_rate": round(sum(container.fill_rate for container in containers) / max(len(containers), 1), 6),
        "max_y_deviation": round(max((y_deviation(container.items) for container in containers), default=0.0), 3),
        "container_summaries": summaries,
    }


def build_output_json(containers: Sequence[Container], team_name: str, execution_time_ms: int) -> Dict[str, object]:
    output_containers = []
    for container in containers:
        # Fail fast if the container-level destination is missing.
        if not container.destination_id:
            raise ValueError(f"container {container.container_id} is missing destination_id")

        # Copy destination_id to each item for evaluator compatibility.
        output_items = [
            {
                "item_id": item.item_id,
                "size_type": item.size_type,
                "dimensions": {"w": item.width, "l": item.length, "h": item.height},
                "position": {"x": item.x, "y": item.y, "z": item.z},
                "weight": round(item.weight, 3),
                "is_rotated": item.is_rotated,
                "destination_id": container.destination_id,
            }
            for item in sorted(container.items, key=lambda placed: placed.item_id)
        ]

        output_containers.append(
            {
                "container_id": container.container_id,
                "destination_id": container.destination_id,
                "total_weight": round(container.total_weight, 3),
                "items": output_items,
            }
        )

    return {
        "project_info": {
            "team_name": team_name,
            "execution_time_ms": execution_time_ms,
        },
        "containers": output_containers,
    }


def validate_output_schema(data: Dict[str, object]) -> None:
    if "project_info" not in data or "containers" not in data:
        raise ValueError("output json must contain project_info and containers")

    if not isinstance(data["project_info"], dict):
        raise ValueError("project_info must be a dict")

    containers = data["containers"]
    if not isinstance(containers, list):
        raise ValueError("containers must be a list")

    for index, container in enumerate(containers, start=1):
        required_container_keys = ["container_id", "destination_id", "total_weight", "items"]
        missing_container_keys = [key for key in required_container_keys if key not in container]
        if missing_container_keys:
            raise ValueError(f"container[{index}] is missing keys: {missing_container_keys}")
        if "destination_id" not in container or not container["destination_id"]:
            raise ValueError(f"container[{index}] is missing destination_id")
        if not isinstance(container["container_id"], int):
            raise ValueError(f"container[{index}].container_id must be an int")
        if not isinstance(container["total_weight"], (int, float)):
            raise ValueError(f"container[{index}].total_weight must be numeric")
        if "items" not in container or not isinstance(container["items"], list):
            raise ValueError(f"container[{index}].items must be a list")

        for item_index, item in enumerate(container["items"], start=1):
            required_item_keys = [
                "item_id",
                "size_type",
                "dimensions",
                "position",
                "weight",
                "is_rotated",
                "destination_id",
            ]
            missing_keys = [key for key in required_item_keys if key not in item]
            if missing_keys:
                raise ValueError(f"container[{index}].items[{item_index}] is missing keys: {missing_keys}")
            if not item["destination_id"]:
                raise ValueError(f"container[{index}].items[{item_index}] has empty destination_id")
            if not isinstance(item["weight"], (int, float)):
                raise ValueError(f"container[{index}].items[{item_index}].weight must be numeric")
            if not isinstance(item["is_rotated"], bool):
                raise ValueError(f"container[{index}].items[{item_index}].is_rotated must be bool")

            dimensions = item["dimensions"]
            if not isinstance(dimensions, dict):
                raise ValueError(f"container[{index}].items[{item_index}].dimensions must be a dict")
            missing_dimension_keys = [key for key in ["w", "l", "h"] if key not in dimensions]
            if missing_dimension_keys:
                raise ValueError(
                    f"container[{index}].items[{item_index}].dimensions is missing keys: {missing_dimension_keys}"
                )
            for dimension_key in ["w", "l", "h"]:
                if not isinstance(dimensions[dimension_key], int):
                    raise ValueError(
                        f"container[{index}].items[{item_index}].dimensions.{dimension_key} must be an int"
                    )

            position = item["position"]
            if not isinstance(position, dict):
                raise ValueError(f"container[{index}].items[{item_index}].position must be a dict")
            missing_position_keys = [key for key in ["x", "y", "z"] if key not in position]
            if missing_position_keys:
                raise ValueError(
                    f"container[{index}].items[{item_index}].position is missing keys: {missing_position_keys}"
                )
            for position_key in ["x", "y", "z"]:
                if not isinstance(position[position_key], int):
                    raise ValueError(
                        f"container[{index}].items[{item_index}].position.{position_key} must be an int"
                    )


def resolve_output_path(output: Path, submission_name: str | None, eval_root: Path | None) -> Path:
    if submission_name and eval_root:
        # Write directly into evaluator batch input/<submission_name>/layout_result.json.
        return eval_root / "input" / submission_name / "layout_result.json"
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run container layout from generated items json")
    parser.add_argument("--input", type=Path, default=Path("generated_items.json"))
    parser.add_argument("--output", type=Path, default=Path("layout_result.json"))
    parser.add_argument("--team-name", default="Team_Alpha")
    parser.add_argument("--submission-name", default=None)
    parser.add_argument("--eval-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.perf_counter()

    table = read_generated_items(args.input)
    items = build_items(table)
    containers = pack_items(items)

    execution_time_ms = int((time.perf_counter() - started_at) * 1000)
    result_json = build_output_json(containers, args.team_name, execution_time_ms)
    validate_output_schema(result_json)
    evaluation = evaluate_solution(containers)
    output_path = resolve_output_path(args.output, args.submission_name, args.eval_root)

    # Create the destination directory so evaluator-oriented paths work without manual setup.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)

    print(f"input_file={args.input.resolve()}")
    print(f"output_file={output_path.resolve()}")
    print(f"container_count={evaluation['container_count']}")
    print(f"average_fill_rate={evaluation['average_fill_rate']}")
    print(f"max_y_deviation={evaluation['max_y_deviation']}")
    print(f"low_fill_container_ids={evaluation['low_fill_container_ids']}")
    print(f"violations={evaluation['violations']}")
    print(json.dumps(evaluation['container_summaries'], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
