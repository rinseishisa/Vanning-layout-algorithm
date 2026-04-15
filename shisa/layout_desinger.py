import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# =========================================================
# 概要
# 1. generate_items.py が生成した items_input.json を読み込む
# 2. destination_id ごとに荷物を分ける
# 3. 同一 destination_id の荷物のみを同一コンテナに積む
# 4. 荷物は天地固定、XY平面での90度回転のみ許可
# 5. バンニング結果を layout_result.json に出力する
#
# 前提
# - 宙に浮かせないため、すべて z=0 に配置する
# - 荷物同士の重なりは禁止
# - コンテナ外へのはみ出しは禁止
# - コンテナ重量上限は 24,000kg
# =========================================================

# -----------------------------
# コンテナ定義
# 座標系:
#   x = 幅方向
#   y = 奥行（長手）方向
#   z = 高さ方向
# -----------------------------
CONTAINER_L = 5900   # y方向（長手）
CONTAINER_W = 2350   # x方向（幅）
CONTAINER_H = 2390   # z方向（高さ）
CONTAINER_MAX_WEIGHT = 24000
CONTAINER_VOLUME = CONTAINER_W * CONTAINER_L * CONTAINER_H

TEAM_NAME = "Team_Alpha"


# =========================================================
# データクラス
# =========================================================

@dataclass
class Item:
    item_id: str
    size_type: str
    w: int
    l: int
    h: int
    weight: float
    destination_id: str

    def volume(self) -> int:
        return self.w * self.l * self.h

    def rotated_dimensions(self, is_rotated: bool) -> Tuple[int, int, int]:
        """
        天地固定。XY平面のみ90度回転可。
        is_rotated = False -> (w, l, h)
        is_rotated = True  -> (l, w, h)
        """
        if is_rotated:
            return self.l, self.w, self.h
        return self.w, self.l, self.h


@dataclass
class FreeRect:
    x: int
    y: int
    w: int
    l: int

    def area(self) -> int:
        return self.w * self.l


@dataclass
class Placement:
    item: Item
    x: int
    y: int
    z: int
    placed_w: int
    placed_l: int
    placed_h: int
    is_rotated: bool

    def volume(self) -> int:
        return self.placed_w * self.placed_l * self.placed_h

    def center_y(self) -> float:
        return self.y + self.placed_l / 2.0


@dataclass
class Container:
    container_id: int
    destination_id: str
    items: List[Placement] = field(default_factory=list)
    free_rects: List[FreeRect] = field(default_factory=list)
    total_weight: float = 0.0
    total_volume: int = 0

    def __post_init__(self):
        if not self.free_rects:
            self.free_rects = [FreeRect(x=0, y=0, w=CONTAINER_W, l=CONTAINER_L)]

    def can_hold_destination(self, destination_id: str) -> bool:
        return self.destination_id == destination_id

    def can_hold_weight(self, item_weight: float) -> bool:
        return self.total_weight + item_weight <= CONTAINER_MAX_WEIGHT

    def fill_rate(self) -> float:
        return self.total_volume / CONTAINER_VOLUME

    def add_placement(self, placement: Placement):
        self.items.append(placement)
        self.total_weight += placement.item.weight
        self.total_volume += placement.volume()


# =========================================================
# 入力JSON読み込み
# =========================================================

def load_items_from_json(path: str) -> List[Item]:
    """
    generate_items.py が出力した items_input.json を読み込み、
    Item オブジェクトのリストに変換する
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_items = data.get("items", [])
    items: List[Item] = []

    for raw in raw_items:
        dims = raw["dimensions"]
        items.append(
            Item(
                item_id=raw["item_id"],
                size_type=raw["size_type"],
                w=int(dims["w"]),
                l=int(dims["l"]),
                h=int(dims["h"]),
                weight=float(raw["weight"]),
                destination_id=raw["destination_id"],
            )
        )

    return items


# =========================================================
# 空き領域処理
# =========================================================

def split_free_rect(free_rect: FreeRect, x: int, y: int, used_w: int, used_l: int) -> List[FreeRect]:
    """
    左下(x, y)に荷物を置いたあと、残り領域を右側と上側に分割
    """
    new_rects: List[FreeRect] = []

    # 右側領域
    remain_w = free_rect.w - used_w
    if remain_w > 0:
        new_rects.append(
            FreeRect(
                x=x + used_w,
                y=y,
                w=remain_w,
                l=free_rect.l
            )
        )

    # 上側領域
    remain_l = free_rect.l - used_l
    if remain_l > 0:
        new_rects.append(
            FreeRect(
                x=x,
                y=y + used_l,
                w=used_w,
                l=remain_l
            )
        )

    return new_rects


def rect_contains(a: FreeRect, b: FreeRect) -> bool:
    return (
        a.x <= b.x and
        a.y <= b.y and
        a.x + a.w >= b.x + b.w and
        a.y + a.l >= b.y + b.l
    )


def prune_free_rects(rects: List[FreeRect]) -> List[FreeRect]:
    pruned: List[FreeRect] = []
    for i, r1 in enumerate(rects):
        contained = False
        for j, r2 in enumerate(rects):
            if i != j and rect_contains(r2, r1):
                contained = True
                break
        if not contained and r1.w > 0 and r1.l > 0:
            pruned.append(r1)
    return pruned


def placement_score(free_rect: FreeRect, item_w: int, item_l: int) -> Tuple[int, int]:
    """
    Best Fit:
    1. 余り面積最小
    2. 短辺余り最小
    """
    leftover_area = free_rect.area() - item_w * item_l
    short_side_leftover = min(free_rect.w - item_w, free_rect.l - item_l)
    return (leftover_area, short_side_leftover)


# =========================================================
# 配置探索
# =========================================================

def try_place_item(container: Container, item: Item) -> Optional[Tuple[Placement, FreeRect]]:
    """
    そのコンテナ内で item を最も良い位置に置く候補を返す
    """
    if not container.can_hold_destination(item.destination_id):
        return None
    if not container.can_hold_weight(item.weight):
        return None

    best_candidate = None
    best_score = None

    for is_rotated in [False, True]:
        pw, pl, ph = item.rotated_dimensions(is_rotated)

        # 天地固定なので高さ h はそのまま
        if ph > CONTAINER_H:
            continue

        for fr in container.free_rects:
            if pw <= fr.w and pl <= fr.l:
                score = placement_score(fr, pw, pl)
                if best_score is None or score < best_score:
                    placement = Placement(
                        item=item,
                        x=fr.x,
                        y=fr.y,
                        z=0,
                        placed_w=pw,
                        placed_l=pl,
                        placed_h=ph,
                        is_rotated=is_rotated,
                    )
                    best_score = score
                    best_candidate = (placement, fr)

    return best_candidate


def apply_placement(container: Container, placement: Placement, used_rect: FreeRect):
    new_rects: List[FreeRect] = []
    for fr in container.free_rects:
        if fr is used_rect:
            new_rects.extend(
                split_free_rect(fr, placement.x, placement.y, placement.placed_w, placement.placed_l)
            )
        else:
            new_rects.append(fr)

    container.free_rects = prune_free_rects(new_rects)
    container.add_placement(placement)


# =========================================================
# パッキング本体
# =========================================================

def pack_items(items: List[Item]) -> List[Container]:
    """
    destination_id ごとに分離し、
    Best Fit Decreasing でコンテナへ詰める
    """
    containers: List[Container] = []
    next_container_id = 1

    # destination ごとにグルーピング
    grouped: Dict[str, List[Item]] = {}
    for item in items:
        grouped.setdefault(item.destination_id, []).append(item)

    for destination_id, group_items in grouped.items():
        # 大きい荷物から入れる
        sorted_items = sorted(
            group_items,
            key=lambda it: (it.volume(), it.weight),
            reverse=True
        )

        dest_containers: List[Container] = []

        for item in sorted_items:
            best_container = None
            best_candidate = None
            best_score = None

            for container in dest_containers:
                result = try_place_item(container, item)
                if result is not None:
                    placement, fr = result
                    score = placement_score(fr, placement.placed_w, placement.placed_l)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_candidate = result
                        best_container = container

            if best_candidate is not None:
                placement, fr = best_candidate
                apply_placement(best_container, placement, fr)
            else:
                new_container = Container(
                    container_id=next_container_id,
                    destination_id=destination_id,
                )
                next_container_id += 1

                result = try_place_item(new_container, item)
                if result is None:
                    raise ValueError(f"Item {item.item_id} cannot fit into an empty container.")

                placement, fr = result
                apply_placement(new_container, placement, fr)
                dest_containers.append(new_container)

        containers.extend(dest_containers)

    return containers


# =========================================================
# JSON出力
# =========================================================

def to_output_json(containers: List[Container], execution_time_ms: int, input_file: str) -> Dict:
    return {
        "project_info": {
            "team_name": TEAM_NAME,
            "execution_time_ms": execution_time_ms,
            "input_file": input_file
        },
        "containers": [
            {
                "container_id": c.container_id,
                "destination_id": c.destination_id,
                "total_weight": round(c.total_weight, 2),
                "items": [
                    {
                        "item_id": p.item.item_id,
                        "size_type": p.item.size_type,
                        "dimensions": {
                            "w": p.placed_w,
                            "l": p.placed_l,
                            "h": p.placed_h
                        },
                        "position": {
                            "x": p.x,
                            "y": p.y,
                            "z": p.z
                        },
                        "weight": p.item.weight,
                        "is_rotated": p.is_rotated,
                        "destination_id": p.item.destination_id
                    }
                    for p in c.items
                ]
            }
            for c in containers
        ]
    }


def save_json(data: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# =========================================================
# 実行
# =========================================================

def main():
    input_path = "items_input.json"
    output_path = "layout_result.json"

    start = time.perf_counter()

    items = load_items_from_json(input_path)
    containers = pack_items(items)

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    output = to_output_json(containers, elapsed_ms, input_path)

    save_json(output, output_path)

    print(f"{output_path} を出力しました。")
    print(f"入力ファイル       : {input_path}")
    print(f"使用コンテナ本数   : {len(containers)}")


if __name__ == "__main__":
    main()