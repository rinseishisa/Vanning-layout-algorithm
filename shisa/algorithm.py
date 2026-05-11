import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# =========================================================
# algorithm_balance.py
#
# 方針:
# - destination_id ごとに完全分離する
# - 重い荷物・大きい荷物を優先して配置する
# - 各配置候補について「置いた後のY軸重心」が中央に近いかを評価する
# - すべて z=0 に置くため、接地違反を避けやすい
# - XY平面での90度回転のみ許可する
# =========================================================


# -----------------------------
# コンテナ定義
# 要件定義書:
# 12000(L) × 2300(W) × 2400(H)
#
# 座標系:
# x = 幅方向
# y = 奥行・長手方向
# z = 高さ方向
# -----------------------------
CONTAINER_L = 12000
CONTAINER_W = 2300
CONTAINER_H = 2400
CONTAINER_MAX_WEIGHT = 24000

CONTAINER_CENTER_Y = CONTAINER_L / 2
CONTAINER_VOLUME = CONTAINER_W * CONTAINER_L * CONTAINER_H

TEAM_NAME = "Team_Balance"


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

    def base_area(self) -> int:
        return self.w * self.l

    def rotated_dimensions(self, is_rotated: bool) -> Tuple[int, int, int]:
        """
        天地固定。
        is_rotated = False -> (w, l, h)
        is_rotated = True  -> (l, w, h)
        """
        if is_rotated:
            return self.l, self.w, self.h
        return self.w, self.l, self.h


@dataclass
class FreeRect:
    """
    z=0 の床面上にある空き長方形領域。
    """
    x: int
    y: int
    w: int
    l: int

    def area(self) -> int:
        return self.w * self.l

    def center_y(self) -> float:
        return self.y + self.l / 2


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
        return self.y + self.placed_l / 2


@dataclass
class Container:
    container_id: int
    destination_id: str
    items: List[Placement] = field(default_factory=list)
    free_rects: List[FreeRect] = field(default_factory=list)
    total_weight: float = 0.0
    total_volume: int = 0
    moment_y: float = 0.0

    def __post_init__(self):
        if not self.free_rects:
            self.free_rects = [
                FreeRect(
                    x=0,
                    y=0,
                    w=CONTAINER_W,
                    l=CONTAINER_L,
                )
            ]

    def can_hold_destination(self, destination_id: str) -> bool:
        return self.destination_id == destination_id

    def can_hold_weight(self, item_weight: float) -> bool:
        return self.total_weight + item_weight <= CONTAINER_MAX_WEIGHT

    def fill_rate(self) -> float:
        return self.total_volume / CONTAINER_VOLUME

    def center_of_gravity_y(self) -> float:
        if self.total_weight <= 0:
            return CONTAINER_CENTER_Y
        return self.moment_y / self.total_weight

    def add_placement(self, placement: Placement):
        self.items.append(placement)
        self.total_weight += placement.item.weight
        self.total_volume += placement.volume()
        self.moment_y += placement.item.weight * placement.center_y()


# =========================================================
# 入力JSON読み込み
# =========================================================

def load_items_from_json(path: str) -> List[Item]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items: List[Item] = []

    for raw in data.get("items", []):
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

def split_free_rect(
    free_rect: FreeRect,
    x: int,
    y: int,
    used_w: int,
    used_l: int,
) -> List[FreeRect]:
    """
    free_rect の左下に荷物を置いたあと、
    残りを「右側」と「奥側」に分割する単純な分割方法。

    注意:
    厳密なMaxRectsではないが、実装が理解しやすく、重複しない配置を作りやすい。
    """
    new_rects: List[FreeRect] = []

    # 右側の空き領域
    remain_w = free_rect.w - used_w
    if remain_w > 0:
        new_rects.append(
            FreeRect(
                x=x + used_w,
                y=y,
                w=remain_w,
                l=used_l,
            )
        )

    # 奥側の空き領域
    remain_l = free_rect.l - used_l
    if remain_l > 0:
        new_rects.append(
            FreeRect(
                x=free_rect.x,
                y=y + used_l,
                w=free_rect.w,
                l=remain_l,
            )
        )

    return new_rects


def rect_contains(a: FreeRect, b: FreeRect) -> bool:
    """
    a が b を完全に含むか判定する。
    """
    return (
        a.x <= b.x
        and a.y <= b.y
        and a.x + a.w >= b.x + b.w
        and a.y + a.l >= b.y + b.l
    )


def prune_free_rects(rects: List[FreeRect]) -> List[FreeRect]:
    """
    他の空き領域に完全に含まれる空き領域を削除する。
    """
    pruned: List[FreeRect] = []

    for i, r1 in enumerate(rects):
        if r1.w <= 0 or r1.l <= 0:
            continue

        contained = False
        for j, r2 in enumerate(rects):
            if i == j:
                continue
            if rect_contains(r2, r1):
                contained = True
                break

        if not contained:
            pruned.append(r1)

    # yが中央に近い空き領域を先に見るために並べ替え
    pruned.sort(key=lambda r: abs(r.center_y() - CONTAINER_CENTER_Y))
    return pruned


# =========================================================
# 配置スコア
# =========================================================

def placement_score(
    container: Container,
    item: Item,
    free_rect: FreeRect,
    placed_w: int,
    placed_l: int,
    x: int,
    y: int,
) -> Tuple[float, float, float, float]:
    """
    スコアが小さい候補ほど良い。

    評価するもの:
    1. 配置後のコンテナ重心Yが中央に近いか
    2. 荷物自体が中央付近に置かれているか
    3. 空き領域に対して無駄が少ないか
    4. 現在のコンテナ充填率が高いか

    Tupleで返すことで、Pythonの比較により
    第1条件 → 第2条件 → 第3条件... の順に優先される。
    """
    item_center_y = y + placed_l / 2

    new_total_weight = container.total_weight + item.weight
    new_moment_y = container.moment_y + item.weight * item_center_y
    new_cog_y = new_moment_y / new_total_weight

    # 1. 置いた後の重心が中央からどれだけズレるか
    cog_penalty = abs(new_cog_y - CONTAINER_CENTER_Y)

    # 2. 荷物単体の中心が中央からどれだけズレるか
    item_center_penalty = abs(item_center_y - CONTAINER_CENTER_Y)

    # 3. その空き領域に置いたときの床面の余り
    leftover_area = free_rect.area() - placed_w * placed_l

    # 4. 短辺の余り
    short_side_leftover = min(
        free_rect.w - placed_w,
        free_rect.l - placed_l,
    )

    return (
        cog_penalty,
        item_center_penalty,
        leftover_area,
        short_side_leftover,
    )


# =========================================================
# 配置探索
# =========================================================

def try_place_item(container: Container, item: Item) -> Optional[Tuple[Placement, FreeRect]]:
    """
    itemをcontainerに入れる候補を探す。
    重心Yが中央に近づく配置を優先する。
    """
    if not container.can_hold_destination(item.destination_id):
        return None

    if not container.can_hold_weight(item.weight):
        return None

    best_candidate: Optional[Tuple[Placement, FreeRect]] = None
    best_score: Optional[Tuple[float, float, float, float]] = None

    # 回転なし・回転ありを試す
    for is_rotated in [False, True]:
        placed_w, placed_l, placed_h = item.rotated_dimensions(is_rotated)

        # 高さチェック
        if placed_h > CONTAINER_H:
            continue

        for free_rect in container.free_rects:
            # 床面に収まるか
            if placed_w > free_rect.w or placed_l > free_rect.l:
                continue

            x = free_rect.x
            y = free_rect.y

            score = placement_score(
                container=container,
                item=item,
                free_rect=free_rect,
                placed_w=placed_w,
                placed_l=placed_l,
                x=x,
                y=y,
            )

            if best_score is None or score < best_score:
                best_score = score
                best_candidate = (
                    Placement(
                        item=item,
                        x=x,
                        y=y,
                        z=0,
                        placed_w=placed_w,
                        placed_l=placed_l,
                        placed_h=placed_h,
                        is_rotated=is_rotated,
                    ),
                    free_rect,
                )

    return best_candidate


def apply_placement(container: Container, placement: Placement, used_rect: FreeRect):
    """
    配置を確定し、空き領域を更新する。
    """
    new_rects: List[FreeRect] = []

    for free_rect in container.free_rects:
        if free_rect is used_rect:
            new_rects.extend(
                split_free_rect(
                    free_rect=free_rect,
                    x=placement.x,
                    y=placement.y,
                    used_w=placement.placed_w,
                    used_l=placement.placed_l,
                )
            )
        else:
            new_rects.append(free_rect)

    container.free_rects = prune_free_rects(new_rects)
    container.add_placement(placement)


# =========================================================
# パッキング本体
# =========================================================

def pack_items(items: List[Item]) -> List[Container]:
    """
    destination_idごとに分けて、重心バランス重視で配置する。
    """
    containers: List[Container] = []
    next_container_id = 1

    grouped: Dict[str, List[Item]] = {}
    for item in items:
        grouped.setdefault(item.destination_id, []).append(item)

    # destination_idごとに処理
    for destination_id in sorted(grouped.keys()):
        group_items = grouped[destination_id]

        # 重くて大きい荷物を優先
        # largeを先に中央付近へ置きやすくする
        sorted_items = sorted(
            group_items,
            key=lambda it: (
                it.weight * it.volume(),
                it.volume(),
                it.weight,
            ),
            reverse=True,
        )

        dest_containers: List[Container] = []

        for item in sorted_items:
            best_container: Optional[Container] = None
            best_candidate: Optional[Tuple[Placement, FreeRect]] = None
            best_score: Optional[Tuple[float, float, float, float, float]] = None

            # 既存コンテナに入るか試す
            for container in dest_containers:
                result = try_place_item(container, item)
                if result is None:
                    continue

                placement, free_rect = result

                score = placement_score(
                    container=container,
                    item=item,
                    free_rect=free_rect,
                    placed_w=placement.placed_w,
                    placed_l=placement.placed_l,
                    x=placement.x,
                    y=placement.y,
                )

                # コンテナ本数削減のため、既存コンテナの利用を少し優先
                extended_score = (
                    score[0],
                    score[1],
                    score[2],
                    score[3],
                    -container.fill_rate(),
                )

                if best_score is None or extended_score < best_score:
                    best_score = extended_score
                    best_candidate = result
                    best_container = container

            # 既存コンテナに置けるなら置く
            if best_candidate is not None and best_container is not None:
                placement, used_rect = best_candidate
                apply_placement(best_container, placement, used_rect)
                continue

            # 置けない場合は新規コンテナを作る
            new_container = Container(
                container_id=next_container_id,
                destination_id=destination_id,
            )
            next_container_id += 1

            result = try_place_item(new_container, item)

            if result is None:
                raise ValueError(
                    f"Item {item.item_id} cannot fit into an empty container. "
                    f"dimensions=({item.w}, {item.l}, {item.h}), "
                    f"weight={item.weight}"
                )

            placement, used_rect = result
            apply_placement(new_container, placement, used_rect)
            dest_containers.append(new_container)

        containers.extend(dest_containers)

    return containers


# =========================================================
# 出力JSON
# =========================================================

def to_output_json(
    containers: List[Container],
    execution_time_ms: int,
    input_file: str,
) -> Dict:
    return {
        "project_info": {
            "team_name": TEAM_NAME,
            "execution_time_ms": execution_time_ms,
            "input_file": input_file,
            "algorithm": "center_balanced_shelf_packing",
        },
        "containers": [
            {
                "container_id": container.container_id,
                "destination_id": container.destination_id,
                "total_weight": round(container.total_weight, 2),
                "fill_rate": round(container.fill_rate(), 4),
                "center_of_gravity_y": round(container.center_of_gravity_y(), 2),
                "items": [
                    {
                        "item_id": placement.item.item_id,
                        "size_type": placement.item.size_type,
                        "dimensions": {
                            "w": placement.placed_w,
                            "l": placement.placed_l,
                            "h": placement.placed_h,
                        },
                        "position": {
                            "x": placement.x,
                            "y": placement.y,
                            "z": placement.z,
                        },
                        "weight": placement.item.weight,
                        "is_rotated": placement.is_rotated,
                        "destination_id": placement.item.destination_id,
                    }
                    for placement in container.items
                ],
            }
            for container in containers
        ],
    }


def save_json(data: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# =========================================================
# 簡易レポート
# =========================================================

def print_report(containers: List[Container]):
    print("====================================")
    print("Packing Report")
    print("====================================")
    print(f"使用コンテナ本数: {len(containers)}")

    if not containers:
        return

    avg_fill_rate = sum(c.fill_rate() for c in containers) / len(containers)
    print(f"平均充填率      : {avg_fill_rate:.2%}")

    for c in containers:
        cog_y = c.center_of_gravity_y()
        cog_diff = abs(cog_y - CONTAINER_CENTER_Y)

        print(
            f"container_id={c.container_id}, "
            f"dest={c.destination_id}, "
            f"items={len(c.items)}, "
            f"weight={c.total_weight:.2f}kg, "
            f"fill={c.fill_rate():.2%}, "
            f"Yg={cog_y:.1f}, "
            f"|Yg-6000|={cog_diff:.1f}"
        )


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

    output = to_output_json(
        containers=containers,
        execution_time_ms=elapsed_ms,
        input_file=input_path,
    )

    save_json(output, output_path)

    print(f"{output_path} を出力しました。")
    print(f"入力ファイル: {input_path}")
    print(f"実行時間    : {elapsed_ms} ms")
    print_report(containers)


if __name__ == "__main__":
    main()
