"""Phase 1 貨物データ生成器。

旧実装 (`weight = uniform(1000, 15000)`) は体積と無関係に重量を振ったため、
コンテナ 24,000kg 制限下で平均 2.94 item/container しか積めず、3D レイアウト問題
として機能していなかった (重量飽和 92.6% / 体積充填 26%)。

本実装は weight を体積比例方式 (`volume_m³ × density`) に切り替え、density レンジ・
サイズ比・目的地分布を case ごとに設定して複数データセットを生成する。
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# -------------------------------------------------------------------
# 物理仕様 (要件定義書v1 / vanning-eval CONTAINER_SPEC_40FT に準拠)
# -------------------------------------------------------------------
CONTAINER_W_MM = 2300
CONTAINER_L_MM = 12000
CONTAINER_H_MM = 2400
CONTAINER_VOLUME_M3 = (CONTAINER_W_MM * CONTAINER_L_MM * CONTAINER_H_MM) / 1e9  # 66.24
CONTAINER_MAX_WEIGHT_KG = 24000

ITEM_WEIGHT_CAP_KG = 12000  # 単品上限 (24,000kg コンテナに対して余裕を残す)
ITEM_WEIGHT_FLOOR_KG = 100  # 単品下限

ITEM_TYPES: Dict[str, Dict[str, int]] = {
    "small":  {"w": 760,  "l": 1130, "h": 550},
    "medium": {"w": 1490, "l": 2260, "h": 900},
    "large":  {"w": 2280, "l": 2550, "h": 2355},
}

DESTINATIONS: List[str] = ["DEST_A", "DEST_B", "DEST_C"]


def _volume_m3(size_type: str) -> float:
    s = ITEM_TYPES[size_type]
    return (s["w"] * s["l"] * s["h"]) / 1e9


# -------------------------------------------------------------------
# ケース定義
# -------------------------------------------------------------------
@dataclass(frozen=True)
class CaseConfig:
    name: str
    density_by_size: Dict[str, Tuple[int, int]]  # kg/m³ per size_type
    size_ratio: Tuple[float, float, float]   # (small, medium, large) 重み
    dest_weights: Tuple[float, float, float]  # (A, B, C) 重み
    item_count: int


CASE_CONFIGS: Dict[str, CaseConfig] = {
    "case_balanced": CaseConfig(
        name="case_balanced",
        density_by_size={"small": (150, 450), "medium": (150, 450), "large": (150, 450)},
        size_ratio=(0.50, 0.35, 0.15),
        dest_weights=(1/3, 1/3, 1/3),
        item_count=100,
    ),
    "case_volume_bound": CaseConfig(
        name="case_volume_bound",
        density_by_size={"small": (80, 250), "medium": (80, 250), "large": (80, 250)},
        size_ratio=(0.50, 0.35, 0.15),
        dest_weights=(1/3, 1/3, 1/3),
        item_count=100,
    ),
    "case_weight_bound": CaseConfig(
        name="case_weight_bound",
        density_by_size={"small": (300, 550), "medium": (300, 550), "large": (300, 500)},
        size_ratio=(0.50, 0.35, 0.15),
        dest_weights=(1/3, 1/3, 1/3),
        item_count=100,
    ),
    "case_small_many": CaseConfig(
        name="case_small_many",
        density_by_size={"small": (120, 350), "medium": (120, 350), "large": (120, 350)},
        size_ratio=(0.65, 0.25, 0.10),
        dest_weights=(1/3, 1/3, 1/3),
        item_count=150,
    ),
    "case_dest_skew": CaseConfig(
        name="case_dest_skew",
        density_by_size={"small": (150, 450), "medium": (150, 450), "large": (150, 450)},
        size_ratio=(0.50, 0.35, 0.15),
        dest_weights=(0.70, 0.20, 0.10),
        item_count=100,
    ),
}

# case × seed の生成セット (plan の「初期 6 本」)
GENERATION_SET: List[Tuple[str, int]] = [
    ("case_balanced",     42),
    ("case_balanced",     7),
    ("case_volume_bound", 42),
    ("case_weight_bound", 42),
    ("case_small_many",   42),
    ("case_dest_skew",    42),
]


def _clamp_weight(raw: float) -> float:
    """重量を [ITEM_WEIGHT_FLOOR_KG, ITEM_WEIGHT_CAP_KG] にクリップ。"""
    return max(ITEM_WEIGHT_FLOOR_KG, min(round(raw, 2), float(ITEM_WEIGHT_CAP_KG)))


# -------------------------------------------------------------------
# 生成本体
# -------------------------------------------------------------------
def generate_items(case: CaseConfig, seed: int) -> Dict:
    rng = random.Random(seed)
    size_types = list(ITEM_TYPES.keys())

    items: List[Dict] = []
    for i in range(1, case.item_count + 1):
        size_type = rng.choices(size_types, weights=case.size_ratio, k=1)[0]
        spec = ITEM_TYPES[size_type]
        volume_m3 = _volume_m3(size_type)

        d_min, d_max = case.density_by_size[size_type]
        density = rng.uniform(d_min, d_max)
        weight = _clamp_weight(volume_m3 * density)

        destination_id = rng.choices(DESTINATIONS, weights=case.dest_weights, k=1)[0]

        items.append({
            "item_id": f"P{i:03d}",
            "size_type": size_type,
            "dimensions": {"w": spec["w"], "l": spec["l"], "h": spec["h"]},
            "weight": weight,
            "destination_id": destination_id,
        })

    dataset_name = f"{case.name}_seed{seed}"
    return {
        "dataset_info": {
            "dataset_name": dataset_name,
            "seed": seed,
            "item_count": case.item_count,
            "case": case.name,
            "density_by_size_kg_per_m3": {k: list(v) for k, v in case.density_by_size.items()},
            "weight_bounds_kg": [ITEM_WEIGHT_FLOOR_KG, ITEM_WEIGHT_CAP_KG],
            "size_ratio": list(case.size_ratio),
            "dest_weights": list(case.dest_weights),
        },
        "items": items,
    }


# -------------------------------------------------------------------
# サニティチェック
# -------------------------------------------------------------------
def sanity_report(data: Dict) -> str:
    items = data["items"]
    total_weight = sum(it["weight"] for it in items)
    total_volume = sum(_volume_m3(it["size_type"]) for it in items)
    max_weight = max(it["weight"] for it in items)
    min_weight = min(it["weight"] for it in items)

    weight_lower_bound = math.ceil(total_weight / CONTAINER_MAX_WEIGHT_KG)
    # 体積側は 80% 充填を仮定した下限
    volume_lower_bound = math.ceil(total_volume / (CONTAINER_VOLUME_M3 * 0.80))

    # clip 発火率
    clipped_floor = sum(1 for it in items if it["weight"] <= ITEM_WEIGHT_FLOOR_KG)
    clipped_cap = sum(1 for it in items if it["weight"] >= ITEM_WEIGHT_CAP_KG)
    clip_rate = (clipped_floor + clipped_cap) / len(items) * 100

    # dest 別 (混載制約下の下限 = 各 dest ごとの max(weight, volume) 下限を合算)
    dest_lines: List[str] = []
    dest_total = 0
    for dest in DESTINATIONS:
        sub = [it for it in items if it["destination_id"] == dest]
        if not sub:
            dest_lines.append(f"    {dest}: count=0")
            continue
        w = sum(it["weight"] for it in sub)
        v = sum(_volume_m3(it["size_type"]) for it in sub)
        w_lb = math.ceil(w / CONTAINER_MAX_WEIGHT_KG)
        v_lb = math.ceil(v / (CONTAINER_VOLUME_M3 * 0.80))
        lb = max(w_lb, v_lb)
        dest_total += lb
        # 推定 item/container (混載制約下では lb = container 数なので item数 / lb)
        est_items_per_container = len(sub) / lb if lb > 0 else float("inf")
        dest_lines.append(
            f"    {dest}: count={len(sub):3d}  weight={w:>10,.0f}kg  "
            f"vol={v:>6.2f}m^3  w_lb={w_lb}  v_lb={v_lb}  -> need>={lb}  "
            f"est_item/cont={est_items_per_container:.1f}"
        )

    # レジーム判定
    if weight_lower_bound > volume_lower_bound:
        regime = "WEIGHT-BOUND"
    elif volume_lower_bound > weight_lower_bound:
        regime = "VOLUME-BOUND"
    else:
        regime = "BALANCED"

    assert max_weight <= ITEM_WEIGHT_CAP_KG, f"単品 weight 上限違反: {max_weight}"
    assert min_weight >= ITEM_WEIGHT_FLOOR_KG, f"単品 weight 下限違反: {min_weight}"

    info = data["dataset_info"]
    lines = [
        f"  dataset        : {info['dataset_name']}",
        f"  items          : {len(items)} "
        f"(size ratio target {info['size_ratio']}, actual "
        f"S={sum(1 for it in items if it['size_type']=='small')} "
        f"M={sum(1 for it in items if it['size_type']=='medium')} "
        f"L={sum(1 for it in items if it['size_type']=='large')})",
        f"  total_weight   : {total_weight:>10,.0f} kg",
        f"  total_volume   : {total_volume:>10.2f} m^3",
        f"  max_item_weight: {max_weight:>10,.0f} kg  (cap {ITEM_WEIGHT_CAP_KG})",
        f"  min_item_weight: {min_weight:>10,.0f} kg  (floor {ITEM_WEIGHT_FLOOR_KG})",
        f"  container_LB   : weight={weight_lower_bound}  volume={volume_lower_bound}  → {regime}",
        f"  clip_rate      : {clip_rate:.1f}% (floor={clipped_floor}, cap={clipped_cap})",
        f"  dest breakdown (weight/volume lower bound per dest):",
        *dest_lines,
        f"  mixing-constrained total LB: {dest_total}",
    ]
    if clip_rate > 50.0:
        lines.append(f"  WARNING: clip_rate {clip_rate:.1f}% > 50% — density ranges may be too extreme")
    return "\n".join(lines)


# -------------------------------------------------------------------
# エントリポイント
# -------------------------------------------------------------------
def _write_dataset(case: CaseConfig, seed: int, out_dir: Path) -> Path:
    data = generate_items(case, seed)
    out_path = out_dir / f"{data['dataset_info']['dataset_name']}.json"
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[generated] {out_path}")
    print(sanity_report(data))
    print()
    return out_path


def _assert_items_per_container_all_cases() -> None:
    """全 case の推定 item/container が 8–40 帯に収まることを assert。"""
    for case_name, case in CASE_CONFIGS.items():
        for seed in [42, 7, 123]:
            data = generate_items(case, seed)
            items = data["items"]
            for dest in DESTINATIONS:
                sub = [it for it in items if it["destination_id"] == dest]
                if not sub:
                    continue
                total_weight = sum(it["weight"] for it in sub)
                total_volume = sum(_volume_m3(it["size_type"]) for it in sub)
                w_lb = math.ceil(total_weight / CONTAINER_MAX_WEIGHT_KG)
                v_lb = math.ceil(total_volume / (CONTAINER_VOLUME_M3 * 0.80))
                lb = max(w_lb, v_lb)
                est = len(sub) / lb if lb > 0 else float("inf")
                assert 8 <= est <= 40, (
                    f"{case_name} seed={seed} {dest}: est_item/container={est:.1f} "
                    f"out of [8, 40] (lb={lb}, sub_items={len(sub)})"
                )


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 1 貨物データ生成器")
    ap.add_argument("--all", action="store_true", help="GENERATION_SET を全生成")
    ap.add_argument("--case", choices=list(CASE_CONFIGS.keys()),
                    help="単体生成する case 名")
    ap.add_argument("--seed", type=int, default=42, help="--case 指定時の seed")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).parent / "datasets",
                    help="出力ディレクトリ (default: rui/datasets)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        targets = GENERATION_SET
    elif args.case:
        targets = [(args.case, args.seed)]
    else:
        targets = [("case_balanced", 42)]
        print("(no args) defaulting to case_balanced seed=42\n")

    for case_name, seed in targets:
        _write_dataset(CASE_CONFIGS[case_name], seed, args.out_dir)

    # 全 case の sanity assert
    _assert_items_per_container_all_cases()
    print("[sanity] All cases pass est_item/container ∈ [8, 40].")


if __name__ == "__main__":
    main()
