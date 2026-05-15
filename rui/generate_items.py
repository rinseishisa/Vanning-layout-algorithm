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

ITEM_WEIGHT_CAP_KG = 20000  # 単品上限 (24,000kg コンテナに対して余裕を残す)

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
    density_range: Tuple[int, int]           # kg/m³
    size_ratio: Tuple[float, float, float]   # (small, medium, large) 重み
    dest_weights: Tuple[float, float, float] # (A, B, C) 重み
    item_count: int


CASE_CONFIGS: Dict[str, CaseConfig] = {
    "case_balanced": CaseConfig(
        name="case_balanced",
        density_range=(300, 700),
        size_ratio=(0.50, 0.35, 0.15),
        dest_weights=(1/3, 1/3, 1/3),
        item_count=100,
    ),
    "case_volume_bound": CaseConfig(
        name="case_volume_bound",
        density_range=(150, 400),
        size_ratio=(0.50, 0.35, 0.15),
        dest_weights=(1/3, 1/3, 1/3),
        item_count=100,
    ),
    "case_weight_bound": CaseConfig(
        name="case_weight_bound",
        density_range=(500, 1200),
        size_ratio=(0.50, 0.35, 0.15),
        dest_weights=(1/3, 1/3, 1/3),
        item_count=100,
    ),
    "case_small_many": CaseConfig(
        name="case_small_many",
        density_range=(300, 700),
        size_ratio=(0.65, 0.25, 0.10),
        dest_weights=(1/3, 1/3, 1/3),
        item_count=150,
    ),
    "case_dest_skew": CaseConfig(
        name="case_dest_skew",
        density_range=(300, 700),
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


# -------------------------------------------------------------------
# 生成本体
# -------------------------------------------------------------------
def generate_items(case: CaseConfig, seed: int) -> Dict:
    rng = random.Random(seed)
    size_types = list(ITEM_TYPES.keys())
    d_min, d_max = case.density_range

    items: List[Dict] = []
    for i in range(1, case.item_count + 1):
        size_type = rng.choices(size_types, weights=case.size_ratio, k=1)[0]
        spec = ITEM_TYPES[size_type]
        volume_m3 = _volume_m3(size_type)

        density = rng.uniform(d_min, d_max)
        weight = min(round(volume_m3 * density, 2), float(ITEM_WEIGHT_CAP_KG))

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
            "density_range_kg_per_m3": list(case.density_range),
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

    weight_lower_bound = math.ceil(total_weight / CONTAINER_MAX_WEIGHT_KG)
    # 体積側は 70% 充填を仮定した下限
    volume_lower_bound = math.ceil(total_volume / (CONTAINER_VOLUME_M3 * 0.70))

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
        v_lb = math.ceil(v / (CONTAINER_VOLUME_M3 * 0.70))
        lb = max(w_lb, v_lb)
        dest_total += lb
        dest_lines.append(
            f"    {dest}: count={len(sub):3d}  weight={w:>10,.0f}kg  "
            f"vol={v:>6.2f}m^3  w_lb={w_lb}  v_lb={v_lb}  -> need>={lb}"
        )

    # レジーム判定
    if weight_lower_bound > volume_lower_bound:
        regime = "WEIGHT-BOUND"
    elif volume_lower_bound > weight_lower_bound:
        regime = "VOLUME-BOUND"
    else:
        regime = "BALANCED"

    assert max_weight <= ITEM_WEIGHT_CAP_KG, f"単品 weight 上限違反: {max_weight}"

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
        f"  container_LB   : weight={weight_lower_bound}  volume={volume_lower_bound}  → {regime}",
        f"  dest breakdown (weight/volume lower bound per dest):",
        *dest_lines,
        f"  mixing-constrained total LB: {dest_total}",
    ]
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


if __name__ == "__main__":
    main()
