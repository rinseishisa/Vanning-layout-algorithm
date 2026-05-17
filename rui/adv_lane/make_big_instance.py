"""大規模インスタンス生成器 (overnight beam reference 用)。

``rui.generate_items.generate_items`` をそのまま再利用し、case_balanced を
ベースに ``item_count`` だけ可変にして ~N コンテナ相当の貨物データを
items_input 形式 (dataset_info + items) で書き出す。

beam reference / calibration の両方がこのモジュールの ``build_big`` を呼ぶ。
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
from typing import Dict

from rui.generate_items import (
    CASE_CONFIGS,
    CONTAINER_VOLUME_M3,
    DESTINATIONS,
    CaseConfig,
    _volume_m3,
    generate_items,
    sanity_report,
)

FILL_TARGET = 0.80  # 要件定義書 3.3 の充填率基準


def _mean_volume_per_item(case: CaseConfig) -> float:
    """size_ratio 加重平均の 1 item 体積 (m^3)。"""
    sizes = ["small", "medium", "large"]
    return sum(r * _volume_m3(s) for r, s in zip(case.size_ratio, sizes))


def estimate_item_count(target_containers: int, case: CaseConfig) -> int:
    """充填率 80% 前提で target_containers 相当になる item_count を見積もる。

    case_balanced は volume-bound レジームなので体積側下限で見積もる
    (weight 側は緩い)。実際の LB は sanity_report で確認すること。
    """
    budget_volume = target_containers * CONTAINER_VOLUME_M3 * FILL_TARGET
    return max(1, math.ceil(budget_volume / _mean_volume_per_item(case)))


def build_big(target_containers: int, case_name: str, seed: int) -> Dict:
    """case_name を item_count 上書きで生成し items_input dict を返す。"""
    base = CASE_CONFIGS[case_name]
    item_count = estimate_item_count(target_containers, base)
    case = dataclasses.replace(base, item_count=item_count, name=f"{base.name}_big{target_containers}c")
    data = generate_items(case, seed)
    # 混載制約下の総下限 (= 期待コンテナ数の目安) を dataset_info に併記
    total_lb = 0
    for dest in DESTINATIONS:
        sub = [it for it in data["items"] if it["destination_id"] == dest]
        if not sub:
            continue
        w = sum(it["weight"] for it in sub)
        v = sum(_volume_m3(it["size_type"]) for it in sub)
        w_lb = math.ceil(w / 24000)
        v_lb = math.ceil(v / (CONTAINER_VOLUME_M3 * FILL_TARGET))
        total_lb += max(w_lb, v_lb)
    data["dataset_info"]["target_containers"] = target_containers
    data["dataset_info"]["mixing_constrained_lb"] = total_lb
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description="overnight beam reference 用 大規模インスタンス生成")
    ap.add_argument("--containers", type=int, default=100, help="目標コンテナ数 (default 100)")
    ap.add_argument("--case", choices=list(CASE_CONFIGS.keys()), default="case_balanced")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True, help="出力 JSON パス")
    args = ap.parse_args()

    data = build_big(args.containers, args.case, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[make_big] wrote {args.out}  item_count={data['dataset_info']['item_count']}")
    print(f"[make_big] mixing_constrained_lb (期待コンテナ数目安) = {data['dataset_info']['mixing_constrained_lb']}")
    print(sanity_report(data))


if __name__ == "__main__":
    main()
