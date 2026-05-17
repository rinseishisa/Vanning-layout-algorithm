"""Adversarial generator for 31-type catalog (design_catalog31.md §4).

Reuses ``rui.generate_items`` constants; does **not** modify it.
Uses ``theta31`` for decoding / feasibility.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from rui.adv_lane.catalog31 import CATALOG_31
from rui.adv_lane.theta31 import THETA_DIM_31, check_feasibility31, decode_theta31
from rui.generate_items import (
    CONTAINER_VOLUME_M3,
    DESTINATIONS,
    ITEM_WEIGHT_CAP_KG,
    ITEM_WEIGHT_FLOOR_KG,
)

# Reuse generator.py budget constants for consistency
from rui.adv_lane.generator import MAX_ITEM_COUNT, N_CONT_TARGET


def _clamp_weight(raw: float) -> float:
    return max(ITEM_WEIGHT_FLOOR_KG, min(round(raw, 2), float(ITEM_WEIGHT_CAP_KG)))


# vanning_eval/schema.py:16 VALID_SIZE_TYPES と一致必須（厳格 enum）。
# 31種カタログラベルを size_type に入れると items_input/layout_result の
# 両方でスキーマ拒否される。size_type は配置・採点に不使用（幾何は
# dimensions、ビューアは opacity 階調のみ）なので、実寸は保ったまま
# 体積で {small,medium,large} へバケットする（dN/regret は不変）。
VALID_SIZE_TYPES = ("small", "medium", "large")
# 既知3アンカー体積 [m³] の幾何中点を境界に（小0.472 / 中3.031 / 大13.69）
_SC_B1 = (0.47234 * 3.03066) ** 0.5  # ≈1.197
_SC_B2 = (3.03066 * 13.69179) ** 0.5  # ≈6.441


def _size_class(w: int, l: int, h: int) -> str:
    """箱体積を要件3クラスへ写像（vanning_eval スキーマ互換のため）。"""
    vol_m3 = (w * l * h) / 1e9
    if vol_m3 < _SC_B1:
        return "small"
    if vol_m3 < _SC_B2:
        return "medium"
    return "large"


def _compute_item_count(size_prob: np.ndarray, type_meta: List[Tuple], s_scale: float) -> int:
    """Volume-budget-driven item count with hard floor/ceiling."""
    avg_vol = sum(
        float(size_prob[i]) * (w * l * h) / 1e9
        for i, (_, _, _, w, l, h) in enumerate(type_meta)
    )
    if avg_vol <= 0:
        avg_vol = 1e-9
    total_vol_budget = s_scale * 0.80 * CONTAINER_VOLUME_M3 * N_CONT_TARGET
    count = max(8 * len(DESTINATIONS), math.ceil(total_vol_budget / avg_vol))
    count = min(count, MAX_ITEM_COUNT)
    return int(count)


def _est_items_per_container(data: Dict) -> List[float]:
    """Replicate sanity_report logic: est_item/cont per destination group."""
    items = data["items"]
    ests: List[float] = []
    for dest in DESTINATIONS:
        sub = [it for it in items if it["destination_id"] == dest]
        if not sub:
            continue
        total_weight = sum(it["weight"] for it in sub)
        total_volume = sum(
            it["dimensions"]["w"] * it["dimensions"]["l"] * it["dimensions"]["h"] / 1e9
            for it in sub
        )
        w_lb = math.ceil(total_weight / 24000.0)
        v_lb = math.ceil(total_volume / (CONTAINER_VOLUME_M3 * 0.80))
        lb = max(w_lb, v_lb)
        if lb > 0:
            ests.append(len(sub) / lb)
    return ests


def build_dataset_31(theta: np.ndarray, seed: int) -> Optional[Dict]:
    """Generate an items_input dict driven by θ (31-type catalog).

    Returns ``None`` when the decoded parameters violate a hard feasibility
    anchor or when estimated items-per-container fall outside [6, 45].
    """
    params = decode_theta31(theta)
    reason = check_feasibility31(params)
    if reason is not None:
        return None

    size_prob = params["size_prob"]
    type_meta = params["type_meta"]
    density_by_material = params["density_by_material"]
    dest_weights = params["dest_weights"]
    s_scale = float(params["s_scale"])

    item_count = _compute_item_count(size_prob, type_meta, s_scale)

    # Build label -> (w,l,h,material) lookup
    meta_by_label = {
        label: (w, l, h, material)
        for (_, material, label, w, l, h) in type_meta
    }
    labels = [label for (_, _, label, _, _, _) in type_meta]
    probs = [float(size_prob[i]) for i in range(len(type_meta))]

    # Try a few seeds around the given one if est bounds fail
    for offset in range(5):
        rng_offset = random.Random(seed + offset)
        items: List[Dict] = []
        for i in range(1, item_count + 1):
            label = rng_offset.choices(labels, weights=probs, k=1)[0]
            w, l, h, material = meta_by_label[label]
            d_lo, d_hi = density_by_material[material]
            density = rng_offset.uniform(d_lo, d_hi)
            vol_m3 = (w * l * h) / 1e9
            weight = _clamp_weight(vol_m3 * density)
            destination_id = rng_offset.choices(DESTINATIONS, weights=list(dest_weights), k=1)[0]
            items.append({
                "item_id": f"P{i:03d}",
                # vanning_eval スキーマ厳格 enum 準拠（catalog ラベル不可）。
                # catalog 箱の同定は一意な dimensions で可能。
                "size_type": _size_class(w, l, h),
                "dimensions": {"w": w, "l": l, "h": h},
                "weight": weight,
                "destination_id": destination_id,
            })

        data = {
            "dataset_info": {
                "dataset_name": f"adv_lane_gen31_seed{seed + offset}",
                "seed": seed + offset,
                "item_count": item_count,
                "case": "adv_lane_gen31",
                "density_by_material_kg_per_m3": {
                    k: [round(float(v[0]), 2), round(float(v[1]), 2)]
                    for k, v in density_by_material.items()
                },
                "weight_bounds_kg": [ITEM_WEIGHT_FLOOR_KG, ITEM_WEIGHT_CAP_KG],
                "size_prob": [round(float(x), 6) for x in size_prob],
                "dest_weights": [round(float(x), 4) for x in dest_weights],
                "adv_lane": {
                    "theta": theta.tolist(),
                    "seed": seed,
                    "params": {
                        "size_entropy_norm": round(float(params["size_entropy_norm"]), 4),
                        "s_scale": round(float(s_scale), 4),
                    },
                },
            },
            "items": items,
        }

        ests = _est_items_per_container(data)
        if not ests:
            continue
        if all(6.0 <= e <= 45.0 for e in ests):
            return data

    # All retries failed
    return None
