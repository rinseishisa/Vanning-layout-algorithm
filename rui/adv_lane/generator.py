"""Adversarial generator parameterised by θ (design.md §3).

Reuses ``rui.generate_items`` density-model logic; does **not** modify it.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from rui.generate_items import (
    CONTAINER_VOLUME_M3,
    DESTINATIONS,
    ITEM_TYPES,
    CaseConfig,
    _volume_m3,
    generate_items,
)

# ------------------------------------------------------------------
# Fixed feasibility-guard hyperparameters (reasonable defaults)
# ------------------------------------------------------------------
H_MIN = 0.30               # minimum size-mix entropy (mode-collapse guard)
P_MIN = 0.08               # minimum per-destination ratio
RHO_LO = 50.0              # kg/m³ inclusive lower bound for all sizes
RHO_HI = 600.0             # kg/m³ inclusive upper bound for all sizes
N_CONT_TARGET = 8          # ≈ D * 2.5  (destinations × 2–3 containers)
MAX_ITEM_COUNT = 320       # keeps est_item/cont roughly ≤ 40

# θ layout (13-D)
# [0:3]   size_logits
# [3:6]   rho_c_raw  (centre density per size)
# [6:9]   rho_w_raw  (half-width density per size)
# [9:12]  dest_logits
# [12]    s_scale_raw

THETA_DIM = 13


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-12)


def _sigmoid_lin(x: float, lo: float, hi: float) -> float:
    return lo + (hi - lo) / (1.0 + math.exp(-x))


def _entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def decode_theta(theta: np.ndarray) -> Dict[str, object]:
    """Decode a real vector θ into human-readable generator parameters."""
    if theta.shape != (THETA_DIM,):
        raise ValueError(f"theta must have shape ({THETA_DIM},), got {theta.shape}")

    size_logits = theta[0:3]
    rho_c_raw = theta[3:6]
    rho_w_raw = theta[6:9]
    dest_logits = theta[9:12]
    s_scale_raw = float(theta[12])

    size_ratio = _softmax(size_logits)
    dest_weights = _softmax(dest_logits)
    s_scale = _sigmoid_lin(s_scale_raw, 0.6, 1.0)

    size_names = list(ITEM_TYPES.keys())  # small, medium, large
    density_by_size: Dict[str, Tuple[float, float]] = {}
    for i, s in enumerate(size_names):
        c = _sigmoid_lin(float(rho_c_raw[i]), RHO_LO, RHO_HI)
        w = _sigmoid_lin(float(rho_w_raw[i]), 0.0, (RHO_HI - RHO_LO) / 2.0)
        lo = max(RHO_LO, c - w)
        hi = min(RHO_HI, c + w)
        # Ensure non-degenerate interval
        if hi < lo + 1.0:
            hi = lo + 1.0
        density_by_size[s] = (lo, hi)

    return {
        "size_ratio": size_ratio,
        "dest_weights": dest_weights,
        "s_scale": s_scale,
        "density_by_size": density_by_size,
        "size_entropy": _entropy(size_ratio),
        "dest_entropy": _entropy(dest_weights),
    }


def _compute_item_count(size_ratio: np.ndarray, s_scale: float) -> int:
    """Volume-budget-driven item count with hard floor/ceiling."""
    avg_vol = sum(size_ratio[i] * _volume_m3(s) for i, s in enumerate(ITEM_TYPES.keys()))
    if avg_vol <= 0:
        avg_vol = 1e-9
    total_vol_budget = s_scale * 0.80 * CONTAINER_VOLUME_M3 * N_CONT_TARGET
    count = max(8 * len(DESTINATIONS), math.ceil(total_vol_budget / avg_vol))
    count = min(count, MAX_ITEM_COUNT)
    return int(count)


def _check_feasibility(params: Dict[str, object]) -> Optional[str]:
    """Return reason string if any hard anchor is violated, else None."""
    if params["size_entropy"] < H_MIN:
        return f"size_entropy={params['size_entropy']:.3f} < {H_MIN}"
    dest_weights = params["dest_weights"]
    if np.min(dest_weights) < P_MIN:
        return f"dest_min={np.min(dest_weights):.3f} < {P_MIN}"
    return None


def _est_items_per_container(data: Dict) -> List[float]:
    """Replicate sanity_report logic: est_item/cont per destination group."""
    items = data["items"]
    ests: List[float] = []
    for dest in DESTINATIONS:
        sub = [it for it in items if it["destination_id"] == dest]
        if not sub:
            continue
        total_weight = sum(it["weight"] for it in sub)
        total_volume = sum(_volume_m3(it["size_type"]) for it in sub)
        w_lb = math.ceil(total_weight / 24000.0)
        v_lb = math.ceil(total_volume / (CONTAINER_VOLUME_M3 * 0.80))
        lb = max(w_lb, v_lb)
        if lb > 0:
            ests.append(len(sub) / lb)
    return ests


def build_dataset(theta: np.ndarray, seed: int) -> Optional[Dict]:
    """Generate an items_input dict driven by θ.

    Returns ``None`` when the decoded parameters violate a hard feasibility
    anchor (entropy / destination balance / estimated item-per-container band).
    """
    params = decode_theta(theta)
    reason = _check_feasibility(params)
    if reason is not None:
        return None

    size_ratio = tuple(float(v) for v in params["size_ratio"])
    dest_weights = tuple(float(v) for v in params["dest_weights"])
    density_by_size = {k: (float(v[0]), float(v[1])) for k, v in params["density_by_size"].items()}

    item_count = _compute_item_count(params["size_ratio"], float(params["s_scale"]))

    case = CaseConfig(
        name="adv_lane_gen",
        density_by_size=density_by_size,  # type: ignore[arg-type]
        size_ratio=size_ratio,              # type: ignore[arg-type]
        dest_weights=dest_weights,          # type: ignore[arg-type]
        item_count=item_count,
    )

    rng = random.Random(seed)
    # Try a few seeds around the given one if est bounds fail
    for offset in range(5):
        data = generate_items(case, seed + offset)
        ests = _est_items_per_container(data)
        if not ests:
            continue
        if all(6.0 <= e <= 45.0 for e in ests):
            # Add adv-lane metadata into dataset_info for traceability
            data["dataset_info"]["adv_lane"] = {
                "theta": theta.tolist(),
                "seed": seed,
                "params": {
                    "size_ratio": [round(x, 4) for x in size_ratio],
                    "dest_weights": [round(x, 4) for x in dest_weights],
                    "s_scale": round(float(params["s_scale"]), 4),
                    "density_by_size": {k: [round(v[0], 2), round(v[1], 2)] for k, v in density_by_size.items()},
                    "size_entropy": round(float(params["size_entropy"]), 4),
                },
            }
            return data
    # All retries failed — degenerate parameter region
    return None


def encode_theta(
    size_ratio: Tuple[float, float, float],
    dest_weights: Tuple[float, float, float],
    s_scale: float,
    density_centres: Tuple[float, float, float],
    density_halfwidths: Tuple[float, float, float],
) -> np.ndarray:
    """Handy inverse: turn *human-readable* params into θ for warm-start."""
    size_logits = np.log(np.maximum(size_ratio, 1e-12))
    dest_logits = np.log(np.maximum(dest_weights, 1e-12))

    def _inv_sigmoid_lin(v: float, lo: float, hi: float) -> float:
        t = (v - lo) / max(hi - lo, 1e-12)
        t = np.clip(t, 1e-12, 1 - 1e-12)
        return float(np.log(t / (1 - t)))

    rho_c_raw = np.array([_inv_sigmoid_lin(c, RHO_LO, RHO_HI) for c in density_centres])
    rho_w_raw = np.array([_inv_sigmoid_lin(w, 0.0, (RHO_HI - RHO_LO) / 2.0) for w in density_halfwidths])
    s_scale_raw = _inv_sigmoid_lin(s_scale, 0.6, 1.0)

    return np.concatenate([
        size_logits,
        rho_c_raw,
        rho_w_raw,
        dest_logits,
        [s_scale_raw],
    ]).astype(float)
