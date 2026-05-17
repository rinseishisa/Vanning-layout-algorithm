"""Tests for generator31 (design_catalog31.md §5).

These tests are authored for pytest but are **not executed** in this headless
environment; the caller (Claude Code) is expected to run them.
"""
import numpy as np
import pytest

from rui.adv_lane.generator import MAX_ITEM_COUNT
from rui.adv_lane.generator31 import VALID_SIZE_TYPES, _size_class, build_dataset_31
from rui.adv_lane.loop import _make_dataframe
from rui.adv_lane.theta31 import THETA_DIM_31, encode_theta31
from rui.algorithm_a import build_items, run_ga
from rui.generate_items import DESTINATIONS


def _feasible_theta() -> np.ndarray:
    """Return a reasonable feasible θ for the 31-type catalog."""
    return encode_theta31(
        p_wood=0.3,
        mu_wood=0.5,
        mu_steel=0.5,
        nu_wood=10.0,
        nu_steel=10.0,
        rho_shift=0.0,
        rho_gain=1.0,
        dest_weights=(1 / 3, 1 / 3, 1 / 3),
        s_scale=0.8,
    )


def test_build_dataset_31_drop_in_compatibility():
    """(a) build_dataset_31 output must feed algorithm_a pipeline without crash."""
    theta = _feasible_theta()
    data = build_dataset_31(theta, seed=42)
    assert data is not None, "feasible theta should produce a dataset"
    df = _make_dataframe(data)
    assert not df.empty
    items = build_items(df)
    containers, eval_result = run_ga(items, generations=2, pop_size=3)
    assert isinstance(eval_result, dict)
    assert "container_count" in eval_result


def test_build_dataset_31_infeasible_dest_skew():
    """(b) Heavily skewed destination weights must yield None."""
    theta = encode_theta31(
        p_wood=0.3,
        mu_wood=0.5,
        mu_steel=0.5,
        nu_wood=10.0,
        nu_steel=10.0,
        rho_shift=0.0,
        rho_gain=1.0,
        dest_weights=(0.95, 0.025, 0.025),
        s_scale=0.8,
    )
    assert build_dataset_31(theta, seed=42) is None


def test_build_dataset_31_item_count_bounds():
    """(c) item_count must lie within [8*len(DESTINATIONS), MAX_ITEM_COUNT]."""
    theta = _feasible_theta()
    data = build_dataset_31(theta, seed=42)
    assert data is not None
    n = data["dataset_info"]["item_count"]
    assert n >= 8 * len(DESTINATIONS)
    assert n <= MAX_ITEM_COUNT


def test_build_dataset_31_dimensions_positive():
    """(d) All items must have positive dimensions originating from catalog31."""
    theta = _feasible_theta()
    data = build_dataset_31(theta, seed=42)
    assert data is not None
    for it in data["items"]:
        d = it["dimensions"]
        assert d["w"] > 0
        assert d["l"] > 0
        assert d["h"] > 0


def test_size_type_conforms_to_vanning_eval_enum():
    """(e) size_type は vanning_eval スキーマ厳格 enum {small,medium,large} 準拠。

    回帰防止: 31種カタログラベル(steel_22 等)を入れると items_input /
    layout_result の両方で SchemaError になる（実地検出済）。
    """
    assert set(VALID_SIZE_TYPES) == {"small", "medium", "large"}
    theta = _feasible_theta()
    data = build_dataset_31(theta, seed=42)
    assert data is not None
    seen = {it["size_type"] for it in data["items"]}
    assert seen, "items should be non-empty"
    assert seen <= set(VALID_SIZE_TYPES), f"non-conforming size_type: {seen}"


def test_size_class_anchor_volumes():
    """既知3アンカー(小/中/大 実寸)が各クラスへ正しく写像される。"""
    assert _size_class(760, 1130, 550) == "small"     # case #24
    assert _size_class(1490, 2260, 900) == "medium"   # case #6/#16
    assert _size_class(2280, 2550, 2355) == "large"   # case #5
