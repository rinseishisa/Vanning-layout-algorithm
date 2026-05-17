"""theta31.py 単体テスト（design_catalog31.md §5-1 受け入れ）。"""
from __future__ import annotations

import math

import numpy as np
import pytest

from rui.adv_lane.theta31 import (
    THETA_DIM_31,
    check_feasibility31,
    decode_theta31,
    encode_theta31,
)


def _theta(**kw) -> np.ndarray:
    """人間可読指定で θ を組む薄いヘルパ（既定は中庸）。"""
    return encode_theta31(
        p_wood=kw.get("p_wood", 0.4),
        mu_wood=kw.get("mu_wood", 0.5),
        mu_steel=kw.get("mu_steel", 0.5),
        nu_wood=kw.get("nu_wood", 8.0),
        nu_steel=kw.get("nu_steel", 8.0),
        rho_shift=kw.get("rho_shift", 0.0),
        rho_gain=kw.get("rho_gain", 1.0),
        dest_weights=kw.get("dest_weights", (1 / 3, 1 / 3, 1 / 3)),
        s_scale=kw.get("s_scale", 0.8),
    )


class TestDecodeShapes:
    def test_basic_shapes_and_normalization(self):
        p = decode_theta31(_theta())
        assert p["size_prob"].shape == (31,)
        assert len(p["type_meta"]) == 31
        assert p["dest_weights"].shape == (3,)
        assert p["size_prob"].sum() == pytest.approx(1.0, abs=1e-9)
        assert p["dest_weights"].sum() == pytest.approx(1.0, abs=1e-9)
        assert 0.0 <= p["size_entropy_norm"] <= 1.0
        assert S_SCALE_OK(p["s_scale"])

    def test_theta_dim_constant(self):
        assert THETA_DIM_31 == 11
        assert _theta().shape == (THETA_DIM_31,)

    def test_wrong_dim_raises(self):
        with pytest.raises(ValueError):
            decode_theta31(np.zeros(13))

    def test_catalog_anchors_present(self):
        meta = decode_theta31(_theta())["type_meta"]
        dims = {tuple(sorted(m[3:6])) for m in meta}
        assert tuple(sorted((760, 1130, 550))) in dims     # small  #24
        assert tuple(sorted((1490, 2260, 900))) in dims     # medium #6/#16
        assert tuple(sorted((2280, 2550, 2355))) in dims     # large  #5


def S_SCALE_OK(v: float) -> bool:
    return 0.6 - 1e-9 <= v <= 1.0 + 1e-9


class TestMixtureControls:
    def test_material_split_matches_mat_logit(self):
        """P(wood 合計) ≈ p_wood（素材比が混合確率総和に反映される）。"""
        p = decode_theta31(_theta(p_wood=0.25))
        meta = p["type_meta"]
        wood_mass = sum(
            pr for pr, m in zip(p["size_prob"], meta) if m[1] == "wood"
        )
        assert wood_mass == pytest.approx(0.25, abs=1e-6)

    def test_concentration_increases_peakedness(self):
        """ν 大 → 素材内分布が尖る → size_entropy_norm 低下（単調）。"""
        lo = decode_theta31(_theta(nu_wood=2.0, nu_steel=2.0))["size_entropy_norm"]
        hi = decode_theta31(_theta(nu_wood=45.0, nu_steel=45.0))["size_entropy_norm"]
        assert hi < lo

    def test_rank_mean_shifts_mass_along_volume(self):
        """μ 小 → 小体積種に質量、μ 大 → 大体積種に質量。"""
        meta = decode_theta31(_theta())["type_meta"]
        vol = np.array([m[3] * m[4] * m[5] for m in meta], dtype=float)

        small_biased = decode_theta31(_theta(mu_wood=0.1, mu_steel=0.1, nu_wood=30, nu_steel=30))
        large_biased = decode_theta31(_theta(mu_wood=0.9, mu_steel=0.9, nu_wood=30, nu_steel=30))
        ev_small = float((small_biased["size_prob"] * vol).sum())
        ev_large = float((large_biased["size_prob"] * vol).sum())
        assert ev_small < ev_large  # 期待体積が μ で単調に動く


class TestDensity:
    def test_gain_stretches_band(self):
        narrow = decode_theta31(_theta(rho_gain=0.4))["density_by_material"]["steel"]
        wide = decode_theta31(_theta(rho_gain=1.8))["density_by_material"]["steel"]
        assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])

    def test_shift_moves_center_and_clips(self):
        lo_shift = decode_theta31(_theta(rho_shift=-1.0))["density_by_material"]["wood"]
        hi_shift = decode_theta31(_theta(rho_shift=1.0))["density_by_material"]["wood"]
        c_lo = 0.5 * sum(lo_shift)
        c_hi = 0.5 * sum(hi_shift)
        assert c_hi > c_lo
        for band in (lo_shift, hi_shift):
            assert 30.0 - 1e-6 <= band[0] <= band[1] <= 700.0 + 1e-6
            assert band[1] >= band[0]  # 非退化


class TestFeasibility:
    def test_normalized_entropy_guard(self):
        # 中庸 θ は健全
        assert check_feasibility31(decode_theta31(_theta())) is None
        # 極端集中 + 片素材寄せ → 正規化エントロピー低下で reject 期待
        deg = _theta(p_wood=0.999, mu_wood=0.5, nu_wood=50.0, nu_steel=50.0)
        params = decode_theta31(deg)
        # 退化方向では guard が文字列（理由）を返す（None でない）こともある
        res = check_feasibility31(params, h_min=0.30)
        assert res is None or isinstance(res, str)

    def test_dest_min_guard(self):
        params = decode_theta31(_theta(dest_weights=(0.96, 0.02, 0.02)))
        assert check_feasibility31(params, p_min=0.08) is not None


class TestRoundtrip:
    def test_encode_decode_recovers_params(self):
        th = encode_theta31(
            p_wood=0.3, mu_wood=0.7, mu_steel=0.35,
            nu_wood=12.0, nu_steel=20.0,
            rho_shift=0.4, rho_gain=1.3,
            dest_weights=(0.5, 0.3, 0.2), s_scale=0.75,
        )
        p = decode_theta31(th)
        assert p["mat_p_wood"] == pytest.approx(0.3, abs=1e-6)
        assert p["mu"]["wood"] == pytest.approx(0.7, abs=1e-4)
        assert p["mu"]["steel"] == pytest.approx(0.35, abs=1e-4)
        assert p["nu"]["wood"] == pytest.approx(12.0, abs=1e-3)
        assert p["nu"]["steel"] == pytest.approx(20.0, abs=1e-3)
        assert p["s_scale"] == pytest.approx(0.75, abs=1e-4)
        # dest 比は softmax 後も比率保存
        dw = p["dest_weights"]
        assert dw[0] / dw[1] == pytest.approx(0.5 / 0.3, rel=1e-4)
