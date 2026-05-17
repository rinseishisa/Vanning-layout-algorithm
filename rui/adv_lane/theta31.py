"""31種カタログ用 θ 再パラメータ化（design_catalog31.md §3.1）。

核心: 31種を「素材軸 × サイズランク軸」の2次元に並べ、その上の
**分布の位置と尖り**だけを少数 knob で操る。各種確率はそこから導出。
→ θ を種数に依存しない ≈11-D に圧縮（CMA-ES 安全域、§3.1 §12）。

既存 13-D（`generator.decode_theta`）は不変。本モジュールは独立追加。
catalog は実データ `catalog31.CATALOG_31`（純データ定数）。
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from rui.adv_lane.catalog31 import CATALOG_31

# ------------------------------------------------------------------
# 固定ハイパー（design_catalog31.md §2.2 / §3.2）
# ------------------------------------------------------------------
NU_LO, NU_HI = 2.0, 50.0          # Beta concentration（小=ほぼ均し / 大=鋭い単峰）
RHO_LO_G, RHO_HI_G = 30.0, 700.0  # 全体現実密度境界 kg/m³
WOOD_RHO0 = (40.0, 180.0)         # 木箱 基底密度帯（§2.2）
STEEL_RHO0 = (120.0, 600.0)       # スチール 基底密度帯（§2.2）
RHO_MAX_SHIFT = 150.0             # rho_shift=±1 で帯中心を ±150 kg/m³
GAIN_LO, GAIN_HI = 0.4, 1.8       # rho_gain で帯半幅を拡縮
S_SCALE_LO, S_SCALE_HI = 0.6, 1.0 # 体積バジェット倍率（generator と同レンジ）
EPS = 1e-12

# θ レイアウト（11-D active。予備次元は表現力不足が出た時のみ design §3.1 で解放）
#  [0]    mat_logit        -> P(wood)
#  [1]    rank_mean_w_raw  -> μ_wood  ∈ (0,1)
#  [2]    rank_mean_s_raw  -> μ_steel
#  [3]    rank_conc_w_raw  -> ν_wood  ∈ [NU_LO,NU_HI]
#  [4]    rank_conc_s_raw  -> ν_steel
#  [5]    rho_shift_raw    -> shift   ∈ (-1,1)  (tanh)
#  [6]    rho_gain_raw     -> gain    ∈ [GAIN_LO,GAIN_HI]
#  [7:10] dest_logits      -> softmax (3)
#  [10]   s_scale_raw      -> s_scale ∈ [S_SCALE_LO,S_SCALE_HI]
THETA_DIM_31 = 11

_MATERIALS = ("wood", "steel")


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-_clamp(x, -60.0, 60.0)))


def _sigmoid_lin(x: float, lo: float, hi: float) -> float:
    return lo + (hi - lo) * _sigmoid(x)


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / (e.sum() + EPS)


def _entropy(p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1.0)
    return float(-(p * np.log(p)).sum())


def _build_rank_table() -> Dict[str, List[Tuple[int, str, str, int, int, int, float, float]]]:
    """素材ごとに体積昇順で並べ、ランク r=(i+0.5)/n ∈ (0,1) を付与。

    開区間にするのは Beta の端点 0/1 特異点（α-1<0 等で発散）回避のため。
    返り値: material -> [(id, material, label, w, l, h, vol_m3, rank), ...]
    """
    table: Dict[str, List] = {m: [] for m in _MATERIALS}
    for (cid, mat, label, w, l, h) in CATALOG_31:
        vol_m3 = (w * l * h) / 1e9
        table[mat].append([cid, mat, label, w, l, h, vol_m3, None])
    for mat, rows in table.items():
        rows.sort(key=lambda r: (r[6], r[0]))  # volume asc, id tiebreak
        n = len(rows)
        for i, r in enumerate(rows):
            r[7] = (i + 0.5) / n
        table[mat] = [tuple(r) for r in rows]
    return table


_RANK_TABLE = _build_rank_table()
# 31種を id 昇順で固定列挙（size_prob のインデックス順 = この順序）
_FLAT_TYPES = sorted(
    [row for rows in _RANK_TABLE.values() for row in rows],
    key=lambda r: r[0],
)


def _beta_logweight(rank: float, mu: float, nu: float) -> float:
    """Beta 形 **対数**重み log[r^(α-1)(1-r)^(β-1)]（正規化定数不要）。

    高 ν では生の r^(α-1)(1-r)^(β-1) が float underflow → 0 になり
    後段の正規化が退化（uniform に潰れ集中度が効かない）。log 空間で
    保持し、素材内で log-sum-exp（max 減算）正規化する。
    """
    alpha = mu * nu
    beta = (1.0 - mu) * nu
    r = _clamp(rank, EPS, 1.0 - EPS)
    return (alpha - 1.0) * math.log(r) + (beta - 1.0) * math.log(1.0 - r)


def decode_theta31(theta: np.ndarray) -> Dict[str, object]:
    """θ(11-D) → 生成パラメータ。

    返り値:
      size_prob          : np.ndarray (31,) — _FLAT_TYPES(=id昇順) に対応、Σ=1
      type_meta          : List[(id, material, label, w, l, h)] — _FLAT_TYPES と同順
      density_by_material: {"wood":(lo,hi), "steel":(lo,hi)}  kg/m³
      dest_weights       : np.ndarray (3,) Σ=1
      s_scale            : float
      size_entropy_norm  : float ∈ [0,1]  (= H / ln(31))
      mat_p_wood, mu, nu : 監査用
    """
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (THETA_DIM_31,):
        raise ValueError(f"theta must be ({THETA_DIM_31},), got {theta.shape}")

    p_wood = _sigmoid(float(theta[0]))
    p_mat = {"wood": p_wood, "steel": 1.0 - p_wood}
    mu = {
        "wood": _clamp(_sigmoid(float(theta[1])), 1e-3, 1.0 - 1e-3),
        "steel": _clamp(_sigmoid(float(theta[2])), 1e-3, 1.0 - 1e-3),
    }
    nu = {
        "wood": _sigmoid_lin(float(theta[3]), NU_LO, NU_HI),
        "steel": _sigmoid_lin(float(theta[4]), NU_LO, NU_HI),
    }
    shift = math.tanh(float(theta[5]))
    gain = _sigmoid_lin(float(theta[6]), GAIN_LO, GAIN_HI)
    dest_weights = _softmax(theta[7:10])
    s_scale = _sigmoid_lin(float(theta[10]), S_SCALE_LO, S_SCALE_HI)

    # --- size_prob: 素材内 Beta 正規化 × 素材比 ---
    prob_by_id: Dict[int, float] = {}
    for mat in _MATERIALS:
        rows = _RANK_TABLE[mat]
        logw = np.array(
            [_beta_logweight(r[7], mu[mat], nu[mat]) for r in rows], dtype=float
        )
        logw -= logw.max()                 # log-sum-exp 安定化
        w = np.exp(logw)
        w = w / w.sum()                    # 素材内で正規化（必ず有限・非退化）
        for row, wi in zip(rows, w):
            prob_by_id[row[0]] = p_mat[mat] * float(wi)
    size_prob = np.array([prob_by_id[t[0]] for t in _FLAT_TYPES], dtype=float)
    size_prob = size_prob / (size_prob.sum() + EPS)

    # --- density: 基底帯を gain で拡縮 + shift で中心移動 + 全体 clip ---
    density_by_material: Dict[str, Tuple[float, float]] = {}
    for mat, (lo0, hi0) in (("wood", WOOD_RHO0), ("steel", STEEL_RHO0)):
        c0 = 0.5 * (lo0 + hi0)
        half = 0.5 * (hi0 - lo0) * gain
        c = c0 + shift * RHO_MAX_SHIFT
        lo = _clamp(c - half, RHO_LO_G, RHO_HI_G)
        hi = _clamp(c + half, RHO_LO_G, RHO_HI_G)
        if hi < lo + 1.0:
            hi = min(RHO_HI_G, lo + 1.0)
        density_by_material[mat] = (lo, hi)

    return {
        "size_prob": size_prob,
        "type_meta": [(t[0], t[1], t[2], t[3], t[4], t[5]) for t in _FLAT_TYPES],
        "density_by_material": density_by_material,
        "dest_weights": dest_weights,
        "s_scale": s_scale,
        "size_entropy_norm": _entropy(size_prob) / math.log(len(_FLAT_TYPES)),
        "mat_p_wood": p_wood,
        "mu": mu,
        "nu": nu,
    }


def encode_theta31(
    p_wood: float,
    mu_wood: float,
    mu_steel: float,
    nu_wood: float,
    nu_steel: float,
    rho_shift: float,
    rho_gain: float,
    dest_weights: Tuple[float, float, float],
    s_scale: float,
) -> np.ndarray:
    """人間可読パラメータ → θ（ウォームスタート/往復テスト用）。"""

    def _logit(p: float) -> float:
        p = _clamp(p, 1e-9, 1.0 - 1e-9)
        return math.log(p / (1.0 - p))

    def _inv_lin(v: float, lo: float, hi: float) -> float:
        t = _clamp((v - lo) / max(hi - lo, EPS), 1e-9, 1.0 - 1e-9)
        return math.log(t / (1.0 - t))

    return np.array(
        [
            _logit(p_wood),
            _logit(mu_wood),
            _logit(mu_steel),
            _inv_lin(nu_wood, NU_LO, NU_HI),
            _inv_lin(nu_steel, NU_LO, NU_HI),
            math.atanh(_clamp(rho_shift, -1.0 + 1e-9, 1.0 - 1e-9)),
            _inv_lin(rho_gain, GAIN_LO, GAIN_HI),
            math.log(max(dest_weights[0], EPS)),
            math.log(max(dest_weights[1], EPS)),
            math.log(max(dest_weights[2], EPS)),
            _inv_lin(s_scale, S_SCALE_LO, S_SCALE_HI),
        ],
        dtype=float,
    )


def check_feasibility31(params: Dict[str, object], h_min: float = 0.30,
                        p_min: float = 0.08) -> "str | None":
    """正規化エントロピー基準（design §3.2）。raw entropy<0.30 は31種で無意味化。"""
    if params["size_entropy_norm"] < h_min:
        return f"size_entropy_norm={params['size_entropy_norm']:.3f} < {h_min}"
    dw = params["dest_weights"]
    if float(np.min(dw)) < p_min:
        return f"dest_min={float(np.min(dw)):.3f} < {p_min}"
    return None
