"""Regret computation for adversarial instance generation (design.md §2)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SolverResult:
    """Minimal solver output needed for regret computation."""

    dq: bool          # disqualified?
    N: int            # number of containers used
    dev: float        # mean |Yg - 6000| over all containers (mm)


def compute_regret(
    p: SolverResult,
    a: SolverResult,
    *,
    eps: float = 1e-4,
    dq_bonus: float = 1e3,
) -> Optional[float]:
    """Compute regret scalar r(θ) = protagonist vs antagonist.

    Returns ``None`` when the instance should be discarded (antagonist also
    disqualified → degenerate / infeasible).

    Boundary semantics (exactly as design.md §2):
    - a.dq=True                → None   (discard)
    - p.dq=True & a.dq=False   → dq_bonus  (best hard instance)
    - both dq=False            → dN + eps * dDev
    """
    if a.dq:
        return None
    if p.dq and not a.dq:
        return dq_bonus
    # Both qualified
    dN = p.N - a.N
    dDev = p.dev - a.dev
    return dN + eps * dDev


def protagonist_pressure(p_min_fill: Optional[float]) -> float:
    """Continuous proxy in [0, 1) of how close GA is to a regret cliff.

    本走診断: regret は整数 dN 支配でプラトーが平坦 → CMA-ES が孤立
    dN=1 スパイクを追尾できない。GA が「ほぼ空のコンテナ」を1本残して
    いる instance は、わずかな摂動で antagonist が dN=1 を達成しうる
    “崖の手前”。GA 最空コンテナの空き率 (1 - min fill) を圧力として
    返し、プラトーに登れる勾配を与える（探索信号専用、採点は不変）。

    ``p_min_fill`` が None（GA 失格/コンテナ無し）の場合は 0.0。
    """
    if p_min_fill is None:
        return 0.0
    return max(0.0, min(1.0, 1.0 - p_min_fill))


def shaped_fitness(
    r: float,
    p_min_fill: Optional[float],
    *,
    lam: float = 0.5,
) -> float:
    """CMA-ES 探索専用の整形 fitness = r + lam · pressure。

    ``lam < 1`` かつ ``pressure < 1`` なので整形項は常に 1 未満 ≤ 真の
    dN ステップ。よって辞書式の dN 境界順位を跨いで並べ替えない
    （compute_regret / hard-instance 保存は純 r のまま）。
    """
    return r + lam * protagonist_pressure(p_min_fill)
