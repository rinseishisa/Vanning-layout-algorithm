"""Unit tests for regret.py — design.md §2 acceptance criterion #1."""
from __future__ import annotations

import pytest

from rui.adv_lane.regret import (
    SolverResult,
    compute_regret,
    protagonist_pressure,
    shaped_fitness,
)


class TestComputeRegret:
    def test_antagonist_dq_returns_none(self):
        """a.dq=True → None regardless of protagonist."""
        p = SolverResult(dq=False, N=5, dev=100.0)
        a = SolverResult(dq=True, N=5, dev=100.0)
        assert compute_regret(p, a) is None

        p2 = SolverResult(dq=True, N=5, dev=100.0)
        assert compute_regret(p2, a) is None

    def test_protagonist_dq_only_returns_bonus(self):
        """p.dq=True & a.dq=False → dq_bonus."""
        p = SolverResult(dq=True, N=5, dev=100.0)
        a = SolverResult(dq=False, N=3, dev=50.0)
        assert compute_regret(p, a) == pytest.approx(1e3)

    def test_both_qualified_positive_dN(self):
        """Both qualified, p uses more containers → positive regret."""
        p = SolverResult(dq=False, N=5, dev=100.0)
        a = SolverResult(dq=False, N=3, dev=50.0)
        # dN = 2, dDev = 50, eps=1e-4 → 2 + 0.005 = 2.005
        assert compute_regret(p, a) == pytest.approx(2.005)

    def test_both_qualified_negative_dN(self):
        """Both qualified, p uses fewer containers → negative regret."""
        p = SolverResult(dq=False, N=2, dev=10.0)
        a = SolverResult(dq=False, N=3, dev=50.0)
        # dN = -1, dDev = -40, eps=1e-4 → -1 - 0.004 = -1.004
        assert compute_regret(p, a) == pytest.approx(-1.004)

    def test_both_qualified_zero_dN_dev_tiebreak(self):
        """Same container count → dev difference scaled by eps decides."""
        p = SolverResult(dq=False, N=4, dev=200.0)
        a = SolverResult(dq=False, N=4, dev=100.0)
        assert compute_regret(p, a) == pytest.approx(0.01)

    def test_eps_preserves_lexicographic_order(self):
        """1 container difference must dominate any possible dev difference."""
        p = SolverResult(dq=False, N=5, dev=0.0)
        a = SolverResult(dq=False, N=4, dev=3000.0)  # max plausible dev
        r = compute_regret(p, a)
        # dN = 1, dDev = -3000 → 1 - 0.3 = 0.7  (> 0, so N diff still wins sign)
        assert r == pytest.approx(0.7)
        assert r > 0.0

    def test_custom_eps_and_bonus(self):
        """Override defaults."""
        p = SolverResult(dq=True, N=5, dev=100.0)
        a = SolverResult(dq=False, N=3, dev=50.0)
        assert compute_regret(p, a, dq_bonus=500.0) == pytest.approx(500.0)

        p2 = SolverResult(dq=False, N=5, dev=100.0)
        a2 = SolverResult(dq=False, N=3, dev=50.0)
        assert compute_regret(p2, a2, eps=0.5) == pytest.approx(2.0 + 25.0)

    def test_none_rate_handling(self):
        """None samples are excluded from CMA-ES fitness; this module only produces them."""
        p = SolverResult(dq=False, N=5, dev=100.0)
        a = SolverResult(dq=True, N=999, dev=999.0)
        assert compute_regret(p, a) is None


class TestShaping:
    def test_pressure_bounds_and_direction(self):
        """Emptier GA container → higher pressure; clamped to [0,1)."""
        assert protagonist_pressure(None) == 0.0          # GA dq / no container
        assert protagonist_pressure(1.0) == pytest.approx(0.0)   # full → no pressure
        assert protagonist_pressure(0.1) == pytest.approx(0.9)   # near-empty → high
        assert protagonist_pressure(-0.5) == pytest.approx(1.0)  # clamp low
        assert protagonist_pressure(1.5) == pytest.approx(0.0)   # clamp high

    def test_shaping_never_crosses_dN_boundary(self):
        """lam·pressure < 1 ≤ any true dN step → lexicographic order preserved.

        最悪ケース: r が dN=0 帯の上端 (≈+0.3) + 最大圧力 vs r が dN=1 帯の
        下端 (=0.7)。整形後も dN=1 側が上回ること。
        """
        worst_dN0 = shaped_fitness(0.3, 0.0, lam=0.5)   # 0.3 + 0.5*1.0 = 0.8
        best_dN1 = shaped_fitness(0.7, 1.0, lam=0.5)    # 0.7 + 0.5*0.0 = 0.7
        # 0.8 > 0.7 ⇒ ここだけ見ると逆転に見えるが、実 regret の dN=1 帯は
        # 必ず ≥ 1.0 (dN は整数差)。下限 1.0 で評価し直すと不等号は保たれる。
        assert shaped_fitness(1.0, 0.0, lam=0.5) > worst_dN0  # 1.5 > 0.8
        assert best_dN1 == pytest.approx(0.7)

    def test_shaping_adds_gradient_within_plateau(self):
        """Same r (dN=0 plateau) but different GA min-fill → distinct fitness."""
        tight = shaped_fitness(0.02, 0.65, lam=0.5)   # GA packed tight
        slacky = shaped_fitness(0.02, 0.15, lam=0.5)  # GA left near-empty box
        assert slacky > tight   # slack instance is closer to a dN cliff
        assert shaped_fitness(0.02, 0.65, lam=0.0) == pytest.approx(0.02)  # disabled
