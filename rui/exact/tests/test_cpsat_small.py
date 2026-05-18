"""Small CP-SAT tests using synthetic instances with known optima."""
from __future__ import annotations

import pytest

pytest.importorskip("ortools")

from rui.exact.slice_instances import synthetic_known
from rui.exact.cpsat_model import solve_min_containers


def test_synthetic_known():
    for name, items, expected in synthetic_known():
        result = solve_min_containers(
            items,
            time_limit_s=30.0,
            max_containers=max(expected, 1),
        )
        assert result["status"] == "OPTIMAL", f"{name}: expected OPTIMAL, got {result['status']}"
        assert result["n_containers"] == expected, f"{name}: expected {expected}, got {result['n_containers']}"
