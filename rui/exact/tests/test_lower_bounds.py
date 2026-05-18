"""Tests for lower_bounds.py — hand-calculated verifications."""
from __future__ import annotations

import pytest

from rui.exact.lower_bounds import mt_l2, volume_lb, weight_lb, per_destination_lb, instance_lb


def test_mt_l2_example_6_6_6_6_cap_10():
    # sizes = [6,6,6,6], cap=10
    # K=0: N1=0, N2=4 (6>5), free=4*10-24=16, N3=0, extra=0, bound=4
    # K=1: N1=0, N2=4, free=16, N3=0, extra=0, bound=4
    # K=2: N1=0, N2=4, free=16, N3=0, extra=0, bound=4
    # K=3: N1=0, N2=4, free=16, N3=0, extra=0, bound=4
    # K=4: N1=0, N2=4, free=16, N3=0, extra=0, bound=4
    # K=5: N1=0, N2=4, free=16, N3=0, extra=0, bound=4
    assert mt_l2([6, 6, 6, 6], 10) == 4


def test_mt_l2_empty():
    assert mt_l2([], 10) == 0


def test_mt_l2_all_small():
    # sizes=[2,2,2], cap=10 -> naive ceil=1, L2 should be 1
    assert mt_l2([2, 2, 2], 10) == 1


def test_volume_lb_simple():
    from rui.algorithm_a import Item
    item = Item("i1", "small", 1000, 1000, 1000, 1.0, "D1", 1_000_000_000)
    cap = 2300 * 12000 * 2400
    assert volume_lb([item]) == 1


def test_weight_lb_simple():
    from rui.algorithm_a import Item
    item = Item("i1", "small", 1, 1, 1, 25000.0, "D1", 1)
    assert weight_lb([item]) == 2


def test_invariant_perdest_ge_max():
    """perdest_lb >= max(volume_lb, weight_lb) must hold for any instance."""
    from rui.algorithm_a import Item
    items = [
        Item("i1", "small", 1000, 1000, 1000, 5000.0, "D1", 1_000_000_000),
        Item("i2", "small", 1000, 1000, 1000, 5000.0, "D1", 1_000_000_000),
        Item("i3", "small", 1000, 1000, 1000, 5000.0, "D2", 1_000_000_000),
    ]
    lb = instance_lb(items)
    assert lb["perdest_lb"] >= max(lb["volume_lb"], lb["weight_lb"])
