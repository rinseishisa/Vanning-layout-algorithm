"""Lower bounds for container count — pure arithmetic, no solver dependency."""
from collections import defaultdict

# Container internal dimensions (mm) and max weight (kg)
_X = 2300
_Y = 12000
_Z = 2400
_MAX_WEIGHT = 24000
_CAP_VOL = _X * _Y * _Z


def volume_lb(items) -> int:
    """ceil(Σ volume / container_volume) using true container volume (no 0.8 factor)."""
    total = sum(it.width * it.length * it.height for it in items)
    return -(-total // _CAP_VOL)


def weight_lb(items) -> int:
    """ceil(Σ weight / max_weight)."""
    total = sum(it.weight for it in items)
    return int(-(-total // _MAX_WEIGHT))


def mt_l2(sizes: list[int], capacity: int) -> int:
    """Martello–Toth L2 lower bound for 1D bin packing.

    sizes: positive integers, each <= capacity.
    """
    if not sizes:
        return 0
    best = 1
    half = capacity // 2
    # The L2 bound is piecewise-constant in K; its maxima occur only at
    # breakpoints {0} ∪ {s_i} ∪ {capacity - s_i} within [0, capacity//2].
    # Iterating every integer up to capacity//2 is intractable for the
    # volume axis (capacity ≈ 6.6e10); the breakpoint set (≤2n+1 values)
    # yields the identical maximum. (Claude review fix — brief vanning_exact_b.)
    candidates = {0}
    for s in sizes:
        if 0 <= s <= half:
            candidates.add(s)
        cs = capacity - s
        if 0 <= cs <= half:
            candidates.add(cs)
    for K in sorted(candidates):
        N1 = sum(1 for s in sizes if s > capacity - K)
        N2_items = [s for s in sizes if capacity - K >= s > capacity // 2]
        N2 = len(N2_items)
        N3_items = [s for s in sizes if capacity // 2 >= s >= K]
        free = N2 * capacity - sum(N2_items)
        extra = max(0, -(-(sum(N3_items) - free) // capacity))
        bound_K = N1 + N2 + extra
        if bound_K > best:
            best = bound_K
    return best


def per_destination_lb(items) -> int:
    """Sum over destinations of max(volume_axis_L2, weight_axis_L2).

    Because a single container may hold only one destination, the sum across
    destinations is a valid (and usually tighter) lower bound.
    """
    groups = defaultdict(list)
    for it in items:
        groups[it.destination_id].append(it)
    total = 0
    for g in groups.values():
        vol_axis = mt_l2([it.width * it.length * it.height for it in g], _CAP_VOL)
        wt_axis = mt_l2([round(it.weight) for it in g], _MAX_WEIGHT)
        total += max(vol_axis, wt_axis)
    return total


def instance_lb(items) -> dict:
    """Return a dict with volume, weight, per-dest LB and destination count."""
    vlb = volume_lb(items)
    wlb = weight_lb(items)
    pdlb = per_destination_lb(items)
    n_dest = len({it.destination_id for it in items})
    return {
        "volume_lb": vlb,
        "weight_lb": wlb,
        "perdest_lb": pdlb,
        "n_dest": n_dest,
    }
