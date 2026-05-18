"""Extract small single-destination groups and synthetic instances."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import Path

# Ensure repo root on sys.path for imports
import sys
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rui.adv_lane.ga_bench import build_suite, _load_items


def iter_small_single_dest(max_items: int):
    """Yield (label, items) for every single-destination group with <= max_items items."""
    for path in build_suite():
        items = _load_items(path)
        groups = defaultdict(list)
        for it in items:
            groups[it.destination_id].append(it)
        for dest_id, g in groups.items():
            if len(g) <= max_items:
                label = f"{path.stem}::{dest_id}"
                yield label, g


def synthetic_known() -> list[tuple[str, list, int]]:
    """Return synthetic instances with known optimal container counts."""
    # syn_one: three small boxes that obviously fit into one container.
    # Container: 2300x12000x2400. Box: 760x1130x550.
    # Place at (0,0,0), (760,0,0), (1520,0,0): X extents 760,1520,2280 <= 2300.
    # Y extent 1130 <= 12000, Z extent 550 <= 2400.
    # Weight is negligible (1 kg each), total 3 kg <= 24000.
    from rui.algorithm_a import Item
    box = Item(
        item_id="b1",
        size_type="small",
        width=760,
        length=1130,
        height=550,
        weight=1.0,
        destination_id="D1",
        volume=760 * 1130 * 550,
    )
    syn_one = [box, replace(box, item_id="b2"), replace(box, item_id="b3")]

    # syn_two: two long items that cannot share a container.
    # Item: 2300x11900x2400. Rotated: 11900x2300x2400 -> X=11900 > 2300, impossible.
    # Unrotated: X=2300<=2300, Y=11900<=12000, Z=2400<=2400.
    # Two items: any dimension doubled exceeds container capacity, and rotation does not help.
    long = Item(
        item_id="l1",
        size_type="large",
        width=2300,
        length=11900,
        height=2400,
        weight=1.0,
        destination_id="D1",
        volume=2300 * 11900 * 2400,
    )
    syn_two = [long, replace(long, item_id="l2")]

    return [
        ("syn_one", syn_one, 1),
        ("syn_two", syn_two, 2),
    ]
