"""Antagonist: deterministic beam-search strong variant (design.md §4).

Reuses placement primitives from ``rui.algorithm_a`` without reimplementation.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

from rui.algorithm_a import (
    Container,
    Item,
    PlacedItem,
    bounding_box_volume,
    can_place,
    candidate_score,
    find_best_placement,
    generate_candidate_points,
    make_placed_item,
    rotated_dims,
    y_deviation,
)
from rui.generate_items import DESTINATIONS

ROTATIONS = [0, 90]


def _partial_lex_key(state: List[Container]) -> Tuple[int, float, int]:
    """Beam-pruning key: smaller is better."""
    n = len(state)
    mean_dev = sum(y_deviation(c.items) for c in state) / max(len(state), 1)
    dead_space = sum(
        bounding_box_volume(c.items) - c.used_volume for c in state
    )
    return (n, mean_dev, dead_space)


def _final_lex_key(state: List[Container]) -> Tuple[int, float]:
    """Final selection key: smaller is better."""
    n = len(state)
    mean_dev = sum(y_deviation(c.items) for c in state) / max(len(state), 1)
    return (n, mean_dev)


def _top_k_placements(
    state: List[Container],
    item: Item,
    k: int,
) -> List[Tuple[Container, PlacedItem]]:
    """Return up to *k* feasible (container, candidate) pairs for *item*."""
    scored: List[Tuple[Tuple, Container, PlacedItem]] = []

    # 既存コンテナへの配置候補のみを収集（新規開封は最後の手段）
    for container in state:
        if container.destination_id != item.destination_id:
            continue
        pts = generate_candidate_points(container)
        for rot in ROTATIONS:
            w, l, h = rotated_dims(item, rot)
            for x, y, z in pts:
                cand = make_placed_item(item, container.container_id, rot, x, y, z)
                if not can_place(container, cand):
                    continue
                score = candidate_score(container, cand)
                scored.append((score, container, cand))
            # Also try centred floor point explicitly (algorithm_a does this inside find_best_placement)
            cx = max(0, min(2300 - w, int(round((2300 - w) / 2.0))))
            cy = max(0, min(12000 - l, int(round((12000 - l) / 2.0))))
            cand = make_placed_item(item, container.container_id, rot, cx, cy, 0)
            if can_place(container, cand):
                score = candidate_score(container, cand)
                scored.append((score, container, cand))

    scored.sort(key=lambda t: t[0])
    unique: List[Tuple[Container, PlacedItem]] = []
    seen = set()
    for _, container, cand in scored:
        # Deduplicate by (container_id, x, y, z, rot)
        sig = (container.container_id, cand.x, cand.y, cand.z, cand.is_rotated)
        if sig in seen:
            continue
        seen.add(sig)
        unique.append((container, cand))
        if len(unique) >= k:
            break

    # 既存に置けるなら新規コンテナは開かない（GA の貪欲単経路に対する
    # ビームの優位性は「既存配置の多様な分岐を保持し新規開封を遅延」で出る）
    if unique:
        return unique

    # 既存コンテナに 1 つも置けない → 新規コンテナを開くしかない
    new_container = Container(
        container_id=len(state) + 1, destination_id=item.destination_id
    )
    cand = find_best_placement(new_container, item)
    if cand is None:
        return []
    return [(new_container, cand)]


def _apply_placement(
    state: List[Container],
    container: Container,
    candidate: PlacedItem,
) -> List[Container]:
    """Return a new state with *candidate* placed into *container*."""
    new_state = [copy.deepcopy(c) for c in state]
    matched = False
    for c in new_state:
        if c.container_id == container.container_id:
            c.items.append(candidate)
            matched = True
            break
    if not matched:
        # brand-new container: deepcopy and renumber sequentially
        nc = copy.deepcopy(container)
        nc.items = [candidate]
        nc.container_id = len(new_state) + 1
        new_state.append(nc)
    return new_state


def _beam_search_for_group(
    items: List[Item],
    beam_width: int,
    branch: int,
) -> Optional[List[Container]]:
    """Run beam search on a single destination group."""
    states: List[List[Container]] = [[]]
    for item in items:
        cand_states: List[List[Container]] = []
        for st in states:
            placements = _top_k_placements(st, item, branch)
            if not placements:
                continue
            for container, candidate in placements:
                cand_states.append(_apply_placement(st, container, candidate))
        if not cand_states:
            # Beam died — cannot place this item
            return None
        # Prune to beam_width best partial states
        cand_states.sort(key=_partial_lex_key)
        states = cand_states[:beam_width]
    if not states:
        return None
    states.sort(key=_final_lex_key)
    return states[0]


def beam_search_strong(
    items: List[Item],
    beam_width: int = 48,
    branch: int = 12,
) -> Tuple[List[Container], bool]:
    """Strong deterministic antagonist (design.md §4).

    Returns ``(containers, dq)`` where *dq* is True when the antagonist could
    not find a feasible layout for at least one destination group.
    """
    # Group by destination (same ordering as algorithm_a.build_items)
    groups: Dict[str, List[Item]] = {dest: [] for dest in DESTINATIONS}
    for item in items:
        groups.setdefault(item.destination_id, []).append(item)

    all_containers: List[Container] = []
    for dest in DESTINATIONS:
        group_items = groups.get(dest, [])
        if not group_items:
            continue
        # Sort heaviest/largest first (same heuristic as algorithm_a)
        group_items = sorted(group_items, key=lambda it: (-it.weight, -it.volume, it.item_id))
        result = _beam_search_for_group(group_items, beam_width, branch)
        if result is None:
            return [], True
        # Renumber container IDs globally so they are unique across dest groups
        for c in result:
            c.container_id = len(all_containers) + 1
        all_containers.extend(result)

    return all_containers, False
