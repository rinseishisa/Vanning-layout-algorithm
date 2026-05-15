"""antagonist.beam_search_strong の出力健全性テスト。

回帰防止: コンテナ id 採番バグ（dest グループ内全コンテナに同一 id を
振り、出力 layout_result の container_id が `1,1,3,3,3,6,6,6,6` のように
重複・歯抜けになる）を捕捉する。dN/regret は len(containers) 基準なので
不変だが、vanning-eval の表示・集計が崩れる cosmetic-but-output 破壊。
"""
from __future__ import annotations

import numpy as np

from rui.adv_lane.antagonist import beam_search_strong
from rui.adv_lane.generator31 import build_dataset_31
from rui.adv_lane.loop import _make_dataframe
from rui.adv_lane.theta31 import encode_theta31
from rui.algorithm_a import build_items


def _items():
    theta = encode_theta31(
        p_wood=0.3, mu_wood=0.5, mu_steel=0.5, nu_wood=10.0, nu_steel=10.0,
        rho_shift=0.0, rho_gain=1.0, dest_weights=(1 / 3, 1 / 3, 1 / 3), s_scale=0.8,
    )
    data = build_dataset_31(theta, seed=42)
    assert data is not None
    return build_items(_make_dataframe(data))


def test_beam_container_ids_unique_sequential():
    containers, dq = beam_search_strong(_items())
    assert not dq, "feasible instance should not disqualify the antagonist"
    ids = [c.container_id for c in containers]
    n = len(containers)
    assert n > 0
    # 一意かつ 1..N の連番（重複・歯抜け禁止）
    assert ids == list(range(1, n + 1)), f"non-sequential/duplicate ids: {ids}"
    assert len(set(ids)) == n


def test_beam_multi_dest_no_id_collision():
    """複数 dest 群でも id が衝突しない（バグの本丸）。"""
    items = _items()
    dests = {it.destination_id for it in items}
    assert len(dests) >= 2, "test instance must span multiple destinations"
    containers, dq = beam_search_strong(items)
    assert not dq
    ids = [c.container_id for c in containers]
    assert len(set(ids)) == len(ids), f"id collision across dest groups: {ids}"
