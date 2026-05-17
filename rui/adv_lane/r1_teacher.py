"""レーン3 R1: beam の勝ち軌跡から配置スコアラの教師データを抽出。

R0 の知見: 固定枝刈りキー ``_partial_lex_key`` は不均一 (hard_01 で逆効果)。
R1 はこのキーを **学習スコアラ** で置換する。教師は強リファレンス
``beam_search_strong`` の **勝ち軌跡** = 各 decode step で beam が選んだ配置。

設計の要:
- antagonist.py は無改変 (pytest/vanning_eval 検証済を壊さない)。本モジュールで
  ``_beam_search_for_group`` を **履歴付きにミラー** し helper だけ import 再利用。
- step ごとに候補を **即 featurize** して (Container,PlacedItem) の重い
  オブジェクトを破棄 → trace は小 np 配列のみ＝メモリ有界。
- featurize は推論 (pack_items_learned) でも同一のものを再利用する。

Step0 規律 ([[feedback_ml_repo_trial_template]]): full 訓練前に
先頭 N inst で #steps/#rows/抽出 wall/sklearn fit 時間を実測する。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rui.adv_lane.antagonist import (  # noqa: E402
    _ITEM_ORDERINGS,
    _apply_placement,
    _final_lex_key,
    _partial_lex_key,
    _top_k_placements,
)
from rui.algorithm_a import (  # noqa: E402
    CONTAINER_HEIGHT_MM,
    CONTAINER_LENGTH_MM,
    CONTAINER_VOLUME_MM3,
    CONTAINER_WIDTH_MM,
    MAX_CONTAINER_WEIGHT_KG,
    Container,
    Item,
    PlacedItem,
    bounding_box_volume,
    y_deviation,
)
from rui.generate_items import DESTINATIONS  # noqa: E402  (antagonist と同一真実源)

# 教師抽出 beam 設定。強さ(教師品質)とコストのトレードオフ。
# beam_search_strong 既定は 48/12 だが教師抽出は全 inst 反復するため
# Step0 でこの値の wall を測ってから確定する。
TEACHER_BEAM_WIDTH = 24
TEACHER_BRANCH = 6

FEATURE_NAMES = [
    "d_dead_space",      # 対象コンテナの bounding-box 死空間増分 (正規化)
    "d_y_dev",           # 対象コンテナの重心ズレ増分 (正規化, Y_LIMIT基準)
    "cand_z",            # 配置 z (積み高さ) 正規化
    "is_rotated",        # 90度回転か
    "opens_new",         # 新規コンテナを開くか
    "fill_after",        # 対象コンテナ充填率 (配置後)
    "weight_after",      # 対象コンテナ重量比 (配置後, MAX基準)
    "item_vol",          # item 体積 / コンテナ容積
    "item_weight",       # item 重量 / MAX
    "footprint",         # 接地面積 / コンテナ床面積
    "n_containers",      # これまでのコンテナ数 / 20
    "remaining_frac",    # 残 item 比
    "is_small",
    "is_medium",
    "is_large",
]
FEATURE_DIM = len(FEATURE_NAMES)

_FLOOR_AREA = CONTAINER_WIDTH_MM * CONTAINER_LENGTH_MM


def _size_onehot(size_type: str) -> Tuple[float, float, float]:
    s = (size_type or "").lower()
    return (
        1.0 if s == "small" else 0.0,
        1.0 if s == "medium" else 0.0,
        1.0 if s == "large" else 0.0,
    )


def featurize(
    state: Sequence[Container],
    target: Container,
    cand: PlacedItem,
    item: Item,
    n_done: int,
    n_total: int,
) -> np.ndarray:
    """(部分状態, 候補配置) → 特徴ベクトル [FEATURE_DIM]。

    既存 primitive のみ使用。固定 lex キー (_partial_lex_key /
    candidate_score) が使う量を素材として渡し、重み・非線形は学習側へ。
    """
    existing_ids = {c.container_id for c in state}
    opens_new = 0.0 if target.container_id in existing_ids else 1.0

    before = list(target.items)
    after = [*before, cand]

    bbv_after = bounding_box_volume(after)
    vol_after = sum(it.width * it.length * it.height for it in after)
    bbv_before = bounding_box_volume(before)
    vol_before = sum(it.width * it.length * it.height for it in before)
    d_dead = ((bbv_after - vol_after) - (bbv_before - vol_before)) / CONTAINER_VOLUME_MM3

    d_ydev = (y_deviation(after) - y_deviation(before)) / 3000.0

    fill_after = vol_after / CONTAINER_VOLUME_MM3
    weight_after = (
        sum(it.weight for it in after) / MAX_CONTAINER_WEIGHT_KG
    )
    s_small, s_med, s_large = _size_onehot(item.size_type)

    return np.array(
        [
            d_dead,
            d_ydev,
            cand.z / CONTAINER_HEIGHT_MM,
            1.0 if cand.is_rotated else 0.0,
            opens_new,
            fill_after,
            weight_after,
            item.volume / CONTAINER_VOLUME_MM3,
            item.weight / MAX_CONTAINER_WEIGHT_KG,
            (cand.width * cand.length) / _FLOOR_AREA,
            len(state) / 20.0,
            (n_total - n_done) / max(n_total, 1),
            s_small,
            s_med,
            s_large,
        ],
        dtype=np.float64,
    )


class _Traced:
    """beam 状態 + 勝ち判定用の決定履歴 (feat 行列, chosen 添字)。"""

    __slots__ = ("containers", "trace")

    def __init__(self, containers: List[Container], trace: List[Tuple[np.ndarray, int]]):
        self.containers = containers
        self.trace = trace


def _traced_beam_for_group(
    items: List[Item],
    beam_width: int,
    branch: int,
    n_total: int,
    n_offset: int,
) -> Optional[_Traced]:
    """``_beam_search_for_group`` のミラー + 決定履歴記録。

    勝ち状態 (states[0]) の ``trace`` が教師軌跡 = 各 step の
    (候補 feat 行列 [n_cand, FEATURE_DIM], beam が選んだ添字)。
    """
    beam: List[_Traced] = [_Traced([], [])]
    for step, item in enumerate(items):
        children: List[_Traced] = []
        for st in beam:
            placements = _top_k_placements(st.containers, item, branch)
            if not placements:
                continue
            n_done = n_offset + step
            feats = np.stack(
                [
                    featurize(st.containers, c, cand, item, n_done, n_total)
                    for (c, cand) in placements
                ]
            )  # [n_cand, FEATURE_DIM]; 兄弟で参照共有 (再計算しない)
            for j, (container, candidate) in enumerate(placements):
                child = _Traced(
                    _apply_placement(st.containers, container, candidate),
                    st.trace + [(feats, j)],
                )
                children.append(child)
        if not children:
            return None
        children.sort(key=lambda t: _partial_lex_key(t.containers))
        beam = children[:beam_width]
    if not beam:
        return None
    beam.sort(key=lambda t: _final_lex_key(t.containers))
    return beam[0]


def extract_instance(
    items: List[Item],
    beam_width: int = TEACHER_BEAM_WIDTH,
    branch: int = TEACHER_BRANCH,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1 instance → (X [N,FEATURE_DIM], y [N], group_ids [N])。

    beam_search_strong と同じく目的地グループ × _ITEM_ORDERINGS で
    最良順序を選び、勝ち軌跡を pointwise BC 用に展開する。
    y=1 が beam の選んだ候補、0 がそれ以外。group_ids は (instance内)
    step 連番 = ランキング学習に切替える時のグループ単位。
    """
    groups = {dest: [] for dest in DESTINATIONS}
    for it in items:
        groups.setdefault(it.destination_id, []).append(it)

    X_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    g_rows: List[int] = []
    step_uid = 0

    for dest in DESTINATIONS:
        gitems = groups.get(dest, [])
        if not gitems:
            continue
        best: Optional[_Traced] = None
        best_key = None
        for _name, key_fn in _ITEM_ORDERINGS:
            ordered = sorted(gitems, key=key_fn)
            res = _traced_beam_for_group(
                ordered, beam_width, branch, n_total=len(items), n_offset=0
            )
            if res is None:
                continue
            k = _final_lex_key(res.containers)
            if best_key is None or k < best_key:
                best_key = k
                best = res
        if best is None:
            continue
        for feats, chosen in best.trace:
            n_cand = feats.shape[0]
            for j in range(n_cand):
                X_rows.append(feats[j])
                y_rows.append(1 if j == chosen else 0)
                g_rows.append(step_uid)
            step_uid += 1

    if not X_rows:
        return (
            np.empty((0, FEATURE_DIM)),
            np.empty((0,), dtype=int),
            np.empty((0,), dtype=int),
        )
    return (
        np.stack(X_rows),
        np.array(y_rows, dtype=int),
        np.array(g_rows, dtype=int),
    )


def build_dataset(paths) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """複数 instance → (X, y, gid)。gid は **全 instance で一意** な step 連番。

    (step0 の gid 衝突バグ修正: extract_instance の step_uid は instance
    ローカルなので連結時に running offset を足す)。
    """
    from rui.adv_lane.ga_bench import _load_items

    Xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    gs: List[np.ndarray] = []
    g_off = 0
    for p in paths:
        items = _load_items(p)
        X, y, g = extract_instance(items)
        if X.shape[0] == 0:
            continue
        Xs.append(X)
        ys.append(y)
        gs.append(g + g_off)
        g_off += int(g.max()) + 1
        print(f"  {p.name}: items={len(items)} steps={len(set(g))} "
              f"rows={X.shape[0]} pos={int(y.sum())}")
    if not Xs:
        return (np.empty((0, FEATURE_DIM)), np.empty((0,), int), np.empty((0,), int))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(gs)


def _top1_acc(proba: np.ndarray, y: np.ndarray, g: np.ndarray) -> Tuple[int, int]:
    """step (gid) ごとに proba argmax が正解候補と一致した数 / step 数。"""
    hit = tot = 0
    for gid in np.unique(g):
        m = g == gid
        if y[m].sum() == 0:
            continue
        hit += int(np.argmax(proba[m]) == int(np.argmax(y[m])))
        tot += 1
    return hit, tot


def _step0() -> None:
    """Step0: 先頭3inst で抽出規模 + sklearn fit 時間 + 模倣精度を実測。"""
    from rui.adv_lane.ga_bench import build_suite

    suite = build_suite()[:3]
    print(f"Step0 teacher extraction on {len(suite)} inst "
          f"(beam {TEACHER_BEAM_WIDTH}/{TEACHER_BRANCH})")
    t0 = time.perf_counter()
    X, y, g = build_dataset(suite)
    ext_wall = time.perf_counter() - t0
    print(f"\ntotal: rows={X.shape[0]} dim={X.shape[1]} steps={len(np.unique(g))} "
          f"pos={int(y.sum())} extract_wall={ext_wall:.1f}s "
          f"({ext_wall/len(suite):.1f}s/inst)")
    print(f"est full 16-inst extract ~{ext_wall/len(suite)*16:.0f}s")

    from sklearn.linear_model import LogisticRegression

    t0 = time.perf_counter()
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X, y)
    fit_wall = time.perf_counter() - t0
    proba = clf.predict_proba(X)[:, 1]
    hit, tot = _top1_acc(proba, y, g)
    rand = 1.0 / (X.shape[0] / max(len(np.unique(g)), 1))
    print(f"LogReg fit={fit_wall:.2f}s  train top1={hit}/{tot}={hit/max(tot,1):.3f} "
          f"(random≈{rand:.3f})")


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8")
            except Exception:
                pass
    _step0()
