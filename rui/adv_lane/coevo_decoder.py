"""レーン3 (coevo) R0: pack_items の beam 幅 k デコード。

2026-05-16 夜間実証の結論 (OVERNIGHT_GA_REPORT §重要な知見):

    dN=1.0 ギャップは「緩いコンテナ」でも「順列の質」でもなく、
    ``pack_items`` の貪欲単経路デコード自体が構造的に持つ差。

現行 ``algorithm_a.pack_items`` は各 item を「最初に置けた既存コンテナ」へ
即 break する単経路デコード (コンテナ比較すらしない)。本モジュールは
**GA が選んだ順列をそのまま** beam 幅 k で分岐デコードする。

設計の肝: antagonist.py に既に pytest + 実 vanning_eval で検証済みの
beam 機構 (``_beam_search_for_group`` / ``_top_k_placements`` /
``_apply_placement``) があるので **再実装せず再利用**する。antagonist の
``beam_search_strong`` は目的地で再グループ化し ``_ITEM_ORDERINGS`` で
順序を作り直す (= GA 順列を捨てる) 強リファレンス用。R0 で欲しいのは
逆に「GA 順列を保ったまま decode だけ分岐」なので、再グループ化も
順序再生成もしない ``_beam_search_for_group`` を順列に直接かける。

ML は一切無い。R0 = 「分岐だけで dN は縮むか」をゼロリスクで測る段。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from rui.algorithm_a import Container, Item, pack_items
from rui.adv_lane.antagonist import _apply_placement, _beam_search_for_group, _top_k_placements

# 「上位 k 候補コンテナの浅い beam」(OVERNIGHT_GA_REPORT §残課題 L66)。
# branch = 各 item で保持する候補配置数 (≒上位候補コンテナ数)。
# beam_width = 保持する部分状態数 (浅さ)。greedy = (1, 1) に縮退する。
DEFAULT_BEAM_WIDTH = 4
DEFAULT_BRANCH = 2


def pack_items_beam(
    items: Sequence[Item],
    beam_width: int = DEFAULT_BEAM_WIDTH,
    branch: int = DEFAULT_BRANCH,
) -> List[Container]:
    """GA 順列 *items* を beam 幅 k で分岐デコードする。

    ``_beam_search_for_group`` は flat な item 列を投入順固定で beam 探索し、
    ``_top_k_placements`` が目的地一致フィルタと「既存に置けないときだけ
    新規コンテナを 1 個開く」を内部処理する。よって目的地混在の GA 順列を
    そのまま渡してよい (再グループ化しない = 順列情報を捨てない)。

    beam が死んだ (None) 場合は貪欲 ``pack_items`` にフォールバックする。
    3 種の積荷はいずれも空コンテナへ必ず 1 個置けるので feasible な
    item 集合では beam は死なない想定だが、評価ループを絶対に
    クラッシュさせない防御として残す。
    """
    result = _beam_search_for_group(list(items), beam_width, branch)
    if result is None:
        return pack_items(items)
    return result


# ---------------------------------------------------------------------------
# R1: 学習スコアラによる greedy デコード (推論 beam 不要 = R0 の ~6x を解消)
# ---------------------------------------------------------------------------
# joblib モデルはプロセス内キャッシュ。run_ga は decode を数百回呼ぶため
# 毎回 load すると致命的に遅い。パス毎に 1 度だけロードする。
_MODEL_CACHE: dict = {}
_DEFAULT_MODEL_PATH = Path(__file__).parent / "runs" / "r1" / "scorer.joblib"
# 推論時の候補数。greedy 単経路ゆえ beam メモリ無し → 教師の branch(6)
# より広く取りモデルに選ばせる。
INFER_K = 12


def _load_model(path: Path):
    key = str(path)
    if key not in _MODEL_CACHE:
        import joblib

        _MODEL_CACHE[key] = joblib.load(path) if path.exists() else None
    return _MODEL_CACHE[key]


def pack_items_learned(
    items: Sequence[Item],
    model_path: Optional[Path] = None,
    k: int = INFER_K,
) -> List[Container]:
    """GA 順列を **学習スコアラで greedy デコード**する。

    各 item で ``_top_k_placements`` が出す候補を ``r1_teacher.featurize``
    で特徴化し、学習モデルの positive 確率 argmax を選んで確定配置する
    (beam 探索しない = 推論は greedy 並み高速)。モデル不在/異常時は
    貪欲 ``pack_items`` にフォールバックし bench を絶対に落とさない。
    """
    from rui.adv_lane.r1_teacher import featurize  # 循環回避の遅延 import

    model = _load_model(model_path or _DEFAULT_MODEL_PATH)
    if model is None:
        return pack_items(items)

    import numpy as np

    items = list(items)
    n_total = len(items)
    state: List[Container] = []
    for step, item in enumerate(items):
        placements = _top_k_placements(state, item, k)
        if not placements:
            # 既存にも空にも置けない = 真に infeasible。貪欲に委譲
            return pack_items(items)
        feats = np.stack(
            [featurize(state, c, cand, item, step, n_total) for (c, cand) in placements]
        )
        try:
            proba = model.predict_proba(feats)[:, 1]
        except Exception:
            return pack_items(items)
        best = int(np.argmax(proba))
        container, candidate = placements[best]
        state = _apply_placement(state, container, candidate)
    return state
