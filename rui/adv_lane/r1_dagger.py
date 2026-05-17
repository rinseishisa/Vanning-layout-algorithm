"""レーン3 R2: DAgger-lite で学習デコーダの ceiling を押し上げる。

R1-MLP は offline BC で floor (greedy/beam 超え dN0.833) を立てた。
残る主因は **分布シフト**: 教師は beam 軌跡状態だが推論は学習器自身の
greedy 軌跡状態 (off-distribution)。DAgger = 学習器が実際に訪れた状態で
oracle ラベルを取り直し base BC データに集約して再訓練、で直撃する。

oracle (lite): 訪問状態から **bounded-horizon beam 先読み** (H item 先・
幅 W、partial lex キーで勝ち枝の第1手を採用)。完全 beam-completion は
コスト過大なので horizon 切り。lookahead 信号注入で機構(b)も改善。

Step-0 規律: --probe で oracle 1コール時間×ラベル状態数×inst を実測し
budget 内を確認してから本走 (beam コスト見積りを甘くして timeout を
踏んだ R0 の反省)。base dataset は npz キャッシュ (再抽出167s回避)。
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

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
from rui.adv_lane.r1_teacher import (  # noqa: E402
    FEATURE_DIM,
    TEACHER_BRANCH,
    _Traced,
    build_dataset,
    featurize,
)
from rui.adv_lane.r1_train import MODEL_DIR, MODEL_PATH  # noqa: E402
from rui.generate_items import DESTINATIONS  # noqa: E402

_HERE = Path(__file__).parent
_CACHE_DIR = _HERE / "runs" / "r1"

# DAgger-lite 既定。probe で実測してから本走。
DEF_HORIZON = 8       # oracle 先読み item 数
DEF_WIDTH = 12        # oracle beam 幅
DEF_MAX_LABEL = 30    # 1 inst で oracle ラベルする状態数の上限 (sparse)


def _base_cache_path(s: int, e: int) -> Path:
    return _CACHE_DIR / f"base_ho{s}_{e}.npz"


def _get_base_dataset(suite, s: int, e: int):
    """base BC データ (window 外で訓練) を npz キャッシュ付きで取得。"""
    cp = _base_cache_path(s, e)
    if cp.exists():
        d = np.load(cp)
        print(f"base dataset: cache hit {cp.name} rows={d['X'].shape[0]}")
        return d["X"], d["y"], d["g"]
    train_paths = suite[:s] + suite[e:]
    print(f"base dataset: extracting (train {len(train_paths)} inst, no cache)")
    X, y, g = build_dataset(train_paths)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cp, X=X, y=y, g=g)
    return X, y, g


def _score_candidates(model, feats: np.ndarray) -> np.ndarray:
    return model.predict_proba(feats)[:, 1]


def _oracle_label(
    state: List,
    item,
    remaining: List,
    width: int,
    horizon: int,
    branch: int,
    n_done: int,
    n_total: int,
    strong: bool = False,
) -> Optional[Tuple[np.ndarray, int]]:
    """訪問状態 *state* + *item* で oracle が選ぶ候補。

    lite (strong=False): [item]+remaining[:horizon-1] を幅 *width* beam で
      先読み、勝ち枝を **partial** lex で選ぶ (短視・安価)。R2 で floor を
      押し下げた弱 oracle。
    strong (strong=True): [item]+**全 remaining** を完走させ、勝ち枝を
      **final** lex (完成解品質) で選ぶ = R1-MLP が模倣した強リファレンス
      (beam_search_strong) と同強度。DAgger の「oracle=強expert」前提を
      満たす正版。horizon は無視。
    どちらも勝ち枝の **第1手** の (候補 feat 行列, 採用添字) を返す。
    """
    if strong:
        seq = [item] + list(remaining)
        sel = _final_lex_key
    else:
        seq = [item] + list(remaining[: max(0, horizon - 1)])
        sel = _partial_lex_key
    beam: List[_Traced] = [_Traced([copy.deepcopy(c) for c in state], [])]
    for step, it in enumerate(seq):
        children: List[_Traced] = []
        for st in beam:
            placements = _top_k_placements(st.containers, it, branch)
            if not placements:
                continue
            feats = np.stack(
                [
                    featurize(st.containers, c, cand, it, n_done + step, n_total)
                    for (c, cand) in placements
                ]
            )
            for j, (cont, cand) in enumerate(placements):
                children.append(
                    _Traced(_apply_placement(st.containers, cont, cand),
                            st.trace + [(feats, j)])
                )
        if not children:
            if step == 0:
                return None
            break
        children.sort(key=lambda t: _partial_lex_key(t.containers))
        beam = children[:width]
    winner = min(beam, key=lambda t: sel(t.containers))
    return winner.trace[0]  # (feats_at_state, oracle_chosen_idx)


def _rollout_and_label(
    items_by_dest, model, width, horizon, branch, max_label, n_total,
    probe_times: Optional[list] = None, strong: bool = False,
):
    """学習器(model) greedy ロールアウト → sparse に oracle ラベル収集。

    1 inst の dest 群を順に。各 dest で weight_desc 1 順序のみ
    (学習器の状態分布を晒すのが目的、順序網羅は不要)。
    """
    rows_X: List[np.ndarray] = []
    rows_y: List[int] = []
    key_fn = _ITEM_ORDERINGS[0][1]  # weight_desc
    n_done = 0
    for dest in DESTINATIONS:
        group = items_by_dest.get(dest, [])
        if not group:
            continue
        ordered = sorted(group, key=key_fn)
        state: List = []
        # この dest でラベルする step を等間隔 sparse 抽出
        n_steps = len(ordered)
        if n_steps == 0:
            continue
        stride = max(1, n_steps // max(1, max_label // max(1, len(DESTINATIONS))))
        for t, it in enumerate(ordered):
            placements = _top_k_placements(state, it, branch)
            if not placements:
                break
            feats = np.stack(
                [featurize(state, c, cand, it, n_done, n_total)
                 for (c, cand) in placements]
            )
            # 学習器の手
            proba = _score_candidates(model, feats)
            lj = int(np.argmax(proba))
            # sparse に oracle ラベル (学習器が訪れた *この* 状態で)
            if t % stride == 0 and len(placements) > 1:
                t0 = time.perf_counter()
                lab = _oracle_label(state, it, ordered[t + 1:], width,
                                    horizon, branch, n_done, n_total,
                                    strong=strong)
                if probe_times is not None:
                    probe_times.append(time.perf_counter() - t0)
                if lab is not None:
                    of, oj = lab
                    for j in range(of.shape[0]):
                        rows_X.append(of[j])
                        rows_y.append(1 if j == oj else 0)
            # 学習器の手で前進 (DAgger: 学習器の状態分布を辿る)
            cont, cand = placements[lj]
            state = _apply_placement(state, cont, cand)
            n_done += 1
    if not rows_X:
        return np.empty((0, FEATURE_DIM)), np.empty((0,), int)
    return np.stack(rows_X), np.array(rows_y, int)


def _load_model():
    import joblib

    if not MODEL_PATH.exists():
        raise SystemExit(f"no base model at {MODEL_PATH}; run r1_train --model mlp first")
    return joblib.load(MODEL_PATH)


def main() -> None:
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8")
            except Exception:
                pass
    ap = argparse.ArgumentParser(description="R2 DAgger-lite (ceiling 押し上げ)")
    ap.add_argument("--ho-start", type=int, default=0)
    ap.add_argument("--ho-len", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=DEF_HORIZON)
    ap.add_argument("--width", type=int, default=DEF_WIDTH)
    ap.add_argument("--branch", type=int, default=TEACHER_BRANCH)
    ap.add_argument("--max-label", type=int, default=DEF_MAX_LABEL)
    ap.add_argument("--strong", action="store_true",
                    help="oracle を full beam-completion + final-lex 化 "
                         "(R2 lite が floor 押下げた弱 oracle の正版, 重い)")
    ap.add_argument("--probe", action="store_true",
                    help="Step0: 1 inst で oracle コスト実測し全体見積りして終了")
    args = ap.parse_args()

    from rui.adv_lane.ga_bench import _load_items, build_suite

    suite = build_suite()
    s, e = args.ho_start, args.ho_start + args.ho_len
    train_paths = suite[:s] + suite[e:]
    print(f"DAgger train pool {len(train_paths)} inst (hold-out suite[{s}:{e}])")
    model = _load_model()

    def _by_dest(items):
        d = {dest: [] for dest in DESTINATIONS}
        for it in items:
            d.setdefault(it.destination_id, []).append(it)
        return d

    if args.probe:
        p = train_paths[0]
        items = _load_items(p)
        probe_times: list = []
        t0 = time.perf_counter()
        X, y = _rollout_and_label(
            _by_dest(items), model, args.width, args.horizon, args.branch,
            args.max_label, len(items), probe_times=probe_times,
            strong=args.strong,
        )
        wall = time.perf_counter() - t0
        avg_oracle = (sum(probe_times) / len(probe_times)) if probe_times else 0.0
        print(f"[probe {p.name}] items={len(items)} labeled_states={len(probe_times)} "
              f"rows={X.shape[0]} wall={wall:.1f}s avg_oracle={avg_oracle*1000:.0f}ms")
        est = wall * len(train_paths)
        print(f"est full DAgger collect ({len(train_paths)} inst) ~{est:.0f}s "
              f"(+ base extract/cache + retrain ~数s)")
        return

    # 本走: base (cache) + DAgger 集約 → MLP 再訓練
    t0 = time.perf_counter()
    Xb, yb, gb = _get_base_dataset(suite, s, e)
    Xd_list, yd_list = [], []
    for p in train_paths:
        items = _load_items(p)
        Xd, yd = _rollout_and_label(
            _by_dest(items), model, args.width, args.horizon, args.branch,
            args.max_label, len(items), strong=args.strong,
        )
        if Xd.shape[0]:
            Xd_list.append(Xd)
            yd_list.append(yd)
        print(f"  {p.name}: dagger_rows={Xd.shape[0]} pos={int(yd.sum()) if Xd.shape[0] else 0}")
    collect_wall = time.perf_counter() - t0

    Xd = np.vstack(Xd_list) if Xd_list else np.empty((0, FEATURE_DIM))
    yd = np.concatenate(yd_list) if yd_list else np.empty((0,), int)
    X = np.vstack([Xb, Xd])
    y = np.concatenate([yb, yd])
    print(f"\naggregate: base={Xb.shape[0]} + dagger={Xd.shape[0]} = {X.shape[0]} rows "
          f"(collect_wall={collect_wall:.1f}s)")

    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=400)),
    ])
    t0 = time.perf_counter()
    pipe.fit(X, y)
    print(f"retrain MLP fit={time.perf_counter()-t0:.2f}s")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(pipe, MODEL_PATH)
    print(f"saved DAgger model -> {MODEL_PATH}")
    print("次: ga_bench --decoder learned --offset {} --limit {} で評価".format(s, args.ho_len))


if __name__ == "__main__":
    main()
