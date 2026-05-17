"""レーン3 R1: beam 教師から配置スコアラを訓練 (sklearn pointwise BC)。

R0 の固定枝刈りキー ``_partial_lex_key`` を学習スコアラで置換するための
モデルを作る。教師は ``r1_teacher.extract_instance`` (beam 勝ち軌跡)。

**リーク回避**: ga_bench の dN 比較は先頭6inst (hard_01..06)。よって
訓練は ``--holdout 6`` = 後半10inst (hard_07..12 + 標準4) のみで行い、
評価インスタンスを訓練に混ぜない (honest dN)。

モデルは StandardScaler + LogisticRegression の Pipeline (LogReg は
スケール依存)。joblib 永続化 = sklearn なので Windows pickle 罠
([[feedback_windows_pytorch_pickle_pitfalls]] は torch 固有) を踏まない。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rui.adv_lane.r1_teacher import FEATURE_NAMES, _top1_acc, build_dataset  # noqa: E402

_HERE = Path(__file__).parent
MODEL_DIR = _HERE / "runs" / "r1"
MODEL_PATH = MODEL_DIR / "scorer.joblib"


def main() -> None:
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8")
            except Exception:
                pass
    ap = argparse.ArgumentParser(description="R1 配置スコアラ訓練 (pointwise BC)")
    ap.add_argument("--ho-start", type=int, default=0,
                    help="hold-out 窓開始 idx (ga_bench --offset と一致させる)")
    ap.add_argument("--ho-len", type=int, default=6,
                    help="hold-out 窓長 (ga_bench --limit と一致させる)")
    ap.add_argument("--model", choices=["logreg", "mlp"], default="logreg")
    args = ap.parse_args()

    from rui.adv_lane.ga_bench import build_suite

    suite = build_suite()
    s, e = args.ho_start, args.ho_start + args.ho_len
    holdout = suite[s:e]
    train_paths = suite[:s] + suite[e:]  # 窓の外側 = 訓練 (リーク無)
    print(f"train on {len(train_paths)} inst "
          f"(hold-out suite[{s}:{e}] = {[p.name for p in holdout]})")

    t0 = time.perf_counter()
    X, y, g = build_dataset(train_paths)
    ext_wall = time.perf_counter() - t0
    print(f"\ndataset: rows={X.shape[0]} dim={X.shape[1]} "
          f"steps={len(np.unique(g))} pos={int(y.sum())} "
          f"extract_wall={ext_wall:.1f}s")
    if X.shape[0] == 0:
        raise SystemExit("empty training set")

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if args.model == "logreg":
        from sklearn.linear_model import LogisticRegression
        est = LogisticRegression(max_iter=2000, class_weight="balanced")
    else:
        from sklearn.neural_network import MLPClassifier
        est = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=400)

    # group (step) 単位 holdout で過学習を確認 (step 内候補は相関するため
    # 行単位 split はリーク)。後半 20% の step を内部検証に。
    uniq = np.unique(g)
    n_val = max(1, len(uniq) // 5)
    val_gids = set(uniq[-n_val:].tolist())
    val_mask = np.array([gid in val_gids for gid in g])

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", est)])
    t0 = time.perf_counter()
    pipe.fit(X[~val_mask], y[~val_mask])
    fit_wall = time.perf_counter() - t0

    for tag, mask in (("train", ~val_mask), ("val", val_mask)):
        proba = pipe.predict_proba(X[mask])[:, 1]
        hit, tot = _top1_acc(proba, y[mask], g[mask])
        rand = 1.0 / (mask.sum() / max(len(np.unique(g[mask])), 1))
        print(f"  {tag}: top1={hit}/{tot}={hit/max(tot,1):.3f} "
              f"(random≈{rand:.3f})")

    # 本番モデルは全 train で再学習 (内部 val 分は捨てない)
    pipe.fit(X, y)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(pipe, MODEL_PATH)
    print(f"\nfit={fit_wall:.2f}s  saved -> {MODEL_PATH}")

    if args.model == "logreg":
        coef = pipe.named_steps["clf"].coef_[0]
        order = np.argsort(-np.abs(coef))
        print("LogReg 係数 (|w| 降順, scaled):")
        for i in order:
            print(f"  {FEATURE_NAMES[i]:>14s}: {coef[i]:+.3f}")


if __name__ == "__main__":
    main()
