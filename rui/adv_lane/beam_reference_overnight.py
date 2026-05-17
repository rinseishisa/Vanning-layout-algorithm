"""Job A 夜間ドライバ: deadline 認識の laddered 並列 beam 参照解。

serial beam は n^2.4 で爆発するので、destination x ordering を並列化した
``run_reference`` を **小→大** のサイズ列で回す。各サイズ完了ごとに
layout_result_n*.json / timing_n*.json が残るので、途中で時間切れでも
「到達できた最大スケールの参照解」が必ず手元に残る (要件: 正直に報告)。

deadline (既定 当日 06:30 JST) を超える/次サイズが収まらないと予測したら
打ち切る。観測 wall から指数 2.4 で次サイズを外挿。
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from rui.adv_lane.beam_reference_scale import run_reference

LADDER = [400, 700, 1100, 1582]  # 1582 ≈ case_balanced 100 コンテナ相当
EXPONENT = 2.4  # calibration 実測 (n=100->42s, n=200->224s)


def _deadline(hhmm: str) -> datetime:
    now = datetime.now()
    h, m = (int(x) for x in hhmm.split(":"))
    d = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if d <= now:  # 既に過ぎていれば翌日… ではなく即終了させたいので now+margin
        d = now + timedelta(minutes=10)
    return d


def main() -> None:
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8")
            except Exception:
                pass
    ap = argparse.ArgumentParser(description="Job A 夜間 laddered 並列 beam 参照解")
    ap.add_argument("--ladder", type=int, nargs="+", default=LADDER)
    ap.add_argument("--workers", type=int, default=10, help="Job B の bench bursts に余地を残す")
    ap.add_argument("--deadline", default="06:30", help="JST HH:MM、これ以降は新規サイズを開始しない")
    ap.add_argument("--margin-min", type=int, default=20, help="deadline 手前マージン(分)")
    ap.add_argument("--case", default="case_balanced")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    deadline = _deadline(args.deadline)
    print(f"[overnight] deadline={deadline:%Y-%m-%d %H:%M}  margin={args.margin_min}m  "
          f"ladder={args.ladder}  workers={args.workers}", flush=True)

    last_n = None
    last_wall = None
    for n in args.ladder:
        now = datetime.now()
        cutoff = deadline - timedelta(minutes=args.margin_min)
        if now >= cutoff:
            print(f"[overnight] {now:%H:%M} >= cutoff {cutoff:%H:%M} — stop (largest done: n={last_n})", flush=True)
            break
        if last_n is not None and last_wall is not None:
            predicted = last_wall * (n / last_n) ** EXPONENT
            remaining = (cutoff - now).total_seconds()
            print(f"[overnight] next n={n}: predicted {predicted:.0f}s, remaining {remaining:.0f}s", flush=True)
            if predicted > remaining:
                print(f"[overnight] predicted exceeds remaining — stop (largest done: n={last_n})", flush=True)
                break
        print(f"[overnight] === start n={n} @ {now:%H:%M} ===", flush=True)
        t0 = time.perf_counter()
        try:
            timing = run_reference(n, args.workers, args.case, args.seed)
        except Exception as exc:  # 1 サイズの事故で全体を落とさない
            print(f"[overnight] n={n} FAILED: {exc!r} — stop", flush=True)
            break
        last_wall = time.perf_counter() - t0
        last_n = n
        print(f"[overnight] === done n={n}  wall={last_wall:.0f}s  "
              f"containers={timing['containers']}  verdict={timing['vanning_eval_verdict']} ===", flush=True)

    print(f"[overnight] FINISHED. largest reference: n={last_n}", flush=True)


if __name__ == "__main__":
    main()
