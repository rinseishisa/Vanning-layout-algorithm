"""beam スケール参照解 runner (Job A)。

- ``--calibrate``: item_count ladder で beam_search_strong を実測し
  power-law (t = a * n^b) を最小二乗 fit、calibration.json に保存。
- ``--reference``: calibration と時間予算から朝までに完了する最大
  item_count を選び、destination x ordering を並列化した beam で
  大規模参照解を出す。実 vanning_eval で feasible 確認。

destination 混載は要件のハード制約 → destination グループ間は厳密に
独立。よって並列化は近似でなく exact。
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]  # .../worksp/vanning-algo
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_VEVAL_SRC = Path(r"G:\マイドライブ\ideamap\worksp\vanning-eval\vanning_eval_rui\src")
if str(_VEVAL_SRC) not in sys.path:
    sys.path.insert(0, str(_VEVAL_SRC))

from rui.adv_lane.antagonist import (  # noqa: E402
    _ITEM_ORDERINGS,
    _beam_search_for_group,
    _final_lex_key,
    beam_search_strong,
)
from rui.algorithm_a import build_items, build_output_json  # noqa: E402
from rui.adv_lane.loop import _make_dataframe  # noqa: E402
from rui.generate_items import CASE_CONFIGS, DESTINATIONS, generate_items  # noqa: E402

_HERE = Path(__file__).parent
OUT_DIR = _HERE / "runs" / "overnight_beam"
CALIB_PATH = OUT_DIR / "calibration.json"
LADDER = [100, 200, 400, 800]
PER_POINT_CAP_S = 1200.0  # 1点が20分超なら ladder 早期停止


def _make_instance(item_count: int, case_name: str = "case_balanced", seed: int = 42) -> Dict:
    base = CASE_CONFIGS[case_name]
    case = dataclasses.replace(base, item_count=item_count, name=f"{base.name}_n{item_count}")
    return generate_items(case, seed)


def _items_from_data(data: Dict):
    return build_items(_make_dataframe(data))


# ------------------------------------------------------------------
# calibration
# ------------------------------------------------------------------
def calibrate(ladder: List[int]) -> Dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    points: List[Dict] = []
    for n in ladder:
        items = _items_from_data(_make_instance(n))
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            containers, dq = beam_search_strong(items)
        dt = time.perf_counter() - t0
        rec = {"item_count": n, "seconds": round(dt, 3),
                "containers": (None if dq else len(containers)), "dq": bool(dq)}
        points.append(rec)
        print(f"[calib] n={n:5d}  t={dt:8.1f}s  containers={rec['containers']}  dq={dq}")
        if dt > PER_POINT_CAP_S:
            print(f"[calib] point exceeded {PER_POINT_CAP_S}s — stopping ladder early")
            break

    fit = None
    valid = [(p["item_count"], p["seconds"]) for p in points if p["seconds"] > 0]
    if len(valid) >= 2:
        ns = np.array([v[0] for v in valid], dtype=float)
        ts = np.array([v[1] for v in valid], dtype=float)
        b, log_a = np.polyfit(np.log(ns), np.log(ts), 1)
        fit = {"a": float(np.exp(log_a)), "b": float(b)}
        print(f"[calib] fit: t ≈ {fit['a']:.3e} * n^{fit['b']:.3f}")

    result = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "points": points, "fit": fit}
    CALIB_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[calib] wrote {CALIB_PATH}")
    return result


def _predict_seconds(fit: Dict, n: int) -> float:
    return fit["a"] * (n ** fit["b"])


def _max_feasible_n(fit: Dict, budget_s: float, hard_cap: int) -> int:
    """予算内に収まる最大 item_count (hard_cap で上限カット)。"""
    lo, hi = 50, hard_cap
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        if _predict_seconds(fit, mid) <= budget_s:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


# ------------------------------------------------------------------
# parallel reference (destination x ordering を並列化)
# ------------------------------------------------------------------
def _group_worker(payload: Tuple[str, int, list, int, int]) -> Tuple[str, int, Optional[list], Optional[tuple]]:
    dest, oidx, items, beam_width, branch = payload
    key_fn = _ITEM_ORDERINGS[oidx][1]
    ordered = sorted(items, key=key_fn)
    with contextlib.redirect_stdout(io.StringIO()):
        result = _beam_search_for_group(ordered, beam_width, branch)
    if result is None:
        return dest, oidx, None, None
    return dest, oidx, result, _final_lex_key(result)


def run_reference(item_count: int, workers: int, case_name: str, seed: int) -> Dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = _make_instance(item_count, case_name, seed)
    inst_path = OUT_DIR / f"big_instance_n{item_count}.json"
    inst_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    items = _items_from_data(data)

    groups: Dict[str, list] = {d: [] for d in DESTINATIONS}
    for it in items:
        groups.setdefault(it.destination_id, []).append(it)

    payloads = [
        (dest, oidx, groups[dest], 48, 12)
        for dest in DESTINATIONS if groups.get(dest)
        for oidx in range(len(_ITEM_ORDERINGS))
    ]
    nw = workers if workers > 0 else max(1, min(len(payloads), (os.cpu_count() or 2) - 1))
    print(f"[ref] item_count={item_count}  tasks={len(payloads)} (dest x ordering)  workers={nw}")

    t0 = time.perf_counter()
    best_per_dest: Dict[str, Tuple[tuple, list]] = {}
    dq = False
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for dest, oidx, result, key in ex.map(_group_worker, payloads):
            if result is None:
                continue
            cur = best_per_dest.get(dest)
            if cur is None or key < cur[0]:
                best_per_dest[dest] = (key, result)
            print(f"  dest={dest} ord={_ITEM_ORDERINGS[oidx][0]}: "
                  f"{'fail' if result is None else f'{len(result)} containers key={key}'}")

    all_containers: list = []
    for dest in DESTINATIONS:
        if groups.get(dest) and dest not in best_per_dest:
            dq = True  # この dest をどの順序でも feasible にできず
            continue
        if dest in best_per_dest:
            all_containers.extend(best_per_dest[dest][1])
    for i, c in enumerate(all_containers, start=1):
        c.container_id = i
    elapsed = time.perf_counter() - t0

    layout = build_output_json(all_containers, "beam_reference", int(elapsed * 1000))
    layout_path = OUT_DIR / f"layout_result_n{item_count}.json"
    layout_path.write_text(json.dumps(layout, indent=2, ensure_ascii=False), encoding="utf-8")

    # 実 vanning_eval で feasible 確認
    from vanning_eval.report import build_report
    from vanning_eval.schema import load_items, load_layout

    report = build_report(load_layout(layout_path), load_items(inst_path))
    verdict = report["verdict"]
    n_containers = report["teacher_score_metrics"]["containers_used"]

    timing = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "item_count": item_count,
        "elapsed_seconds": round(elapsed, 1),
        "containers": n_containers,
        "antagonist_dq": dq,
        "vanning_eval_verdict": verdict,
        "n_violations": len(report["disqualifications"]),
        "instance": str(inst_path),
        "layout": str(layout_path),
    }
    (OUT_DIR / f"timing_n{item_count}.json").write_text(
        json.dumps(timing, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[ref] DONE in {elapsed:.1f}s  containers={n_containers}  "
          f"verdict={verdict}  dq={dq}  violations={timing['n_violations']}")
    print(f"[ref] layout={layout_path}")
    return timing


def main() -> None:
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8")
            except Exception:
                pass
    ap = argparse.ArgumentParser(description="beam スケール参照解 runner")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--ladder", type=int, nargs="+", default=LADDER)
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--budget-seconds", type=float, default=None,
                    help="reference: この秒数内に収まる最大 item_count を自動選択")
    ap.add_argument("--item-count", type=int, default=None,
                    help="reference: item_count を明示指定 (budget より優先)")
    ap.add_argument("--target-containers", type=int, default=100,
                    help="hard cap 用の目標コンテナ数 (item_count 上限算出)")
    ap.add_argument("--case", default="case_balanced")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    if args.calibrate:
        calibrate(args.ladder)
        return

    if args.reference:
        if args.item_count is not None:
            n = args.item_count
        else:
            if not CALIB_PATH.exists():
                raise SystemExit("calibration.json がない。先に --calibrate するか --item-count 指定")
            calib = json.loads(CALIB_PATH.read_text(encoding="utf-8"))
            fit = calib.get("fit")
            if fit is None:
                raise SystemExit("calibration に fit がない (有効点 < 2)")
            from rui.adv_lane.make_big_instance import build_big
            hard_cap = build_big(args.target_containers, args.case, args.seed)["dataset_info"]["item_count"]
            budget = args.budget_seconds if args.budget_seconds else 6 * 3600
            n = _max_feasible_n(fit, budget, hard_cap)
            print(f"[ref] budget={budget:.0f}s  hard_cap(n@{args.target_containers}c)={hard_cap}  "
                  f"-> chosen item_count={n}  (predicted {_predict_seconds(fit, n):.0f}s)")
        run_reference(n, args.workers, args.case, args.seed)
        return

    ap.error("--calibrate か --reference のいずれかを指定")


if __name__ == "__main__":
    main()
