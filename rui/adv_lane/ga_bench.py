"""GA ベンチハーネス (overnight 自動改善の客観指標)。

固定スイート (curated hard 31cat + 標準 case) で ``run_ga`` を回し、
**実バリデータ** ``vanning_eval`` を import して合否・コンテナ数・重心ズレを
算出する (中間経路でなく下流の本物を叩く = 罠6 教訓)。

dN = GA コンテナ数 - beam 参照コンテナ数。beam 参照は重いので
``--build-beam-ref`` で 1 度だけ算出してキャッシュし、ループ中は再計算しない。

verdict (baseline.json 比):
  IMPROVED  : 平均コンテナ数が減 かつ 失格不増 かつ wall <= 3x baseline
  REGRESSED : 平均コンテナ数が増 or 失格増 or wall > 3x baseline
  NOCHANGE  : それ以外

OpenCode はこの数値で証明できない限り「成功」と報告してはならない。
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# cwd 非依存化: vanning-algo リポジトリ root と vanning_eval src を sys.path へ。
# (OpenCode は cwd=vault root から起動されるため child spawn でも import 可能に)
_REPO_ROOT = Path(__file__).resolve().parents[2]  # .../worksp/vanning-algo
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_VEVAL_SRC = Path(
    r"G:\マイドライブ\ideamap\worksp\vanning-eval\vanning_eval_rui\src"
)
if str(_VEVAL_SRC) not in sys.path:
    sys.path.insert(0, str(_VEVAL_SRC))

from rui.adv_lane.antagonist import beam_search_strong  # noqa: E402
from rui.adv_lane.coevo_decoder import (  # noqa: E402
    pack_items_beam,
    pack_items_learned,
    pack_items_portfolio,
)
from rui.adv_lane.loop import _make_dataframe  # noqa: E402  (tolerant loader)
from rui.algorithm_a import build_items, build_output_json, pack_items, run_ga  # noqa: E402

# R0/R1: GA の順列→コンテナ列デコーダを差し替え可能化。
# greedy  = 既存 baseline とビット同一
# beam    = R0 pack_items_beam (固定枝刈り浅い分岐, ~6x遅)
# learned = R1 学習スコアラ greedy デコード (推論 beam 不要 = 高速)
_DECODERS = {
    "greedy": pack_items,
    "beam": pack_items_beam,
    "learned": pack_items_learned,
    "portfolio": pack_items_portfolio,
}

_HERE = Path(__file__).parent
RUNS_DIR = _HERE / "runs" / "overnight_ga"
BEAM_REF_PATH = RUNS_DIR / "beam_ref.json"
BASELINE_PATH = RUNS_DIR / "baseline.json"
LAST_PATH = RUNS_DIR / "last_bench.json"
PROGRESS_LOG = RUNS_DIR / "progress.log"
TMP_DIR = RUNS_DIR / "tmp"

# ベンチ固定 GA 設定 (baseline と改善後で同一 = 公平比較)。
# OpenCode はこの値ではなく run_ga 内部の種付け/局所探索/演算子を改善する。
BENCH_GA_GEN = 20
BENCH_GA_POP = 14
BENCH_SEED = 1234
# 絶対時間天井。夜間は Job A(beam) と CPU 競合し bench wall がノイジーに
# なるため、ノイジーな baseline 比 (相対) でなく固定上限で「GA が
# 病的に遅くなっていないか」だけを守る。GA は数十〜数百秒、beam は時間
# 単位なので 1500s でも桁違いに高速。改善の真否は mean_containers が主。
ABS_TIME_CEIL_S = 1500.0

# スイート: honban_cat31 curated hard を先頭 N + 標準 dataset
N_HARD = 12
_HARD_DIR = _HERE / "hard_instances" / "honban_cat31"
_DATASET_DIR = _HERE.parent / "datasets"
_STD_DATASETS = [
    "case_balanced_seed42.json",
    "case_weight_bound_seed42.json",
    "case_volume_bound_seed42.json",
    "case_small_many_seed42.json",
]


def build_suite() -> List[Path]:
    """評価対象インスタンスのパス一覧 (決定的順序)。"""
    suite: List[Path] = []
    hard = sorted(_HARD_DIR.glob("hard_*_g15_p12.json"))[:N_HARD]
    suite.extend(hard)
    for name in _STD_DATASETS:
        p = _DATASET_DIR / name
        if p.exists():
            suite.append(p)
    return suite


def _load_items(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    df = _make_dataframe(data)
    return build_items(df)


# ------------------------------------------------------------------
# beam 参照キャッシュ構築 (Phase 0 で 1 回、重い)
# ------------------------------------------------------------------
def _beam_ref_worker(path_str: str) -> Tuple[str, Optional[int], bool]:
    path = Path(path_str)
    items = _load_items(path)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            containers, dq = beam_search_strong(items)
    except Exception:
        return path.name, None, True
    return path.name, (None if dq else len(containers)), bool(dq)


def build_beam_ref(workers: int) -> Dict[str, Dict]:
    suite = build_suite()
    nw = workers if workers > 0 else max(1, min(len(suite), (os.cpu_count() or 2) - 1))
    print(f"[beam-ref] {len(suite)} instances, workers={nw} (this is slow, ~once)")
    ref: Dict[str, Dict] = {}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for name, n, dq in ex.map(_beam_ref_worker, [str(p) for p in suite]):
            ref[name] = {"N": n, "dq": dq}
            print(f"  {name}: beam_N={n} dq={dq}")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    BEAM_REF_PATH.write_text(json.dumps(ref, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[beam-ref] wrote {BEAM_REF_PATH}  ({time.perf_counter() - t0:.1f}s)")
    return ref


# ------------------------------------------------------------------
# GA 1 インスタンス評価 (実 vanning_eval で検証)
# ------------------------------------------------------------------
def _ga_worker(payload: Tuple[str, int, str]) -> Dict:
    path_str, idx, decoder_name = payload
    path = Path(path_str)
    decoder = _DECODERS[decoder_name]
    random.seed(BENCH_SEED + idx)
    t0 = time.perf_counter()
    items = _load_items(path)
    with contextlib.redirect_stdout(io.StringIO()):
        containers, _eval = run_ga(
            items, generations=BENCH_GA_GEN, pop_size=BENCH_GA_POP, decoder=decoder
        )
        layout = build_output_json(containers, "overnight_ga", int((time.perf_counter() - t0) * 1000))
    wall = time.perf_counter() - t0

    # --- 実バリデータ (vanning_eval) で検証: 中間経路でなく本物を叩く ---
    from vanning_eval.report import build_report
    from vanning_eval.schema import load_items, load_layout

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp_layout = TMP_DIR / f"layout_{idx}.json"
    tmp_layout.write_text(json.dumps(layout, ensure_ascii=False), encoding="utf-8")
    lr = load_layout(tmp_layout)
    ii = load_items(path)
    report = build_report(lr, ii)

    verdict = report["verdict"]
    teacher = report["teacher_score_metrics"]
    n_real = int(teacher["containers_used"])
    devs = teacher.get("cog_dev_per_container") or [0.0]
    mean_dev = sum(devs) / len(devs) if devs else 0.0
    return {
        "instance": path.name,
        "ga_containers": n_real,
        "verdict": verdict,
        "disqualified": verdict != "pass",
        "n_violations": len(report["disqualifications"]),
        "mean_cog_dev": round(mean_dev, 1),
        "wall_s": round(wall, 2),
    }


def run_bench(
    workers: int, decoder_name: str = "greedy", limit: int = 0, offset: int = 0
) -> Dict:
    suite = build_suite()
    # suite[offset : offset+limit] (limit=0 は offset 以降全部)。
    # 既定 offset=0/limit=0 = 全16, contract 不変。fold 検証用に窓指定可。
    suite = suite[offset:]
    if limit > 0:
        suite = suite[:limit]
    if not suite:
        raise SystemExit("empty suite: hard_instances/honban_cat31 or datasets missing")
    nw = workers if workers > 0 else max(1, min(len(suite), (os.cpu_count() or 2) - 1))
    beam_ref: Dict[str, Dict] = {}
    if BEAM_REF_PATH.exists():
        beam_ref = json.loads(BEAM_REF_PATH.read_text(encoding="utf-8"))

    t0 = time.perf_counter()
    payloads = [(str(p), i, decoder_name) for i, p in enumerate(suite)]
    with ProcessPoolExecutor(max_workers=nw) as ex:
        rows = list(ex.map(_ga_worker, payloads))
    total_wall = time.perf_counter() - t0

    for r in rows:
        ref = beam_ref.get(r["instance"])
        if ref and ref.get("N") is not None:
            r["beam_N"] = ref["N"]
            r["dN"] = r["ga_containers"] - ref["N"]
        else:
            r["beam_N"] = None
            r["dN"] = None

    n = len(rows)
    mean_containers = sum(r["ga_containers"] for r in rows) / n
    dns = [r["dN"] for r in rows if r["dN"] is not None]
    mean_dN = (sum(dns) / len(dns)) if dns else None
    total_disq = sum(1 for r in rows if r["disqualified"])
    return {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "decoder": decoder_name,
        "ga_gen": BENCH_GA_GEN,
        "ga_pop": BENCH_GA_POP,
        "instances": n,
        "mean_containers": round(mean_containers, 3),
        "mean_dN": None if mean_dN is None else round(mean_dN, 3),
        "total_disqualified": total_disq,
        "total_wall_s": round(total_wall, 2),
        "rows": rows,
    }


def _verdict(cur: Dict, base: Optional[Dict]) -> str:
    if base is None:
        return "BASELINE"
    eps = 1e-9
    worse_n = cur["mean_containers"] > base["mean_containers"] + eps
    better_n = cur["mean_containers"] < base["mean_containers"] - eps
    more_disq = cur["total_disqualified"] > base["total_disqualified"]
    over_time = cur["total_wall_s"] > ABS_TIME_CEIL_S
    if worse_n or more_disq or over_time:
        return "REGRESSED"
    if better_n and not more_disq and not over_time:
        return "IMPROVED"
    return "NOCHANGE"


def main() -> None:
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8")
            except Exception:
                pass
    ap = argparse.ArgumentParser(description="GA ベンチ (実 vanning_eval 検証)")
    ap.add_argument("--mode", choices=["bench", "baseline"], default="bench")
    ap.add_argument("--build-beam-ref", action="store_true", help="beam 参照キャッシュを構築して終了 (Phase 0, 1回)")
    ap.add_argument("--workers", type=int, default=0, help="0=自動")
    ap.add_argument(
        "--decoder",
        choices=sorted(_DECODERS),
        default="greedy",
        help="GA 順列デコーダ。greedy=既存baseline同一 / beam=R0 浅い分岐",
    )
    ap.add_argument("--limit", type=int, default=0, help="N inst のみ (0=offset以降全部)")
    ap.add_argument("--offset", type=int, default=0, help="suite[offset:] から開始 (fold 検証用)")
    args = ap.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    if args.build_beam_ref:
        build_beam_ref(args.workers)
        return

    cur = run_bench(args.workers, args.decoder, args.limit, args.offset)
    LAST_PATH.write_text(json.dumps(cur, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.mode == "baseline":
        BASELINE_PATH.write_text(json.dumps(cur, indent=2, ensure_ascii=False), encoding="utf-8")
        line = (
            f"BASELINE  mean_containers={cur['mean_containers']}  "
            f"mean_dN={cur['mean_dN']}  disq={cur['total_disqualified']}  "
            f"wall={cur['total_wall_s']}s  instances={cur['instances']}"
        )
        print(line)
        return

    base = json.loads(BASELINE_PATH.read_text(encoding="utf-8")) if BASELINE_PATH.exists() else None
    verdict = _verdict(cur, base)
    if base is None:
        line = f"VERDICT: {verdict} (no baseline.json)  mean_containers={cur['mean_containers']}"
    else:
        line = (
            f"VERDICT: {verdict}  mean_containers {base['mean_containers']}->{cur['mean_containers']}  "
            f"mean_dN {base['mean_dN']}->{cur['mean_dN']}  "
            f"disq {base['total_disqualified']}->{cur['total_disqualified']}  "
            f"wall {base['total_wall_s']}s->{cur['total_wall_s']}s "
            f"(ceil {ABS_TIME_CEIL_S:.0f}s)"
        )
    line = f"[decoder={args.decoder}] {line}"
    print(line)
    with PROGRESS_LOG.open("a", encoding="utf-8") as f:
        f.write(f"{cur['ts']}  {line}\n")


if __name__ == "__main__":
    main()
