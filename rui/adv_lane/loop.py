"""CMA-ES adversarial loop (design.md §5/§6)."""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from rui.adv_lane.antagonist import beam_search_strong
from rui.adv_lane.generator import THETA_DIM, build_dataset, decode_theta
from rui.adv_lane.generator31 import build_dataset_31
from rui.adv_lane.regret import SolverResult, compute_regret, shaped_fitness
from rui.adv_lane.theta31 import THETA_DIM_31, decode_theta31
from rui.algorithm_a import REQUIRED_COLUMNS, build_items, evaluate_solution, run_ga

# ------------------------------------------------------------------
# Optional CMA-ES library
# ------------------------------------------------------------------
try:
    import cma

    HAS_CMA = True
except Exception:  # pragma: no cover
    HAS_CMA = False

# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------
DEFAULT_G = 15
DEFAULT_POP = 12
DEFAULT_GA_GEN = 10
DEFAULT_GA_POP = 10
SMOKE_G = 3
SMOKE_POP = 4
SMOKE_GA_GEN = 5
SMOKE_GA_POP = 6
TOP_K_SAVE = 20


def _make_dataframe(data: Dict) -> pd.DataFrame:
    """Replicate algorithm_a.read_generated_items logic for an in-memory dict."""
    df = pd.DataFrame(data["items"])
    if "dimensions" in df.columns:
        dims = df["dimensions"].apply(lambda v: v if isinstance(v, dict) else {})
        df = df.assign(
            width=dims.apply(lambda v: v.get("w")),
            length=dims.apply(lambda v: v.get("l")),
            height=dims.apply(lambda v: v.get("h")),
        )
    df = df[REQUIRED_COLUMNS].copy()
    df = df.dropna(subset=REQUIRED_COLUMNS)
    for col in ["width", "length", "height", "weight"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["width", "length", "height", "weight"])
    df["item_id"] = df["item_id"].astype(str).str.strip()
    df["size_type"] = df["size_type"].astype(str).str.strip().str.lower()
    df["destination_id"] = df["destination_id"].astype(str).str.strip()
    df["width"] = df["width"].astype(int)
    df["length"] = df["length"].astype(int)
    df["height"] = df["height"].astype(int)
    df["weight"] = df["weight"].astype(float)
    return df.reset_index(drop=True)


def evaluate_instance(
    theta: np.ndarray,
    seed: int,
    ga_gen: int,
    ga_pop: int,
    catalog: int = 3,
) -> Tuple[Optional[float], Optional[Dict], Optional[Dict], Optional[Dict]]:
    """Build dataset, run protagonist & antagonist, return regret and raw evals."""
    if catalog == 31:
        data = build_dataset_31(theta, seed)
    else:
        data = build_dataset(theta, seed)
    if data is None:
        return None, None, None, None

    df = _make_dataframe(data)
    if df.empty:
        return None, None, None, None

    items = build_items(df)

    # Protagonist (GA)
    try:
        p_containers, eval_p = run_ga(items, generations=ga_gen, pop_size=ga_pop)
    except Exception as exc:
        # If GA crashes (rare), treat as DQ
        eval_p = {
            "disqualified": True,
            "container_count": 999,
            "mean_y_deviation": 999.0,
            "violations": [str(exc)],
        }

    # Antagonist (beam search)
    try:
        a_containers, dq_a = beam_search_strong(items)  # strong defaults (48/12)
    except Exception as exc:
        dq_a = True
        a_containers = []

    if dq_a:
        eval_a = {
            "disqualified": True,
            "container_count": 999,
            "mean_y_deviation": 999.0,
            "violations": [],
        }
    else:
        eval_a = evaluate_solution(a_containers)

    p = SolverResult(
        dq=bool(eval_p.get("disqualified", False)),
        N=int(eval_p.get("container_count", 999)),
        dev=float(eval_p.get("mean_y_deviation", 999.0)),
    )
    a = SolverResult(
        dq=bool(eval_a.get("disqualified", False)),
        N=int(eval_a.get("container_count", 999)),
        dev=float(eval_a.get("mean_y_deviation", 999.0)),
    )
    r = compute_regret(p, a)
    return r, data, eval_p, eval_a


def _format_theta(theta: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:.3f}" for x in theta) + "]"


def _dump_gen_summary(path: Path, summaries: List[Dict]) -> None:
    """gen_summary を CSV へ書き出す（冪等・毎世代呼べる）。

    最終一括書き出しだとクラッシュ/kill で全消失し本走中インスペクトも不可
    だった反省（scale up 前提改修）。毎世代これを呼べば途中まで必ず
    ディスクに残り、別プロセスから進捗を読める。summary 行は両経路で固定
    キー（gen/best_regret/mean_regret/std_regret/none_rate/mean_entropy）。
    """
    if not summaries:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)


def _simple_es_loop(
    dim: int,
    generations: int,
    popsize: int,
    fitness_fn,
    x0: Optional[np.ndarray] = None,
    sigma0: float = 0.5,
) -> Tuple[np.ndarray, List[Dict]]:
    """Bare-bones (mu,lambda)-ES fallback when ``cma`` is unavailable."""
    mu = max(1, popsize // 2)
    if x0 is None:
        x0 = np.zeros(dim)
    mean = x0.copy()
    sigma = sigma0
    gen_logs: List[Dict] = []
    for gen in range(generations):
        pop = [mean + sigma * np.random.randn(dim) for _ in range(popsize)]
        fits = [fitness_fn(p, gen, i) for i, p in enumerate(pop)]
        # minimize
        idx = np.argsort(fits)
        elite = [pop[i] for i in idx[:mu]]
        mean = np.mean(elite, axis=0)
        # simple 1/5 success rule approx
        success_rate = mu / popsize
        if success_rate > 0.2:
            sigma *= 1.1
        else:
            sigma *= 0.9
        gen_logs.append({
            "gen": gen + 1,
            "best_fitness": float(fits[idx[0]]),
            "mean_fitness": float(np.mean(fits)),
            "sigma": float(sigma),
        })
    return mean, gen_logs


NONE_PENALTY = 1e6


def _eval_one(payload: Tuple) -> Dict:
    """Top-level (picklable) per-individual evaluation for ProcessPool.

    payload = (theta_list, gen, idx, base_seed, popsize, ga_gen, ga_pop,
               catalog, shaping_lambda)
    Returns a serializable dict; the parent does trajectory/hard_buffer
    bookkeeping and feeds ``fitness`` to CMA-ES.
    """
    (theta_list, gen, idx, base_seed, popsize, ga_gen, ga_pop,
     catalog, shaping_lambda) = payload
    theta = np.asarray(theta_list, dtype=float)
    s = base_seed + gen * popsize + idx
    # 並列ワーカ間で GA を再現可能に（旧直列は random 未シードで非決定だった）
    random.seed(s)
    np.random.seed(s % (2 ** 32 - 1))

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            r, data, eval_p, eval_a = evaluate_instance(
                theta, s, ga_gen, ga_pop, catalog=catalog
            )
    except Exception as exc:  # 1 ワーカの事故でプール全体を落とさない
        return {
            "row": {"gen": gen + 1, "idx": idx, "regret": None,
                    "shaped": None, "p_min_fill": None, "p_dq": True,
                    "p_N": None, "p_dev": None, "a_dq": None, "a_N": None,
                    "a_dev": None, "size_entropy": None, "item_count": None},
            "fitness": NONE_PENALTY, "regret": None,
            "data": None, "theta": theta_list,
        }

    p_min_fill = None
    if eval_p is not None and not eval_p.get("disqualified", False):
        fills = [
            cs.get("fill_rate")
            for cs in (eval_p.get("container_summaries") or [])
            if cs.get("fill_rate") is not None
        ]
        if fills:
            p_min_fill = min(fills)
    shaped = None if r is None else shaped_fitness(r, p_min_fill, lam=shaping_lambda)

    row = {
        "gen": gen + 1,
        "idx": idx,
        "regret": None if r is None else round(r, 6),
        "shaped": None if shaped is None else round(shaped, 6),
        "p_min_fill": None if p_min_fill is None else round(p_min_fill, 6),
        "p_dq": eval_p.get("disqualified", False) if eval_p is not None else None,
        "p_N": eval_p.get("container_count", None) if eval_p is not None else None,
        "p_dev": eval_p.get("mean_y_deviation", None) if eval_p is not None else None,
        "a_dq": eval_a.get("disqualified", False) if eval_a is not None else None,
        "a_N": eval_a.get("container_count", None) if eval_a is not None else None,
        "a_dev": eval_a.get("mean_y_deviation", None) if eval_a is not None else None,
    }
    if data is not None:
        if catalog == 31:
            row["size_entropy"] = round(float(decode_theta31(theta)["size_entropy_norm"]), 4)
        else:
            row["size_entropy"] = round(float(decode_theta(theta)["size_entropy"]), 4)
        row["item_count"] = data["dataset_info"]["item_count"]
    else:
        row["size_entropy"] = None
        row["item_count"] = None

    return {
        "row": row,
        "fitness": (-shaped if shaped is not None else NONE_PENALTY),
        "regret": r,
        "data": data if (r is not None and data is not None) else None,
        "theta": theta_list,
    }


def _resolve_workers(requested: int, popsize: int) -> int:
    """0/負 → 自動（min(pop, CPU-1)）。1 で直列、それ以外は指定値。"""
    if requested and requested > 0:
        return min(requested, popsize)
    return max(1, min(popsize, (os.cpu_count() or 2) - 1))


def run_loop(
    generations: int,
    popsize: int,
    ga_gen: int,
    ga_pop: int,
    smoke: bool,
    out_dir: Path,
    hard_dir: Path,
    seed: int = 42,
    shaping_lambda: float = 0.5,
    catalog: int = 3,
    workers: int = 0,
) -> None:
    """Main CMA-ES / ES loop."""
    out_dir.mkdir(parents=True, exist_ok=True)
    hard_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    base_seed = rng.randint(0, 1_000_000)

    # Replay buffer for hard instances
    hard_buffer: List[Tuple[float, Dict, np.ndarray]] = []

    # Logging structures
    trajectory: List[Dict] = []
    gen_summaries: List[Dict] = []

    # We minimize fitness = -regret (None → large penalty)
    NONE_PENALTY = 1e6

    def _fitness(theta: np.ndarray, gen: int, idx: int) -> float:
        nonlocal hard_buffer  # L220 で再代入するため（無いと UnboundLocalError）
        s = base_seed + gen * popsize + idx
        r, data, eval_p, eval_a = evaluate_instance(theta, s, ga_gen, ga_pop, catalog=catalog)
        # 整形 fitness 用: GA 最空コンテナの fill（崖の手前圧力）
        p_min_fill = None
        if eval_p is not None and not eval_p.get("disqualified", False):
            fills = [
                cs.get("fill_rate")
                for cs in (eval_p.get("container_summaries") or [])
                if cs.get("fill_rate") is not None
            ]
            if fills:
                p_min_fill = min(fills)
        shaped = None if r is None else shaped_fitness(r, p_min_fill, lam=shaping_lambda)
        # store trajectory row
        row = {
            "gen": gen + 1,
            "idx": idx,
            "regret": None if r is None else round(r, 6),
            "shaped": None if shaped is None else round(shaped, 6),
            "p_min_fill": None if p_min_fill is None else round(p_min_fill, 6),
            "p_dq": eval_p.get("disqualified", False) if eval_p is not None else None,
            "p_N": eval_p.get("container_count", None) if eval_p is not None else None,
            "p_dev": eval_p.get("mean_y_deviation", None) if eval_p is not None else None,
            "a_dq": eval_a.get("disqualified", False) if eval_a is not None else None,
            "a_N": eval_a.get("container_count", None) if eval_a is not None else None,
            "a_dev": eval_a.get("mean_y_deviation", None) if eval_a is not None else None,
        }
        trajectory.append(row)
        # hard-instance buffering
        if r is not None and data is not None:
            hard_buffer.append((r, data, theta.copy()))
            hard_buffer.sort(key=lambda t: -t[0])
            if len(hard_buffer) > TOP_K_SAVE * 2:
                hard_buffer = hard_buffer[:TOP_K_SAVE * 2]
        # mode-collapse metrics
        if data is not None:
            if catalog == 31:
                params = decode_theta31(theta)
                row["size_entropy"] = round(float(params["size_entropy_norm"]), 4)
            else:
                params = decode_theta(theta)
                row["size_entropy"] = round(float(params["size_entropy"]), 4)
            row["item_count"] = data["dataset_info"]["item_count"]
        # CMA-ES は整形 fitness で探索（プラトーに勾配を付与）。
        # gen summary / hard-instance 保存は純 regret のまま（fidelity 保全）。
        return -shaped if shaped is not None else NONE_PENALTY

    print(f"[adv_lane] Starting loop: G={generations}, pop={popsize}, ga_gen={ga_gen}, ga_pop={ga_pop}, catalog={catalog}")
    print(f"[adv_lane] CMA-ES available: {HAS_CMA}")
    print(f"[adv_lane] Output dir: {out_dir}")
    print(f"[adv_lane] Hard instance dir: {hard_dir}")

    dim = THETA_DIM_31 if catalog == 31 else THETA_DIM
    if HAS_CMA:
        x0 = np.zeros(dim)
        sigma0 = 0.5
        opts = {
            "popsize": popsize,
            "verbose": -9,
            "CMA_stds": [0.5] * dim,
        }
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        nw = _resolve_workers(workers, popsize)
        print(f"[adv_lane] Parallel workers: {nw} (pop={popsize}, cpu={os.cpu_count()})")
        pool = ProcessPoolExecutor(max_workers=nw) if nw > 1 else None
        try:
            for gen in range(generations):
                solutions = es.ask()
                payloads = [
                    (np.asarray(sol).tolist(), gen, i, base_seed, popsize,
                     ga_gen, ga_pop, catalog, shaping_lambda)
                    for i, sol in enumerate(solutions)
                ]
                if pool is not None:
                    results = list(pool.map(_eval_one, payloads))
                else:
                    results = [_eval_one(p) for p in payloads]
                # parent 側で順序保持しつつ trajectory / hard_buffer を更新
                for res in results:
                    trajectory.append(res["row"])
                    if res["regret"] is not None and res["data"] is not None:
                        hard_buffer.append(
                            (res["regret"], res["data"], np.asarray(res["theta"]))
                        )
                hard_buffer.sort(key=lambda t: -t[0])
                del hard_buffer[TOP_K_SAVE * 2:]
                fitnesses = [res["fitness"] for res in results]
                es.tell(solutions, fitnesses)
                # generation summary
                regrets = [trajectory[-(popsize - i)]["regret"] for i in range(popsize) if trajectory[-(popsize - i)]["regret"] is not None]
                none_count = sum(1 for i in range(popsize) if trajectory[-(popsize - i)]["regret"] is None)
                entropies = [trajectory[-(popsize - i)].get("size_entropy") for i in range(popsize) if trajectory[-(popsize - i)].get("size_entropy") is not None]
                summary = {
                    "gen": gen + 1,
                    "best_regret": max(regrets) if regrets else None,
                    "mean_regret": sum(regrets) / len(regrets) if regrets else None,
                    "std_regret": (sum((x - sum(regrets) / len(regrets)) ** 2 for x in regrets) / len(regrets)) ** 0.5 if regrets else None,
                    "none_rate": none_count / popsize,
                    "mean_entropy": sum(entropies) / len(entropies) if entropies else None,
                }
                gen_summaries.append(summary)
                br = "n/a" if summary["best_regret"] is None else f"{summary['best_regret']:.3f}"
                mr = "n/a" if summary["mean_regret"] is None else f"{summary['mean_regret']:.3f}"
                sr = "n/a" if summary["std_regret"] is None else f"{summary['std_regret']:.3f}"
                en = "n/a" if summary["mean_entropy"] is None else f"{summary['mean_entropy']:.3f}"
                print(
                    f"Gen {gen+1:02d}/{generations}  best_r={br}  "
                    f"mean_r={mr}  std_r={sr}  "
                    f"none={summary['none_rate']:.2%}  entropy={en}"
                )
                # 毎世代 flush 書き出し（クラッシュ/kill 生存・別プロセスから進捗読取）
                _dump_gen_summary(out_dir / "gen_summary.csv", gen_summaries)
        finally:
            if pool is not None:
                pool.shutdown(wait=True)
        best_theta = es.result.xbest
    else:
        print("[adv_lane] Warning: cma not installed — falling back to simple (mu,lambda)-ES")
        best_theta, gen_logs = _simple_es_loop(
            dim, generations, popsize, _fitness, sigma0=0.5
        )
        for g, gl in enumerate(gen_logs):
            # fitness は -regret or NONE_PENALTY(1e6) のため集計に使わない。
            # 当該世代の trajectory スライスから実 regret で算出する。
            rows = trajectory[g * popsize : (g + 1) * popsize]
            regs = [r["regret"] for r in rows if r["regret"] is not None]
            ents = [
                r.get("size_entropy")
                for r in rows
                if r.get("size_entropy") is not None
            ]
            none_ct = sum(1 for r in rows if r["regret"] is None)
            mean_r = sum(regs) / len(regs) if regs else None
            std_r = (
                (sum((x - mean_r) ** 2 for x in regs) / len(regs)) ** 0.5
                if regs
                else None
            )
            summary = {
                "gen": gl["gen"],
                "best_regret": max(regs) if regs else None,
                "mean_regret": mean_r,
                "std_regret": std_r,
                "none_rate": none_ct / popsize if popsize else None,
                "mean_entropy": sum(ents) / len(ents) if ents else None,
            }
            gen_summaries.append(summary)
            br = "n/a" if summary["best_regret"] is None else f"{summary['best_regret']:.3f}"
            mr = "n/a" if summary["mean_regret"] is None else f"{summary['mean_regret']:.3f}"
            print(
                f"Gen {gl['gen']:02d}/{generations}  best_r={br}  mean_r={mr}  "
                f"none={summary['none_rate']:.0%}  sigma={gl['sigma']:.4f}"
            )
            _dump_gen_summary(out_dir / "gen_summary.csv", gen_summaries)

    # ------------------------------------------------------------------
    # Persist outputs
    # ------------------------------------------------------------------
    # Trajectory CSV
    traj_path = out_dir / "trajectory.csv"
    with traj_path.open("w", newline="", encoding="utf-8") as f:
        if trajectory:
            writer = csv.DictWriter(f, fieldnames=trajectory[0].keys())
            writer.writeheader()
            writer.writerows(trajectory)

    # Generation summary CSV（毎世代の冪等書き出しと同一経路＝最終も一貫）
    _dump_gen_summary(out_dir / "gen_summary.csv", gen_summaries)

    # Best theta
    theta_path = out_dir / "best_theta.json"
    theta_path.write_text(
        json.dumps({"best_theta": best_theta.tolist()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Hard instances (top K)
    hard_buffer.sort(key=lambda t: -t[0])
    saved = 0
    for rank, (r, data, theta) in enumerate(hard_buffer[:TOP_K_SAVE], start=1):
        tag = f"g{generations}_p{popsize}"
        path = hard_dir / f"hard_{rank:02d}_{tag}.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        saved += 1
    print(f"[adv_lane] Saved {saved} hard instances to {hard_dir}")

    # Smoke sanity report
    if smoke:
        all_regrets = [row["regret"] for row in trajectory if row["regret"] is not None]
        none_rate = sum(1 for row in trajectory if row["regret"] is None) / max(len(trajectory), 1)
        print("\n=== Smoke Summary ===")
        print(f"Generations: {generations}, Population: {popsize}")
        print(f"Total evaluated: {len(trajectory)}")
        print(f"Valid (non-None) regrets: {len(all_regrets)}")
        print(f"None rate: {none_rate:.2%}")
        if all_regrets:
            print(f"Regret min: {min(all_regrets):.4f}")
            print(f"Regret max: {max(all_regrets):.4f}")
            print(f"Regret mean: {sum(all_regrets)/len(all_regrets):.4f}")
            print(f"Regret std:  {(sum((x-sum(all_regrets)/len(all_regrets))**2 for x in all_regrets)/len(all_regrets))**0.5:.4f}")
        if saved:
            print(f"Hard instances saved: {saved}")
        if none_rate > 0.6:
            print("WARNING: None rate > 60% — possible mode collapse or infeasible region")
        if all_regrets and max(all_regrets) - min(all_regrets) < 1e-6:
            print("WARNING: Regret variance is near-zero — signal is constant")
        print("=====================\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adversarial instance generation loop")
    parser.add_argument("--gen", type=int, default=DEFAULT_G, help="Number of generations")
    parser.add_argument("--pop", type=int, default=DEFAULT_POP, help="Population size")
    parser.add_argument("--ga-gen", type=int, default=DEFAULT_GA_GEN, help="GA generations for protagonist")
    parser.add_argument("--ga-pop", type=int, default=DEFAULT_GA_POP, help="GA population for protagonist")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test (G=3,pop=4, light GA)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--shaping-lambda",
        type=float,
        default=0.5,
        help="regret 整形係数 (CMA-ES 探索専用)。0 で整形無効＝純 regret 探索（A/B 比較用）",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--hard-dir", type=Path, default=None)
    parser.add_argument("--catalog", type=int, choices={3, 31}, default=3, help="Catalog mode: 3 (legacy) or 31 (extended)")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="並列ワーカ数。0=自動(min(pop,CPU-1))、1=直列、N=指定。世代内 pop 評価をプロセス並列化（結果不変）",
    )
    return parser.parse_args()


def _setup_utf8_streams() -> None:
    """Windows の CP932 端末でも非 ASCII (em-dash 等) を文字化け/クラッシュさせない。

    併せて line_buffering を有効化する。ログをファイル/パイプへリダイレクトすると
    stdout はブロックバッファリングになり、本走中ほぼ何も出ず進捗が見えなかった
    （世代行は生成済だがバッファ滞留）。改行ごと flush で即時に進捗が見える。
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", line_buffering=True)
            except Exception:
                pass


def main() -> None:
    _setup_utf8_streams()
    args = parse_args()

    if args.smoke:
        generations = SMOKE_G
        popsize = SMOKE_POP
        ga_gen = SMOKE_GA_GEN
        ga_pop = SMOKE_GA_POP
    else:
        generations = args.gen
        popsize = args.pop
        ga_gen = args.ga_gen
        ga_pop = args.ga_pop

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (Path(__file__).parent / "runs" / ts)
    hard_dir = args.hard_dir or (Path(__file__).parent / "hard_instances")

    run_loop(
        generations=generations,
        popsize=popsize,
        ga_gen=ga_gen,
        ga_pop=ga_pop,
        smoke=args.smoke,
        out_dir=out_dir,
        hard_dir=hard_dir,
        seed=args.seed,
        shaping_lambda=args.shaping_lambda,
        catalog=args.catalog,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
