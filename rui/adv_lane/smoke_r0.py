"""R0 Step-0 スモーク: pack_items_beam の feasibility + デコード時間計測。

全16inst bench (gen20×pop14) を盲目起動する前に、1-2 inst で
  (a) beam デコードが feasible 解を返すか (disq 0)
  (b) 1 デコードの実時間 (greedy 比の倍率)
を測り、full bench の概算 wall を見積もる (Step-0 規律 / LeWM 反省)。
"""
from __future__ import annotations

import contextlib
import io
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rui.adv_lane.coevo_decoder import pack_items_beam
from rui.adv_lane.ga_bench import _load_items, build_suite
from rui.algorithm_a import evaluate_solution, pack_items, run_ga


def _summ(containers):
    ev = evaluate_solution(containers)
    return ev["container_count"], bool(ev["disqualified"])


def main() -> None:
    suite = build_suite()
    print(f"suite size = {len(suite)}; smoke on first 2")
    for path in suite[:2]:
        items = _load_items(path)
        n_items = len(items)

        t0 = time.perf_counter()
        g = pack_items(items)
        t_g = time.perf_counter() - t0
        gn, gdq = _summ(g)

        t0 = time.perf_counter()
        b = pack_items_beam(items)
        t_b = time.perf_counter() - t0
        bn, bdq = _summ(b)

        ratio = t_b / t_g if t_g > 0 else float("inf")
        print(
            f"\n[{path.name}] items={n_items}\n"
            f"  greedy: N={gn} dq={gdq} decode={t_g*1000:.1f}ms\n"
            f"  beam  : N={bn} dq={bdq} decode={t_b*1000:.1f}ms  (x{ratio:.1f} greedy)"
        )

    # 1 inst だけ小 GA (gen3/pop4) を beam デコードで回し run_ga 統合 + 概算
    path = suite[0]
    items = _load_items(path)
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        containers, _ = run_ga(items, generations=3, pop_size=4, decoder=pack_items_beam)
    t_run = time.perf_counter() - t0
    rn, rdq = _summ(containers)
    print(
        f"\n[run_ga beam gen3/pop4 on {path.name}] N={rn} dq={rdq} wall={t_run:.1f}s"
    )
    # full bench 概算: 16 inst, gen20/pop14。run_ga eval 回数 ~= gen*pop。
    # gen3/pop4 = 12 eval 相当 → スケール係数 (20*14)/(3*4) ≈ 23.3
    scale = (20 * 14) / (3 * 4)
    est_serial = t_run * scale * 16
    print(
        f"full bench (16 inst, gen20/pop14) 概算: serial ~{est_serial:.0f}s "
        f"/ 15並列 ~{est_serial/15:.0f}s  (ABS_TIME_CEIL=1500s)"
    )


if __name__ == "__main__":
    main()
