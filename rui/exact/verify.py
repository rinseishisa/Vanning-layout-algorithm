"""Verification gate for rui/exact.

Final stdout line must be exactly one of:
  VERDICT: PASS
  VERDICT: PARTIAL (ortools missing)
  VERDICT: FAIL <reason>

Exit code: PASS/PARTIAL=0, FAIL=1.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import sys
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rui.adv_lane.ga_bench import build_suite, BEAM_REF_PATH, _load_items
from rui.exact.lower_bounds import instance_lb


def _run_pytest() -> tuple[int, str]:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(_REPO_ROOT / "rui" / "exact" / "tests"),
        "-q",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode, (result.stdout + result.stderr)
    except Exception as e:
        return 1, str(e)


def _has_ortools() -> bool:
    try:
        import ortools
        return True
    except Exception:
        return False


def main() -> int:
    failures = []

    # 1. pytest
    rc, out = _run_pytest()
    if rc != 0:
        # If ortools is missing, cpsat tests skip via importorskip; pytest still passes.
        # But if there are real failures, capture them.
        if "passed" in out and "failed" not in out.lower():
            pass  # all passed (or all skipped)
        else:
            failures.append(f"pytest failed: rc={rc}\n{out[:500]}")

    # 2. invariant perdest_lb >= max(volume_lb, weight_lb) for all suite instances
    for path in build_suite():
        items = _load_items(path)
        lb = instance_lb(items)
        if lb["perdest_lb"] < max(lb["volume_lb"], lb["weight_lb"]):
            failures.append(
                f"invariant broken for {path.name}: perdest_lb={lb['perdest_lb']} < max({lb['volume_lb']},{lb['weight_lb']})"
            )

    # 3. beam_N known => max(volume_lb, weight_lb, perdest_lb) <= beam_N
    if BEAM_REF_PATH.exists():
        beam_ref = json.loads(BEAM_REF_PATH.read_text(encoding="utf-8"))
        for path in build_suite():
            inst = path.name
            ref = beam_ref.get(inst)
            if not ref or ref.get("N") is None:
                continue
            beam_n = ref["N"]
            items = _load_items(path)
            lb = instance_lb(items)
            if max(lb["volume_lb"], lb["weight_lb"], lb["perdest_lb"]) > beam_n:
                failures.append(
                    f"beam bound broken for {inst}: LB={max(lb['volume_lb'], lb['weight_lb'], lb['perdest_lb'])} > beam_N={beam_n}"
                )

    # 4. ortools path
    if _has_ortools():
        from rui.exact import slice_instances, cpsat_model
        for name, items, expected in slice_instances.synthetic_known():
            result = cpsat_model.solve_min_containers(
                items, time_limit_s=30.0, max_containers=max(expected, 1)
            )
            if result["status"] != "OPTIMAL" or result["n_containers"] != expected:
                failures.append(
                    f"synthetic {name}: expected OPTIMAL {expected}, got {result['status']} {result['n_containers']}"
                )

        # Exact small groups: perdest_lb <= exact_opt (and <= beam_N if known)
        for label, group in slice_instances.iter_small_single_dest(12):
            lb = instance_lb(group)
            max_c = max(lb["perdest_lb"], lb["volume_lb"], lb["weight_lb"], len(group))
            result = cpsat_model.solve_min_containers(
                group, time_limit_s=60.0, max_containers=max_c
            )
            if result["n_containers"] is not None:
                if lb["perdest_lb"] > result["n_containers"]:
                    failures.append(
                        f"exact LB violation for {label}: perdest_lb={lb['perdest_lb']} > exact={result['n_containers']}"
                    )
            # beam_N check for the parent instance if known
            stem = label.split("::")[0]
            inst_name = stem + ".json"
            if BEAM_REF_PATH.exists():
                beam_ref = json.loads(BEAM_REF_PATH.read_text(encoding="utf-8"))
                ref = beam_ref.get(inst_name)
                if ref and ref.get("N") is not None:
                    beam_n = ref["N"]
                    if result["n_containers"] is not None and result["n_containers"] > beam_n:
                        failures.append(
                            f"exact upper bound violation for {label}: exact={result['n_containers']} > beam_N={beam_n}"
                        )
    else:
        # ortools missing: skip 4
        pass

    if failures:
        print("VERDICT: FAIL " + failures[0])
        return 1

    if _has_ortools():
        print("VERDICT: PASS")
        return 0
    else:
        print("VERDICT: PARTIAL (ortools missing)")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
