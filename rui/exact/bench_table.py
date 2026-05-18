"""Build lower-bound / exact-opt table and audit artifacts.

Does NOT re-run GA or beam; values are read from existing JSON files.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repo root on sys.path for imports
import sys
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rui.adv_lane.ga_bench import build_suite, BEAM_REF_PATH, BASELINE_PATH, LAST_PATH, _load_items
from rui.exact.lower_bounds import instance_lb
from rui.exact import slice_instances
from rui.exact import cpsat_model

RUNS_DIR = Path(__file__).parent / "runs"
EXACT_MAX_ITEMS = 12
EXACT_STRETCH = 18  # reserved for future use
EXACT_TL = 60.0


def _ga_rows() -> List[Dict[str, Any]]:
    path = LAST_PATH if LAST_PATH.exists() else (BASELINE_PATH if BASELINE_PATH.exists() else None)
    if path is None:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("rows", [])


def _beam_ref() -> Dict[str, Any]:
    if BEAM_REF_PATH.exists():
        return json.loads(BEAM_REF_PATH.read_text(encoding="utf-8"))
    return {}


def build_lb_table(lb_only: bool = False) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ga_rows = {r["instance"]: r for r in _ga_rows()}
    beam_ref = _beam_ref()

    rows: List[Dict[str, Any]] = []
    exact_rows: List[Dict[str, Any]] = []

    for path in build_suite():
        inst_name = path.name
        stem = path.stem
        items = _load_items(path)
        lb = instance_lb(items)
        ga_n = ga_rows.get(inst_name, {}).get("ga_containers")
        beam_n = beam_ref.get(inst_name, {}).get("N")
        row = {
            "instance": inst_name,
            "stem": stem,
            "n_items": len(items),
            "n_dest": lb["n_dest"],
            "GA_N": ga_n,
            "beam_N": beam_n,
            "volume_lb": lb["volume_lb"],
            "weight_lb": lb["weight_lb"],
            "perdest_lb": lb["perdest_lb"],
            "exact_opt": None,
            "opt_certified": False,
        }
        rows.append(row)

    if not lb_only:
        for label, group in slice_instances.iter_small_single_dest(EXACT_MAX_ITEMS):
            # label format: "stem::dest_id"
            stem = label.split("::")[0]
            lb = instance_lb(group)
            max_containers = max(lb["perdest_lb"], lb["volume_lb"], lb["weight_lb"], len(group))
            # If the full instance has a known beam_N, cap max_containers there to keep model small
            # Find any row with matching stem
            beam_n = None
            for r in rows:
                if r["stem"] == stem:
                    beam_n = r.get("beam_N")
                    break
            if beam_n is not None:
                max_containers = min(max_containers, beam_n)
                if max_containers < lb["perdest_lb"]:
                    max_containers = beam_n  # safety: at least beam_N (known upper bound)
            result = cpsat_model.solve_min_containers(group, time_limit_s=EXACT_TL, max_containers=max_containers)
            exact_rows.append({
                "label": label,
                "n_items": len(group),
                "perdest_lb": lb["perdest_lb"],
                "exact_opt": result["n_containers"],
                "opt_certified": result["certified"],
                "status": result["status"],
            })

    return rows, exact_rows


def _fmt_table(rows: List[Dict[str, Any]]) -> str:
    lines = []
    header = f"{'instance':<30} {'n':>4} {'dest':>4} {'GA':>4} {'beam':>4} {'volLB':>5} {'wtLB':>5} {'pdLB':>5} {'exact':>5} {'cert':>4}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        lines.append(
            f"{r['instance']:<30} {r['n_items']:>4} {r['n_dest']:>4} "
            f"{str(r['GA_N']):>4} {str(r['beam_N']):>4} "
            f"{r['volume_lb']:>5} {r['weight_lb']:>5} {r['perdest_lb']:>5} "
            f"{str(r['exact_opt']):>5} {'Y' if r['opt_certified'] else 'N':>4}"
        )
    return "\n".join(lines)


def _build_hardness_audit(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    audit = []
    for r in rows:
        if not r["instance"].startswith("hard_"):
            continue
        beam_n = r.get("beam_N")
        if beam_n is None:
            continue
        gap = beam_n - r["perdest_lb"]
        if gap <= 0:
            cls = "beam_optimal_artifact"
        elif gap == 1:
            cls = "marginal"
        else:
            cls = "genuinely_hard"
        audit.append({
            "instance": r["instance"],
            "stem": r["stem"],
            "beam_N": beam_n,
            "perdest_lb": r["perdest_lb"],
            "gap": gap,
            "class": cls,
        })
    return {"audit": audit}


def _build_adv_bounds(rows: List[Dict[str, Any]], exact_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    iso = time.strftime("%Y-%m-%d")
    bounds: Dict[str, Any] = {}
    # Map exact results by stem
    exact_by_stem: Dict[str, List[Dict[str, Any]]] = {}
    for er in exact_rows:
        stem = er["label"].split("::")[0]
        exact_by_stem.setdefault(stem, []).append(er)

    for r in rows:
        if not r["instance"].startswith("hard_"):
            continue
        stem = r["stem"]
        # exact_opt for the full instance is unknown; we only have per-dest exacts.
        # Leave opt null unless all dests of this instance were solved and sum matches.
        opt = None
        certified = False
        if stem in exact_by_stem:
            # If the instance had only one small dest group and it is certified,
            # we can treat that as the instance opt.
            ers = exact_by_stem[stem]
            if len(ers) == 1 and ers[0]["opt_certified"]:
                opt = ers[0]["exact_opt"]
                certified = True
        bounds[stem] = {
            "lb": r["perdest_lb"],
            "opt": opt,
            "opt_certified": certified,
            "source": f"rui/exact bench_table {iso}",
        }
    return bounds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lb-only", action="store_true", help="Skip CP-SAT; compute lower bounds only")
    args = ap.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    rows, exact_rows = build_lb_table(lb_only=args.lb_only)

    # Merge exact results into rows for display (best effort: match stem)
    exact_by_stem: Dict[str, List[Dict[str, Any]]] = {}
    for er in exact_rows:
        stem = er["label"].split("::")[0]
        exact_by_stem.setdefault(stem, []).append(er)

    for r in rows:
        ers = exact_by_stem.get(r["stem"], [])
        if ers:
            # If any exact result exists for this instance, show the best certified opt
            certifieds = [e for e in ers if e["opt_certified"]]
            if certifieds:
                best = min(certifieds, key=lambda e: e["exact_opt"] if e["exact_opt"] is not None else 9999)
                r["exact_opt"] = best["exact_opt"]
                r["opt_certified"] = True
            else:
                best = min(ers, key=lambda e: e["exact_opt"] if e["exact_opt"] is not None else 9999)
                r["exact_opt"] = best["exact_opt"]
                r["opt_certified"] = False

    table_str = _fmt_table(rows)
    print(table_str)

    (RUNS_DIR / "lb_table.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (RUNS_DIR / "hardness_audit.json").write_text(
        json.dumps(_build_hardness_audit(rows), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (RUNS_DIR / "adv_bench_bounds.json").write_text(
        json.dumps(_build_adv_bounds(rows, exact_rows), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote {RUNS_DIR / 'lb_table.json'}")
    print(f"Wrote {RUNS_DIR / 'hardness_audit.json'}")
    print(f"Wrote {RUNS_DIR / 'adv_bench_bounds.json'}")


if __name__ == "__main__":
    main()
