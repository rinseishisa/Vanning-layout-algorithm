"""Claude design spec: brief vanning_exact_b §2

CP-SAT formulation for min-containers 3D bin packing (single destination groups only).
Chen–Lee–Shen relative-position 3D-OPP, reified with CP-SAT.
"""
from __future__ import annotations

import os

from ortools.sat.python import cp_model

# Container dimensions (mm)
_X = 2300
_Y = 12000
_Z = 2400
_MAX_WEIGHT = 24000


def solve_min_containers(items, time_limit_s: float, max_containers: int, with_support: bool = False) -> dict:
    """Solve min containers for a single-destination item group.

    Returns {"status": str, "n_containers": int|None, "certified": bool}
    """
    n = len(items)
    if n == 0:
        return {"status": "OPTIMAL", "n_containers": 0, "certified": True}

    C = max_containers
    m = cp_model.CpModel()

    # --- assignment ---
    cont = [m.new_int_var(0, C - 1, f"cont_{i}") for i in range(n)]
    used = [m.new_bool_var(f"used_{c}") for c in range(C)]

    b = {}
    for i in range(n):
        for c in range(C):
            b[i, c] = m.new_bool_var(f"b_{i}_{c}")
            m.add(cont[i] == c).only_enforce_if(b[i, c])
            m.add(cont[i] != c).only_enforce_if(b[i, c].Not())
        m.add(sum(b[i, c] for c in range(C)) == 1)

    # used linkage + symmetry breaking
    for c in range(C):
        m.add(sum(b[i, c] for i in range(n)) >= 1).only_enforce_if(used[c])
        for i in range(n):
            m.add(b[i, c] <= used[c])
    for c in range(C - 1):
        m.add(used[c] >= used[c + 1])

    # --- position + rotation (XY 90° only) ---
    x = [m.new_int_var(0, _X, f"x_{i}") for i in range(n)]
    y = [m.new_int_var(0, _Y, f"y_{i}") for i in range(n)]
    z = [m.new_int_var(0, _Z, f"z_{i}") for i in range(n)]
    rot = [m.new_bool_var(f"rot_{i}") for i in range(n)]
    ax = [m.new_int_var(0, _X, f"ax_{i}") for i in range(n)]
    ay = [m.new_int_var(0, _Y, f"ay_{i}") for i in range(n)]

    for i in range(n):
        wi = int(items[i].width)
        li = int(items[i].length)
        hi = int(items[i].height)
        m.add(ax[i] == wi).only_enforce_if(rot[i].Not())
        m.add(ay[i] == li).only_enforce_if(rot[i].Not())
        m.add(ax[i] == li).only_enforce_if(rot[i])
        m.add(ay[i] == wi).only_enforce_if(rot[i])
        m.add(x[i] + ax[i] <= _X)
        m.add(y[i] + ay[i] <= _Y)
        m.add(z[i] + hi <= _Z)

    # --- non-overlap (same-container pairs only) ---
    samec_pairs: dict = {}  # (i,j) i<j -> samec bool (reused by support)
    for i in range(n):
        hi = int(items[i].height)
        for j in range(i + 1, n):
            hj = int(items[j].height)
            samec = m.new_bool_var(f"samec_{i}_{j}")
            m.add(cont[i] == cont[j]).only_enforce_if(samec)
            m.add(cont[i] != cont[j]).only_enforce_if(samec.Not())
            samec_pairs[(i, j)] = samec
            lft = m.new_bool_var(f"lft_{i}_{j}")
            rgt = m.new_bool_var(f"rgt_{i}_{j}")
            frt = m.new_bool_var(f"frt_{i}_{j}")
            bck = m.new_bool_var(f"bck_{i}_{j}")
            dwn = m.new_bool_var(f"dwn_{i}_{j}")
            up = m.new_bool_var(f"up_{i}_{j}")
            m.add(x[i] + ax[i] <= x[j]).only_enforce_if(lft)
            m.add(x[j] + ax[j] <= x[i]).only_enforce_if(rgt)
            m.add(y[i] + ay[i] <= y[j]).only_enforce_if(frt)
            m.add(y[j] + ay[j] <= y[i]).only_enforce_if(bck)
            m.add(z[i] + hi <= z[j]).only_enforce_if(dwn)
            m.add(z[j] + hj <= z[i]).only_enforce_if(up)
            m.add_bool_or([lft, rgt, frt, bck, dwn, up]).only_enforce_if(samec)

    # --- weight per container ---
    for c in range(C):
        m.add(
            sum(round(float(items[i].weight)) * b[i, c] for i in range(n)) <= _MAX_WEIGHT
        )

    # --- partial-support relaxation (VALID lower-bound model) ---
    # True support (algorithm_a.is_supported): z=0 OR ∃ base with
    #   base.z2 == z[i] AND base footprint fully contains item i's footprint.
    # We RELAX to: z[i]=0 OR ∃ j≠i (same container) with z[j]+h[j] == z[i].
    # True-support ⇒ this (that base IS such a j) ⇒ feasible(C') ⊇ feasible(C)
    # ⇒ min-used(C') ≤ true optimum ⇒ solver.best_objective_bound is a
    # CERTIFIED valid lower bound. Strictly weaker than full support (no XY
    # containment) = leanest model that still forbids items floating with
    # nothing at their base height anywhere in the container.
    if with_support:
        for i in range(n):
            zi = z[i]
            of = m.new_bool_var(f"onfloor_{i}")
            m.add(zi == 0).only_enforce_if(of)
            m.add(zi != 0).only_enforce_if(of.Not())
            lits = [of]
            for j in range(n):
                if j == i:
                    continue
                hj = int(items[j].height)
                samec = samec_pairs[(i, j) if i < j else (j, i)]
                sup = m.new_bool_var(f"sup_{i}_by_{j}")
                # sup ⇒ same container ∧ j's top exactly at i's bottom.
                # (no reverse implication needed: sup is only set when it
                #  helps satisfy the disjunction, and then it must be real.)
                m.add(z[j] + hj == zi).only_enforce_if(sup)
                m.add_implication(sup, samec)
                lits.append(sup)
            m.add_bool_or(lits)

    # --- objective ---
    m.minimize(sum(used))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = max(1, (os.cpu_count() or 2) - 1)

    st = solver.solve(m)
    status_name = solver.status_name(st)

    certified = status_name == "OPTIMAL"
    if status_name in ("OPTIMAL", "FEASIBLE"):
        n_containers = int(round(solver.objective_value))
    else:
        n_containers = None

    # best_objective_bound is a SOUND dual lower bound for this minimization
    # regardless of proven optimality (== objective_value when OPTIMAL).
    import math

    try:
        dual = solver.best_objective_bound
        dual_lb = int(math.ceil(dual - 1e-6)) if dual is not None else None
    except Exception:
        dual_lb = None

    return {
        "status": status_name,
        "n_containers": n_containers,
        "certified": certified,
        "dual_lb": dual_lb,
    }
