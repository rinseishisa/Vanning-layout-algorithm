"""Regret computation for adversarial instance generation (design.md §2)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SolverResult:
    """Minimal solver output needed for regret computation."""

    dq: bool          # disqualified?
    N: int            # number of containers used
    dev: float        # mean |Yg - 6000| over all containers (mm)


def compute_regret(
    p: SolverResult,
    a: SolverResult,
    *,
    eps: float = 1e-4,
    dq_bonus: float = 1e3,
) -> Optional[float]:
    """Compute regret scalar r(θ) = protagonist vs antagonist.

    Returns ``None`` when the instance should be discarded (antagonist also
    disqualified → degenerate / infeasible).

    Boundary semantics (exactly as design.md §2):
    - a.dq=True                → None   (discard)
    - p.dq=True & a.dq=False   → dq_bonus  (best hard instance)
    - both dq=False            → dN + eps * dDev
    """
    if a.dq:
        return None
    if p.dq and not a.dq:
        return dq_bonus
    # Both qualified
    dN = p.N - a.N
    dDev = p.dev - a.dev
    return dN + eps * dDev
