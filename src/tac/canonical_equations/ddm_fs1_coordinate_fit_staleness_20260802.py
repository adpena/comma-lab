# SPDX-License-Identifier: MIT
"""ddm_fs1 — COORDINATE-FIT STALENESS: did a solved coefficient outlive its partner?

THE DEFECT CLASS (measured 2026-08-02, ddm_v4c/mq1 photometric stage).

Coordinate descent solves x_1 with {x_2..x_n} held at their CURRENT values, then
solves x_2 with x_1 frozen, and so on.  Every stage's own delta is correctly
negative, so every receipt reads healthy.  But when a LATER coordinate moves, the
EARLIER one is left fitted against a value that no longer exists -- and nothing in
the receipt records which value it was fitted against, so the staleness is
invisible and can only be recovered by hand-archaeology across arms.

MEASURED instance that motivated this (v4c/pw1/mq1 photometric stage, n600):
  * v4c solves auto-exposure (a,b) by 2-param GN at beta=0, then picks the
    rolling-shutter beta with (a,b) FROZEN, searching the grid {0.0, 0.5, 1.0}.
  * pw1 then DERIVED the beta table from the magnitudes the solve actually chose;
    mq1 refined beta continuously to 132 distinct values.  Neither re-solved (a,b)
    -- mq1's refinement set is {p0, p1, p2, beta}, which excludes gain/bias.
  * Result: 244 of 600 pairs ship a non-identity (a,b) together with a non-zero
    beta, and **59 of those ship |beta| > 1.0 -- strictly OUTSIDE the {0,0.5,1.0}
    hull the (a,b) was fitted against**, out to |beta| = 7.5.

That last set is the sharp one.  Staleness INSIDE the fitted hull is interpolation
error; staleness OUTSIDE it is EXTRAPOLATION -- the coefficient is being applied at
a geometry the solve never sampled.  The two deserve different verdicts and
different priority, which is why this discriminator separates them.

THE VACUITY RULE, inherited from ddm_pw1 and the scope-ledger work: a row with NO
recorded fit context is ``UNDETERMINED_NO_CONTEXT``.  It is never ``FRESH``.
Absence of evidence of staleness is not evidence of freshness -- that equivalence
is the exact bug this module exists to make impossible.

Consumers: any coordinate-descent / alternating solver that ships per-item
coefficients -- experiments/ddm_v4c_resolve.py (photo stage), the mq1 joint
refiner, tac.optimization.terminal_pose_gn, experiments/ddm_pfs1_ep_warp_pose_solve.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

FRESH = "FRESH"
STALE_INTERPOLATED = "STALE_INTERPOLATED"
STALE_EXTRAPOLATED = "STALE_EXTRAPOLATED"
UNDETERMINED_NO_CONTEXT = "UNDETERMINED_NO_CONTEXT"

#: Measured anchor: the v4c photometric grid the shipped (a,b) was fitted against.
V4C_AB_FIT_MENU: tuple[float, ...] = (0.0, 0.5, 1.0)

#: Measured 2026-08-02 over ddm_v4d_20260731/mq1_emit/final_mq1.jsonl (600 pairs).
V4C_AB_STALENESS_CENSUS: dict[str, int] = {
    "pairs": 600,
    "beta_nonzero": 248,
    "ab_non_identity": 592,
    "stale_set": 244,          # both a non-identity (a,b) AND a non-zero beta
    "extrapolated": 59,        # of the stale set, |beta| outside the fitted hull
}


def fit_staleness(
    *,
    coefficient: str,
    fitted_against: Mapping[str, float] | None,
    shipped_partners: Mapping[str, float],
    fit_menu: Sequence[float] | None = None,
    partner_key: str | None = None,
    tol: float = 1e-9,
) -> dict[str, Any]:
    """Is ``coefficient`` still fitted against the partner values it ships with?

    ``fitted_against`` is the co-coordinate state AT FIT TIME.  ``shipped_partners``
    is that same state in the artifact that actually ships.  ``fit_menu``, when the
    partner was chosen from a discrete grid, is that grid -- it is what separates
    interpolated staleness from extrapolation.

    Returns a dict carrying the verdict, the drifted keys, and -- when a menu is
    supplied -- whether the shipped partner escaped the fitted hull.  Never raises
    on a missing context; that is a verdict, not an error.
    """
    if not fitted_against:
        return {
            "coefficient": coefficient,
            "verdict": UNDETERMINED_NO_CONTEXT,
            "sufficient_for_verdict": False,
            "insufficiency_reason": (
                "no fit context recorded -- the solver did not stamp which partner "
                "values this coefficient was solved against, so freshness is "
                "UNKNOWABLE from the artifact. Absence is not FRESH."
            ),
            "drifted": (),
            "outside_fit_hull": None,
        }

    drifted = tuple(
        k for k, v in fitted_against.items()
        if k in shipped_partners and abs(float(shipped_partners[k]) - float(v)) > tol
    )
    if not drifted:
        return {
            "coefficient": coefficient,
            "verdict": FRESH,
            "sufficient_for_verdict": True,
            "insufficiency_reason": None,
            "drifted": (),
            "outside_fit_hull": False,
        }

    outside = None
    if fit_menu:
        key = partner_key if partner_key is not None else drifted[0]
        if key in shipped_partners:
            lo, hi = min(fit_menu), max(fit_menu)
            v = float(shipped_partners[key])
            outside = bool(v < lo - tol or v > hi + tol)

    return {
        "coefficient": coefficient,
        "verdict": STALE_EXTRAPOLATED if outside else STALE_INTERPOLATED,
        "sufficient_for_verdict": True,
        "insufficiency_reason": None,
        "drifted": drifted,
        "outside_fit_hull": outside,
        # EXTRAPOLATED is the priority class: the coefficient is applied at a
        # partner value the solve never sampled, so its error is unbounded by the
        # fit, not merely interpolated within it.
    }


def census(rows: Sequence[Mapping[str, Any]], **kw: Any) -> dict[str, int]:
    """Aggregate ``fit_staleness`` over an artifact's rows -- the tool-side report.

    Emits every verdict as its own count so an empty or context-free population is
    visible as such rather than collapsing into a clean-looking zero.
    """
    out = {FRESH: 0, STALE_INTERPOLATED: 0, STALE_EXTRAPOLATED: 0,
           UNDETERMINED_NO_CONTEXT: 0, "rows": len(rows)}
    for r in rows:
        v = fit_staleness(
            coefficient=kw.get("coefficient", "?"),
            fitted_against=r.get("fitted_against"),
            shipped_partners=r.get("shipped_partners", {}),
            fit_menu=kw.get("fit_menu"),
            partner_key=kw.get("partner_key"),
        )["verdict"]
        out[v] += 1
    return out
