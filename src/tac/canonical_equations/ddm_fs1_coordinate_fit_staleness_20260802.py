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
STALE_FOREIGN_PARTNER = "STALE_FOREIGN_PARTNER"
UNDETERMINED_NO_CONTEXT = "UNDETERMINED_NO_CONTEXT"

#: The canonical key a producer stamps its fit context under.  One name, so a
#: consumer can find the context without knowing which solver wrote the row.
FIT_CONTEXT_KEY = "fitted_against"


class StaleFitContextError(RuntimeError):
    """A solved value was consumed without a usable fit context.

    Raised by :func:`require_fit_context`.  It exists so that an UNSTAMPED
    coefficient REFUSES at the point of consumption instead of silently reading
    as fresh -- the failure mode this whole module was written against.
    """


def _is_numeric(v: Any) -> bool:
    """Numeric partners admit a hull; identity partners do not.

    ``bool`` is excluded deliberately: ``True`` is a categorical partner that
    happens to be a Python int, and comparing it on a tolerance would silently
    place a flag inside a numeric hull it has no business in.
    """
    return isinstance(v, (int, float)) and not isinstance(v, bool)

#: Measured anchor: the v4c photometric grid the shipped (a,b) was fitted against.
V4C_AB_FIT_MENU: tuple[float, ...] = (0.0, 0.5, 1.0)

#: Measured 2026-08-02 over ddm_v4d_20260731/mq1_emit/final_mq1.jsonl (600 pairs).
#: SUPERSEDED IN PLACE by ddm_ft1 the same day.  BOTH original counts were
#: UNDER-counts, and each miss is a distinct blind spot in the FIRST version of
#: this module -- errors of SCOPE, not of measurement, so the originals are kept
#: as labelled history rather than deleted:
#:   * stale_set 244 -> 267.  The original counted the BETA partner ONLY.  ft1
#:     measured that pw1 moved the POSE on 89 rows and mq1 moved p1/p2 on 128
#:     more, and neither re-solved (a,b) (zero rows changed).  So 178 pairs ship
#:     an (a,b) fitted against a pose THAT NO LONGER EXISTS -- a second drifted
#:     partner a beta-only census structurally cannot see.
#:   * extrapolated 59 -> 101 = 38 magnitude-escape + 63 SIGN-escape.  At fit
#:     time the sign was PINNED TO YAW (beta = g*yaw_sign, g >= 0), so an
#:     opposing-sign shipment is a direction the fit never sampled while sitting
#:     INSIDE [min(menu), max(menu)].  A magnitude-only min/max hull cannot see
#:     it; that is why ``fit_staleness`` takes ``fitted_sign``.
V4C_AB_STALENESS_CENSUS: dict[str, int] = {
    "pairs": 600,
    "beta_nonzero": 248,
    "ab_non_identity": 592,
    "stale_set": 267,           # ANY drifted partner (beta OR pose), per ft1
    "stale_beta_only": 244,     # the original beta-only count, retained as history
    "stale_pose_partner": 178,  # (a,b) fitted against a pose pw1/mq1 later moved
    "extrapolated": 101,        # 38 magnitude-escape + 63 SIGN-escape
    "extrapolated_magnitude_only": 59,  # the original count, retained as history
}


def fit_staleness(
    *,
    coefficient: str,
    fitted_against: Mapping[str, float] | None,
    shipped_partners: Mapping[str, float],
    fit_menu: Sequence[float] | None = None,
    partner_key: str | None = None,
    fitted_sign: float | None = None,
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
            "drifted_numeric": (),
            "drifted_identity": (),
            "outside_fit_hull": None,
            "transport_admissible": False,
        }

    # Two partner kinds, two comparison laws.  A NUMERIC partner (a beta, a pose
    # dim) drifts by a distance and can be inside or outside a fitted hull.  An
    # IDENTITY partner (a base archive, a vehicle, a checkpoint, a scorer
    # snapshot) has no hull at all -- it either is the thing that was fitted
    # against or it is a foreign one.  Float-casting the second kind is how the
    # pose/celldrop50 and C1/TR1 instances stayed unrepresentable.
    drifted_numeric: list[str] = []
    drifted_identity: list[str] = []
    for k, v in fitted_against.items():
        if k not in shipped_partners:
            continue
        s = shipped_partners[k]
        if _is_numeric(v) and _is_numeric(s):
            if abs(float(s) - float(v)) > tol:
                drifted_numeric.append(k)
        elif s != v:
            drifted_identity.append(k)
    drifted = tuple(drifted_numeric) + tuple(drifted_identity)
    if not drifted:
        return {
            "coefficient": coefficient,
            "verdict": FRESH,
            "sufficient_for_verdict": True,
            "insufficiency_reason": None,
            "drifted": (),
            "drifted_numeric": (),
            "drifted_identity": (),
            "outside_fit_hull": False,
            "transport_admissible": True,
        }

    outside = None
    if fit_menu:
        key = partner_key if partner_key is not None else (
            drifted_numeric[0] if drifted_numeric else None)
        if key is not None and key in shipped_partners and _is_numeric(
                shipped_partners[key]):
            lo, hi = min(fit_menu), max(fit_menu)
            v = float(shipped_partners[key])
            outside = bool(v < lo - tol or v > hi + tol)
            # SIGN ESCAPE.  A min/max hull is a MAGNITUDE test.  When the menu
            # was sampled at a pinned sign -- v4c pinned beta to the pose yaw,
            # beta = g*yaw_sign with g >= 0 -- an OPPOSING sign is a direction
            # the fit never sampled even though |v| sits inside [lo, hi].
            # MEASURED: 63 such pairs, invisible to the magnitude test alone,
            # which is why the first census under-counted 59 -> 101.
            if (fitted_sign is not None and abs(v) > tol
                    and v * float(fitted_sign) < 0.0):
                outside = True

    if drifted_identity:
        # FOREIGN outranks both numeric classes and is reported first: the base
        # or vehicle the value was solved against is not the one it ships with,
        # so there is no fitted hull to be inside of and no Taylor step home.
        # The only cure is a re-solve on the live partner.  This is the
        # pose-solved-on-celldrop50-applied-to-ep854 class and the
        # C1-lattice-as-teacher-for-a-TR1-student class.
        verdict = STALE_FOREIGN_PARTNER
    else:
        # EXTRAPOLATED is the priority numeric class: the coefficient is applied
        # at a partner value the solve never sampled, so its error is unbounded
        # by the fit, not merely interpolated within it.
        verdict = STALE_EXTRAPOLATED if outside else STALE_INTERPOLATED

    return {
        "coefficient": coefficient,
        "verdict": verdict,
        "sufficient_for_verdict": True,
        "insufficiency_reason": None,
        "drifted": drifted,
        "drifted_numeric": tuple(drifted_numeric),
        "drifted_identity": tuple(drifted_identity),
        "outside_fit_hull": outside,
        "transport_admissible": bool(not drifted_identity and outside is False),
    }


def stamp_fit_context(
    *,
    coefficient: str,
    partners: Mapping[str, Any],
    base: str | None = None,
    vehicle: str | None = None,
    fit_menu: Sequence[float] | None = None,
    fit_sign: float | None = None,
) -> dict[str, Any]:
    """Build the fit context a producer embeds under :data:`FIT_CONTEXT_KEY`.

    This is THE convention.  A solver that has just solved ``coefficient`` calls
    this with the co-coordinate state it solved AT -- every partner it held
    fixed -- plus the ``base``/``vehicle`` identity it solved against, and writes
    the result into the row it emits.  Cost: one dict per row.  What it buys: the
    freshness question becomes answerable from the artifact alone, instead of by
    hand-archaeology across arms after the partner has already moved.

    ``base`` and ``vehicle`` are folded in as ordinary identity partners so a
    single comparison path covers all three measured drift kinds (co-coordinate,
    base archive, vehicle).  Stamping a partner you did NOT hold fixed is worse
    than stamping nothing: it manufactures a freshness claim.  Stamp exactly the
    state the solve actually saw.
    """
    if not coefficient:
        raise ValueError("coefficient must be a non-empty name")
    if not partners and base is None and vehicle is None:
        # A stamp with nothing in it would read as FRESH against any shipment --
        # strictly worse than no stamp, which at least reads UNDETERMINED.
        raise ValueError(
            "refusing to stamp an empty fit context: it would certify freshness "
            "against partners that were never recorded. Pass the co-coordinate "
            "state, or a base/vehicle identity, or do not stamp."
        )
    ctx: dict[str, Any] = {str(k): v for k, v in partners.items()}
    if base is not None:
        ctx["base"] = str(base)
    if vehicle is not None:
        ctx["vehicle"] = str(vehicle)
    ctx["_coefficient"] = str(coefficient)
    if fit_menu is not None:
        ctx["_fit_menu"] = [float(x) for x in fit_menu]
    if fit_sign is not None:
        # The menu was sampled at ONE sign (v4c pinned beta to the pose yaw).
        # Carrying it is what lets a consumer see a sign escape; a magnitude
        # menu alone silently certifies the 63 opposing-sign pairs as in-hull.
        ctx["_fit_sign"] = float(fit_sign)
    return ctx


def require_fit_context(
    row: Mapping[str, Any],
    *,
    coefficient: str,
    shipped_partners: Mapping[str, Any],
    fit_menu: Sequence[float] | None = None,
    partner_key: str | None = None,
    allow_stale: bool = True,
) -> dict[str, Any]:
    """Read a solved value's freshness, REFUSING when the row is unstamped.

    This is the fail-closed consumption path.  :func:`fit_staleness` reports
    ``UNDETERMINED_NO_CONTEXT`` as a verdict, which a caller is free to ignore;
    this wrapper turns that same state into a raise, so a solved value cannot be
    consumed as if fresh merely because nobody recorded what it was fitted to.

    ``allow_stale`` defaults True because a KNOWN stale value is a ranked, curable
    finding -- ft1's whole priority ordering operates on them.  Set it False at a
    boundary that may only accept fresh coefficients (a shipping build).
    """
    ctx = row.get(FIT_CONTEXT_KEY)
    menu, sign = fit_menu, None
    if isinstance(ctx, Mapping):
        if menu is None:
            m = ctx.get("_fit_menu")
            if isinstance(m, Sequence) and not isinstance(m, (str, bytes)):
                menu = [float(x) for x in m]
        s = ctx.get("_fit_sign")
        if _is_numeric(s):
            sign = float(s)
    clean = ({k: v for k, v in ctx.items() if not k.startswith("_")}
             if isinstance(ctx, Mapping) else None)
    verdict = fit_staleness(
        coefficient=coefficient,
        fitted_against=clean,
        shipped_partners=shipped_partners,
        fit_menu=menu,
        partner_key=partner_key,
        fitted_sign=sign,
    )
    if verdict["verdict"] == UNDETERMINED_NO_CONTEXT:
        raise StaleFitContextError(
            f"{coefficient!r} carries no {FIT_CONTEXT_KEY!r}: which partner values "
            f"it was solved against is UNKNOWABLE from this row, so its freshness "
            f"cannot be asserted. Absence of a fit context is not freshness. "
            f"Stamp it at the producer with stamp_fit_context()."
        )
    if not allow_stale and verdict["verdict"] != FRESH:
        raise StaleFitContextError(
            f"{coefficient!r} is {verdict['verdict']} (drifted={verdict['drifted']}) "
            f"and this boundary requires FRESH."
        )
    return verdict


def census(rows: Sequence[Mapping[str, Any]], **kw: Any) -> dict[str, int]:
    """Aggregate ``fit_staleness`` over an artifact's rows -- the tool-side report.

    Emits every verdict as its own count so an empty or context-free population is
    visible as such rather than collapsing into a clean-looking zero.
    """
    out = {FRESH: 0, STALE_INTERPOLATED: 0, STALE_EXTRAPOLATED: 0,
           STALE_FOREIGN_PARTNER: 0, UNDETERMINED_NO_CONTEXT: 0,
           "rows": len(rows)}
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
