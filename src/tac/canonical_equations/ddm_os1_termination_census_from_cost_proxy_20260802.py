# SPDX-License-Identifier: MIT
"""Canonical law: a damped-GN solver's TERMINATION CENSUS is reconstructible, at ZERO
new compute, from a cost proxy the solve already wrote down (ddm_os1).

``ddm_sv1`` established the genus -- a capped numeric solver that stops on a BOUND while
its receipt reads like a completed solve -- and cured one site by EMITTING ``stop_reason``.
Emission fixes the FUTURE.  This law fixes the PAST: wherever a receipt recorded a
forward/evaluation COUNT, the census can be recovered retroactively without re-running
anything, because the loop's cost is an exact affine function of how it terminated.

THE MODEL (the shape at ``experiments/ddm_pfs1_ep_warp_pose_solve.py:183``,
``experiments/ddm_p3v2_optimal_form_pose_resolve.py:212`` and the pre-cure
``ddm_v4c_resolve`` loop kept verbatim at
``src/tac/tests/test_ddm_sv1_ab_gn_termination.py:189``)::

    n = init + fd * R + sum_{i=1..R} L_i

with ``R`` relinearizations entered (``1 <= R <= relin_bound``), ``fd`` forwards per
relinearization for the finite-difference Jacobian, and ``L_i`` line-search evaluations
inside relin ``i``.  The damping ladder gives ``L_max = ladder_levels *
line_search_points``.  Two structural facts make the census decidable:

* an ACCEPTING relin costs ``1 <= L_i <= L_max`` (it breaks out on the first improvement);
* the FINAL relin of a ladder-exhausted solve costs EXACTLY ``L_max`` -- every level and
  every line-search point was tried and none improved.  That is what "damp_cap" MEANS.

So each terminal state occupies a KNOWN interval in ``n``, and an observed ``n`` admits
only the states whose interval contains it.  When exactly one does, the termination is
PROVED from a number already on disk.

WHY THIS IS NOT THE SAME QUESTION AS "IS THERE A CRITERION" -- a MEASURED NEGATIVE that
bounds the whole approach and is recorded rather than patched away.  ddm_os1 first built
a static classifier over loop SHAPE (does a non-bound exit exist?) and ran it against the
only pair of loops with two-sided ground truth: sv1's pre-cure copy (converged provably
unreachable, 0/60) and its post-cure replacement (criterion present).  The classifier
returned the SAME verdict for both -- 1 true positive, 1 false positive, discrimination
ZERO.  The reason is structural, not a tuning failure: the exit CONDITION is
byte-identical across the cure (``if not accepted: break``); sv1's criterion lives in the
stop-reason ASSIGNMENT inside the break body.  A convergence CRITERION and a convergence
LABEL are different objects and the loop's exit shape carries neither.  Loop-shape
analysis therefore CANNOT rank these sites; a cost-proxy census can, because it reads
what the solve DID rather than what it could have done.

MEASURED (ddm_os1, 2026-08-02), n600, on the LIVE v4d pose chain -- ``ddm_v4c_resolve.py``
imports ``ddm_pfs1_ep_warp_pose_solve`` at ``:61`` and consumes its D2 solve at ``:68``, so
this solve produces the STARTING POINT that v4c's rung-B (a,b) GN then refines.  Receipt
``/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d2/d2_ep_solve.partial.jsonl``, 600 rows.
The producing revision is ``8eb3d14594`` (identical to HEAD), whose cost accounting is
``n = 2 + 7R + Sum L_i`` -- ``init=2`` (one initial ``pose6_of`` plus one trailing
``d_pose_shipped``), ``fd=7`` (six pose FD columns plus the ``s_t`` column), ladder 4,
line-search 2:

* **converged 0 / 600 = 0.0%** -- EXACT and MODEL-INDEPENDENT: the solve's only genuine
  criterion is ``cur < 1e-6``, no pair's recorded ``d_pose_solved`` is below it, and the
  closest pair sits 15.5x above it.  This is the load-bearing reading.
* **stopped on a BOUND at least 512 / 600 = 85.3%**, carrying 66.8% of d_pose mass
  (222 provably ladder-exhausted, 290 ladder-OR-relin -- either way a bound).
* **88 / 600 = 14.7% of rows (33.2% of mass) are INFEASIBLE under the model, so the law
  REFUSES: ``sufficient_for_verdict=False``.**  48 of them record fewer than the 17
  forwards a ladder-exhausted single relinearization costs, which is the signature of the
  ``LinAlgError`` break shortening ``L_i`` -- exactly the limit named below.  The bound
  fraction is therefore a LOWER BOUND, not a census.
* A second receipt from the same solver, ``ddm_ps1_20260730/ps1_ladder.partial.jsonl``
  (``relin_bound=3`` INFERRED -- the receipt does not record its config): converged
  **0 / 600** again, bound at least 459 / 600 = 76.5%, 141 infeasible.

CORRECTION, recorded rather than quietly rewritten (2026-08-02, same session).  The first
version of this law was anchored with ``init=1, fd=6`` -- read off the WORKING TREE, which
carried an uncommitted sibling rewrite of the solver -- instead of the revision that
produced the receipt.  Under those wrong parameters the census read "600/600 bound-stopped,
0 infeasible", and the zero was cited as a positive control on the model.  It was an
artifact of wrong parameters coincidentally fitting.  With the correct parameters the
infeasible bucket fires and the law refuses.  **The instrument behaved as designed** -- it
declined to emit a confident census on a shape it could not explain -- and the episode is
the sharpest available argument for the ``n_infeasible`` guard.  What survives unchanged is
the ``converged = 0/600`` reading, because it never depended on the cost model.

So sv1's (a,b) finding -- 0% converged -- reproduces on the SIX-parameter pose GN at 600
pairs instead of 60, for ZERO scorer evaluations where sv1 spent 1,385.  The companion
"100% bound-stopped" does NOT reproduce at full strength here: 85.3% is proved and 14.7%
is undetermined pending a singular-aware model.

SHARPEST FORM OF THE DEFECT AT THAT SITE: the two exits are FUSED into a single
condition, ``if not accepted or cur < 1e-6: break`` (``:212``).  Even a caller who
inspected the source cannot tell convergence from ladder-exhaustion from the receipt,
because both write the same absence.  Fusing a criterion and a bound into one predicate
destroys the census at the point of writing, not at the point of reading.

REACH, with the denominator (this is the honest limit).  0 of 21,700 ``.omx/research``
JSON receipts and 19 of 8,204 SSD receipts (0.23%) carry any iteration/evaluation count.
The method applies wherever the proxy exists and NOWHERE ELSE -- which is precisely why
sv1 had to spend scorer evaluations to answer the same question on a site that recorded
none.  The cheap cure is therefore not this law but the habit it depends on: record the
cost proxy, and the census stays recoverable forever.

Sister of ``tac.canonical_equations.ddm_pw1_menu_saturation_discriminator_v1`` (occupancy
of a discrete MENU) -- that law reads WHICH VALUES were reachable, this one reads WHETHER
THE SEARCH FINISHED.  Both are zero-cost readings of data a solve already produced, and
both REFUSE rather than guess when the evidence is insufficient.

Receipt: .omx/research/ddm_os1_optimization_sweep_termination_census_20260802.md
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.canonical_equations.evaluators import register_evaluator
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "ddm_os1_termination_census_from_cost_proxy_v1"
REPO = Path(__file__).resolve().parents[3]
RECEIPT = REPO / ".omx/research/ddm_os1_optimization_sweep_termination_census_20260802.md"

#: terminal states.  BOUND_EITHER is not a hedge -- it is the honest reading when both
#: bound states are feasible, and it is still a BOUND (never a convergence).
CONVERGED = "converged"
BOUND_LADDER = "bound_ladder_exhausted"
BOUND_RELIN = "bound_relin_exhausted"
BOUND_EITHER = "bound_either"
INFEASIBLE = "infeasible_under_model"

#: measured anchor -- ddm_pfs1 D2 solve, n600, live v4d pose chain.
#: Shape read from the RECEIPT-PRODUCING revision 8eb3d14594 (== HEAD), not the working
#: tree: init=2 (initial pose6_of + trailing d_pose_shipped), fd=7 (6 pose FD columns +
#: the s_t column). See the CORRECTION paragraph in the module docstring.
PFS1_PAIRS = 600
PFS1_RELINS = 4
PFS1_INIT_COST = 2
PFS1_FD_PER_RELIN = 7
PFS1_LADDER_LEVELS = 4
PFS1_LINE_SEARCH_POINTS = 2
PFS1_TOLERANCE = 1e-6
#: model-INDEPENDENT: read from the objective, not the cost
PFS1_CONVERGED = 0
#: LOWER BOUND -- 88 rows are infeasible under the model, so the law refuses
PFS1_BOUND_LOWER_BOUND = 512
PFS1_PROVABLY_LADDER = 222
PFS1_INFEASIBLE = 88
PFS1_BOUND_MASS_FRACTION = 0.668
PFS1_INFEASIBLE_MASS_FRACTION = 0.332


def _feasible_states(n: int, *, relin_bound: int, fd: int, l_max: int, init: int) -> set[str]:
    """Terminal states whose exact cost interval contains ``n``.

    Ladder-exhausted at relin ``R``: ``init + fd*R + L_max`` for the failing relin, plus
    ``S`` over the ``R-1`` accepting ones with ``S in [R-1, (R-1)*L_max]``.
    Relin-exhausted: ``R == relin_bound`` and every relin accepted, so
    ``S in [relin_bound, relin_bound*L_max]``.
    """
    out: set[str] = set()
    for r in range(1, relin_bound + 1):
        base = init + fd * r + l_max
        if base + (r - 1) <= n <= base + (r - 1) * l_max:
            out.add(BOUND_LADDER)
            break
    lo = init + fd * relin_bound + relin_bound
    hi = init + fd * relin_bound + relin_bound * l_max
    if lo <= n <= hi:
        out.add(BOUND_RELIN)
    return out


def termination_census(
    cost_counts: Sequence[int],
    *,
    relin_bound: int,
    fd_per_relin: int,
    ladder_levels: int,
    line_search_points: int,
    init_cost: int = 1,
    objective: Sequence[float] | None = None,
    tolerance: float | None = None,
) -> dict[str, Any]:
    """Reconstruct a damped-GN termination census from recorded per-solve cost counts.

    ``cost_counts[i]`` is the forward/evaluation count the solve recorded for item ``i``
    (e.g. ``n_forwards``).  ``objective[i]`` is that item's final objective value; supply
    it WITH ``tolerance`` to decide the convergence leg -- without both, convergence is
    UNDECIDABLE and is reported as such rather than assumed absent.

    Returns per-item states, the aggregate census by count and by objective mass, and
    ``sufficient_for_verdict``.  A ``bound_*`` verdict is a MEASUREMENT REQUEST -- free
    the bound and re-measure -- never a claim that freeing it will pay.

    MODEL LIMIT, stated because it is the one that can bite: a ``LinAlgError`` break
    inside the ladder shortens ``L_i`` below the model's floor.  It cannot manufacture a
    false ``converged`` (convergence is read from the objective, never from the cost),
    but it CAN make a genuinely-ladder-exhausted row read INFEASIBLE.  So a nonzero
    ``n_infeasible`` means the model is wrong for this solver, not that the solves were
    strange -- check for singular handling before reading the census.  ``n_infeasible ==
    0`` is the model's own positive control.
    """
    n_items = len(cost_counts)
    if n_items == 0:
        return {
            "verdict": "UNDETERMINED_EMPTY",
            "n_items": 0,
            "note": "no solves recorded — VACUOUS scope, never a clean bill",
            "sufficient_for_verdict": False,
            "insufficiency_reason": "empty_population_report_the_denominator",
        }
    for name, v in (("relin_bound", relin_bound), ("fd_per_relin", fd_per_relin),
                    ("ladder_levels", ladder_levels),
                    ("line_search_points", line_search_points)):
        if int(v) < 1:
            raise ValueError(f"{name} must be >= 1; got {v!r}")
    if init_cost < 0:
        raise ValueError("init_cost must be non-negative")
    if (objective is None) != (tolerance is None):
        raise ValueError(
            "objective and tolerance must be supplied together — a tolerance without "
            "the objective it is compared against cannot decide convergence"
        )
    if objective is not None and len(objective) != n_items:
        raise ValueError("objective must align 1:1 with cost_counts")
    if objective is not None and not all(math.isfinite(float(x)) for x in objective):
        # A NaN/inf objective is a diverged solve, not a datum. Absorbing it would
        # poison every mass fraction silently; refuse and make the caller decide.
        raise ValueError(
            "objective contains a non-finite entry — a diverged solve must be excluded "
            "or repaired explicitly, never averaged into the mass fractions"
        )

    l_max = int(ladder_levels) * int(line_search_points)
    states: list[str] = []
    for i, raw in enumerate(cost_counts):
        n = int(raw)
        if objective is not None and tolerance is not None:
            val = float(objective[i])
            if math.isfinite(val) and val < float(tolerance):
                states.append(CONVERGED)
                continue
        feas = _feasible_states(n, relin_bound=int(relin_bound), fd=int(fd_per_relin),
                                l_max=l_max, init=int(init_cost))
        if not feas:
            states.append(INFEASIBLE)
        elif len(feas) == 1:
            states.append(next(iter(feas)))
        else:
            states.append(BOUND_EITHER)

    mass = [float(x) for x in objective] if objective is not None else None
    mass_total = sum(mass) if mass else 0.0

    def _agg(pred) -> dict[str, Any]:
        idx = [i for i in range(n_items) if pred(states[i])]
        # the mass key is ALWAYS present — None when undecidable — so a consumer that
        # ranks by mass fails loudly on absence instead of KeyError-ing on some inputs
        # and silently succeeding on others.
        share = (sum(mass[i] for i in idx) / mass_total
                 if mass and mass_total > 0.0 else None)
        return {"count": len(idx), "fraction": len(idx) / n_items,
                "objective_mass_fraction": share}

    bound_states = {BOUND_LADDER, BOUND_RELIN, BOUND_EITHER}
    census = {
        CONVERGED: _agg(lambda s: s == CONVERGED),
        BOUND_LADDER: _agg(lambda s: s == BOUND_LADDER),
        BOUND_RELIN: _agg(lambda s: s == BOUND_RELIN),
        BOUND_EITHER: _agg(lambda s: s == BOUND_EITHER),
        INFEASIBLE: _agg(lambda s: s == INFEASIBLE),
        "stopped_on_a_bound": _agg(lambda s: s in bound_states),
    }
    n_infeasible = census[INFEASIBLE]["count"]
    convergence_decidable = objective is not None
    sufficient = n_infeasible == 0 and convergence_decidable
    reason = None
    if not convergence_decidable:
        reason = "no_objective_and_tolerance_convergence_leg_undecidable"
    elif n_infeasible:
        reason = "rows_infeasible_under_model_check_singular_step_handling"

    return {
        "verdict": (
            "ALL_STOPPED_ON_A_BOUND"
            if census["stopped_on_a_bound"]["count"] == n_items and n_items
            else "MIXED" if census[CONVERGED]["count"] else "UNDETERMINED"
        ),
        "n_items": n_items,
        "states": states,
        "census": census,
        "l_max_per_relin": l_max,
        "max_possible_cost": int(init_cost) + int(fd_per_relin) * int(relin_bound)
        + int(relin_bound) * l_max,
        "convergence_decidable": convergence_decidable,
        "verdict_is_a_measurement_request": census["stopped_on_a_bound"]["count"] > 0,
        "sufficient_for_verdict": bool(sufficient),
        "insufficiency_reason": reason,
    }


def _evaluate(inputs: Mapping[str, Any]) -> dict[str, Any]:
    required = {"cost_counts", "relin_bound", "fd_per_relin", "ladder_levels",
                "line_search_points"}
    allowed = required | {"init_cost", "objective", "tolerance"}
    keys = set(inputs)
    if not required <= keys or not keys <= allowed:
        raise ValueError(
            "termination-census inputs differ from the canonical callable contract "
            "(required: cost_counts, relin_bound, fd_per_relin, ladder_levels, "
            "line_search_points; optional: init_cost, objective, tolerance)"
        )
    return termination_census(
        inputs["cost_counts"],
        relin_bound=inputs["relin_bound"],
        fd_per_relin=inputs["fd_per_relin"],
        ladder_levels=inputs["ladder_levels"],
        line_search_points=inputs["line_search_points"],
        init_cost=inputs.get("init_cost", 1),
        objective=inputs.get("objective"),
        tolerance=inputs.get("tolerance"),
    )


register_evaluator(EQUATION_ID, _evaluate)


def build_ddm_os1_termination_census_from_cost_proxy_v1(
    *,
    source_receipt: Path = RECEIPT,
) -> CanonicalEquation:
    """Build the retroactive termination-census law with its measured n600 anchor."""

    provenance = build_provenance_for_research_sidecar(
        source_receipt,
        reactivation_criteria=(
            "Re-anchor if an instrumented re-run of ddm_pfs1 solve_pair_gn reports a "
            "stop-reason census materially different from the reconstruction here "
            "(that would falsify the cost model, not the genus), if a solver in this "
            "family is found to break the ladder early on a singular step often enough "
            "to produce infeasible rows, or if the loop shape changes so that a failing "
            "relin no longer costs exactly ladder_levels*line_search_points. The law is "
            "solver-shape-specific by construction; the SHAPE is the assumption to check."
        ),
        measurement_axis="[macOS-CPU frozen-PoseNet advisory]",
        hardware_substrate="darwin_arm64_cpu_reconstruction_from_existing_receipt",
        captured_at_utc="2026-08-02T00:00:00Z",
    )
    anchor = EmpiricalAnchor(
        anchor_id="ddm_os1_pfs1_d2_termination_census_n600_20260802",
        measurement_utc="2026-08-02T00:00:00Z",
        inputs={
            "receipt": "/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d2/"
                       "d2_ep_solve.partial.jsonl",
            "site": "experiments/ddm_pfs1_ep_warp_pose_solve.py:183 solve_pair_gn",
            "on_live_chain": "ddm_v4c_resolve.py:61 imports it; :68 consumes the D2 solve",
            "population": PFS1_PAIRS,
            "receipt_producing_revision": "8eb3d14594 (identical to HEAD)",
            "relin_bound": PFS1_RELINS,
            "init_cost": PFS1_INIT_COST,
            "fd_per_relin": PFS1_FD_PER_RELIN,
            "ladder_levels": PFS1_LADDER_LEVELS,
            "line_search_points": PFS1_LINE_SEARCH_POINTS,
            "tolerance": PFS1_TOLERANCE,
            "new_scorer_evaluations": 0,
        },
        predicted_output={
            # sv1 measured 0% converged / 100% bound-stopped on the 2-parameter (a,b) GN
            # over 60 pairs; the prediction carried here is that the same genus holds on
            # the 6-parameter pose GN at n600.
            "converged_fraction": 0.0,
            "bound_stopped_fraction": 1.0,
        },
        empirical_output={
            # model-INDEPENDENT, the load-bearing reading
            "converged": PFS1_CONVERGED,
            "converged_fraction": 0.0,
            "closest_pair_multiple_of_tolerance": 15.5,
            # LOWER BOUND: the law refuses because 88 rows are infeasible
            "stopped_on_a_bound_at_least": PFS1_BOUND_LOWER_BOUND,
            "stopped_on_a_bound_fraction_at_least": PFS1_BOUND_LOWER_BOUND / PFS1_PAIRS,
            "stopped_on_a_bound_mass_fraction_at_least": PFS1_BOUND_MASS_FRACTION,
            "provably_ladder_exhausted": PFS1_PROVABLY_LADDER,
            "n_infeasible": PFS1_INFEASIBLE,
            "n_infeasible_mass_fraction": PFS1_INFEASIBLE_MASS_FRACTION,
            "sufficient_for_verdict": False,
            "insufficiency_reason": "rows_infeasible_under_model_check_singular_step_handling",
            "sister_receipt_ps1_converged": 0,
            "sister_receipt_ps1_bound_at_least": 459,
            "cost_proxy_receipt_reach_omx_research": "0 of 21700",
            "cost_proxy_receipt_reach_ssd": "19 of 8204",
        },
        # The converged leg matched the prediction exactly (0.0 vs 0.0). The bound leg did
        # NOT: predicted 1.0, proved 0.853 with 0.147 undetermined. Residual is that gap,
        # recorded rather than rounded away.
        residual=1.0 - PFS1_BOUND_LOWER_BOUND / PFS1_PAIRS,
        source_artifact=(
            "/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d2/d2_ep_solve.partial.jsonl"
        ),
        measurement_method=(
            "Retroactive reconstruction from an existing receipt; ZERO new scorer "
            "evaluations. score_claim=false, promotable=false; the exact contest pointer "
            "is UNMOVED. The converged=0/600 leg is EXACT — the objective is read against "
            "the literal 1e-6 tolerance — and is NOT inferred from the cost model; only "
            "the ladder-vs-relin split relies on the cost intervals. 0 infeasible rows is "
            "the model's own positive control."
        ),
        provenance=provenance,
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Retroactive termination census of a damped-GN solve from a recorded cost proxy (ddm_os1)",
        one_line_summary=(
            "Damped-GN terminal state occupies a known interval in its forward count, so a "
            "census is recoverable from an old receipt at zero cost. Measured n600: "
            "converged 0/600 exact; bound >= 512/600."
        ),
        latex_form=(
            r"n=\iota+\phi R+\sum_{i=1}^{R}L_i,\quad L_{\max}=\lambda\pi,\quad "
            r"\text{ladder-exhausted at }R:\ n\in[\iota+\phi R+L_{\max}+(R-1),\ "
            r"\iota+\phi R+L_{\max}+(R-1)L_{\max}],\quad "
            r"\text{relin-exhausted}:\ n\in[\iota+\phi R_b+R_b,\ \iota+\phi R_b+R_bL_{\max}]"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.ddm_os1_termination_census_from_cost_proxy_20260802"
            ":termination_census"
        ),
        domain_of_validity={
            "object": "a relinearized damped Gauss-Newton (Levenberg-Marquardt-shaped) "
                      "solve with a finite-difference Jacobian, a multiplicative damping "
                      "ladder and a fixed line-search point set, run once per item",
            "requires": "a per-item recorded forward/evaluation count; optionally the "
                        "final objective and its tolerance to decide the convergence leg",
            "evidence_axis": "[macOS-CPU frozen-PoseNet advisory] on the anchor; the "
                             "RECONSTRUCTION itself is axis-free (it reads counts)",
            "research_only": True,
            "score_claim": False,
            "verdict_scope": "FORMULATION_DAMPED_GN_TERMINATION_CENSUS",
            "excluded": [
                "a claim that freeing a bound will pay — a bound_* verdict is a "
                "measurement request, exactly as in the sister pw1 law",
                "solvers whose failing relinearization does NOT cost exactly L_max "
                "(early singular break, adaptive ladders, wall-clock cuts) — those "
                "produce infeasible rows, which is the model refusing, not a census",
                "loop-SHAPE inference of whether a criterion exists — MEASURED "
                "discrimination zero; use this cost reading instead",
                "any promotion or submission use",
            ],
        },
        canonical_producers=(
            "experiments/ddm_pfs1_ep_warp_pose_solve.py",
            "experiments/ddm_ps1_pose_stage.py",
        ),
        # Consumers are surfaces that actually CALL this law. ddm_v4c_resolve consumes
        # pfs1's SOLVE OUTPUT, not this equation, and listing it here would be a claim
        # the code does not honour.
        canonical_consumers=(
            "tools/os1_termination_census_report.py",
            "src/tac/tests/test_ddm_os1_termination_census.py",
        ),
        empirical_anchors=(anchor,),
        units_in={
            "cost_counts": "forward/evaluation counts per solved item (dimensionless ints)",
            "relin_bound": "relinearization cap (dimensionless int)",
            "fd_per_relin": "forwards per Jacobian column sweep (dimensionless int)",
            "ladder_levels": "damping ladder length (dimensionless int)",
            "line_search_points": "line-search points per damping level (dimensionless int)",
            "objective": "final objective value per item (objective units, optional)",
            "tolerance": "convergence tolerance in objective units (optional)",
        },
        units_out={
            "verdict": "enum {ALL_STOPPED_ON_A_BOUND, MIXED, UNDETERMINED, "
                       "UNDETERMINED_EMPTY}",
            "census": "per-state counts, count fractions in [0,1], and objective-mass "
                      "fractions in [0,1]",
            "states": "per-item enum {converged, bound_ladder_exhausted, "
                      "bound_relin_exhausted, bound_either, infeasible_under_model}",
        },
        predicted_vs_empirical_residual={
            "converged_fraction_absolute": 0.0,
            "bound_stopped_fraction_absolute": 1.0 - PFS1_BOUND_LOWER_BOUND / PFS1_PAIRS,
        },
        last_calibration_utc="2026-08-02T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        provenance=provenance,
    )
