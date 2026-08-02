# SPDX-License-Identifier: MIT
"""Canonical law: the OCCUPANCY-SATURATION discriminator for discrete search menus (ddm_pw1).

A discrete menu -- a bounded search bracket, a fixed magnitude table, a sign forced
from a heuristic, a boolean flag -- is a MODELLING CHOICE that our code has been
routinely presenting as a closed question.  This law makes the question falsifiable
from data the solve already produces, at zero extra cost: the OCCUPANCY HISTOGRAM of
the selected indices.

    Mass piling at a bound  => OUR MENU is clipping the solution (the bound binds).
    Mass strictly interior  => the menu is honestly CLOSED (the optimum is inside).

MEASURED (ddm_pw1, 2026-08-01, n600, [macOS-CPU frozen-PoseNet advisory]), on the LIVE
v4d pose chain -- three menus in ONE run, giving a positive control, a second positive,
and a negative control that shares every other condition:

* ``_refine_dim0`` (bracket +-0.048 coarse / +-0.006 fine).  |move| histogram decays
  ``103, 93, 67, 39, 51, 34, 46, 28, 15`` and then **JUMPS to 124 at the bound** --
  a terminal-to-last-interior ratio of **8.27x** against a decaying interior.
  124/600 pairs = 20.7% by count but **37.4% of pose mass**, because the clipped pairs
  carry **2.3x** the interior mean d_pose.  => SATURATED.
* ``_beta_select`` (table ``(0, 0.5, 1.0)``, sign forced from yaw).  **76 pairs pinned
  at the top entry** = 12.7% by count, **26.4% of mass**.  => SATURATED.
* ``s_t`` (11-point grid).  Occupied indices **6-9 only**; zero at 0-5 AND zero at 10.
  => NOT saturated.  This is the negative control, and it is what makes the finding
  specific rather than the vacuous "all our menus are too small".

REALIZED CONSEQUENCE, exact-evaluated (``upstream/evaluate.py``, n600, archive
360,323 B): replacing both bounds with self-terminating brackets moved the own-vehicle
composed score **S 0.9639878 -> 0.9476091 (dS -0.0163787) at +85 archive bytes** --
entirely on the pose axis (d_pose 0.00858145 -> 0.00764555, -10.91%); d_seg bit-identical
(same tokens).  Byte-closed prediction was 0.9476066: **error +2.5e-06**.

THE SECOND DISCRIMINATOR -- bound-limited vs resolution-limited.  An occupancy spike
alone does not say WHICH.  The attribution does: **97.89% of arm A's realized gain lies
OUTSIDE the shipped search's 0.054 reach**, i.e. unreachable at ANY refinement budget
inside the old bracket.  So the defect was the BOUND, not the step size.  Symmetrically,
arm B's dominant win (29 pairs, 0.2196 d) required BOTH sign freedom AND magnitude
> 1.0 -- so a longer same-sign table would NOT have bought it, and "sign from yaw" was
a binding constraint rather than a free modelling choice.

WHY THE CURE IS NOT A BIGGER CONSTANT.  Replacing a saturated bound with a larger
bound re-poses the same unfalsified question one notch further out.  The measured cure
is a Swann outward bracket that terminates by PROOF -- an accepted step strictly
decreases a quantity bounded below by zero, and the doubling exhausts the f16 range in
``ceil(log2(65504/step0))`` = 23/17 doublings -- costing exactly **2 extra evaluations
on a pair whose bound did not bind**.  Menus that were honestly closed pay ~nothing.

RATE NOTE: both removals were rate-free because the receiver already read its table
from the manifest and applied ``beta_mags[idx] * yaw_sign``; negative and >1.0 entries
needed ZERO receiver change.  The +85 B is the extended table itself.  A saturated menu
is NOT generally free to widen -- the receiver contract decides.

Receipt: .omx/research/ddm_pw1_pose_menu_saturation_20260801.md (commit 5ea9cd3f0a);
exact-eval receipt: the ``stage_v4d_realized_gate.sh cpu pw1`` run, 2026-08-01.
"""

from __future__ import annotations

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

EQUATION_ID = "ddm_pw1_menu_saturation_discriminator_v1"
REPO = Path(__file__).resolve().parents[3]
RECEIPT = REPO / ".omx/research/ddm_pw1_pose_menu_saturation_20260801.md"

#: terminal/last-interior occupancy ratio at or above which a decaying interior is
#: called SATURATED.  Derived, not chosen: the measured separation is 8.27x (dim0,
#: saturated) vs 0.0 (s_t, unsaturated with an EMPTY terminal bin), so any threshold
#: in (0, 8.27] separates the observed classes.  1.0 is the weakest defensible line --
#: "the terminal bin is not smaller than the bin before it, against a decaying trend"
#: -- and is deliberately conservative: it over-reports suspicion rather than missing
#: a clipped menu.  A menu flagged here is a MEASUREMENT REQUEST, never a verdict.
SATURATION_RATIO_THRESHOLD = 1.0

#: measured occupancy histograms, ddm_pw1 n600 (index order; last entry = at-bound)
DIM0_MOVE_HISTOGRAM: tuple[int, ...] = (103, 93, 67, 39, 51, 34, 46, 28, 15, 124)
BETA_PINNED_AT_TOP = 76
BETA_MENU_SIZE = 3
ST_OCCUPIED_INDICES: tuple[int, ...] = (6, 7, 8, 9)
ST_GRID_SIZE = 11

#: exact-eval anchor (upstream/evaluate.py, n600, archive 360,323 B)
S_BEFORE = 0.9639878
S_AFTER = 0.9476091
S_PREDICTED = 0.9476066
BYTES_ADDED = 85
#: fraction of the realized gain lying OUTSIDE the pre-fix bracket's reach
UNREACHABLE_GAIN_FRACTION = 0.9789


def menu_saturation(
    occupancy: Sequence[int],
    *,
    terminal_index: int | None = None,
    objective_mass: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Classify a discrete menu from the occupancy histogram of its selected indices.

    ``occupancy[i]`` = how many times index ``i`` was selected.  ``terminal_index``
    defaults to the last index (the bound); pass it explicitly for a menu whose bound
    is elsewhere (e.g. a signed table bounded at both ends -- call twice).
    ``objective_mass[i]`` = total objective contribution of the items that chose ``i``
    (e.g. summed d_pose).  **Supply it whenever it exists** -- see the measured false
    negative below.

    Returns the terminal mass fraction, the terminal/last-interior ratio, and a verdict
    in {SATURATED_MEASURE_BEYOND_BOUND, CLOSED_INTERIOR_OPTIMUM, UNDETERMINED_EMPTY}.

    A SATURATED verdict is a request for ONE measurement -- free the bound and
    re-measure -- not a claim that freeing it will pay.  Whether the bound or the
    resolution was binding is settled only by the post-free attribution: the fraction
    of realized gain lying outside the old reach (0.9789 in the anchor).

    MEASURED FALSE NEGATIVE of the count-only reading (recorded, not patched away).
    On ``_beta_select`` -- a genuine positive, 26.4% of pose mass pinned at the top of a
    3-entry table -- the count-only ratio reads CLOSED.  Two structural reasons, both
    of which bound this law's scope:

    1. **Short menus have no interior trend.**  With 3 entries there is no decaying
       interior for the terminal bin to violate.  The count ratio is uninformative
       below roughly 5 entries.
    2. **Count is the wrong weight.**  The pinned pairs carried 2.3x the mean objective,
       so mass (26.4%) and count (12.7%) disagree by ~2x.  Passing ``objective_mass``
       recovers the SATURATED verdict; omitting it is how a real positive was missed.

    And a limit no 1-D histogram can reach: beta's dominant win (29 pairs, 0.2196 d)
    required BOTH sign freedom AND magnitude > 1.0.  A menu whose bound is a CONJUNCTION
    of constraints needs a JOINT occupancy over those axes; the marginal histogram
    cannot see it.  ``sufficient_for_verdict`` is False in exactly these cases.
    """
    h = [int(x) for x in occupancy]
    if not h:
        raise ValueError("occupancy histogram is empty; a menu with no selections is "
                         "VACUOUS, not CLOSED (report the denominator)")
    if any(x < 0 for x in h):
        raise ValueError("occupancy counts must be non-negative")
    total = sum(h)
    if total == 0:
        return {
            "verdict": "UNDETERMINED_EMPTY",
            "n_selections": 0,
            "terminal_mass_fraction": None,
            "terminal_to_last_interior_ratio": None,
            "note": "no selections recorded — VACUOUS scope, never a CLOSED verdict",
        }
    ti = len(h) - 1 if terminal_index is None else int(terminal_index)
    if not (0 <= ti < len(h)):
        raise ValueError("terminal_index out of range for the given occupancy histogram")
    terminal = h[ti]
    interior = [h[i] for i in range(len(h)) if i != ti]
    last_interior = interior[-1] if interior else 0
    ratio = (float(terminal) / float(last_interior)) if last_interior > 0 else (
        float("inf") if terminal > 0 else 0.0
    )

    mass_ratio: float | None = None
    mass_fraction: float | None = None
    if objective_mass is not None:
        m = [float(x) for x in objective_mass]
        if len(m) != len(h):
            raise ValueError("objective_mass must align 1:1 with occupancy")
        if any(x < 0.0 for x in m):
            raise ValueError("objective_mass entries must be non-negative")
        m_total = sum(m)
        if m_total > 0.0:
            m_interior = [m[i] for i in range(len(m)) if i != ti]
            m_last = m_interior[-1] if m_interior else 0.0
            mass_fraction = m[ti] / m_total
            mass_ratio = (m[ti] / m_last) if m_last > 0.0 else (
                float("inf") if m[ti] > 0.0 else 0.0
            )

    # Mass, when supplied, is the AUTHORITY -- the count-only reading missed a real
    # positive (see the docstring's measured false negative).
    decisive = mass_ratio if mass_ratio is not None else ratio
    decisive_terminal = (
        (mass_fraction or 0.0) > 0.0 if mass_ratio is not None else terminal > 0
    )
    saturated = decisive >= SATURATION_RATIO_THRESHOLD and decisive_terminal

    # A short menu has no interior trend for the ratio to violate.  With a trend, the
    # count ratio is informative on its own (dim0: 8.27x against 9 decaying bins);
    # without one, only objective mass can decide (beta: 3 bins -- the measured miss).
    # TREND, not strict monotonicity: the measured dim0 interior is noisy-but-falling
    # (103,93,67,39,51,34,46,28,15 -- two local rises), so a monotone test would call
    # the strongest positive we have "no trend".  The signal is the DECLINE across the
    # menu: last interior bin <= half the first.
    interior_decaying = (
        (interior[-1] <= 0.5 * interior[0]) if len(interior) > 1 and interior[0] > 0
        else (None if len(interior) <= 1 else False)
    )
    sufficient = len(h) >= 5 and (
        interior_decaying is True or objective_mass is not None or terminal == 0
    )
    return {
        "verdict": (
            "SATURATED_MEASURE_BEYOND_BOUND" if saturated else "CLOSED_INTERIOR_OPTIMUM"
        ),
        "n_selections": total,
        "terminal_count": terminal,
        "terminal_mass_fraction": float(terminal) / float(total),
        "terminal_to_last_interior_ratio": ratio,
        "objective_mass_fraction": mass_fraction,
        "objective_mass_ratio": mass_ratio,
        "decided_by": "objective_mass" if mass_ratio is not None else "count",
        "threshold": SATURATION_RATIO_THRESHOLD,
        "interior_is_decaying": interior_decaying,
        "verdict_is_a_measurement_request": bool(saturated),
        "sufficient_for_verdict": bool(sufficient),
        "insufficiency_reason": (
            None if sufficient
            else "menu_shorter_than_5_entries_no_interior_trend" if len(h) < 5
            else "no_interior_trend_and_no_objective_mass_supply_objective_mass"
        ),
    }


def _evaluate(inputs: Mapping[str, Any]) -> dict[str, Any]:
    keys = set(inputs)
    allowed = {"occupancy", "terminal_index", "objective_mass"}
    if not keys <= allowed or "occupancy" not in keys:
        raise ValueError(
            "menu-saturation inputs differ from the canonical callable contract "
            "(expected 'occupancy' and optionally 'terminal_index'/'objective_mass')"
        )
    return menu_saturation(
        inputs["occupancy"],
        terminal_index=inputs.get("terminal_index"),
        objective_mass=inputs.get("objective_mass"),
    )


register_evaluator(EQUATION_ID, _evaluate)


def build_ddm_pw1_menu_saturation_discriminator_v1(
    *,
    source_receipt: Path = RECEIPT,
) -> CanonicalEquation:
    """Build the menu-saturation discriminator with its exact-evaluated anchor."""

    provenance = build_provenance_for_research_sidecar(
        source_receipt,
        reactivation_criteria=(
            "Re-anchor if a menu flagged SATURATED is freed and does NOT improve the "
            "objective (would show the occupancy spike is not sufficient and demand a "
            "second conjunct), if a menu called CLOSED is later beaten from outside its "
            "bound (would falsify the interior-optimum reading), or if the threshold is "
            "re-derived from a population of menus rather than the three measured here. "
            "The discriminator is scale-free and vehicle-independent by construction; "
            "the THRESHOLD is anchored on n=3 menus and is the weak part."
        ),
        measurement_axis="[macOS-CPU advisory]",
        hardware_substrate="darwin_arm64_cpu_upstream_evaluate_n600",
        captured_at_utc="2026-08-01T00:00:00Z",
    )
    anchor = EmpiricalAnchor(
        anchor_id="ddm_pw1_menu_saturation_v4d_pose_n600_20260801",
        measurement_utc="2026-08-01T00:00:00Z",
        inputs={
            "menus_examined": 3,
            "population": "600 pairs, live v4d pose chain (ddm_v4c_resolve -> ddm_v4d_resolve)",
            "dim0_occupancy": list(DIM0_MOVE_HISTOGRAM),
            "beta_menu_size": BETA_MENU_SIZE,
            "beta_pinned_at_top": BETA_PINNED_AT_TOP,
            "s_t_grid_size": ST_GRID_SIZE,
            "s_t_occupied_indices": list(ST_OCCUPIED_INDICES),
            "scorer_evaluations": 4644,
        },
        predicted_output={
            # byte-closed prediction, made BEFORE the exact-eval gate fired
            "composed_S": S_PREDICTED,
            "delta_S": -0.0163792,
        },
        empirical_output={
            "dim0_verdict": "SATURATED_MEASURE_BEYOND_BOUND",
            "dim0_terminal_to_last_interior_ratio": 124.0 / 15.0,
            "dim0_terminal_mass_fraction_by_count": 124.0 / 600.0,
            "dim0_mass_fraction_objective_weighted": 0.374,
            "beta_verdict": "SATURATED_MEASURE_BEYOND_BOUND",
            "beta_mass_fraction_objective_weighted": 0.264,
            "s_t_verdict": "CLOSED_INTERIOR_OPTIMUM",
            "exact_eval_composed_S": S_AFTER,
            "exact_eval_delta_S": S_AFTER - S_BEFORE,
            "prediction_error": S_AFTER - S_PREDICTED,
            "archive_bytes_added": BYTES_ADDED,
            "d_pose_relative_change": -0.1091,
            "d_seg_change": 0.0,
            "unreachable_gain_fraction": UNREACHABLE_GAIN_FRACTION,
            "bound_limited_not_resolution_limited": True,
            "cure_cost_on_unbinding_menu_evaluations": 2,
        },
        # the byte-closed prediction vs the exact-evaluated realization
        residual=abs(S_AFTER - S_PREDICTED) / S_AFTER,
        source_artifact=str(RECEIPT.relative_to(REPO)),
        measurement_method=(
            "occupancy histograms read from the live n600 pose solve; A/B/AB continuation "
            "arms (4,644 scorer evaluations) with an EXACT canary on 577/600 pairs and a "
            "diagnosed 1.08e-5 floor; byte-closed through the real receiver with #417 "
            "parse-back; then upstream/evaluate.py n600 on the 360,323 B archive. "
            "Regression guard MEASURED not asserted: rebuilding from the pre-fix JSONL is "
            "byte-identical (sha f1f3288062, 360,238 B)"
        ),
        provenance=provenance,
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
    )
    # SECOND, INDEPENDENT INSTANCE -- found the same day, by a different arm, on a
    # different object class: a SOLVER ITERATION CAP rather than a value menu.  This is
    # what lifts the law above INSTANCE scope: the discriminator was not tuned to it.
    dc1_anchor = EmpiricalAnchor(
        anchor_id="ddm_dc1_qa03_max_quanta_cap_saturation_20260801",
        measurement_utc="2026-08-01T00:00:00Z",
        inputs={
            "menu": "tools/sb1_seg_batch.py --max-quanta (solver relinearization cap)",
            "menu_bound": 4,
            "n_instances": 120,
            "note": "a convergence test WAS present and correct; the cap outranked it",
        },
        predicted_output={
            "cap_is_a_menu_and_should_show_terminal_pileup": True,
        },
        empirical_output={
            "instances_stopped_at_cap": 51,
            "fraction_of_instances_at_cap": 51.0 / 120.0,
            "fraction_of_realized_flips_from_capped_instances": 0.647,
            "terminal_spike_vs_decaying_interior": 2.68,
            "verdict": "SATURATED_MEASURE_BEYOND_BOUND",
            "consequence": (
                "the 1,866-flip result is a strict LOWER bound; the 'not reachable by "
                "this formulation class' verdict was drawn on a CENSORED solve"
            ),
            "cure_landed": "default 4 -> 32, per-instance stop_reason, receipt-level "
                           "n_cap_saturated so censoring can never again be invisible",
        },
        residual=0.0,
        source_artifact=".omx/research/ddm_dc1_correction_stream_label_cost_20260801.md",
        measurement_method=(
            "per-instance stop-reason instrumentation over 120 QA03 solver instances; "
            "step histogram read directly rather than assumed"
        ),
        provenance=provenance,
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Occupancy-saturation discriminator for discrete search menus (ddm_pw1)",
        one_line_summary=(
            "Occupancy at a menu's bound => OUR menu clips the solution; strictly "
            "interior => honestly closed. Measured 8.27x spike vs an interior control; "
            "freeing two menus realized exact dS -0.0163787 at +85 B."
        ),
        latex_form=(
            r"r=\frac{h[T]}{h[T-1]},\quad "
            r"r\geq 1\ \wedge\ h\ \text{decaying}\ \Rightarrow\ \text{SATURATED};\quad "
            r"h[T]=0\ \Rightarrow\ \text{CLOSED};\quad "
            r"\text{bound-limited iff}\ \frac{\Delta_{\text{outside old reach}}}"
            r"{\Delta_{\text{total}}}\to 1\ (0.9789)"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.ddm_pw1_menu_saturation_discriminator_20260801"
            ":menu_saturation"
        ),
        domain_of_validity={
            "object": "any discrete/bounded choice whose selected index is observable per "
                      "item: search brackets, magnitude tables, sign heuristics, boolean "
                      "flags (a 2-entry menu), grid resolutions, top-k caps",
            "requires": "an occupancy histogram over the menu's own index set — the solve "
                        "already produces it; no extra scorer cost",
            "evidence_axis": "[macOS-CPU advisory] on the anchor; the DISCRIMINATOR itself "
                             "is axis-free (it reads occupancy, not score)",
            "threshold_provenance": "separation measured on n=3 menus (8.27x vs 0.0); the "
                                    "1.0 line is the weakest defensible cut, deliberately "
                                    "over-reporting suspicion",
            "research_only": True,
            "score_claim": False,
            "verdict_scope": "FORMULATION_DISCRETE_MENU_SATURATION",
            "excluded": [
                "a claim that freeing a SATURATED bound will pay — the verdict is a "
                "measurement request; only the post-free attribution settles it",
                "a claim that widening is rate-free — the receiver contract decides "
                "(pw1's was free because the table was already manifest-read; a menu "
                "needing a schema bump is not)",
                "continuous parameters with no index (use a gradient/trust-region test)",
                "menus with zero selections — that is VACUOUS scope, never CLOSED",
                "any promotion or submission use",
            ],
        },
        canonical_producers=(
            "tools/pw1_pose_menu_saturation_ab.py",
            "experiments/ddm_v4d_resolve.py",
        ),
        canonical_consumers=(
            "src/tac/optimization/lane_guard.py",
            "experiments/ddm_v4d_resolve.py",
        ),
        empirical_anchors=(anchor, dc1_anchor),
        units_in={
            "occupancy": "counts per menu index (dimensionless integers)",
            "terminal_index": "menu index of the bound (dimensionless, optional)",
        },
        units_out={
            "verdict": "enum {SATURATED_MEASURE_BEYOND_BOUND, CLOSED_INTERIOR_OPTIMUM, "
                       "UNDETERMINED_EMPTY}",
            "terminal_mass_fraction": "dimensionless fraction in [0,1]",
            "terminal_to_last_interior_ratio": "dimensionless ratio in [0, inf]",
        },
        predicted_vs_empirical_residual={
            "composed_S_relative": abs(S_AFTER - S_PREDICTED) / S_AFTER,
            "composed_S_absolute": abs(S_AFTER - S_PREDICTED),
        },
        last_calibration_utc="2026-08-01T00:00:00Z",
        provenance=provenance,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
    )
