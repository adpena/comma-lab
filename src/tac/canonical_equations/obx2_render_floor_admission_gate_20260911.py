# SPDX-License-Identifier: MIT
"""Canonical equation: the PRE-BURN render-floor admission gate (ddm_obx2, 2026-09-11).

WHAT IT DECIDES.  Before any edge-local correction, lattice head, or implicit-correction burn is
funded on a candidate generator, decompose that generator's ``d_seg`` into the two terms the
correction family can and cannot reach:

    d_seg_total = d_seg_partition + d_seg_render_floor

``d_seg_partition`` is argmax error caused by the generator putting the region boundary in the
wrong PLACE; an edge-local correction can move it.  ``d_seg_render_floor`` is the argmax error
that REMAINS when the partition is already correct -- photometric jitter around a correct
boundary -- and no boundary-moving correction can reach it.  The gate is therefore

    ADMIT  iff  d_seg_render_floor  <  SEG_CEILING

and it is a NECESSARY condition, never a sufficient one: clearing it does not make the object
admissible, it only means the floor is not on its own disqualifying.

WHICH CEILING.  Two numbers are in play and the looser one must not silently replace the tighter.
The design-A closure memo states the ceiling as 4.0e-4.  The burn spec's PRE-REGISTERED hard gate
is tighter and depends on the Pose leg it is paired with: ``d_seg < 2.8765207719095085e-4`` at
``d_pose <= 1e-5``, and ``d_seg < 3.1990253844713356e-4`` at a pointer-like ``d_pose = 4.59e-6``.
4.0e-4 is a rounded stand-in that is 25 % LOOSER than the pre-registered gate, so
:func:`admits` defaults to the pre-registered pointer-like gate and the closure memo's 4.0e-4 is
available explicitly.  On this object the choice does not change the verdict -- the floor is
5.77x over the loose ceiling and 7.22x over the pre-registered one -- but on a successor it can.

MEASURED on the ddm_obx2 design-A object, n600, frozen CPU SegNet argmax:

  * d_seg_total                 0.02271701         (the gate needs 56.8x on the total)
  * render floor share          10.17 %  of the total error
  * d_seg_render_floor          0.0023099568  = 272,494 argmax errors AT A CORRECT PARTITION
  * against the 4.0e-4 ceiling  5.77x OVER, on its own
  * partition share             89.83 %

So the family's own ceiling -- what a PERFECT edge-local correction would leave -- is already
5.77x past the seg ceiling.  That is what closed design A at FORMULATION scope: the mechanism was
validated (the lattice captured 53.1 % of the render floor available to it, for 575 B, at 6.4x on
Pose) and the OBJECT was refused.

THE FLOOR BEHAVES LIKE JITTER, which is why it is a floor and not a target: 65.11 % of its errors
lie within ONE cell of a correct boundary.  A correction that moves boundaries cannot remove an
error whose boundary is already in the right place.

TRANSFER BOUNDARY (binding).  This is a GATE on a generator, not a property of the correction
family.  It says nothing about how large a render floor a DIFFERENT generator has; it says that
whatever that generator's floor is, it must be measured with
``experiments/ddm_obx2_trainer.py::decompose_seg_error`` BEFORE the burn is funded, because the
burn cannot go below it.  Design A reactivates on a generator that measures a render floor below
4.0e-4; this object's is 0.00231.

Axis ``[n600 frozen CPU scorer argmax]``; ``score_claim=false``; the pointer is UNMOVED by this
equation.  Registered by ddm_cons1 on 2026-09-11 from the arm's own closure memo.
"""

from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "obx2_render_floor_admission_gate_v1"

_UTC = "2026-09-11T00:00:00Z"
_AXIS = "[n600 frozen CPU scorer argmax]"
_CLOSURE = ".omx/research/ddm_obx2_design_a_closure_20260911.md"
_ARM = ".omx/research/ddm_obx2_edge_local_implicit_correction_20260911.md"
_PRODUCER = "experiments/ddm_obx2_trainer.py"

# MEASURED, ddm_obx2 design-A object, n600.
# The closure memo's stated (rounded, LOOSER) ceiling.
SEG_CEILING_CLOSURE_MEMO = 4.0e-4
# The burn spec's PRE-REGISTERED hard gates -- "hard gates, not predictions".
SEG_CEILING_PREREGISTERED_AT_DPOSE_1E5 = 2.8765207719095085e-4
SEG_CEILING_PREREGISTERED_AT_POINTER_DPOSE = 3.1990253844713356e-4
# What :func:`admits` uses unless the caller names another.
SEG_CEILING = SEG_CEILING_PREREGISTERED_AT_POINTER_DPOSE
D_SEG_TOTAL = 0.02271701
RENDER_FLOOR_D_SEG = 0.0023099568
RENDER_FLOOR_ARGMAX_ERRORS = 272_494
RENDER_FLOOR_SHARE = 0.1017
PARTITION_SHARE = 0.8983
FLOOR_WITHIN_ONE_CELL = 0.6511
LATTICE_FLOOR_CAPTURE = 0.531
LATTICE_BYTES = 575
LATTICE_POSE_COST_FACTOR = 6.4


def render_floor_over_ceiling(
    render_floor_d_seg: float = RENDER_FLOOR_D_SEG,
    seg_ceiling: float = SEG_CEILING,
) -> float:
    """How many times the seg ceiling a generator's render floor is, on its own."""
    return render_floor_d_seg / seg_ceiling


def admits(render_floor_d_seg: float, seg_ceiling: float = SEG_CEILING) -> bool:
    """The NECESSARY pre-burn gate: a correction family cannot go below the render floor.

    Defaults to the burn spec's PRE-REGISTERED pointer-like gate, not the closure memo's
    rounded 4.0e-4 -- a gate must never loosen by being restated.
    """
    return render_floor_d_seg < seg_ceiling


def total_factor_needed(
    d_seg_total: float = D_SEG_TOTAL,
    seg_ceiling: float = SEG_CEILING,
) -> float:
    """How many times the TOTAL seg error must fall for the burn gate to be met."""
    return d_seg_total / seg_ceiling


def build_obx2_render_floor_admission_gate_v1() -> CanonicalEquation:
    """Build the pre-burn render-floor admission gate (ddm_obx2 design-A closure)."""
    provenance = build_provenance_for_research_sidecar(
        sidecar_path=_CLOSURE,
        reactivation_criteria=(
            "design A reactivates on a candidate generator that MEASURES a render floor below "
            "4.0e-4 using experiments/ddm_obx2_trainer.py::decompose_seg_error, together with a "
            "partition error the correction family can actually reach; this object measured "
            "0.00231 (5.77x over) and was refused at OBJECT scope, not at mechanism scope"
        ),
        measurement_axis=_AXIS,
        hardware_substrate="macOS CPU (frozen torch scorers)",
        captured_at_utc=_UTC,
    )

    floor_anchor = EmpiricalAnchor(
        anchor_id="obx2_design_a_render_floor_n600_20260911",
        measurement_utc=_UTC,
        inputs={
            "object": "the ddm_obx2 design-A generator's rendered frames, all 600 pairs",
            "instrument": (
                "experiments/ddm_obx2_trainer.py::decompose_seg_error -- argmax error at the "
                "generator's own partition vs argmax error with the partition forced correct"
            ),
            "seg_ceiling_closure_memo": SEG_CEILING_CLOSURE_MEMO,
            "seg_ceiling_preregistered_at_pointer_dpose": SEG_CEILING_PREREGISTERED_AT_POINTER_DPOSE,
            "seg_ceiling_preregistered_at_dpose_1e5": SEG_CEILING_PREREGISTERED_AT_DPOSE_1E5,
            "scorer_runs": "frozen CPU SegNet argmax, n600",
            "modal_runs": 0,
        },
        predicted_output={
            "charter_expectation": (
                "an edge-local implicit correction can reach the seg ceiling because the seg "
                "error is dominated by boundary PLACEMENT"
            ),
            "implied_render_floor_share": "small enough not to bind",
        },
        empirical_output={
            "d_seg_total": D_SEG_TOTAL,
            "render_floor_d_seg": RENDER_FLOOR_D_SEG,
            "render_floor_argmax_errors": RENDER_FLOOR_ARGMAX_ERRORS,
            "render_floor_share_of_total": RENDER_FLOOR_SHARE,
            "partition_share_of_total": PARTITION_SHARE,
            "render_floor_over_closure_memo_ceiling": 5.77,
            "render_floor_over_preregistered_ceiling": 7.22,
            "total_factor_needed_closure_memo_ceiling": 56.8,
            "total_factor_needed_preregistered_ceiling": 71.0,
            "floor_errors_within_one_cell": FLOOR_WITHIN_ONE_CELL,
            "mechanism_validated": (
                f"the lattice captured {LATTICE_FLOOR_CAPTURE:.1%} of the render floor available "
                f"to it for {LATTICE_BYTES} B, at {LATTICE_POSE_COST_FACTOR}x on Pose"
            ),
            "verdict": (
                "CLOSED AT FORMULATION SCOPE -- MECHANISM VALIDATED, OBJECT REFUSED: 89.83% of "
                "the seg error is partition-level and the render floor alone is 5.77x the ceiling"
            ),
        },
        # |predicted 'does not bind' (<= 1.0x ceiling) - measured 5.77x| / 5.77x
        residual=0.8267,
        source_artifact=_CLOSURE,
        measurement_method="decompose_seg_error_partition_vs_render_floor_n600_frozen_cpu_argmax",
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )

    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Pre-burn render-floor admission gate for edge-local correction families",
        one_line_summary=(
            "A correction family cannot go below the render floor: gate the burn on "
            "d_seg_floor < 3.199e-4 (pre-registered); obx2's object measured 0.00231 = 7.22x over"
        ),
        latex_form=(
            r"d_{\mathrm{seg}} = d_{\mathrm{seg}}^{\mathrm{partition}} + "
            r"d_{\mathrm{seg}}^{\mathrm{floor}},\quad "
            r"\text{ADMIT} \iff d_{\mathrm{seg}}^{\mathrm{floor}} < 4.0\times10^{-4}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.obx2_render_floor_admission_gate_20260911:admits"
        ),
        domain_of_validity={
            "applies_to": (
                "any candidate generator being considered for an edge-local / implicit / "
                "boundary-moving correction burn against the contest SegNet argmax"
            ),
            "measured_on": "the ddm_obx2 design-A object only (n=1 generator)",
            "ceiling_provenance": (
                "the PRE-REGISTERED burn-spec gate (3.1990253844713356e-4 at pointer-like "
                "d_pose 4.59e-6; 2.8765207719095085e-4 at d_pose <= 1e-5) is the default. The "
                "closure memo's 4.0e-4 is a rounded stand-in 25% looser and is kept only as a "
                "named constant -- a restatement must never loosen a pre-registered gate"
            ),
            "necessary_not_sufficient": (
                "clearing the gate does not admit an object; failing it disqualifies the burn"
            ),
            "excluded": (
                "correction families that move PHOTOMETRY rather than boundaries -- they act on "
                "the floor term itself and this gate does not price them"
            ),
        },
        units_in={
            "render_floor_d_seg": "SegNet argmax disagreement rate (dimensionless)",
            "d_seg_total": "SegNet argmax disagreement rate (dimensionless)",
        },
        units_out={
            "admits": "boolean admission verdict",
            "render_floor_over_ceiling": "multiples of the 4.0e-4 seg ceiling",
            "total_factor_needed": "multiples of the 4.0e-4 seg ceiling",
        },
        empirical_anchors=(floor_anchor,),
        predicted_vs_empirical_residual={"render_floor_over_ceiling": 0.8267},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(_CLOSURE, _ARM, "experiments/ddm_obx2_trainer.py"),
        canonical_producers=(_PRODUCER, "experiments/ddm_obx2_edge_local_implicit_correction.py"),
        provenance=provenance,
    )
