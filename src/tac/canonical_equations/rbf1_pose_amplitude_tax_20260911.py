# SPDX-License-Identifier: MIT
"""Canonical equation: the quadratic out-of-distribution Pose tax on a free post-render edit (rbf1).

THE LAW.  Editing the rendered frame after the render -- free receiver code, zero counted bytes --
costs PoseNet a tax that is exactly quadratic in the edit's amplitude and does not care about its
direction:

    d_pose(tau) - d_pose(0) = k * tau**2,      k = 9.0e-06 at a ~22,000-pixel support
    => Delta S_pose ~= 6.6e-03 * tau**2

``tau`` is a per-channel uint8 amplitude bound on the treated frame's delta against the retained
baseline -- ONE QUANTISATION LEVEL per unit.  MEASURED excess-d_pose/tau**2 at tau = 1, 2, 4, 8 is
8.763e-06, 9.062e-06, 9.249e-06, 8.829e-06: a 5.5 % spread over an 8x amplitude range, which the
arm rounds as "constant to within 5 %".

WHY IT CLOSES THE FAMILY.  At tau = 1 -- a SINGLE quantisation level, the smallest edit that
exists -- the pose tax alone is 6.5e-03 S, which is 323 bars, against a measured seg benefit of
THREE pixels.  Both terms vanish as tau -> 0, so the infimum of Delta S over the whole family is
0, reached only by not treating.  **There is no payable amplitude.**  Two further readings kill
the obvious escapes: selectivity does NOT improve at small amplitude (harm/benefit 11.67 at
tau = 1 vs 9.86 at tau = 4), and the operator is barely better than its own reverse (tau = -8
gives Delta S_seg +0.005447 against +0.005404 -- indistinguishable), so the direction information
the token plane carries into this actuator is weak.  The tax is direction-blind and
symmetry-proof.

WHERE THE 6.6e-03 COMES FROM.  The contest objective's own derivative at the operating point:
``d/dx sqrt(10 x) = 5 / sqrt(10 x)`` = 738 at ``d_pose`` 4.59e-06, and 738 * 9.0e-06 = 6.6e-03.
The COEFFICIENT is therefore operating-point dependent -- on a vehicle with a higher d_pose the
same k buys a smaller Delta S, which is exactly why the arm's reactivation criterion names "a
vehicle whose d_pose operating point is more than 10x higher".

SUPPORT, STATED HONESTLY (this is the part that must travel with the number).  The amplitude arm
is **n = 24 seeded-random pairs (seed 1, guided source, 355 s) -- NOT n600**, declared by the arm
as scope, a random sample rather than a prefix per the prefix-bias law.  The companion SUPPORT
law (``Delta S_pose ~= 0.0024 * f**1.16 * tau**2``, f = percent of frame_1's pixels edited) rests
on only **4 seeded-random pairs (seed 5)** and is NOT registered here.

STANDING OBJECTION, CARRIED DELIBERATELY.  ddm_rbf1 declined to register this itself, verbatim:
"It is a CLOSURE price, not a lever: registering it would put a two-parameter fit with a 4-pair
support arm and a 24-pair amplitude arm into the registry with no consumer ... the successor that
prices a counted boundary edit against this tax is the right registrant, and it should first
widen the support arm from 4 seeded-random pairs to n>=120."  ddm_cons1 registers only the
AMPLITUDE clause -- the one the arm's own n600 closure line quotes -- with its n=24 support in the
domain, names the consumer the arm asked for, and leaves the two-parameter support fit out.  The
objection stands: widen the support arm before the f**1.16 clause is registered.

Axis ``[n=24 seeded-random pairs; frozen CPU PoseNet]``; ``score_claim=false``.  The n600 exact
closure rows the arm measured (SDF +0.088367, SSAA +0.093865, guided +0.315443, composition
+0.397397) are a different, larger population and are recorded as context, not as this law's fit.
"""

from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "rbf1_post_render_pose_amplitude_tax_v1"

_UTC = "2026-09-11T00:00:00Z"
_AXIS = "[n=24 seeded-random pairs; frozen CPU PoseNet]"
_LEDGER = ".omx/research/ddm_rbf1_free_post_render_boundary_treatment_20260911.md"
_RETAINED = ".omx/research/ddm_rbf1_20260911/POSE_TAX_v1.json"
_PRODUCER = "experiments/ddm_rbf1_amplitude.py"

# MEASURED, n=24 seeded-random pairs (seed 1), ~22,000-pixel support.
K_D_POSE_PER_TAU_SQUARED = 9.0e-06
DELTA_S_PER_TAU_SQUARED = 6.6e-03
EXCESS_OVER_TAU_SQUARED_BY_TAU = {1: 8.763e-06, 2: 9.062e-06, 4: 9.249e-06, 8: 8.829e-06}
SPREAD_OVER_8X_RANGE = 0.055
SUPPORT_PIXELS = 22_000
AMPLITUDE_ARM_PAIRS = 24
SUPPORT_ARM_PAIRS = 4
POSE_WEIGHT_AT_OPERATING_POINT = 738.0
OPERATING_POINT_D_POSE = 4.59e-06
HARM_BENEFIT_BY_TAU = {1: 11.67, 4: 9.86, 8: 10.81, -8: 14.53}
N600_CLOSURE_DELTA_S = {
    "sdf": 0.088367,
    "ssaa": 0.093865,
    "guided": 0.315443,
    "composition": 0.397397,
}


def pose_tax_d_pose(tau: float) -> float:
    """Excess ``d_pose`` over the untreated render at amplitude ``tau`` quantisation levels.

    Uses the FITTED k = 9.0e-06, so at tau = 1 this returns 6.642e-03 S while the arm quotes
    6.5e-03 -- the arm quotes the MEASURED tau = 1 point (8.763e-06 * 738 = 6.467e-03). The 2.7 %
    gap is the fit's own 5.5 % spread, not a disagreement. Use :data:`EXCESS_OVER_TAU_SQUARED_BY_TAU`
    when the measured point at a specific tau is wanted instead of the fit.
    """
    return K_D_POSE_PER_TAU_SQUARED * tau * tau


def pose_tax_delta_s(tau: float, pose_weight: float = POSE_WEIGHT_AT_OPERATING_POINT) -> float:
    """Score cost of the Pose tax at amplitude ``tau``, at a given operating-point Pose weight."""
    return pose_weight * pose_tax_d_pose(tau)


def payable_amplitude_exists(seg_credit_delta_s: float) -> bool:
    """True only if some tau > 0 buys more seg than its Pose tax costs.

    MEASURED false for this family at every tau: the tax is superlinear while the seg credit is
    not, so the infimum of Delta S is reached only at tau -> 0 (not treating).
    """
    return seg_credit_delta_s > pose_tax_delta_s(1.0)


def build_rbf1_post_render_pose_amplitude_tax_v1() -> CanonicalEquation:
    """Build the quadratic post-render Pose-tax equation (ddm_rbf1, amplitude arm n=24)."""
    provenance = build_provenance_for_research_sidecar(
        sidecar_path=_LEDGER,
        reactivation_criteria=(
            "the family reopens on an operator that measures harm/benefit below 1 on a seeded "
            "RANDOM n>=120 sample AND leaves d_pose within 3e-08 of the shipped render, or on a "
            "vehicle whose d_pose operating point is more than 10x higher (the 6.6e-03 "
            "coefficient is operating-point dependent through 5/sqrt(10 d_pose)). The COMPANION "
            "support clause (f**1.16) is NOT registered and must not be until its 4-pair arm is "
            "widened to n>=120 -- ddm_rbf1's own standing objection, carried here deliberately"
        ),
        measurement_axis=_AXIS,
        hardware_substrate="macOS CPU (frozen torch PoseNet)",
        captured_at_utc=_UTC,
    )

    amplitude_anchor = EmpiricalAnchor(
        anchor_id="rbf1_pose_tax_amplitude_sweep_n24_20260911",
        measurement_utc=_UTC,
        inputs={
            "object": "move-44 rendered frame_1, token-edge band, ~22,000-pixel support",
            "amplitudes": sorted(EXCESS_OVER_TAU_SQUARED_BY_TAU),
            "sign_control": -8,
            "producer_reproduction_check": "tau = 255 reproduces the producer exactly",
            "population": (
                "n = 24 seeded-random pairs (seed 1, guided source, 355 s) -- a RANDOM sample, "
                "not a prefix; NOT n600"
            ),
            "operating_point_d_pose": OPERATING_POINT_D_POSE,
        },
        predicted_output={
            "charter_hope": (
                "a gentler edit is a more SELECTIVE edit, so some small amplitude pays: the seg "
                "credit survives while the pose tax shrinks away"
            ),
            "expected_payable_amplitude": "some tau >= 1",
        },
        empirical_output={
            "excess_d_pose_over_tau_squared": EXCESS_OVER_TAU_SQUARED_BY_TAU,
            "k_d_pose_per_tau_squared": K_D_POSE_PER_TAU_SQUARED,
            "delta_s_per_tau_squared": DELTA_S_PER_TAU_SQUARED,
            "spread_over_8x_amplitude_range": SPREAD_OVER_8X_RANGE,
            "tax_at_tau_1_delta_s": 6.5e-03,
            "tax_at_tau_1_in_bars": 323,
            "seg_benefit_at_tau_1": "3 pixels",
            "harm_benefit_by_tau": HARM_BENEFIT_BY_TAU,
            "selectivity_improves_at_small_amplitude": False,
            "direction_blind": (
                "tau = -8 gives Delta S_seg +0.005447 against the forward +0.005404 -- "
                "indistinguishable; the operator is barely better than its own reverse"
            ),
            "payable_amplitude_exists": False,
            "n600_closure_delta_s": N600_CLOSURE_DELTA_S,
            "verdict": (
                "the free post-render boundary treatment family is CLOSED at FORMULATION scope on "
                "the move-44 vehicle, on two independent n600 channels (seg harm/benefit 12.0-16.9 "
                "where break-even needs 186x better-than-random selectivity; pose the quadratic tax)"
            ),
        },
        # |predicted 'some payable tau exists' - measured 'none, infimum at tau -> 0'| : total miss
        residual=1.0,
        source_artifact=_RETAINED,
        measurement_method="amplitude_sweep_tau_1_2_4_8_plus_sign_control_n24_seeded_random_pairs",
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )

    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Quadratic out-of-distribution Pose tax on a free post-render edit",
        one_line_summary=(
            "A free post-render edit of amplitude tau costs Delta S_pose ~= 6.6e-3*tau^2 "
            "(k=9.0e-6 d_pose, 5% constant over tau=1..8, n=24): no payable amplitude exists"
        ),
        latex_form=(
            r"d_{\mathrm{pose}}(\tau) - d_{\mathrm{pose}}(0) = k\tau^{2},\ k=9.0\times10^{-6}"
            r"\ \Rightarrow\ \Delta S_{\mathrm{pose}} \approx 6.6\times10^{-3}\,\tau^{2},\quad "
            r"\inf_{\tau\ge 0}\Delta S = 0 \text{ at } \tau=0"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.rbf1_pose_amplitude_tax_20260911:pose_tax_delta_s"
        ),
        domain_of_validity={
            "applies_to": (
                "deterministic post-render operators on the token-edge band of the rendered "
                "frames -- free receiver code, no counted side information -- on the move-44 vehicle"
            ),
            "support": (
                f"n = {AMPLITUDE_ARM_PAIRS} seeded-random pairs (seed 1), ~{SUPPORT_PIXELS:,} "
                "pixels, tau in {1,2,4,8} plus a -8 sign control. NOT n600. The arm's n600 rows "
                "are the mode-composition closure, a different population"
            ),
            "excluded": (
                "the COMPANION support-scaling clause Delta S_pose ~= 0.0024*f^1.16*tau^2, whose "
                f"support arm is only {SUPPORT_ARM_PAIRS} seeded-random pairs -- deliberately NOT "
                "registered until widened to n>=120 (ddm_rbf1's own precondition); and any "
                "COUNTED edit, which pays bytes this law does not price"
            ),
            "operating_point_dependence": (
                "the 6.6e-03 coefficient is 5/sqrt(10*d_pose) = 738 times k at d_pose 4.59e-06; "
                "re-derive the coefficient at any other operating point"
            ),
        },
        units_in={"tau": "uint8 quantisation levels of per-channel amplitude bound",
                  "pose_weight": "dS per unit d_pose at the operating point"},
        units_out={
            "pose_tax_d_pose": "PoseNet MSE on the first 6 pose dimensions",
            "pose_tax_delta_s": "contest score units (S)",
        },
        empirical_anchors=(amplitude_anchor,),
        predicted_vs_empirical_residual={"payable_amplitude_exists": 1.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(_LEDGER, _RETAINED),
        canonical_producers=(_PRODUCER, "experiments/ddm_rbf1_boundary_treatments.py"),
        provenance=provenance,
    )
