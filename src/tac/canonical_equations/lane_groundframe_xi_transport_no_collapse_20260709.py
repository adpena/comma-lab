# SPDX-License-Identifier: MIT
"""Canonical equation: LANE GROUND-FRAME xi-TRANSPORT NO-COLLAPSE LAW (MEASURED negative).

The corollary law from the FEED-v8-lane-xi measurement (memo
`v8_roadlane_ego_compensated_rate_20260709.md`): ego-freeze does NOT transfer to any coordinate
chart that has ALREADY absorbed the ego DOF. Concretely, for a lane centerline already parametrized
in the ego-GROUND frame (`lateral = poly(forward)` via a fixed IPM `image_to_ground`), ego-advected
predictive coding (LBND3, exact closed-form SE(2) transport) yields an innovation stream whose L1
magnitude is >= the LBND2 identity temporal delta:

    ||innov_xi||_1  >=  ||Delta Q||_1          for ALL xi predictors tested

i.e. the plain identity predictor (Q[t] - Q[t-1]) beats every ego-rigid predictor, INCLUDING the
lane-OPTIMAL xi (fit to minimize innovation = the achievable floor) and a smoothed variant. The IPM
already quotients out the dominant ego-forward-translation component at FIT time, so the residual
temporal delta is the IRREDUCIBLE part (real lane-curvature evolution + multi-instance slot churn +
per-frame fit noise) which a rigid ego-advection model actively MIS-predicts.

Corollary (the durable generalization): the horizon's 14.6x ego-freeze ratio does NOT transfer to any
representation whose coordinate chart has already absorbed the dominant ego DOF. The horizon won its
14.6x precisely because its polynomial is NOT ground-canonicalized (it lives in IMAGE ROWS, where
inter-frame motion IS a single removable ego-pitch intercept).

THE MEASURED VERDICT (n600, real gt_poses xi, exact transport, bit-exact roundtrips; BOTH accounting
modes -- full-blob-including-ego AND isolated-lane-payload-ego-free):
  * LBND2 identity temporal delta (lane payload only)  = 41,085 B   ||Delta Q||_1 = 7,584,060  (1.0x)
  * xi PoseNet-affine (best xi arm)                    = 42,017 B   ||innov||_1  = 8,252,100  (0.98x WORSE)
  * xi lane-optimal no-forward (dy+yaw)                = 42,195 B   ||innov||_1  = 7,727,778  (0.97x WORSE)
  * xi lane-optimal smooth-7                           = 43,136 B   ||innov||_1  = 8,670,182  (0.95x WORSE)
  * xi lane-optimal (fwd+dy+yaw)                       = 43,964 B   ||innov||_1  = 9,983,228  (0.93x WORSE)
Full-blob (incl. ~2.1-2.7 KB ego stream) is strictly worse still (0.92x-0.87x). Identity wins on BOTH
bytes AND L1, robustly to the accounting choice. Per-coeff nonzero-fraction: NO coeff collapses;
offset(c0)/heading(c1) get DENSER under advection (ego over-predicts the bulk motion the steered ego
cancels).

means != ends: this is a MEASURED characterization of a NEGATIVE lever (a NO-GO for the lane rate axis)
on the DECODE side, NOT a converged-witness score. The pointer moves only through a byte-closed
upstream/evaluate.py exact-eval row; the d_seg half (#205) remains the true blocker. verdict_scope =
FORMULATION (xi-advection on the ground-frame-canonicalized lane chart) -- the PARADIGM is intact:
xi remains DECISIVE for POSE (dual-use) and for the image-frame horizon (14.6x). Snapshot-scope (n600
GT argmax labels at this cache), advisory-class, promotion_eligible=false. Pointer 0.19110 UNMOVED,
#205 untouched.

DSL leg: N/A (a MEASUREMENT of a NEGATIVE lever -- no trainer lever / launch / curriculum change; the
law tells you WHERE xi-transport pays: only UN-canonicalized charts). EQUATIONS leg = this module.
DAG: `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md` (FEED-v8-lane-xi).
Memo: `.omx/research/v8_roadlane_ego_compensated_rate_20260709.md`.
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_research_sidecar,
)

EQUATION_ID = "lane_groundframe_xi_transport_no_collapse_v1"

_UTC = "2026-07-09T00:00:00Z"
_ADVISORY = "[macOS-CPU advisory]"
_DAG = ".omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md"
_MEMO = ".omx/research/v8_roadlane_ego_compensated_rate_20260709.md"

# MEASURED L1 innovation magnitudes (n600, isolated lane payload, ego-free -- xi's fairest shot).
_L1_IDENTITY = 7_584_060      # LBND2 identity temporal delta (the FLOOR the negative proves unbeaten)
_L1_XI_BEST_AFFINE = 8_252_100  # PoseNet-affine (smallest-bytes xi arm, still WORSE on L1)
_L1_XI_LANEOPT = 9_983_228    # lane-optimal (fwd+dy+yaw) -- fit to minimize innovation, still WORSE

# MEASURED brotli bytes (isolated lane payload).
_B_IDENTITY = 41_085
_B_XI_BEST = 42_017           # PoseNet-affine = the smallest ANY xi arm achieves; > identity


def build_lane_groundframe_xi_transport_no_collapse_v1() -> CanonicalEquation:
    """Build the ground-frame xi-transport NO-COLLAPSE law with its two MEASURED accounting-mode anchors.

    Anchor 1 = the isolated-lane-payload mode (ego overhead REMOVED -- xi's fairest, dual-use best
    case): identity 41,085 B / L1 7.58M beats every xi predictor on BOTH bytes and L1.
    Anchor 2 = the full-LBND3-blob mode (ego stream INCLUDED, conservative no-dual-use-credit): every
    xi arm is strictly larger still (0.92x-0.87x vs the LBND2 baseline). The negative is robust to the
    accounting choice."""

    anchor_lane_payload = EmpiricalAnchor(
        anchor_id="lane_groundframe_xi_no_collapse_isolated_payload_measured_20260709",
        measurement_utc=_UTC,
        inputs={
            "mode": "isolated lane payload (ego block EXCLUDED -- xi's fairest, dual-use best case)",
            "edge": "Road<->Lane (comma10k class Road0/Lane1)",
            "labels": (
                "experiments/results/mlx_fleet_gt_cache/gt_n600.npz['lstars'] "
                "(600x384x512 int64 argmax)"
            ),
            "xi_source": (
                "experiments/results/mlx_fleet_gt_cache/gt_n600.npz['gt_poses'] (600,6) = REAL "
                "frozen-PoseNet ego readout (pose section, dual-use, ZERO synthetic proxy); ALSO the "
                "lane-OPTIMAL xi (per-pair LSQ ds/dy/dpsi = the achievable innovation FLOOR)"
            ),
            "estimators": (
                "tac.boundary_math.ego_xi_trajectory.{PoseTargetEgoEstimator(affine/geometry),"
                "LaneOptimalEgoEstimator}; transport = advect_centerline_coeffs "
                "(exact closed-form SE(2): Pascal Taylor-shift + dy + dpsi)"
            ),
            "coder": (
                "tac.boundary_math.analytic_lane_render_band.serialize_lane_band_rd3 "
                "(LBND3 ego-predictive P-frame; xi=0 => innovation == LBND2 temporal delta; "
                "brotli q11; bit-exact roundtrip vs LBND2 dequantized geometry)"
            ),
        },
        predicted_output={
            "law": "||innov_xi||_1 >= ||Delta Q||_1  for ALL xi (identity predictor is unbeaten)",
            "hoped_for_collapse": "horizon-class 14.6x transfer to lanes -- FALSIFIED",
        },
        empirical_output={
            "identity_bytes": _B_IDENTITY,
            "identity_L1": _L1_IDENTITY,
            "xi_affine_bytes": _B_XI_BEST,
            "xi_affine_L1": _L1_XI_BEST_AFFINE,     # 8.25M > 7.58M identity
            "xi_laneopt_L1": _L1_XI_LANEOPT,        # 9.98M -- the achievable-floor predictor STILL worse
            "best_xi_vs_identity_bytes_x": 0.98,    # every xi arm LARGER
            "per_coeff_collapse": "NONE (c0/c1 DENSER under advection)",
            "note": (
                "identity (Q[t]-Q[t-1]) wins on BOTH bytes AND L1; the best-possible ego predictor "
                "(lane-optimal, fit to minimize innovation) still loses -> not a bad-xi artifact; the "
                "residual delta is IRREDUCIBLE (curvature evolution + slot churn + fit noise), not a "
                "rigid ego transport the IPM did not already quotient out"
            ),
        },
        residual=0.0,
        source_artifact=_MEMO,
        measurement_method="n600_argmax_real_gt_poses_xi_exact_transport_lbnd3_bit_exact_roundtrip",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria=(
                "an UN-canonicalized lane chart (image-frame coeffs, ego DOF NOT pre-absorbed) OR a "
                "new xi predictor class that beats identity on ||innov||_1 at n600 would RE-OPEN the "
                "formulation; a byte-closed exact-eval on a v8 archive that ships ground-frame lane "
                "coeffs converts the advisory rate finding to contest authority"
            ),
            measurement_axis=_ADVISORY,
            hardware_substrate="macos_arm64",
        ),
    )

    anchor_full_blob = EmpiricalAnchor(
        anchor_id="lane_groundframe_xi_no_collapse_full_blob_measured_20260709",
        measurement_utc=_UTC,
        inputs={
            "mode": (
                "full LBND3 blob INCLUDING the ~2.1-2.7 KB ego stream (conservative -- no dual-use "
                "credit; rules out 'the ego bytes are the whole regression')"
            ),
            "labels": "gt_n600.npz['lstars'] (comma10k order Road0/Lane1)",
            "baseline": (
                "LBND2 coherent-slot temporal delta 41,303 B -> S=0.02750 (matches FEED-v8-roadlane "
                "headline exactly)"
            ),
        },
        predicted_output={
            "law": "S(LBND3 xi-blob) >= S(LBND2 delta) for ALL xi",
        },
        empirical_output={
            "lbnd2_baseline_S": 0.02750,             # 41,303 B, HELD
            "xi_affine_S": 0.02990,                  # 44,908 B, 0.92x WORSE
            "xi_geometry_S": 0.03075,                # 46,178 B, 0.89x WORSE
            "xi_laneopt_S": 0.03160,                 # 47,453 B, 0.87x WORSE
            "roundtrip": "bit-exact (LBND3 dequantized geometry == LBND2 dequantized, all 3 arms)",
            "verdict": (
                "every xi arm is strictly LARGER whether the ego stream is counted or excluded -> "
                "the NO-GO is robust to the accounting choice; honest v8 Road<->Lane = 0.0275 S HELD"
            ),
        },
        residual=0.0,
        source_artifact=_MEMO,
        measurement_method="n600_argmax_lbnd3_full_blob_incl_ego_stream_bit_exact_roundtrip",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria=(
                "same as the isolated-payload anchor: an un-canonicalized chart or a new xi class "
                "that beats identity re-opens the formulation"
            ),
            measurement_axis=_ADVISORY,
            hardware_substrate="macos_arm64",
        ),
    )

    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=(
            "lane ground-frame xi-transport no-collapse: ego-freeze does NOT transfer to a chart that "
            "already absorbed the ego DOF; MEASURED ||innov_xi||_1 >= ||Delta Q||_1 for all xi "
            "(n600, isolated payload: identity 41,085 B / 7.58M beats best xi 42,017 B / 8.25M); the "
            "horizon's 14.6x does NOT transfer to the ground-canonicalized lane chart"
        ),
        one_line_summary=(
            "Ego-advected predictive coding LOSES to the identity temporal delta on a ground-frame lane "
            "chart (IPM already quotiented the ego DOF); MEASURED NO-GO, pointer UNMOVED."
        ),
        latex_form=(
            r"\lVert \mathrm{innov}_{\xi} \rVert_1 \;\geq\; \lVert \Delta Q \rVert_1"
            r"\quad \forall\, \xi \ \text{(ground-frame chart, IPM ego-quotiented)};"
            r"\quad \text{corollary: horizon-}14.6\times\ \text{transfers only to UN-canonicalized charts}"
        ),
        # Reproduction / producer callable: the real LBND3 ego-predictive coder that measured the
        # innovation stream (importable dotted path; this is a MEASUREMENT law, not a closed-form fn).
        python_callable_module_path=(
            "tac.boundary_math.analytic_lane_render_band:serialize_lane_band_rd3"
        ),
        domain_of_validity={
            "vehicle": ["v8_edge_centric_per_class_carrier"],
            "labels": "n600 GT argmax (gt_n600.npz['lstars'] + real gt_poses xi); bit-exact roundtrips",
            "verdict_scope": (
                "FORMULATION -- xi-advection on the ground-frame-canonicalized lane chart. PARADIGM "
                "INTACT: xi stays decisive for POSE (dual-use) and the image-frame horizon (14.6x)."
            ),
            "scope": (
                "SNAPSHOT-scope RATE-half NEGATIVE characterization on the DECODE side at THIS label "
                "cache -- NOT a converged-witness fact and NOT a byte-closed exact-eval score"
            ),
            "measurement_axis": ["macOS-CPU advisory"],
            "promotion_eligible": False,
            "note": (
                "means != ends: a MEASURED NO-GO for the lane rate axis. The law's positive value is "
                "chart-SELECTION guidance: xi-transport pays ONLY on un-canonicalized charts (image "
                "rows), NEVER on a chart whose IPM already absorbed the dominant ego DOF."
            ),
        },
        units_in={"lane_coeff_stream": "quantized_int_delta", "xi": "se2_ds_dy_dpsi_per_pair"},
        units_out={"innovation_l1": "sum_abs_quantized_delta", "s_rate_term": "contest_rate_score"},
        empirical_anchors=(anchor_lane_payload, anchor_full_blob),
        predicted_vs_empirical_residual={
            "isolated_payload_measured": 0.0,
            "full_blob_measured": 0.0,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            ".omx/research/SPEC_v8_perclass_decomposition_20260708.md",   # SPEC_v8 carrier-table
            ".omx/research/v8_increment1_design_draft_20260709.md",        # v8 increment-1 roll-up (use 0.0275)
            "v8_chart_selection_decision",                                 # future: WHERE xi-transport pays
        ),
        canonical_producers=(
            "tac.boundary_math.ego_xi_trajectory",            # xi estimators + exact SE(2) transport
            "tac.boundary_math.analytic_lane_render_band",    # LBND3 ego-predictive coder surface
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria=(
                "an un-canonicalized lane chart or a new xi predictor class beating identity on "
                "||innov||_1 at n600; OR a byte-closed exact-eval on a v8 archive shipping the lane "
                "generator (advisory -> contest rate authority)"
            ),
            measurement_axis=_ADVISORY,
            hardware_substrate="macos_arm64",
        ),
    )


def populate_lane_groundframe_xi_transport_no_collapse_equation(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration of the ground-frame xi-transport no-collapse law (mirrors
    ``populate_v8_geometric_rate_decomposition_equation``; latest-row-wins). EQUATIONS leg of the
    FEED-v8-lane-xi measurement; DSL leg = N/A (a measurement of a NEGATIVE lever, no trainer lever /
    launch / curriculum change)."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_lane_groundframe_xi_transport_no_collapse_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes="lane_groundframe_xi_transport_no_collapse_20260709 (equations leg of FEED-v8-lane-xi; "
              "MEASURED NO-GO -- ego-freeze does not transfer to a ground-canonicalized chart; DSL leg "
              "N/A = measurement of a negative lever; advisory NON-PROMOTABLE)",
    )
    return eq


__all__ = [
    "EQUATION_ID",
    "build_lane_groundframe_xi_transport_no_collapse_v1",
    "populate_lane_groundframe_xi_transport_no_collapse_equation",
]
