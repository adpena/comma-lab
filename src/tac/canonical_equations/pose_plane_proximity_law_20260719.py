# SPDX-License-Identifier: MIT
"""Canonical pose plane-proximity law (operator reframe 2026-07-19, CONFIRMED).

THE LAW: under the frozen scorers, d_pose is a corollary of the shared
384x512x3 scorer-plane's PROXIMITY to the source plane, not an independent
axis. PoseNet reads yuv6(A(frame)) (resize FIRST, upstream/modules.py:71-75),
i.e. the SAME plane SegNet reads; a smooth global 6-dim functional of that
plane barely moves inside a neighborhood of the source plane. Consequences:

1. A well-conditioned SOLVED d_seg (plane inside seg margin-bands around the
   SOURCE plane) carries pose essentially for free — the pose constraint is
   CONFIRMED INACTIVE at in-band solutions (pre-registered 2026-07-19 AM for
   the VJP/#549 bindingness harvest; MEASURED same day: 96/96 hard-oracle
   Pose constraints inactive/slack across 4 wide-band operating points on the
   n24 corpus, d_pose 7.7e-6..2.5e-5 at Seg scales 1e-4..1e-3 — merge
   6704c3857c, .omx/research/vjp_custody_positive_bands_20260719_codex.md).
2. A generator whose plane is FAR from source (the c2 witness: seg-only
   trained, plane RMSE ~25/255) destroys pose REGARDLESS of its d_seg — the
   yhat-ladder rung-B d_pose~63 was a GENERATOR artifact (verdict_scope:
   instance), never a family property of the inverse solve.
3. Marginal slack is enormous: pose only binds at d_pose ~ 2.5e-4 (crossover
   of d(sqrt(10*d))/dd = 5/sqrt(10*d) with seg's constant-100 marginal) —
   ~5.7 orders of magnitude above the ~5e-10 measured at solved planes
   (2.5e-4 / 5.35e-10 = 4.7e5; the earlier "~9 orders" wording was an
   arithmetic error, corrected 2026-07-19 per the fresh-eyes verification
   spec_v10_reconciliation_and_kkt_verify_20260719_fable.md V-1).

MEASURED anchors span the mechanism: near-source planes 5.35e-10..1.14e-9
(per-row MEANS at plane RMSE ~= 0; observed per-pair maxima 3.69e-9 n6 pair
424 and 2.04e-9 #549 — inside the x10 predicted band); far-off-source plane
63. Regime THRESHOLDS (rmse<1.0 near / rmse>=12.52 far) are ASSUMED design
radii — pre-registered falsifiable predictions, NOT measured regime edges;
measured support sits at rmse~=0 and rmse~=25 only. NO score/promotion claim;
this law constrains v10 CARRIER DESIGN (solve near source +
residual-vs-predictor bytes; do NOT spend training-side forces making a
far-off generator "pose-shaped").
"""

from __future__ import annotations

import math
from typing import Any

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "pose_plane_proximity_corollary_v1"
SOURCE_MEMO = ".omx/research/yhat_rd_ladder_20260719_codex.md"
DAG_FEED = ".omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md#FEED-pose-falls-out"
_UTC = "2026-07-19T07:10:00Z"

# MEASURED regime endpoints (plane RMSE in uint8 units vs source; d_pose 1/6-mean).
NEAR_SOURCE_DPOSE_MAX = 1.14e-9   # worst of the 3 solved-row MEANS (rung A); per-pair max observed 3.69e-9
FAR_PLANE_RMSE = 25.044688        # c2-witness yhat rung B
FAR_PLANE_DPOSE = 63.031066895    # its measured d_pose
POSE_BIND_CROSSOVER = 2.5e-4      # d_pose where pose marginal 5/sqrt(10 d) = seg's 100


def pose_regime_from_plane_proximity(
    plane_rmse_vs_source: float,
    *,
    seg_band_solved: bool,
) -> dict[str, Any]:
    """Classify the pose regime of a candidate ŷ from its plane proximity.

    Returns the design verdict the v10 carrier search consumes: planes solved
    inside seg margin-bands around the source (``seg_band_solved=True``, small
    RMSE) sit in the POSE_FREE regime (measured 5.35e-10..1.14e-9 at plane
    RMSE ~= 0, ~5.7 orders of magnitude below the 2.5e-4 binding crossover;
    the <1.0 RMSE radius is an ASSUMED pre-registered design radius — the
    measurements sit at rmse~=0); planes far from source sit in POSE_DESTROYED
    regardless of their d_seg (measured 63 at RMSE 25; the >=0.5*FAR threshold
    is likewise ASSUMED interpolation from that single instance). The
    intermediate region is UNMEASURED — the bytes(tau_pose) curve the VJP arm
    owes. Advisory design classifier, never a score.
    """

    if plane_rmse_vs_source < 0:
        raise ValueError("plane_rmse_vs_source must be >= 0")
    if seg_band_solved and plane_rmse_vs_source < 1.0:
        regime = "POSE_FREE_near_source"
        predicted_dpose_band = (0.0, NEAR_SOURCE_DPOSE_MAX * 10.0)
    elif plane_rmse_vs_source >= FAR_PLANE_RMSE * 0.5:
        regime = "POSE_DESTROYED_far_generator"
        predicted_dpose_band = (1.0, math.inf)
    else:
        regime = "UNMEASURED_intermediate_needs_tau_pose_curve"
        predicted_dpose_band = (NEAR_SOURCE_DPOSE_MAX, FAR_PLANE_DPOSE)
    return {
        "regime": regime,
        "predicted_dpose_band": predicted_dpose_band,
        "pose_bind_crossover": POSE_BIND_CROSSOVER,
        "pose_constraint_predicted_inactive": regime == "POSE_FREE_near_source",
        "score_claim": False,
        "promotion_eligible": False,
    }


def build_pose_plane_proximity_law_v1() -> CanonicalEquation:
    """Build the law with its measured regime-endpoint anchors."""

    def _prov(path: str):
        return build_provenance_for_research_sidecar(
            sidecar_path=path,
            reactivation_criteria=(
                "VJP/#549 harvest: bindingness maps (pose inactive at in-band "
                "solutions?) + bytes(tau_pose) curve fill the UNMEASURED "
                "intermediate regime; each new (plane_rmse, d_pose) point "
                "appends an anchor and recalibrates the regime boundaries."
            ),
            measurement_axis="[macOS-CPU advisory]",
            hardware_substrate="macos_arm64",
            captured_at_utc=_UTC,
        )

    anchor_near = EmpiricalAnchor(
        anchor_id="near_source_planes_pose_free_20260719",
        measurement_utc=_UTC,
        inputs={
            "rows": [
                "rung A source-exact yhat (n24): d_pose 1.14e-9",
                "n6 composition (both frames solved): 9.3e-10",
                "#549 zero-band joint control (f0 solve variable): 5.35e-10",
            ],
            "plane_rmse_vs_source": 0.0,
        },
        predicted_output="pose free at near-source planes",
        empirical_output=f"d_pose 5.35e-10..{NEAR_SOURCE_DPOSE_MAX} across 3 independent rows",
        residual=NEAR_SOURCE_DPOSE_MAX,
        source_artifact=".omx/research/yhat_rd_ladder_20260719_codex.json",
        measurement_method=(
            "exact lattice realization -> full upstream DistortionNet on real "
            "gt_n600 pairs (chunked receipts)"
        ),
        provenance=_prov(SOURCE_MEMO),
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
    )
    anchor_bindingness = EmpiricalAnchor(
        anchor_id="bindingness_harvest_pose_inactive_96of96_20260719",
        measurement_utc="2026-07-19T08:51:50Z",
        inputs={
            "corpus": "n24 real pairs, 4 wide-band operating points (96 rows)",
            "seg_scales": [1e-4, 1e-3],
            "tau_pose": [1e-4, 2.5e-4],
            "plane_rmse_vs_source": "in-band (positive Seg radii, all 56,623,104 channels)",
        },
        predicted_output=(
            "pre-registered: pose constraint inactive at in-band solutions "
            "(codimension argument, KKT derivation §5)"
        ),
        empirical_output=(
            "CONFIRMED: 96/96 hard-oracle Pose constraints inactive/slack at "
            "declared tau_pose; d_seg=0 every accepted row; measured d_pose "
            "7.67e-6..2.52e-5 (all >=10x under the 2.5e-4 crossover)"
        ),
        residual=2.521975392375284e-5,
        source_artifact=".omx/research/vjp_custody_positive_bands_20260719_codex.md",
        measurement_method=(
            "96 content-hashed bindingness NPZ sidecars; frozen hard oracle "
            "(native-fp32 CPU-torch SegNet/PoseNet) as sole admission authority; "
            "repair<=9 levels per row"
        ),
        provenance=_prov(".omx/research/vjp_custody_positive_bands_20260719_codex.md"),
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
    )
    anchor_far = EmpiricalAnchor(
        anchor_id="far_generator_plane_destroys_pose_20260719",
        measurement_utc=_UTC,
        inputs={
            "row": "rung B c2-witness yhat (seg-only-trained generator)",
            "plane_rmse_vs_source": FAR_PLANE_RMSE,
            "its_d_seg": 0.003455480,
        },
        predicted_output=(
            "operator hypothesis: d_pose~63 is a generator artifact of the "
            "messed-up-d_seg carrier, not a family property"
        ),
        empirical_output=f"d_pose {FAR_PLANE_DPOSE} at plane RMSE {FAR_PLANE_RMSE} (verdict_scope: instance)",
        residual=FAR_PLANE_DPOSE,
        source_artifact=".omx/research/yhat_rd_ladder_20260719_codex.json",
        measurement_method="same harness as the near-source rows; source frame0 policy",
        provenance=_prov(SOURCE_MEMO),
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
    )
    anchor_intermediate = EmpiricalAnchor(
        anchor_id="intermediate_regime_measured_secant_curve_20260719",
        measurement_utc="2026-07-19T10:11:42Z",
        inputs={
            "corpus": "n24 real pairs x 9 coarsening points (margin/precision/spatial families)",
            "coarseness_axis": (
                "description fidelity = plane proximity (coarser residual -> farther plane)"
            ),
        },
        predicted_output=(
            "law: pose free NEAR source, destroyed FAR; intermediate UNMEASURED — "
            "coarsening should raise d_pose monotonically toward the crossover then past it"
        ),
        empirical_output=(
            "MEASURED, mechanism CONFIRMED: margin family (nearest source) 96/96 inactive "
            "(d_pose 1.8e-8..8.4e-7); precision drop-1 24/24 inactive (3.9e-5); precision "
            "drop-2/3 = 6 per-pair crossover VIOLATIONS (3.03e-4..5.84e-4 vs 2.5e-4); "
            "spatial stride-8/16 (farthest plane) 48/48 violated, mean d_pose 0.85-1.02 = "
            "far-generator regime. Unrestricted all-in-band-rows inactivity REFUTED at n24 "
            "instance scope; pose inactivity is PLANE-PROXIMITY-CONDITIONAL exactly as the "
            "law states — the selected KKT segment (margin_m0p3<->precision_drop1) is "
            "120/120 inactive"
        ),
        residual=5.84e-4,
        source_artifact=".omx/research/seg_secant_rd_curve_n24_20260719_v2.json",
        measurement_method=(
            "full native-fp32 CPU-Torch hard oracle per pair/point; reachable uint8 "
            "preimages via generated-predictor block replacement / bit-plane truncation / "
            "stride-sampled reconstruction; merge c6b798f146"
        ),
        provenance=_prov(".omx/research/seg_secant_rd_curve_n24_20260719_v2.json"),
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="pose is a corollary of plane proximity to source",
        one_line_summary=(
            "d_pose tracks plane proximity to source, not compression: "
            "in-band solved planes 5e-10..1e-9 (~5.7 orders under the 2.5e-4 "
            "crossover); plane RMSE-25 off source gives 63 at any d_seg."
        ),
        latex_form=(
            r"d_{\mathrm{pose}}(\hat y)\approx d_{\mathrm{pose}}(y)+O(\|\hat y-y\|)"
            r"\ \ \text{(smooth 6-dim functional of the shared plane)};\quad"
            r"\text{binds only at } d_{\mathrm{pose}}^{*}=2.5\times10^{-4}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.pose_plane_proximity_law_20260719:"
            "pose_regime_from_plane_proximity"
        ),
        domain_of_validity={
            "scorers": "frozen SegNet + PoseNet, shared bilinear A (modules.py:71-75)",
            "claim_type": "v10 carrier-design regime classifier (advisory)",
            "intermediate_regime": (
                "MEASURED 2026-07-19 (secant 9-point curve, n24): monotone toward the "
                "crossover then past it — see anchor intermediate_regime_measured_"
                "secant_curve_20260719; bytes(tau_pose) full curve still owed at n600"
            ),
            "verdict_scope": (
                "far-generator anchor is instance-scoped (c2-witness-as-"
                "generator); near-source regime spans 3 independent rows; "
                "intermediate regime measured at n24 instance scope"
            ),
        },
        units_in={"plane_rmse_vs_source": "uint8 plane units (0..255)"},
        units_out={"regime": "categorical", "predicted_dpose_band": "d_pose (1/6-mean MSE)"},
        empirical_anchors=(anchor_near, anchor_bindingness, anchor_far, anchor_intermediate),
        predicted_vs_empirical_residual={
            "near_source_dpose_max": NEAR_SOURCE_DPOSE_MAX,
            "far_plane_dpose": FAR_PLANE_DPOSE,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools/measure_joint_seg_pose_rate.py (tau_pose sweep sizing)",
            "SPEC_v10 carrier design (solve-near-source + residual-vs-predictor)",
            ".omx/tmp/codex arm vjp_custody_positive_bands_20260719 (bindingness prediction)",
        ),
        canonical_producers=(
            ".omx/research/yhat_rd_ladder_20260719_codex.json",
            ".omx/research/joint_seg_pose_inverse_solve_receipt_n24_20260719.json",
            DAG_FEED,
        ),
        provenance=_prov(SOURCE_MEMO),
    )


# ======================================================================
# SUCCESSOR (append-only, 2026-07-19 Task #570; never mutate the corollary
# above). Source: #564 surprise review §1/§2 — the frozen Pose term is ONE
# global pooled Euclidean L2 ball, NOT 600 per-pair caps, and the 2.5e-4
# "binding crossover" is a coordinate-derivative identity, not a feasibility
# wall. DERIVED exactly from upstream/modules.py:82-84 (per-pair MSE q_i =
# ||e_i||^2/6) + upstream/evaluate.py:81-92 (global pool D_pose = (1/N) sum_i
# q_i; S_pose = sqrt(10*D_pose)). At N=600 this collapses to the native error
# norm S_pose(e) = ||e||_2 / sqrt(360), whose gradient has CONSTANT norm
# 1/sqrt(360) = 0.05270462766947299 away from e=0 (the 1/sqrt(D) blow-up of
# the D-coordinate derivative cancels the O(||e||) MSE gradient). No score /
# promotion claim; this successor re-expresses the frozen Pose term for the
# v10 C4/C9 objective (one global dual, per-pair telemetry as diagnostic).
# ======================================================================

EQUATION_ID_POOLED = "pose_global_norm_pooled_dual_v1"
_POOLED_UTC = "2026-07-19T18:30:00Z"
INV_SQRT_360 = 0.05270462766947299  # = 1/sqrt(360); S_pose per unit ||e||_2 at N=600
POSE_BIND_CROSSOVER_DPOSE = 2.5e-4  # coordinate-derivative identity, NOT a wall


def pose_global_norm_pooled_dual(
    per_pair_q: list[float],
    *,
    tau_dpose: float = POSE_BIND_CROSSOVER_DPOSE,
    n_pairs: int = 600,
) -> dict[str, Any]:
    """Frozen Pose term as ONE pooled global-norm constraint + the two feasible sets.

    ``per_pair_q[i]`` is the frozen per-pair MSE ``q_i = ||e_i||_2^2 / 6``
    (modules.py:82-84). The evaluator pools these into the single global mean
    ``D_pose = (1/N) sum_i q_i`` and scores ``S_pose = sqrt(10*D_pose)`` — which
    at ``n_pairs=600`` equals the native error norm ``||e||_2 / sqrt(360)``.

    Returns the score term plus BOTH feasibility verdicts so the v10 C4/C9
    objective can drop the 600 per-pair vetoes for the one pooled dual the
    evaluator actually implements:

      * ``pooled_feasible`` — ``D_pose < tau_dpose`` (the frozen score-term
        sublevel set: one ball ``sum_i ||e_i||^2 <= 6*N*tau`` in R^(6N)).
      * ``per_pair_veto_feasible`` — ``all(q_i < tau_dpose)`` (the strictly
        stricter C4 rule; a strict SUBSET that prohibits cross-pair allocation
        the score explicitly allows).

    The per-pair marginal ``dS_pose/dq_i = sqrt(10)/(2*N*sqrt(D_pose))`` (= 1/6
    at N=600, D=2.5e-4) is returned as the diagnostic; the 2.5e-4 crossover is
    the D-coordinate derivative identity (``5/sqrt(10*D)=100``), NOT a
    feasibility boundary. Advisory design classifier, never a score.
    """

    N = int(n_pairs)
    if N <= 0:
        raise ValueError("n_pairs must be > 0")
    if not per_pair_q:
        raise ValueError("per_pair_q must be non-empty")
    if any(q < 0 for q in per_pair_q):
        raise ValueError("per_pair_q entries are MSE magnitudes (>= 0)")
    if tau_dpose <= 0:
        raise ValueError("tau_dpose must be > 0")
    d_pose = sum(per_pair_q) / N
    s_pose = math.sqrt(10.0 * d_pose)
    e_norm = math.sqrt(6.0 * N * d_pose)  # ||e||_2 from the pooled mean
    marginal = math.sqrt(10.0) / (2.0 * N * math.sqrt(d_pose)) if d_pose > 0 else math.inf
    return {
        "d_pose_pooled": d_pose,
        "s_pose": s_pose,
        # Native-norm identity: sqrt(10*D_pose) == ||e||_2 / sqrt(6*N)
        # (== ||e||_2 / sqrt(360) at N=600). Both computed; equal by algebra.
        "s_pose_via_global_norm": e_norm * math.sqrt(10.0 / (6.0 * N)),
        "global_norm_coeff_at_n600": INV_SQRT_360,
        "pooled_feasible": d_pose < tau_dpose,
        "per_pair_veto_feasible": all(q < tau_dpose for q in per_pair_q),
        "per_pair_marginal_dS_dq": marginal,
        "pose_bind_crossover_is_coordinate_identity": True,
        "score_claim": False,
        "promotion_eligible": False,
    }


def build_pose_global_norm_pooled_law_v1() -> CanonicalEquation:
    """Successor to pose_plane_proximity_corollary_v1: the frozen Pose term is a
    single global L2 ball (pooled dual), and 2.5e-4 is a coordinate identity."""

    def _prov(path: str):
        return build_provenance_for_research_sidecar(
            sidecar_path=path,
            reactivation_criteria=(
                "any exact recomposition that re-ranks candidates under the "
                "pooled Pose dual vs the per-pair veto appends an anchor; a MAIN "
                "review that locates an executable per-pair-cap consumer with a "
                "separate operator/axis-robustness authority reopens the "
                "optional typed tail-risk guard (default OFF)."
            ),
            measurement_axis="[macOS-CPU advisory]",
            hardware_substrate="macos_arm64",
            captured_at_utc=_POOLED_UTC,
        )

    # DERIVED identity anchor: S_pose = sqrt(10*D_pose) == ||e||/sqrt(360) at N=600.
    anchor_identity = EmpiricalAnchor(
        anchor_id="pose_global_norm_identity_20260719",
        measurement_utc=_POOLED_UTC,
        inputs={
            "source": "upstream/modules.py:82-84 (per-pair MSE) + evaluate.py:81-92 (pool+score)",
            "n_pairs": 600,
            "identity": "sqrt(10*D_pose) == ||e||_2 / sqrt(6*N); at N=600 == ||e||_2/sqrt(360)",
        },
        predicted_output="gradient of S_pose(e) has constant norm 1/sqrt(360) away from e=0",
        empirical_output=(
            f"1/sqrt(360) = {INV_SQRT_360}; the 1/sqrt(D) D-coordinate blow-up cancels "
            "the O(||e||) MSE gradient -> 2.5e-4 is a coordinate-derivative identity, "
            "not a feasibility wall"
        ),
        residual=0.0,  # exact algebraic identity (source-derived)
        source_artifact=".omx/research/v10_frozen_space_surprises_20260719_codex.md",
        measurement_method="frozen-source linear/differential algebra (fp64 verification)",
        provenance=_prov(".omx/research/v10_frozen_space_surprises_20260719_codex.md"),
        empirical_verification_status="VERIFIED_VIA_SOURCE_INSPECTION",
    )
    # MEASURED recomposition witness: pooled dual admits lower-objective
    # allocations the per-pair veto rejects. Both are rate-dead (n600-scaled
    # CONDITIONAL range-payload; NOT viable archives) — a gate-logic witness only.
    anchor_recompose = EmpiricalAnchor(
        anchor_id="pooled_vs_perpair_veto_recomposition_20260719",
        measurement_utc=_POOLED_UTC,
        inputs={
            "custodied_rows": ".omx/research/seg_secant_rd_curve_n24_20260719_v2.json (n24 precision drops)",
            "tau_dpose": POSE_BIND_CROSSOVER_DPOSE,
            "byte_model": "n600-scaled CONDITIONAL range-payload (brotli_q11 B/pair * 600 / 37,545,489)",
        },
        predicted_output=(
            "per-pair veto is STRICTLY stricter than the frozen score term -> it can "
            "reject a lower-objective allocation whose global D_pose is feasible"
        ),
        empirical_output=(
            "CONFIRMED: drop-1 S=707.575004098 (D=3.87e-5, 0 viol, veto-PASS); "
            "drop-2 S=524.636100821 (D=7.53e-5, 2 viol, veto-REJECT, pooled-FEASIBLE); "
            "drop-3 S=457.546999260 (D=1.42e-4, 4 viol, veto-REJECT, pooled-FEASIBLE); "
            "drop-3 lower than veto-admitted drop-1 by 250.02800483805515. Pooled dual "
            "admits drop-2/drop-3 (both global D<2.5e-4) that the veto rejects. Rate-dead "
            "(hundreds) -> gate-logic witness, NOT viable archives / not a contest score"
        ),
        residual=250.02800483805515,  # score gap drop-1 - drop-3 (the veto's suppressed objective)
        source_artifact=".omx/research/seg_secant_rd_curve_n24_20260719_v2.json",
        measurement_method=(
            "$0 pure recomposition of custodied per-pair D_pose/d_seg/conditional-byte rows "
            "(no scorer re-run); [macOS-CPU advisory] conditional payload"
        ),
        provenance=_prov(".omx/research/task570_pose_law_successor_and_probes_20260719.md"),
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID_POOLED,
        name="frozen Pose term is one pooled global L2 ball, not 600 per-pair caps",
        one_line_summary=(
            "S_pose(e)=||e||/sqrt(360) at N=600 (one pooled dual); per-pair 2.5e-4 "
            "cap is a coordinate-derivative identity, strictly stricter than the score"
        ),
        latex_form=(
            r"S_{\mathrm{pose}}(e)=\sqrt{10\,D_{\mathrm{pose}}}"
            r"=\|e\|_2\big/\sqrt{6N}\ \xrightarrow{N=600}\ \|e\|_2/\sqrt{360};\quad"
            r"D_{\mathrm{pose}}=\tfrac1N\sum_i \|e_i\|_2^2/6;\ "
            r"\nabla S_{\mathrm{pose}}\ \text{has const norm } 1/\sqrt{360}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.pose_plane_proximity_law_20260719:"
            "pose_global_norm_pooled_dual"
        ),
        domain_of_validity={
            "scorers": "frozen PoseNet, per-pair MSE pooled to one global mean (evaluate.py:81-92)",
            "claim_type": "v10 C4/C9 Pose objective/dual reformulation (advisory, source-derived)",
            "supersedes_prose": (
                "the '600 per-pair q_i<2.5e-4 caps' reading of pose_plane_proximity_"
                "corollary_v1 / C4 / C9 / #536 — the evaluator has ONE global pool"
            ),
            "verdict_scope": (
                "identity is source-derived (paradigm); recomposition witness is n24 "
                "instance-scoped and rate-dead (gate-logic only, NOT a viable archive)"
            ),
        },
        units_in={
            "per_pair_q": "per-pair MSE ||e_i||^2/6 (pose units^2)",
            "tau_dpose": "d_pose (1/6-mean MSE)",
            "n_pairs": "count",
        },
        units_out={
            "s_pose": "score-law pose contribution",
            "d_pose_pooled": "d_pose (1/6-mean MSE)",
            "per_pair_marginal_dS_dq": "score per unit q_i",
        },
        empirical_anchors=(anchor_identity, anchor_recompose),
        predicted_vs_empirical_residual={
            "global_norm_identity": 0.0,
            "pooled_vs_veto_gap_score": 250.02800483805515,
        },
        last_calibration_utc=_POOLED_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "SPEC_v10 C4 Pose objective (one global norm/dual, not 600 caps)",
            "SPEC_v10 C9 KKT (one shared Pose lambda, per-pair telemetry as diagnostic)",
            "tools/measure_joint_seg_pose_rate.py (pooled tau sizing)",
        ),
        canonical_producers=(
            ".omx/research/v10_frozen_space_surprises_20260719_codex.md",
            ".omx/research/seg_secant_rd_curve_n24_20260719_v2.json",
            ".omx/research/task570_pose_law_successor_and_probes_20260719.md",
        ),
        provenance=_prov(".omx/research/v10_frozen_space_surprises_20260719_codex.md"),
    )
