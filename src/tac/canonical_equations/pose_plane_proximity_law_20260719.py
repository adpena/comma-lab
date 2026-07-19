# SPDX-License-Identifier: MIT
"""Canonical pose plane-proximity law (operator reframe 2026-07-19, CONFIRMED).

THE LAW: under the frozen scorers, d_pose is a corollary of the shared
384x512x3 scorer-plane's PROXIMITY to the source plane, not an independent
axis. PoseNet reads yuv6(A(frame)) (resize FIRST, upstream/modules.py:71-75),
i.e. the SAME plane SegNet reads; a smooth global 6-dim functional of that
plane barely moves inside a neighborhood of the source plane. Consequences:

1. A well-conditioned SOLVED d_seg (plane inside seg margin-bands around the
   SOURCE plane) carries pose essentially for free — the pose constraint is
   predicted INACTIVE at in-band solutions (pre-registered for the VJP/#549
   bindingness harvest).
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
            "unmeasured": "intermediate plane-RMSE regime — the bytes(tau_pose) curve",
            "verdict_scope": (
                "far-generator anchor is instance-scoped (c2-witness-as-"
                "generator); near-source regime spans 3 independent rows"
            ),
        },
        units_in={"plane_rmse_vs_source": "uint8 plane units (0..255)"},
        units_out={"regime": "categorical", "predicted_dpose_band": "d_pose (1/6-mean MSE)"},
        empirical_anchors=(anchor_near, anchor_far),
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
