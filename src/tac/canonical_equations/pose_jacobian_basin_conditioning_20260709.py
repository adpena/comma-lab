# SPDX-License-Identifier: MIT
"""Canonical equation: the ξ→PoseNet Jacobian conditioning BASIN law — render coherence gates the
pose-descent basin, and its measurable sensor is σ_min(J_ξ) (2026-07-09).

Spec: ``.omx/research/pose_jacobian_conditioning_basin_trigger_formalization_20260709.md`` +
build memo ``.omx/research/pose_jacobian_basin_telemetry_build_20260709.md``. Extends
``morse_smale_stratified_parallax_dpose_v1`` (the cheap-carrier d_pose floor) with the OBSERVABILITY
axis it was missing.

THE LAW (DERIVED from the scorer's own structure; §1–§3 of the spec):

  J_ξ = ∂ PoseNet(R(θ,ξ))[:6] / ∂ξ  ∈ ℝ^{6×6} = J_P · J_R  (chain rule).
  On a flat/incoherent (piecewise-constant) render, ∇source appears as a MULTIPLICATIVE factor in
  J_R and is supported ONLY on the codim-1 boundary set (the annulus) — so few, near-straight edges ⇒
  few distinct edge-normals ⇒ the observable ξ-subspace has LOW rank ⇒ σ_min(J_ξ)→0 (some ego-motion
  direction is INVISIBLE through render+PoseNet). As d_seg converges the partition acquires more, finer,
  CURVED boundaries ⇒ richer normals ⇒ σ_min↑ and plateaus. σ_min is the basin variable.

  THE REACHABILITY FLOOR (the load-bearing callable): the pose-finish target sits at output-distance
  d_0 = ‖p(ξ_init) − t‖ = √(6·d_pose_init) (from ``modules.py`` compute_distortion: d_pose = ‖p−t‖²/6).
  Within a finish-stage trust-region of radius ρ_budget the target is reachable when σ_min·ρ_budget ≳ d_0,
  i.e. the pre-registered floor

      σ*(d_pose_init, ρ_budget) = √(6·d_pose_init) / ρ_budget.

  The (observer-only) fire criterion expresses this RELATIVE to the observed plateau:
  would-fire ⟺ median σ_min ≥ f_basin·σ_min^plateau  AND  basin_frac ≥ q.

⛔ HONESTY BOUND (from the aperture FALSIFICATION, ``pose_aperture_probe_measured_20260708``): σ_min>0
is NECESSARY (observability) but NOT SUFFICIENT — the reachable set must ALSO contain the target
(flow-model CONTENT correctness). A fixed cheap carrier fails the CONTENT axis (d_pose WORSE with
texture). σ_min is the correct basin sensor FOR JOINT DESCENT (θ co-adapts to supply the CONTENT the
fixed carrier lacked); it is NOT a resurrection of the fixed-cheap-carrier (dead, verdict_scope
FORMULATION).

THE seg⊥pose KERNEL (§4, DERIVED EXACT): ``modules.py:108`` — SegNet reads ONLY frame1; the pose ξ
shapes ONLY the seg-free frame0 ⇒ ∂d_seg/∂ξ ≡ 0 EXACTLY. This is the Jacobian-geometry reason the
~99.95% seg⊥pose null (#206/#227) is not a coincidence, and why engaging pose EARLIER is SAFE at any
epoch (the θ-channel residual coupling is the MEASURED ~0.05%, not exactly zero — OWED-on-measurement).

VERDICT (honest, NO-FAKE): the σ_min basin is the INSTRUMENT that makes the pose-timing question
MEASURABLE; it is NOT a proof that earlier engage wins — that optimality is the run-2 resume-A/B on
f_basin. The telemetry is score-NEUTRAL (read-only OBSERVER; B1 byte-identity VERIFIED by an
ON-vs-OFF checkpoint compare). Advisory axis (``[macOS-MLX advisory] NON-PROMOTABLE``); pointer
0.19110 UNMOVED.

Producer: ``experiments/train_levelset_witness_realized_through_R_mlx.py`` (the live T0/T1 sensor) +
``src/tac/witness_control/jacobian_basin.py`` (the σ_min/conditioning core). Consumers: the run-2
``TerminalPoseFinish(start_event='jacobian_basin')`` actuator (the f_basin<1 earlier-engage A/B) +
the offline σ_min(epoch) basin-curve reconstruction over the per-stage EMA checkpoints.
"""
from __future__ import annotations

import math

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

EQUATION_ID = "pose_jacobian_basin_conditioning_v1"

_UTC = "2026-07-09T00:00:00Z"
_ADVISORY = "[macOS-MLX advisory]"
_SPEC = ".omx/research/pose_jacobian_conditioning_basin_trigger_formalization_20260709.md"
_BUILD = ".omx/research/pose_jacobian_basin_telemetry_build_20260709.md"

# The scored PoseNet output width (modules.py compute_distortion uses the first out//2 == 6 dims).
N_POSE = 6


def reachability_floor_sigma_star(d_pose_init: float, rho_budget: float) -> float:
    """σ*(d_pose_init, ρ_budget) = √(6·d_pose_init) / ρ_budget — the pre-registered basin floor (spec §3).

    The pose-finish target is reachable within the finish trust-region when median σ_min(J_ξ) ≥ σ*.
    ``d_pose_init`` is the verdict d_pose row (PoseNet6 MSE); ρ_budget is the finish-stage ξ trust radius.
    Pure; ρ_budget<=0 ⇒ +inf (no reachable ball). DERIVED (not a fit); advisory / NON-PROMOTABLE."""
    d0 = math.sqrt(max(0.0, float(N_POSE) * float(d_pose_init)))
    r = float(rho_budget)
    return (d0 / r) if r > 0.0 else float("inf")


def would_fire_basin(
    median_sigma_min: float, sigma_min_plateau: float, basin_frac: float,
    *, f_basin: float = 1.0, quorum_q: float = 0.8,
) -> bool:
    """The observer-only relative fire criterion (spec §3): median σ_min ≥ f_basin·σ_min^plateau AND
    basin_frac ≥ q. f_basin=1.0 reproduces the current TERMINAL policy; f_basin<1 = the run-2 A/B."""
    return bool(
        float(median_sigma_min) >= float(f_basin) * float(sigma_min_plateau)
        and float(basin_frac) >= float(quorum_q)
    )


def build_pose_jacobian_basin_conditioning_v1() -> CanonicalEquation:
    """Build the ξ→PoseNet Jacobian conditioning-basin canonical equation (design law + advisory anchor)."""
    # Advisory anchor: the FIRST live σ_min basin readout on the crucible-v7 (minimal) render at the
    # incoherent `ce` baseline (n8 B1/B4 verification smoke 2026-07-09; MLX advisory NON-PROMOTABLE).
    # It CONFIRMS the DERIVED prediction: at low render-coherence J_ξ is near rank-deficient — σ_min
    # small, cond huge, effective-rank ≈ 1 — i.e. the pose descent is basin-EMPTY at ce (why terminal
    # engage-at-convergence is the sealed default). Plus the B1 score-neutrality receipt.
    anchor = EmpiricalAnchor(
        anchor_id="jacobian_basin_ce_baseline_and_byte_identity_20260709",
        measurement_utc=_UTC,
        inputs={
            "config": "crucible_v7 minimal render (self-orient + chroma + fused-R), n8, seg_form=ce",
            "k_pairs": 8,
            "base_point": "ξ0 = 0 (identity warp; observability of infinitesimal ego-motion)",
            "authority": "frozen MLX PoseNet through the real R; NON-PROMOTABLE (MLX advisory, NEVER a score)",
            "sensor": "mx.vjp 6-row assembly of J_ξ (6×6) + numpy SVD; finite-diff parity UNIT-TESTED",
        },
        predicted_output={
            "derived_prediction": "at low render-coherence (ce): σ_min(J_ξ)→small, cond→large, r_eff≈1 "
            "(near rank-deficient ⇒ basin-EMPTY); σ_min RISES as d_seg converges",
            "score_neutral": "telemetry is read-only ⇒ ON vs OFF trained artifact bit-identical (B1)",
        },
        empirical_output={
            "median_sigma_min_ce_ep1": 0.0633,
            "median_cond_ce_ep1": 8203.0,
            "median_r_eff_ce_ep1": 1.274,
            "basin_frac_ce": 1.0,
            "t0_render_grad_energy_ce_ep1": 15.48,
            "b1_byte_identity": "VERIFIED — telemetry ON vs OFF EMA + resume-state checkpoints "
            "bit-identical (max_abs 0.0) at matched epoch",
            "b4_launch_path": "GREEN — run starts, renders, hits the verdict path with the sensor on, "
            "emits basin rows (actuated:false); 0 fail-open skips on the happy path",
            "note": "r_eff≈1.27 at ce CONFIRMS the near-rank-deficient basin-empty prediction; the "
            "σ_min(epoch) curve to convergence is OWED (offline over per-stage checkpoints)",
        },
        residual=0.0,
        source_artifact=_BUILD,
        measurement_method="mx_vjp_jacobian_svd_conditioning_plus_on_vs_off_byte_identity_smoke_n8",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_BUILD,
            reactivation_criteria=(
                "reconstruct the full σ_min(epoch) basin curve offline over the run-1 per-stage EMA "
                "checkpoints; then the run-2 resume-A/B on f_basin<1; n600 + exact-eval before any "
                "promotable pose number"
            ),
            measurement_axis=_ADVISORY,
            hardware_substrate="m5_max_mlx_gpu",
        ),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )

    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=("ξ→PoseNet Jacobian conditioning BASIN: render coherence gates the pose-descent basin, "
              "measured by σ_min(J_ξ); σ*=√(6·d_pose_init)/ρ_budget is the reachability floor"),
        one_line_summary=(
            "σ_min(J_ξ=∂PoseNet(R(θ,ξ))[:6]/∂ξ)→0 on a flat render (∇source-gated boundary aperture), "
            "rises with d_seg; reachability floor σ*=√(6·d_pose_init)/ρ_budget"
        ),
        latex_form=(
            r"J_\xi=\partial \mathrm{PoseNet}(R(\theta,\xi))_{[:6]}/\partial\xi=J_P J_R;\ "
            r"\sigma_{\min}(J_\xi)\downarrow 0\ \text{as edge-normal diversity}\downarrow;\ "
            r"\sigma^*=\sqrt{6\,d_{pose}^{init}}/\rho_{budget}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.pose_jacobian_basin_conditioning_20260709:reachability_floor_sigma_star"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness", "coord_inr_seg_witness"],
            "carrier": "JOINT pose-descent (render θ co-adapts) — the observability axis; NOT the fixed "
                       "cheap-carrier (dead on the CONTENT axis per pose_aperture_probe_measured_20260708)",
            "measurement_axis": [_ADVISORY, "[predicted]"],
            "result_type": ("OBSERVABILITY sensor for the pose-timing question (score-NEUTRAL telemetry); "
                            "NOT a proof that earlier engage wins — that is the run-2 f_basin A/B"),
            "seg_perp_pose_kernel": "∂d_seg/∂ξ ≡ 0 EXACT (SegNet reads only frame1; ξ shapes only frame0)",
            "honesty_bound": "σ_min>0 NECESSARY not SUFFICIENT; reachable set must contain the target "
                             "(flow-model CONTENT correctness)",
            "owed": "the σ_min(epoch) curve to d_seg convergence (offline); n600 + exact-eval",
        },
        units_in={"d_pose_init": "posenet6_mse", "rho_budget": "xi_trust_region_radius_se3"},
        units_out={"sigma_star": "smallest_singular_value_floor"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "derived_near_rank_deficient_basin_empty_at_ce_confirmed": 0.0,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "experiments/train_levelset_witness_realized_through_R_mlx.py",
        ),
        canonical_producers=(
            "src/tac/witness_control/jacobian_basin.py",
        ),
        provenance=build_provenance_for_predicted(
            model_id="pose_jacobian_basin_conditioning.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_ADVISORY,
            hardware_substrate="m5_max_mlx_gpu",
        ),
    )


__all__ = [
    "EQUATION_ID",
    "N_POSE",
    "build_pose_jacobian_basin_conditioning_v1",
    "reachability_floor_sigma_star",
    "would_fire_basin",
]
