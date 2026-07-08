# SPDX-License-Identifier: MIT
"""Canonical equation: Morse-Smale-stratified parallax warp — MEASURED d_pose floor of the cheap
task-space pose carrier (task #365; council-flagged in the design memo, anchored here at the advisory tier).

The store-nothing pose carrier synthesizes the consistent generated pair
``[warp(witness_f0_render, H(xi)), witness_f1_render]`` and PoseNet reads d_pose from it. The design memo
(pose_taskspace_native_morse_smale_depth_warp_design_20260708) predicted that stratifying the warp by the
argmax partition's depth (ground H(xi) + off-plane parallax) collapses d_pose; the extension predicted a
per-pair 6-DOF pose-space solve reaches ~0.0011. BOTH are REFUTED at the formulation level by the measured
M-ladder on the crucible run-1 checkpoint (epoch 200, w_pose=1.0):

    d_pose_floor(dof) = { 0 -> 1.685 (A0 deterministic global ground-H),
                          6 -> 1.486 (A2 pose-space LM/GN over xi_eff),
                         12 -> 1.223 (A2+ + 6 off-plane affine steering DOF, ORACLE GT mask) }   (medians)

The ladder DESCENDS monotonically but SHALLOWLY (~10-14%/rung) and floors ~1.2 — orders above the 0.019
target. ROOT CAUSE (measured, Rung 0): the off-plane finite-depth parallax MASS is ~0.5% (area ~2.7%) and
corr(d_pose, |ego translation|) is NEGATIVE (-0.45 n24 / -0.68 n8) -> the wall is NOT recoverable
off-plane parallax, it is the cartoon-pair appearance/flow-consistency vs the real photometric pair, which
no low-DOF (<=12) warp of the generated source escapes. (Free-pixel full-rank frame0 reaches ~2.7e-7,
#249, but is rate-prohibitive + adversarial -> firewall'd; the ancestor's 3.4e-5 is a BORROWED photometric
pair, never witness-validated.)

means != ends: advisory anchors, NON-PROMOTABLE; pointer 0.19110 UNMOVED. n600 + exact-eval OWED.
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    ASSUMED_AWAITING_VERIFICATION,
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

EQUATION_ID = "morse_smale_stratified_parallax_dpose_v1"

_UTC = "2026-07-08T00:00:00Z"
_ADVISORY = "[macOS-CPU advisory]"
_PREDICTED = "[predicted]"
_MEMO = ".omx/research/pose_mladder_depthwarp_measured_20260708.md"

# --- MEASURED constants (medians, frozen CPU-torch PoseNet through the real R; this checkpoint) ---------
DPOSE_A0_DETERMINISTIC = 1.685    # 0-DOF global ground-H warp (store-nothing), n24 median (mean 1.734)
DPOSE_A2_6DOF_SOLVE = 1.486       # per-pair 6-DOF pose-space LM solve over xi_eff, n24 median (mean 1.521)
DPOSE_A2PLUS_12DOF = 1.223        # + 6 off-plane affine steering DOF (ORACLE GT mask), n8 median (mean 1.567)
OFFPLANE_PARALLAX_MASS = 0.005    # row-weighted (1/Z) off-plane mass fraction (GT partition, Rung 0)
OFFPLANE_AREA_FRAC = 0.027        # off-plane finite-depth area fraction (movable + undrivable-below-horizon)
CORR_DPOSE_VS_EGO_TRANS = -0.446  # NEGATIVE (n24); d_pose does NOT rise with ego translation
DPOSE_TARGET_CONTRIB = 0.019      # sqrt(10*d_pose) target for pose to be a non-issue (~3.4e-5 ancestor)

_FLOOR_BY_DOF = {0: DPOSE_A0_DETERMINISTIC, 6: DPOSE_A2_6DOF_SOLVE, 12: DPOSE_A2PLUS_12DOF}


def measured_dpose_floor(dof: int) -> float:
    """The MEASURED cheap-warp-carrier d_pose floor (median) by DOF tier on the crucible run-1 witness.
    dof in {0 (A0 deterministic), 6 (A2 xi-solve), 12 (A2+ xi + off-plane affine, oracle mask)}.
    Fail-closed on an unmeasured tier (never a silent interpolation)."""
    if dof not in _FLOOR_BY_DOF:
        raise ValueError(f"dof {dof!r} not measured; measured tiers = {sorted(_FLOOR_BY_DOF)} "
                         "(0=A0, 6=A2, 12=A2+). n600/finer-DOF are OWED, not interpolated.")
    return _FLOOR_BY_DOF[dof]


def build_morse_smale_stratified_parallax_dpose_v1() -> CanonicalEquation:
    """Build the stratified-parallax d_pose-floor equation with its measured advisory anchors."""

    anchor_ladder = EmpiricalAnchor(
        anchor_id="mladder_a0_a2_a2plus_crucible_run1_20260708",
        measurement_utc=_UTC,
        inputs={
            "checkpoint": "crucible run-1 EMA (levelset_witness_ema_mlx.npz, ep200, n_pairs=600, "
                          "params=117527, self_orient, w_pose=1.0)",
            "authority": "frozen CPU-torch PoseNet, through the real R (uint8 warp); positive control "
                         "d_pose([gt_f0,gt_f1])=1.2e-12; NEVER MPS",
            "pair": "consistent generated: frame1=witness render, frame0=warp(witness f0 render)",
            "n_pairs": "24 (A0/A2) / 8 (A2+); bounded subset, direction-only",
        },
        predicted_output={
            "design_claim": "depth-stratified parallax warp collapses d_pose",
            "extension_claim": "6-DOF pose-space solve reaches ~0.0011",
        },
        empirical_output={
            "d_pose_A0_0dof_median": DPOSE_A0_DETERMINISTIC,
            "d_pose_A2_6dof_median": DPOSE_A2_6DOF_SOLVE,
            "d_pose_A2plus_12dof_median": DPOSE_A2PLUS_12DOF,
            "verdict": ("BOTH predictions REFUTED at formulation level: the cheap warp carrier floors "
                        "~1.2-1.7 (orders above the 0.019 target); the ladder descends only ~10-14%/rung; "
                        "the 6-DOF solve does NOT reach ~0.0011 (that #249 ~0 was a full-rank FREE-PIXEL "
                        "solve, rate-prohibitive). A2+ off-plane depth steering (oracle mask) adds only "
                        "-10% over A2 -> off-plane parallax is NOT the wall"),
        },
        residual=0.0,
        source_artifact=_MEMO,
        measurement_method="M-ladder per-pair LM/GN over warp DOF through frozen CPU-torch PoseNet (#365)",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria=("re-measure at n600 + exact-eval; re-measure on any pose-descent "
                                   "training run (R1-class) where the RENDER co-adapts (post-hoc warp "
                                   "cannot reproduce joint-training pose descent)"),
            measurement_axis=_ADVISORY,
            hardware_substrate="apple_m5_max_cpu",
        ),
    )
    anchor_rootcause = EmpiricalAnchor(
        anchor_id="mladder_rung0_offplane_mass_tiny_negcorr_20260708",
        measurement_utc=_UTC,
        inputs={"partition": "authoritative GT SegNet argmax (matches CLAUDE.md L80 class order)",
                "horizon_row": 437, "camera": "rectified pinhole (eon K, focal 910, principal centered)"},
        predicted_output={"design_hypothesis": "off-plane parallax loss drives d_pose ∝ |t|"},
        empirical_output={
            "offplane_parallax_mass": OFFPLANE_PARALLAX_MASS,
            "offplane_area_frac": OFFPLANE_AREA_FRAC,
            "corr_dpose_vs_ego_translation": CORR_DPOSE_VS_EGO_TRANS,
            "verdict": ("off-plane finite-depth mass ~0.5% (area ~2.7%); corr(d_pose,|t|) NEGATIVE "
                        "-> the depth-warp addressable fraction is tiny; bounds A1/A2+ win small BEFORE "
                        "any warp is built (confirmed by A2+ +10% only)"),
        },
        residual=0.0,
        source_artifact=_MEMO,
        measurement_method="Rung-0 free companion: partition off-plane mass + d_pose-vs-|xi| scatter",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria="recompute if clip/partition geometry changes (tac.clip_profile)",
            measurement_axis=_ADVISORY,
            hardware_substrate="apple_m5_max_cpu",
        ),
    )
    anchor_owed = EmpiricalAnchor(
        anchor_id="mladder_n600_exacteval_owed_20260708",
        measurement_utc=_UTC,
        inputs={"scope": "the advisory medians generalize to n600 + move the exact pointer"},
        predicted_output={"floor_by_dof": {str(k): v for k, v in _FLOOR_BY_DOF.items()}},
        empirical_output={"status": ("OWED — n600 + upstream/evaluate.py exact-eval; and re-validation "
                                     "of R1's cited 0.0011 through byte-close (a RUN not a carrier)")},
        residual=0.0,
        source_artifact=_MEMO,
        measurement_method="none yet (advisory n8-n24 only)",
        empirical_verification_status=ASSUMED_AWAITING_VERIFICATION,
        provenance=build_provenance_for_predicted(
            model_id="morse_smale_stratified_parallax_dpose.n600_owed",
            inputs_sha256="0" * 64,
            measurement_axis=_PREDICTED,
            hardware_substrate="apple_m5_max_cpu",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=("Morse-Smale stratified parallax warp: cheap task-space pose carrier floors d_pose ~1.2-1.7 "
              "(A0 0-DOF 1.685 -> A2 6-DOF 1.486 -> A2+ 12-DOF-oracle 1.223), orders above the 0.019 "
              "target; off-plane parallax mass ~0.5% is NOT the wall"),
        one_line_summary=("cheap warp pose carrier d_pose floors ~1.2 (even 12-DOF oracle-depth solve); "
                          "depth-stratification + 6-DOF solve BOTH refuted; wall = cartoon-pair "
                          "consistency, not off-plane parallax"),
        latex_form=(r"d_{pose}^{floor}(\mathrm{dof})=\{0{:}1.685,\ 6{:}1.486,\ 12{:}1.223\}\gg "
                    r"(\text{target }0.019),\quad \text{offplane-mass}\approx0.005"),
        python_callable_module_path=(
            "tac.canonical_equations.morse_smale_stratified_parallax_dpose_20260708:measured_dpose_floor"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness", "coord_inr_seg_witness"],
            "carrier": "store-nothing consistent generated pair (warp of witness render); post-hoc warp "
                       "solve (NOT joint pose-descent training)",
            "checkpoint": "crucible run-1 ep200 w_pose=1.0",
            "measurement_axis": ["macOS-CPU advisory", "predicted"],
            "note": ("FORMULATION-scoped negative on THIS carrier; named reformulations NOT refuted: "
                     "dedicated joint pose-descent RUN (R1-class, render co-adapts) — the only measured "
                     "path to low pose; free-pixel is rate-prohibitive; store-real-appearance is DEAD"),
        },
        units_in={"dof": "warp_degrees_of_freedom_int"},
        units_out={"d_pose": "posenet6_mse_median"},
        empirical_anchors=(anchor_ladder, anchor_rootcause, anchor_owed),
        predicted_vs_empirical_residual={
            # extension predicted A2 ~0.0011; MEASURED 1.486 -> the prediction miss is the finding.
            "a2_prediction_miss_vs_0p0011": abs(DPOSE_A2_6DOF_SOLVE - 0.0011),
            # A2+ off-plane-steering benefit over A2 (should be large per design; measured tiny).
            "a2plus_benefit_over_a2": DPOSE_A2_6DOF_SOLVE - DPOSE_A2PLUS_12DOF,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.boundary_math.stratified_depth_warp",   # the warp op the ladder measures
        ),
        canonical_producers=(
            "tools/pose_frame0_inverse_solve_probe.py",
            "experiments/train_levelset_witness_realized_through_R_mlx.py",
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria="n600 + exact-eval; re-validate R1 0.0011 through byte-close",
            measurement_axis=_ADVISORY,
            hardware_substrate="apple_m5_max_cpu",
        ),
    )


def populate_morse_smale_stratified_parallax_dpose_equation(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration of the stratified-parallax d_pose-floor law (equations leg of
    task #365; the prototype module is tac.boundary_math.stratified_depth_warp; the memo is the DAG leg)."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_morse_smale_stratified_parallax_dpose_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes="morse_smale_stratified_parallax_dpose_20260708 (task #365 M-ladder; advisory anchors, "
              "n600+exact-eval OWED)",
    )
    return eq


__all__ = [
    "CORR_DPOSE_VS_EGO_TRANS",
    "DPOSE_A0_DETERMINISTIC",
    "DPOSE_A2PLUS_12DOF",
    "DPOSE_A2_6DOF_SOLVE",
    "DPOSE_TARGET_CONTRIB",
    "EQUATION_ID",
    "OFFPLANE_AREA_FRAC",
    "OFFPLANE_PARALLAX_MASS",
    "build_morse_smale_stratified_parallax_dpose_v1",
    "measured_dpose_floor",
    "populate_morse_smale_stratified_parallax_dpose_equation",
]
