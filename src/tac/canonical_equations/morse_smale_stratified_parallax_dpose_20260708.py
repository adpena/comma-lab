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
_MEMO_APERTURE = ".omx/research/pose_aperture_probe_measured_20260708.md"
_MEMO_A1T = ".omx/research/pose_stratified_texture_probe_measured_20260708.md"
_MEMO_L2 = ".omx/research/pose_l2_truedepth_probe_measured_20260708.md"

# --- MEASURED constants (medians, frozen CPU-torch PoseNet through the real R; this checkpoint) ---------
DPOSE_A0_DETERMINISTIC = 1.685    # 0-DOF global ground-H warp (store-nothing), n24 median (mean 1.734)
DPOSE_A2_6DOF_SOLVE = 1.486       # per-pair 6-DOF pose-space LM solve over xi_eff, n24 median (mean 1.521)
DPOSE_A2PLUS_12DOF = 1.223        # + 6 off-plane affine steering DOF (ORACLE GT mask), n8 median (mean 1.567)
OFFPLANE_PARALLAX_MASS = 0.005    # row-weighted (1/Z) off-plane mass fraction (GT partition, Rung 0)
OFFPLANE_AREA_FRAC = 0.027        # off-plane finite-depth area fraction (movable + undrivable-below-horizon)
CORR_DPOSE_VS_EGO_TRANS = -0.446  # NEGATIVE (n24); d_pose does NOT rise with ego translation
DPOSE_TARGET_CONTRIB = 0.019      # sqrt(10*d_pose) target for pose to be a non-issue (~3.4e-5 ancestor)
DPOSE_A0T_BEST_TEXTURED = 15.14   # BEST (least-bad) observable-texture A0T point (P=32px, peak amp=4),
#   n24 median; the derived-spec grid ranges 15.1..117.5, ALL >> A0 1.685 -> aperture-fix FALSIFIED.
#   Diagnostic: warp(frame,xi) vs same frame = d_pose 166.8 (f0r) / 186.1 (f1r) -> the carrier's
#   ground-homography flow is NOT the true ego-motion; observable texture makes PoseNet read that WRONG
#   flow -> d_pose RISES. Mechanism = wrong-flow-observability (NOT aperture, NOT semantic-prior dominance).
DPOSE_A1T_BEST_STRATIFIED_TEXTURED = 2.608  # BEST (least-bad) A1T point (P=24px, peak amp=2), n24 median:
#   texture advected by the PER-CELL STRATIFIED flow (ground H(xi) + sky rotation-only R(xi) + hood identity
#   + off-plane H(xi)) -- the untested T x D cell. A1T grid range 2.6 .. 145 (ALL >= A0 flat 1.685, ABOVE
#   the A2+ 1.223 floor), monotone in amplitude, and per-cell is WORSE than global-H at matched settings
#   (P32/amp4: A1T 34.9 vs A0T 15.1). SELF-PAIR DIAGNOSTIC: per-cell flow reads d_pose 183.5 vs global-H
#   165.6 -> the piecewise per-cell flow is NOT more correct as PoseNet reads it (a discontinuous composite,
#   not a single rigid ego-motion). SCALE SWEEP (s*xi, s in {0.25..4} + translation flip): NO sharp optimum
#   away from s=1; monotone toward the flat floor as s->0 -> NO units/convention bug. Wrong-flow-observability
#   REFUTED for the geometric stratified carrier: correct flow needs real per-clip DEPTH (Option-A), not a
#   piecewise-geometric approximation. d_seg flips 0.8%(amp2)..9.5%(amp8) of f1 argmax (moot; d_pose refutes).

DPOSE_HPLAN_REAL_SELFFIT = 0.878   # L2 probe: homography self-fit on REAL luma (n24 median, warp,src);
#   == INVHOMOG (bit-parity validity gate). The real-luma flow reference (homography is a near-perfect flow
#   for the median pair). 15/24 pairs < 2; heavy tail (3 catastrophic ~97/169/184) -> mean 20.2.
DPOSE_L2_REAL_TRUEDEPTH = 1.296    # L2 probe: homography base + TRUE off-the-shelf mono-depth parallax
#   (Depth-Anything-V2-Small-hf, ground-plane scale+shift aligned) on REAL luma (n24 median). SLIGHTLY WORSE
#   than the plane homography (0.878) -> true dense depth does NOT help; net-neutral-to-harmful per-pair.
DPOSE_L2_WITNESS_TRUEDEPTH = 171.77  # L2 probe: same true-depth flow on WITNESS luma (n24 median) ~ the
#   zero-motion null (HPLAN_WITNESS 166.8) -> witness/cartoon render is pose-BLIND regardless of flow.

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
    anchor_r1_custody = EmpiricalAnchor(
        anchor_id="r1_0011_custody_revalidation_20260708",
        measurement_utc=_UTC,
        inputs={
            "run": "experiments/results/levelset_n600_R1_storenothing_descent_ev1_20260703T004906Z "
                   "(#245, warm-start converged v2_attrclean mod-dim 26, ep1001 inside Muon, w_pose=1.0, "
                   "pose-carrier generated + per-pair dxi table)",
            "authority": "frozen CPU-torch PoseNet, through real R (uint8 warp), n600 (verdict-pairs 0, "
                         "verdict-batch 32); EMA-shadow (conservative during descent); NEVER MPS/MLX",
            "target": "gt_poses = _cpu_pose_raw(PoseNet, real_f0, real_f1)[:6] (cache gt_n600.npz) => "
                      "d_pose = MSE(PoseNet(generated)[:6], PoseNet(real)[:6]) = EXACT contest definition",
        },
        predicted_output={
            "symposium_claim": "R1 d_pose 0.0011 = SOLID SHIPPABLE pose floor (contribution 0.105)",
            "charter_prime_suspect": "0.0011 is a xi-parameter-space MSE, not PoseNet-output d_pose",
        },
        empirical_output={
            "d_pose_R1_descent": "97.2(ep1000) -> 62.4(ep1001) -> 0.00334(ep1021) -> 0.00108(ep1074) "
                                 "-> 0.001012(ep1108); d_seg held ~0.0046; ep_loss 143-713 (live)",
            "prime_suspect": "REFUTED — verdict routes cpu_verdict_d_pose_batch => MSE vs PoseNet(real_pair)"
                             "[:6], NOT xi-space; gt_poses are PoseNet OUTPUTS (verified cache).",
            "mechanism": "the descent lives ENTIRELY in the trained per-pair pose_carrier.dxi (600,6) table "
                         "(checkpoint absmean 0.0038); xi_stored-alone read-back = 2.562 (the 'cap ~2.5').",
            "shippability": "NOT byte-closed — the serializer recomputes deterministic calibration xi and "
                            "does NOT ship dxi (arms_measured:80) => a byte-close reports the ~1.99 no-dxi "
                            "floor, not 0.0011. #238 must serialize xi_eff=xi_stored+dxi (~7.2KB, ~0.0005 "
                            "rate) then re-measure through inflate.",
            "verdict": ("(a)-with-caveat: joint pose-descent (render co-adapts) is a REAL, validly-measured "
                        "path to low d_pose (prime suspect DEAD) -> a dedicated run is justified; the "
                        "specific 'SOLID shippable 0.105' is OVER-STATED -> downgrade to 'SOLID training-side "
                        "advisory; shippability PENDING #238'. NOT an artifact (row (b) does not fire). The "
                        "run-1 1.79 vs 0.0011 gap = DIFFERENT checkpoints (mod-32 ep200 un-descended vs "
                        "mod-26 ep1108 descended), NOT a contradiction. FEED-posesolve 'w_pose=0' is wrong "
                        "(checkpoint __cfg_w_pose=1.0)."),
        },
        residual=0.0,
        source_artifact=".omx/research/r1_0011_custody_revalidation_20260708.md",
        measurement_method="custody re-validation: measurement-path source-inspection + relaunch.log n600 "
                           "verdict rows + checkpoint dxi npz inspection (read-only, no re-execution)",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=".omx/research/r1_0011_custody_revalidation_20260708.md",
            reactivation_criteria=("#238: serialize xi_eff=xi_stored+dxi in store_nothing mode, re-measure "
                                   "d_pose through the real inflate/decode at n600 on R1's checkpoint (or a "
                                   "fresh dedicated joint pose-descent arm carried to convergence)"),
            measurement_axis=_ADVISORY,
            hardware_substrate="apple_m5_max_cpu",
        ),
    )
    anchor_aperture = EmpiricalAnchor(
        anchor_id="aperture_probe_A0T_falsified_crucible_run1_20260708",
        measurement_utc=_UTC,
        inputs={
            "probe": "A0T = A0 consistent pair + seeded ISOTROPIC band-limited texture advected by the "
                     "SAME H(xi); derived spec (posenet_scorer_side_deepdive): period 16/24/32/48 px@874, "
                     "peak amp 4/8/16/32 uint8, luma+chroma & DC-match toggles",
            "authority": "frozen CPU-torch PoseNet through real R (uint8); poscontrol 2.1e-12; NEVER MLX",
            "n_pairs": "24 (A0T grid) / 8 (A2T best-shot solve); |t|-stratified",
        },
        predicted_output={
            "aperture_hypothesis": "observable xi-consistent texture DROPS d_pose (selective low-|t| rescue)",
        },
        empirical_output={
            "d_pose_A0_flat_median": DPOSE_A0_DETERMINISTIC,
            "d_pose_A0T_best_textured_median": DPOSE_A0T_BEST_TEXTURED,  # P=32/amp4, ~9x WORSE
            "d_pose_A0T_grid_range": "15.1 .. 117.5 (ALL >> A0 1.685; monotone in amplitude)",
            "low_t_rescue": "ABSENT (low-|t| tracks/exceeds high-|t|)",
            "d_pose_A2T_solve_on_textured_median": 12.65,  # 6-DOF solve on textured pair (P32/amp4), n8;
            # ~8.5x WORSE than flat-pair A2 1.486 -> texture does NOT revive the solve either,
            "diagnostic_warp_vs_self_dpose": "166.8 (f0r) / 186.1 (f1r) -> carrier flow != true ego-motion",
            "verdict": ("APERTURE HYPOTHESIS FALSIFIED (formulation-scoped): observable texture RAISES "
                        "d_pose because the only flow a cheap store-nothing carrier can paint is the "
                        "ground-homography flow, which is WRONG; making it legible amplifies the error. "
                        "Semantic-prior dominance ALSO not indicated (PoseNet responds strongly to injected "
                        "flow: 1.7->118). Wall = FLOW-MODEL CORRECTNESS, not observability. NOT refuted: "
                        "joint pose-descent RUN (render co-adapts), Option-A depth-warp (correct flow)."),
        },
        residual=0.0,
        source_artifact=_MEMO_APERTURE,
        measurement_method="A0T advected-texture grid + A2T 6-DOF solve on textured pair, frozen CPU-torch PoseNet",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO_APERTURE,
            reactivation_criteria=("re-measure on a CORRECT-flow carrier (Option-A depth-warp, per-clip "
                                   "stored depth) or on a joint pose-descent RUN where the render co-adapts; "
                                   "n600 + exact-eval"),
            measurement_axis=_ADVISORY,
            hardware_substrate="apple_m5_max_cpu",
        ),
    )
    anchor_a1t = EmpiricalAnchor(
        anchor_id="a1t_stratified_texture_falsified_crucible_run1_20260708",
        measurement_utc=_UTC,
        inputs={
            "probe": "A1T = seeded isotropic band-limited LUMA texture (DC-matched) advected by the PER-CELL "
                     "STRATIFIED flow: ground(0,1)->H(xi); sky(2 above horizon)->rotation-only R(xi),t=0; "
                     "hood(4)->identity; off-plane movable(3)+struct(2 below)->H(xi) (zero affine; off-plane "
                     "mass ~0.5% so un-fit affine ~ zero). GT argmax = oracle cells. period 24/32/48 px@874 x "
                     "peak amp 2/4/8 uint8. The untested T x D cell (A0T=texture x global-H; A2+=solve x flat).",
            "authority": "frozen CPU-torch PoseNet through real R (uint8); poscontrol 2.1e-12; NEVER MLX",
            "n_pairs": "24 (A1T grid + self-pair diagnostic n8 + scale sweep n24); |t|-stratified",
        },
        predicted_output={
            "wrong_flow_observability": "per-cell-CORRECT flow made legible collapses d_pose << 1.223",
        },
        empirical_output={
            "d_pose_A0_flat_median": DPOSE_A0_DETERMINISTIC,
            "d_pose_A2plus_flat_stratified_median": DPOSE_A2PLUS_12DOF,
            "d_pose_A1T_best_median": DPOSE_A1T_BEST_STRATIFIED_TEXTURED,  # P24/amp2; ABOVE both flat arms
            "d_pose_A1T_grid_range": "2.6 .. 145 (monotone in amp; P32/amp4 34.9 WORSE than A0T 15.1)",
            "self_pair_diagnostic": "per-cell flow d_pose 183.5 vs global-H 165.6 -> per-cell NOT more correct",
            "scale_sweep": "s*xi in {0.25,0.5,1,2,4} + tflip: monotone toward flat as s->0, NO sharp optimum "
                           "away from s=1 (s0.25=22.8, s1=34.9), tflip only mildly lower (24.1) -> NO "
                           "units/convention bug",
            "dseg_flip_frac": "0.8% (amp2) .. 9.5% (amp8) of f1 SegNet argmax (moot; d_pose refutes first)",
            "verdict": ("WRONG-FLOW-OBSERVABILITY REFUTED for the geometric stratified carrier "
                        "(formulation-scoped): the piecewise per-cell flow (ground-H + sky-rotation + "
                        "hood-static) is a DISCONTINUOUS composite, NOT a single rigid ego-motion; PoseNet "
                        "reads it as WORSE motion than global-H (183 vs 165), so making it legible RAISES "
                        "d_pose (best 2.6 > A2+ 1.223 > A0 1.685). No scale/sign bug hides a correct flow. "
                        "Correct scene flow requires real per-clip DEPTH (Option-A depth-warp), not a "
                        "piecewise-geometric approximation. NOT refuted: Option-A (stored depth), joint "
                        "pose-descent RUN (render co-adapts), pose-as-budget-item."),
        },
        residual=0.0,
        source_artifact=_MEMO_A1T,
        measurement_method="A1T per-cell-advected-texture grid + self-pair flow-correctness diagnostic + "
                           "s*xi/sign scale sweep, frozen CPU-torch PoseNet",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO_A1T,
            reactivation_criteria=("re-measure on Option-A depth-warp (per-clip STORED depth D + stored xi, "
                                   "SfMLearner K T(xi) D K^-1 -> a CORRECT non-piecewise flow) or on a joint "
                                   "pose-descent RUN where the render co-adapts; n600 + exact-eval"),
            measurement_axis=_ADVISORY,
            hardware_substrate="apple_m5_max_cpu",
        ),
    )
    anchor_l2 = EmpiricalAnchor(
        anchor_id="l2_truedepth_falsified_crucible_run1_20260708",
        measurement_utc=_UTC,
        inputs={
            "probe": "L2 = SELF-PAIR (frame1 = depth-warp of frame0 by GT ego-motion) isolating flow x "
                     "appearance. Depth = Depth-Anything-V2-Small-hf (HF rev 5426e4f0, 24.8M, inverse-depth) "
                     "scale+shift aligned to the calibrated ground plane on GT-Road pixels. Flow = the true "
                     "parallax the plane homography MISSES: delta = proj(K(R Z_true ray + t)) - proj(K(R "
                     "Z_plane ray + t)), SAME (R,t)=exp_se3(xi) as the homography -> delta=0 on-plane "
                     "(median 3.91px off-plane). Apparatus = INVERSE homography warp + delta correction "
                     "(hole-free; base bit-identical to warp_frame0 -> validity gate).",
            "authority": "frozen CPU-torch PoseNet through real R (uint8); poscontrol 5.8e-12; NEVER MLX",
            "n_pairs": "24 (n8 confirms); warp,src ordering (deployment self-fit)",
        },
        predicted_output={
            "decisive_question": "is the pose wall (a) crude depth/flow MODELS (true dense depth fixes it) "
                                 "or (b) PoseNet needs real scene content beyond correct flow?",
        },
        empirical_output={
            "d_pose_HPLAN_REAL_selffit_median": DPOSE_HPLAN_REAL_SELFFIT,   # 0.878 == INVHOMOG (validity ok)
            "d_pose_L2_REAL_truedepth_median": DPOSE_L2_REAL_TRUEDEPTH,     # 1.296 -> WORSE than plane
            "d_pose_L2_WITNESS_truedepth_median": DPOSE_L2_WITNESS_TRUEDEPTH,  # 171.8 -> pose-blind
            "offplane_parallax_px_median": 3.91,
            "raft_ceiling": "INVALID (half-res raft_small backward flow degenerate ~0.49px -> ~188 null); "
                            "not load-bearing (HPLAN_REAL 0.878 IS the real-luma flow reference)",
            "verdict": ("TRUE-DEPTH-FLOW FALSIFIED (formulation-scoped): off-the-shelf true dense mono-depth "
                        "does NOT beat the plane homography on real luma (1.296 vs 0.878, net-harmful per-"
                        "pair) and does NOT rescue the pose-blind witness (171.8). The wall is PHOTOMETRIC "
                        "APPEARANCE, not flow-model crudeness (witness cartoon = pose-blind ~167; the gap "
                        "real-selffit 0.9 -> deployable 1.7-1.8 is appearance). Decision row 2->3: cure = "
                        "JOINT pose-descent RUN (render co-adapts, #238), NOT paid stored-depth Option-A. "
                        "Closes A1T's 'correct flow needs real depth' reformulation: true depth measured, "
                        "does not help. REFUTED as levers: Option-A stored-depth, store-real-appearance."),
        },
        residual=0.0,
        source_artifact=_MEMO_L2,
        measurement_method="L2 self-pair: inverse homography warp + true-mono-depth parallax delta, "
                           "real-vs-witness luma, frozen CPU-torch PoseNet (validity gate HPLAN==selffit)",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO_L2,
            reactivation_criteria=("re-measure on a JOINT pose-descent RUN where the render co-adapts "
                                   "(R1-class, #238) -- the only open path to low pose; n600 + exact-eval; "
                                   "a valid full-res dense-flow ceiling"),
            measurement_axis=_ADVISORY,
            hardware_substrate="apple_m5_max_cpu",
        ),
    )
    anchor_r1_dxi_shippability = EmpiricalAnchor(
        anchor_id="r1_dxi_shippability_byteclose_20260708",
        measurement_utc=_UTC,
        inputs={
            "run": "R1 checkpoint levelset_witness_ema_mlx.npz (mod-dim 26, ep1108; pose_carrier.xi_stored "
                   "absmean 0.2296 + trained pose_carrier.dxi absmean 0.00382; __cfg_w_pose=1.0)",
            "connector": "tools/levelset_byte_close_and_eval.py --pose-carrier-xi-from-ckpt "
                         "[--pose-carrier-dxi-scale S]: ships ξ_eff = xi_stored + S*dxi (store-nothing v2 "
                         "coded ξ payload) instead of the deterministic xi_from_pose_calibration recompute",
            "authority": "realized d_pose on the INFLATED .raw (frozen CPU-torch PoseNet, through real R, "
                         "gt_n600 cache), n24 (n STATED; archive/ξ section built for all 600 pairs)",
            "geom": "pitch=0.0 (--pc-pitch 0.0) == trainer GroundHomographyGeom.eon(pitch=0.0); H derived "
                    "from the SHIPPED ξ (rule-118 FREE), K/n/d bit-identical to the deriver",
        },
        predicted_output={
            "question_238": "does serialized ξ_eff=xi_stored+dxi reproduce R1's trained d_pose 0.0011 "
                            "through the REAL inflate/decode (SHIPPABLE), or degrade (GAP)?",
        },
        empirical_output={
            "ship_dxi_realized_d_pose": "0.001127 (n24) vs R1 training-side 0.001012 (ep1108) => SURVIVES "
                                        "byte-close; pose contribution sqrt(10*d_pose)=0.106 (~symposium 0.105)",
            "matched_no_dxi": "0.02197 (SAME fitted xi_stored, dxi off via --dxi-scale 0); dxi = a 20x "
                              "refinement on an already pose-legible co-adapted witness pair",
            "calibration_wrong_s_t": "26.04 (default s_t=0.16, the naive no-calibration byte-close reads "
                                     "this) -> s_t calibration matters; the matched isolate uses R1's fitted xi_stored",
            "ceiling_real_warp_real": "2.561 (no-dxi, == custody 'cap 2.56') -> 1.958 (dxi)",
            "counted_bytes": "ξ_eff store-nothing section 7,195 B (6,634 B coded, raw ref 7,232 B; dxi "
                             "jitter kills delta smoothness), rate 25*7195/37.5M = 0.004791; full archive "
                             "89,772 B; +3,433 B / +0.0023 rate over matched-no-dxi for the 20x gain",
            "frame0_decode_caveat": "bit-exact=False max_abs 14/16/18 (calib/no-dxi/dxi) in ALL configs => "
                                    "pre-existing store_nothing warp-reproduction (inflate-vs-oracle numerics), "
                                    "NOT dxi-specific; d_pose is on the shipped deterministic .raw (authority)",
            "verdict": ("SHIPPABLE: R1's trained pose ships in ~7.2KB COUNTED ξ and reproduces d_pose 0.0011 "
                        "through the real byte-closed inflate. The custody downgrade ('shippability PENDING "
                        "#238') is RESOLVED. verdict_scope: pose HALF on THIS checkpoint via the store-nothing "
                        "ξ_eff carrier; does NOT touch the d_seg blocker (0.40 seg term) nor claim a frontier "
                        "move. Advisory; upstream/evaluate.py CPU remains the promotion authority."),
        },
        residual=abs(0.001127 - 0.001012),  # realized-vs-training-side d_pose gap (survives byte-close)
        source_artifact=".omx/research/r1_dxi_shippability_byteclose_20260708.md",
        measurement_method="byte-close ξ_eff serialization + full inflate -> frozen CPU-torch PoseNet on "
                           "inflated .raw; 3-point A/B (calibration / matched-no-dxi scale0 / ship-dxi scale1)",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=".omx/research/r1_dxi_shippability_byteclose_20260708.md",
            reactivation_criteria=("n600 realized-d_pose row (in progress) + exact upstream/evaluate.py CPU; "
                                   "a dedicated from-converged pose-finishing descent could push toward the "
                                   "0.018-class ancestor (operator-GO, NOT launched)"),
            measurement_axis=_ADVISORY,
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
        empirical_anchors=(anchor_ladder, anchor_rootcause, anchor_aperture, anchor_a1t, anchor_l2, anchor_owed, anchor_r1_custody, anchor_r1_dxi_shippability),
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
    "DPOSE_A1T_BEST_STRATIFIED_TEXTURED",
    "DPOSE_A2PLUS_12DOF",
    "DPOSE_A2_6DOF_SOLVE",
    "DPOSE_HPLAN_REAL_SELFFIT",
    "DPOSE_L2_REAL_TRUEDEPTH",
    "DPOSE_L2_WITNESS_TRUEDEPTH",
    "DPOSE_TARGET_CONTRIB",
    "EQUATION_ID",
    "OFFPLANE_AREA_FRAC",
    "OFFPLANE_PARALLAX_MASS",
    "build_morse_smale_stratified_parallax_dpose_v1",
    "measured_dpose_floor",
    "populate_morse_smale_stratified_parallax_dpose_equation",
]
