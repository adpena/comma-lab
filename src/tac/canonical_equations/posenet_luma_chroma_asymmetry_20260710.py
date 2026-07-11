# SPDX-License-Identifier: MIT
"""Canonical equation: SCORER DIMENSION ASYMMETRY — luma≫chroma for pose, frame_0 seg-free.

Operator re-read (2026-07-10) of ``upstream/evaluate.py`` + ``modules.py`` + ``frame_utils.py`` across
ALL dimensions (chroma/luma/yuv/hydra/order/dtype), in light of the palette finding
(``palette_artifact_probe_20260710``: SegNet argmax is TEXTURE/CONTEXT-dominated, Road/Lane never win on
colour alone). Two structural pose-free seg DOFs surface, both UNPRICED in the current witness/v8 build.

MEASURED (real GT pair, PoseNet weights, per-channel Jacobian of the 6 scored pose dims w.r.t. the 12
yuv6 input channels; ``upstream_scorer_alldim_reread_20260710``):

  * **Luma is 11.1x more pose-sensitive PER PLANE than chroma** (aggregate 22.2x over 8 luma vs 4 chroma
    planes). Per-channel Jacobian energy: luma planes ~5.0e-3 each, U/V ~3e-4 to 7e-4 each.
  * **frame1/frame0 pose energy = 0.86x** — both frames carry pose ~equally (pose is differential).

DERIVED-FROM-CODE (exact, not measured):

  * **Chroma above the 2x2 yuv-grid is EXACTLY pose-null.** ``rgb_to_yuv6`` (frame_utils:65-72) forms
    U_sub/V_sub as a 2x2 BOX-AVERAGE of the 384x512 chroma -> a sinc frequency response that nulls at the
    2x2 grid; RGB chroma detail above that grid is box-averaged to a constant before PoseNet sees it.
  * **Luma is carried LOSSLESS at 384x512** — the 4 polyphase planes (y00,y10,y01,y11) are the full-res
    luma re-tiled into 4 channels at 192x256; no luma information is lost (frame_utils:74-77).
  * **frame_0 is SegNet-FREE.** ``SegNet.preprocess_input`` slices ``x[:, -1, ...]`` (modules:108) — only
    frame_1 is scored by SegNet; frame_0's d_seg constraint is identically zero (structurally validated
    seg_delta=0.0, cascade_c_prime_..._20260526). SegNet also takes RAW RGB (no yuv, no norm) at full
    384x512, so chroma feeds SegNet at full resolution while it is pose-null at high freq.

CONSEQUENCE (the two overlooked levers, both pose-free seg DOFs, both unpriced in witness/v8):
  1. frame_0 -> pose-only carrier (zero seg-texture obligation; half the rendered pixels).
  2. chroma-high-freq -> seg TEXTURE carrier (free-for-pose x needed-for-seg; resolves the palette
     dilemma — SegNet is texture-dominated and colour-flat kills Road/Lane, but chroma-high-freq gives
     SegNet texture at ~zero pose cost).

Authority: ``[macOS-CPU advisory . real-GT-pair . NON-PROMOTABLE]`` — a Jacobian energy is NEVER a score;
pointer contest-CPU 0.19110 UNMOVED (MEANS). The pointer moves only through a byte-closed n600 exact eval.
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    VERIFIED_VIA_SOURCE_INSPECTION,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "posenet_luma_chroma_sensitivity_asymmetry_v1"

_UTC = "2026-07-10T00:00:00Z"
_ADVISORY = "[macOS-CPU advisory . real-GT-pair PoseNet-Jacobian . NON-PROMOTABLE]"
_DAG = ".omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md"
_MEMO = ".omx/research/upstream_scorer_alldim_reread_20260710.md"

# --- MEASURED per-channel PoseNet Jacobian energy (Σ over 6 scored pose dims of grad², real GT pair) ---
LUMA_PER_PLANE_AVG = 5.006e-3         # mean of the 8 luma-plane energies
CHROMA_PER_PLANE_AVG = 4.502e-4       # mean of the 4 chroma-plane energies
LUMA_CHROMA_PER_PLANE_RATIO = 11.1    # luma/chroma per-plane pose sensitivity
LUMA_CHROMA_AGGREGATE_RATIO = 22.2    # 8 luma vs 4 chroma total
FRAME1_OVER_FRAME0_POSE = 0.86        # both frames carry pose ~equally
FRAME0_SEGNET_SEG_DELTA = 0.0         # frame_0 is seg-free (x[:,-1]); structurally validated


def build_posenet_luma_chroma_asymmetry_v1() -> CanonicalEquation:
    """Build the scorer-dimension-asymmetry law (luma≫chroma pose, frame_0 seg-free)."""

    anchor = EmpiricalAnchor(
        anchor_id="posenet_luma_chroma_jacobian_realgt_20260710",
        measurement_utc=_UTC,
        inputs={
            "harness": "per-channel autograd Jacobian of PoseNet pose[:6] w.r.t. yuv6 12ch input",
            "weights": "upstream/models/posenet.safetensors (FastViT-T12, IN_CHANS=12)",
            "data": "experiments/results/mlx_fleet_gt_cache/gt_n6.npz real GT pairs (n<=4)",
            "code_refs": "modules:64-84,107-109; frame_utils:51-78",
        },
        predicted_output={
            "question": "which YUV directions are in PoseNet's null space; luma vs chroma cost for pose",
        },
        empirical_output={
            "luma_per_plane_energy_avg": LUMA_PER_PLANE_AVG,
            "chroma_per_plane_energy_avg": CHROMA_PER_PLANE_AVG,
            "luma_chroma_per_plane_ratio": LUMA_CHROMA_PER_PLANE_RATIO,
            "luma_chroma_aggregate_ratio": LUMA_CHROMA_AGGREGATE_RATIO,
            "frame1_over_frame0_pose_energy": FRAME1_OVER_FRAME0_POSE,
            "chroma_high_freq_pose_null_derived": (
                "U_sub/V_sub are a 2x2 box-average (frame_utils:65-72) -> sinc nulls at the 2x2 grid -> "
                "RGB chroma above that grid is EXACTLY pose-null (derived from code)"
            ),
            "luma_lossless_derived": (
                "y00,y10,y01,y11 = 2x2 polyphase of 384x512 luma -> full-res luma, no info lost"
            ),
            "frame0_segnet_free_derived": (
                "SegNet.preprocess_input x[:,-1] (modules:108) -> only frame_1 scored; frame_0 d_seg==0 "
                "(validated seg_delta=0.0, cascade_c_prime_..._20260526)"
            ),
            "verdict": (
                "Two structural pose-free seg DOFs, both UNPRICED in witness/v8: (1) frame_0 seg-free -> "
                "pose-only carrier (half the pixels owe nothing to d_seg); (2) chroma-high-freq free-for-"
                "pose x needed-for-seg -> seg TEXTURE carrier resolving the palette dilemma. means != "
                "ends: pointer 0.19110 UNMOVED."
            ),
            "verdict_scope": "FORMULATION (scorer-dimension characterisation; no kill)",
        },
        residual=0.0,
        source_artifact=_MEMO,
        measurement_method=(
            "autograd per-channel Jacobian energy Σ_k Σ_spatial (∂pose_k/∂yuv_c)² over 6 scored pose dims "
            "on a real GT pair + exact-from-code derivation of the 2x2-box-average chroma null and the "
            "frame_0 SegNet slice"
        ),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria=(
                "re-measure at n600 through the byte-close decode when a v8 frame_0-pose-only carrier or "
                "chroma-high-freq seg-texture carrier is built; confirm d_seg drop on Movable/MyCar/Road "
                "with d_pose unchanged"
            ),
            measurement_axis=_ADVISORY,
            hardware_substrate="apple_m5_max_cpu",
        ),
    )
    anchor_kernel = EmpiricalAnchor(
        # (c) EXACT 6-ATOM POSE-PREPROCESS NULL KERNEL — advisory-fold 2026-07-10
        # (ADVISORY_evaluator_video_geometry §Pose geometry; the v7.5.3 exact-D texture home law).
        # This equation already holds the SUFFICIENT chroma-box-average sinc null; this anchor pins
        # the EXACT/COMPLETE linear kernel of the Pose preprocessing map at the 384x512 scorer grid
        # (rgb_to_yuv6 keeps all 4 luma polyphases + averages U,V per 2x2 block; frame_utils:50-78).
        anchor_id="pose_preprocess_exact_6atom_null_kernel_20260710",
        measurement_utc="2026-07-10T00:00:00Z",
        inputs={
            "map": "rgb_to_yuv6 (frame_utils:50-78): 4 luma polyphases kept per 2x2 scorer-grid block; "
                   "U,V box-averaged over the block",
            "frozen_checkout_git": "991b317c41fe3aac657e0f0cb88fd831b2e4185a",
            "custody_sha256": "frame_utils.py d689aca7d263997cb2fb980d6098d503f955e56e8642cd0a04cc437"
                              "f0ffdab90; modules.py 065961ba97023e393e27818760b0dc8efaa8dd53c5d4cc70a"
                              "2db8ee1b3cf49aa",
            "advisory": ".omx/research/ADVISORY_evaluator_video_geometry_20260710.md",
        },
        predicted_output={
            "question": "the EXACT linear kernel (all pose-null perturbations) of the Pose preprocess",
        },
        empirical_output={
            "per_pixel_luma_null": "0.299·dR + 0.587·dG + 0.114·dB = 0  (determines dG)",
            "per_2x2_block_zero_sum": "sum(dR)=0 and sum(dB)=0  (3 dof each)",
            "local_kernel_dim_per_block": 6,
            "basis": (
                "3 zero-sum Haar patterns h_x=[[1,-1],[1,-1]], h_y=[[1,1],[-1,-1]], "
                "h_xy=[[1,-1],[-1,1]] CROSSED with c_U=(0,-0.3441362862010222,1.772) and "
                "c_V=(1.402,-0.7141362862010221,0) -> 6 atoms h_k⊗c_U, h_k⊗c_V"
            ),
            "total_dof_per_frame": 294912,           # 49,152 blocks × 6
            "basis_residual": "< 5.6e-17 (direct float64 algebra)",
            "four_binding_caveats_that_break_the_null": (
                "(1) texture composed PRE-sigmoid — unequal per-channel derivative destroys RGB luma-"
                "nullness; (2) soft per-pixel class placement + annulus gates destroy the blockwise "
                "zero-sums; (3) bicubic camera-grid lift + evaluator bilinear resize can leak out of "
                "the scorer-grid kernel; (4) uint8 rounding + RGB/YUV clamp make the feasible kernel "
                "piecewise"
            ),
            "home_law": (
                "PROJECT AFTER the last learned RGB nonlinearity, solve a reachable camera-grid "
                "preimage through the exact resize, guard the active set, require the first-6 Pose "
                "outputs BIT-STABLE after fresh .raw reload. A proxy tensor equality is NOT enough; "
                "'high-frequency chroma' alone is NOT an exact-null theorem. frame1-only (frame_0 is "
                "seg-free -> pose-only carrier)."
            ),
            "verdict_scope": (
                "DERIVED-FROM-CODE (linear scorer-grid model; the 4 caveats are the reachability "
                "boundary). This is a CANDIDATE home law, NOT proof the current period-[4,8] RGB "
                "texture bank inhabits it. means != ends; pointer 0.19110 UNMOVED."
            ),
        },
        residual=0.0,
        source_artifact=".omx/research/ADVISORY_evaluator_video_geometry_20260710.md",
        measurement_method=(
            "exact linear algebra of the rgb_to_yuv6 preprocessing map at the 384x512 scorer grid "
            "(4 luma polyphase constraints + 2 chroma block-average constraints per 2x2 block)"
        ),
        empirical_verification_status=VERIFIED_VIA_SOURCE_INSPECTION,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=".omx/research/ADVISORY_evaluator_video_geometry_20260710.md",
            reactivation_criteria=(
                "when the v7.5.3 exact-D texture trunk is built (src/tac/boundary_math/texture_trunk"
                ".py), require post-nonlinearity projection into this kernel + camera-grid preimage + "
                "first-6-Pose bit-stability after fresh raw reload; verify the built bank inhabits it"
            ),
            measurement_axis=_ADVISORY,
            hardware_substrate="apple_m5_max_cpu",
        ),
    )
    anchor_mirror = EmpiricalAnchor(
        # (d) d_pose-side MIRROR of the covariance-totality audit — 2026-07-11 (Fable crux audit).
        # Closes the "texture-trunk-lives-on-POSE" UNDECIDED edge left by
        # covariance_totality_texture_trunk_verdict_20260710: WHICH luma band carries pose, and is
        # the trunk's pose job real or dominated?
        anchor_id="dpose_photometric_band_mirror_audit_20260711",
        measurement_utc="2026-07-11T00:00:00Z",
        inputs={
            "harness": (
                "n48-strided frozen CPU-torch PoseNet band-limit ladder (Gaussian sigma_cam on BOTH "
                "GT frames, uint8 re-quantized, exact cpu_verdict_d_pose) + n600 cached-code mirror "
                "regression (xi <- rank-8 partition code, same code definition as the d_seg audit)"
            ),
            "data": "experiments/results/mlx_fleet_gt_cache/gt_n600.npz (cached; baseline parity 2.6e-12)",
            "tools": ".omx/tmp/dpose_mirror_audit/{mirror_partA,mirror_partB_posefreq,mirror_partB2_biasvar}.py",
        },
        predicted_output={
            "question": (
                "does per-pair high-freq photometric texture carry pose-legible signal BEYOND the "
                "se(3) xi channel — i.e. does the texture trunk have a POSE job?"
            ),
        },
        empirical_output={
            "pose_legibility_band_spectrum_dpose_n48": {
                "sigma_cam_1": 4.197e-3, "sigma_cam_2": 5.726e-2,
                "sigma_cam_4": 5.582e-1, "sigma_cam_8": 1.676,
            },
            "blur_shift_bias_fraction": {"sigma1": 0.627, "sigma2": 0.764},
            "sigma1_pair_specific_ceiling_after_linear_xi_calibration": 1.303e-3,
            "f0_vs_f1_sigma4_attribution": {"f0_only": 8.141e-2, "f1_only": 8.887e-2},
            "xi_from_code8_knn_cv_r2_n600": 0.329,
            "xi_from_code16_knn_cv_r2_n600": 0.387,
            "cca_code8_vs_xi6_n600": (0.79, 0.65, 0.36, 0.31, 0.27, 0.05),
            "twonn_joint_minus_code_local_dof": "+2.1 to +2.4 (pose adds ~2 DOF beyond the partition code)",
            "pose_residual_vs_luma_hf_spearman_n600": -0.089,
            "verdict": (
                "ASYMMETRIC MIRROR: unlike d_seg (residual has ZERO texture -> dead by ABSENCE), the "
                "pose readout IS broadband/HF-legible (top camera octave alone = d_pose 4.2e-3) — but "
                "63-76% of the band's effect is a CONSTANT bias (6-scalar-correctable) and the pair-"
                "specific ceiling after a linear-xi calibration is ~1.3e-3, the same order as R1's "
                "already-banked 1.61e-3 and 10-40x ABOVE the S2 output-space-steering target at "
                ">=100-1000x the bytes (image-space stores measured RATE-PROHIBITIVE, #249). Texture "
                "trunk DEAD-for-pose by DOMINANCE. Capstone = single covariant trunk on BOTH axes; "
                "pose's +2 DOF route through the dedicated dxi/steering channel (6+k), NOT the trunk "
                "mod vector. means != ends: pointer 0.19108282 UNMOVED."
            ),
            "verdict_scope": (
                "FORMULATION (texture-trunk-as-pose-carrier under measured byte economics; "
                "reactivation: S1/S2 output-space solve fails <=0.0011 at n600 byte-close)"
            ),
        },
        residual=0.0,
        source_artifact=".omx/research/dpose_covariance_mirror_audit_20260711.md",
        measurement_method=(
            "frozen CPU-torch PoseNet on Gaussian-band-limited GT pairs (n48 strided, NO-FAKE baseline "
            "parity 2.6e-12) + bias/variance/linear-xi decomposition of the readout shift + n600 "
            "cached-code kNN/CCA/TwoNN mirror regression (rank-8 parity 0.9321 vs d_seg audit 0.9316)"
        ),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=".omx/research/dpose_covariance_mirror_audit_20260711.md",
            reactivation_criteria=(
                "re-price the photometric pose path ONLY IF the S1/S2 output-space per-pair solve "
                "(FEED-posesolve rungs A2/A2+) fails to reach <=0.0011 at n600 through byte-close; "
                "even then the calibrated HF ceiling ~1.3e-3 barely matches R1 -> a NEW mechanism, "
                "not this one, would be required"
            ),
            measurement_axis=_ADVISORY,
            hardware_substrate="apple_m5_max_cpu",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=(
            "Scorer-dimension asymmetry: PoseNet luma is 11.1x more sensitive per plane than chroma "
            "(chroma-high-freq exactly pose-null via 2x2 box-average); SegNet sees ONLY frame_1 (frame_0 "
            "seg-free) at raw full-res RGB -> two pose-free seg DOFs unpriced in witness/v8"
        ),
        one_line_summary=(
            "luma/chroma pose sensitivity = 11.1x/plane; chroma-high-freq pose-null (box-avg); frame_0 "
            "seg-free (x[:,-1]) -> frame-0 pose-only carrier + chroma-high-freq seg-texture carrier."
        ),
        latex_form=(
            r"\frac{\|\partial p/\partial \mathrm{luma}\|^2}{\|\partial p/\partial \mathrm{chroma}\|^2}"
            r"\approx 11.1\ (\text{per plane}),\ \ \widehat{U},\widehat{V}=\mathrm{box}_{2\times2}\!*\!C"
            r"\Rightarrow \partial p/\partial C_{>\text{grid}}=0,\ \ \mathrm{d\_seg}(f_0)\equiv 0"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.posenet_luma_chroma_asymmetry_20260710:"
            "build_posenet_luma_chroma_asymmetry_v1"
        ),
        domain_of_validity={
            "vehicle": ["coord_inr_seg_witness", "v8_perclass_carriers", "pose_carrier_sidecar"],
            "measurement_axis": ["macOS-CPU advisory . real-GT-pair PoseNet-Jacobian . NON-PROMOTABLE"],
            "note": (
                "Jacobian measured on real GT pair (n<=4, advisory NON-PROMOTABLE); the chroma-null and "
                "frame_0-seg-free legs are EXACT from code. The overlooked levers must be re-measured at "
                "n600 through the byte-close decode before any score claim."
            ),
        },
        units_in={"yuv6_channel": "posenet_input_channel", "frame_index": "pair_position"},
        units_out={"jacobian_energy": "sum_grad_squared", "seg_delta": "dseg_fraction"},
        empirical_anchors=(anchor, anchor_kernel, anchor_mirror),
        predicted_vs_empirical_residual={
            "luma_chroma_asymmetry_confirmed": 0.0,
            "pose_preprocess_exact_6atom_kernel_derived": 0.0,
            "dpose_photometric_band_mirror_measured": 0.0,
        },
        last_calibration_utc="2026-07-11T00:00:00Z",  # dpose_photometric_band_mirror anchor added
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "experiments/train_levelset_witness_realized_through_R_mlx.py",  # frame-0 + chroma carriers
            "src/tac/boundary_math/texture_trunk.py",  # v7.5.3 exact-D texture home (6-atom kernel)
        ),
        canonical_producers=(
            "tac.canonical_equations.posenet_luma_chroma_asymmetry_20260710",
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria="see anchor reactivation_criteria (n600 through-byte-close re-measure)",
            measurement_axis=_ADVISORY,
            hardware_substrate="apple_m5_max_cpu",
        ),
    )


def populate_posenet_luma_chroma_asymmetry_equation(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration of the scorer-dimension-asymmetry law (latest-row-wins).

    Equations leg of the upstream all-dimension re-read; DAG leg = FEED-alldim; memo =
    ``upstream_scorer_alldim_reread_20260710``.
    """
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_posenet_luma_chroma_asymmetry_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes="posenet_luma_chroma_asymmetry_20260710 (equations leg of upstream all-dim re-read; luma "
              "11.1x/plane pose vs chroma; chroma-high-freq pose-null; frame_0 seg-free; two v8 levers)",
    )
    return eq


__all__ = [
    "CHROMA_PER_PLANE_AVG",
    "EQUATION_ID",
    "FRAME0_SEGNET_SEG_DELTA",
    "FRAME1_OVER_FRAME0_POSE",
    "LUMA_CHROMA_AGGREGATE_RATIO",
    "LUMA_CHROMA_PER_PLANE_RATIO",
    "LUMA_PER_PLANE_AVG",
    "build_posenet_luma_chroma_asymmetry_v1",
    "populate_posenet_luma_chroma_asymmetry_equation",
]
