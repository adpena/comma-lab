# SPDX-License-Identifier: MIT
"""Canonical equation: BLIND-COORDINATE RATE LEVER (#401, DERIVED + MEASURED-PROVEN).

Camera pixels invisible to BOTH scorers, filled by a rule-118 generic decode-time rule so the
encoder stores NOTHING there — a rate cut at PROVABLY zero Δd_seg / Δd_pose.

DERIVED (VERIFIED_VIA_SOURCE_INSPECTION — a deterministic property of the pinned resize kernel):
  Both scorers put every frame through the SAME preprocess resize
  ``(CAMERA_H=874, CAMERA_W=1164) -> (SEG_H=384, SEG_W=512)`` bilinear, align_corners=False,
  antialias=False (``upstream/modules.py`` SegNet.preprocess_input:109 + PoseNet.preprocess_input:73;
  ``rgb_to_yuv6`` runs AFTER the resize on the 384x512 grid, reads every resized pixel, adds NO
  blindness). Downsampling by ~2.28x WITHOUT antialias gives DISJOINT 2-tap supports, so many
  input rows/cols receive weight EXACTLY 0.0 (bilinear, not epsilon). A camera pixel is blind iff
  its row OR its col is all-zero in the extracted 1-D operator:
    * 106 / 874 camera ROWS blind, 140 / 1164 camera COLS blind,
    * blind px/frame = 106*1164 + 140*874 - 106*140 = 230,904 (= 22.6969% of 1,017,336),
    * the RETAINED (non-blind) set is the exact product = a regular 768 x 1024 sub-grid (786,432).
  Same mask for frame0 (PoseNet-only) and frame1 (both) because the resize is identical.

MEASURED-PROVEN (VERIFIED_VIA_EMPIRICAL_ANCHOR; n600 real gt camera frames, real torch preprocess):
  * BIT-IDENTITY THROUGH R (n600): over ALL 600 pairs, filling the blind set with ARBITRARY
    random content in BOTH frames leaves the real torch scorer-input tensors (posenet_in +
    segnet_in) bit-for-bit identical: max|Δpose| = max|Δseg| = 0.0, 0 failures. Because the
    scorer INPUT is bit-identical, d_seg and d_pose are UNCHANGED by construction — no scorer
    forward needed. This is the PROOF the fill earns zero score change.
  * BYTE DELTA (32 real gt frames, brotli-q11): full frame 1,438,886 B -> retained 768x1024
    sub-grid 1,142,811 B => 296,075 B/frame saved = 20.55% of the stored content (< the 22.70%
    blind fraction because brotli already exploits some spatial redundancy). Real lossless
    byte-close of real camera-res video content.

THE FILL is a rule-118 GENERIC ALGORITHM: the receiver reconstructs every blind pixel by
separable linear interpolation over the KNOWN retained-index coordinates — data-INDEPENDENT
(mask is a fixed kernel property, re-derivable in inflate.py FREE), reading only stored
video-derived (COUNTED) pixels. The generator ALGORITHM is free; the retained payload is counted.
Smuggling a blind-region table into "code" is the hide-data-in-code FAKE (NO-FAKE #6/#7).

SCOPE (honest): the byte delta is the saving available to ANY camera-resolution-storing
representation (raw frames, per-frame residual, HF sidecar). A pure GENERATOR witness archive
(e.g. the 69,984-byte levelset packet used as byte-closed context) stores no camera pixels, so
its DIRECT saving is 0 until it carries a camera-res residual/sidecar section — then this lever
drops 22.7% of that section's spatial content.

means != ends: a rate MEANS. blind-mask + bit-identity are EXACT PROOFs; the byte delta is
``[macOS-CPU advisory / derivation]`` NON-PROMOTABLE. A promotable ΔS needs the full n600
byte-close + exact eval shipping the retained sub-grid on the chosen chain. Pointer 0.19108 UNMOVED.

DSL leg: N/A (a receiver/byte-close rate lever, not a trainer lever / launch / curriculum change;
it is wired into the #406 apply-pass as stage 6 + tools/blind_coordinate_proof.py). EQUATIONS leg =
this module. Mechanism: :mod:`tac.through_r.blind_coordinate` (reuses the #391 exact resize kernel
:func:`tac.through_r.flip_inverse.resize_matrix_1d`). Sister DERIVED anchor: the
``evaluator_resize_blind_coordinate_law_20260710`` anchor on ``resize_exploit_flip_fix_frontier_v1``
(same 230,904 number; this equation MEASURES + PROVES + BUILDS it).
DAG: ``.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`` (FEED-blindcoord-401).
Memo: ``.omx/research/blind_coordinate_401_20260711T221447Z.md``.
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

EQUATION_ID = "blind_coordinate_rate_lever_v1"

_UTC = "2026-07-11T00:00:00Z"
_ADVISORY = "[macOS-CPU advisory]"
_DAG = ".omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md"
_MEMO = ".omx/research/blind_coordinate_401_20260711T221447Z.md"
_MODULE = "src/tac/through_r/blind_coordinate.py"
_PROOF = "experiments/results/blind_coord_401_20260711T221447Z/blind_coordinate_proof.json"

# --- DERIVED geometry (exact kernel property; verified in unit tests) ---
_N_BLIND_ROWS = 106
_N_BLIND_COLS = 140
_N_BLIND_PX = 230_904
_N_RETAINED_PX = 786_432
_N_TOTAL_PX = 1_017_336
_BLIND_FRACTION = 0.22696926089315625
_RETAINED_HW = (768, 1024)
_MIN_NONZERO_WEIGHT = 0.002604166666628771  # smallest nonzero bilinear weight (clean 0-separation)

# --- MEASURED-PROVEN n600 (real gt frames, real torch preprocess; proof artifact) ---
_N_PAIRS = 600
_MAX_ABS_DIFF_POSE = 0.0
_MAX_ABS_DIFF_SEG = 0.0
_N_FAILURES = 0
_BYTE_DELTA_FRAMES = 32
_BYTES_FULL_MEAN = 1_438_886.5625
_BYTES_RETAINED_MEAN = 1_142_811.3125
_BYTE_DELTA_MEAN = 296_075.25
_DELTA_FRACTION_MEAN = 0.2054733412185899
_ARCHIVE_CONTEXT_BYTES = 69_984  # levelset_packet_20260710T165204Z/archive.zip (byte-closed context)


def build_blind_coordinate_rate_lever_v1() -> CanonicalEquation:
    """Build the blind-coordinate rate-lever law: one DERIVED anchor (exact blind-mask geometry,
    source-inspection) and one MEASURED-PROVEN anchor (n600 bit-identity-through-R + real byte
    delta)."""

    anchor_derived = EmpiricalAnchor(
        anchor_id="blind_coordinate_mask_geometry_20260711",
        measurement_utc=_UTC,
        inputs={
            "mechanism": "tac.through_r.blind_coordinate.build_blind_mask "
                         "(reuses tac.through_r.flip_inverse.resize_matrix_1d — impulse-probed torch)",
            "resize_contract": (
                "both scorers: interpolate(size=(384,512), bilinear, align_corners=False, "
                "antialias=False); rgb_to_yuv6 AFTER resize on the 384x512 grid (no reach-back) "
                "-> blindness is ONE resize's zero-weight set"
            ),
            "source": "upstream/modules.py:73 (PoseNet.preprocess_input) + :109 (SegNet.preprocess_input) "
                      "+ upstream/frame_utils.py:rgb_to_yuv6",
            "tests": "src/tac/through_r/tests/test_blind_coordinate.py (24 passing, $0)",
        },
        predicted_output={
            "law": "camera pixel blind iff (blind row OR blind col); retained = regular subgrid",
        },
        empirical_output={
            "n_blind_rows_of_874": _N_BLIND_ROWS,
            "n_blind_cols_of_1164": _N_BLIND_COLS,
            "n_blind_px_per_frame": _N_BLIND_PX,               # 230,904 == advisory D21
            "n_retained_px": _N_RETAINED_PX,                    # 786,432 = 768*1024
            "retained_subgrid_hw": list(_RETAINED_HW),
            "blind_fraction": _BLIND_FRACTION,                  # 22.6969%
            "inclusion_exclusion": "106*1164 + 140*874 - 106*140 == 230904 (exact)",
            "edge_rows_cols_not_blind": True,                   # interior samples skipped, not edges
            "min_nonzero_bilinear_weight": _MIN_NONZERO_WEIGHT,  # clean 0-separation (no epsilon blindness)
            "reproduces_advisory": "matches D21 / ADVISORY_evaluator_video_geometry 230,904 EXACTLY",
            "verdict": (
                "the blind set is a deterministic property of the pinned bilinear resize kernel: "
                "230,904 px/frame (22.70%) read by NO scorer, retained content is a regular 768x1024 "
                "sub-grid. Verified by impulse-probing the real torch interpolate (not re-derived)."
            ),
        },
        residual=0.0,
        source_artifact=_MODULE,
        measurement_method="source inspection + impulse-probe of torch.interpolate + 24 NO-FAKE unit tests",
        empirical_verification_status=VERIFIED_VIA_SOURCE_INSPECTION,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MODULE,
            reactivation_criteria=(
                "re-derive if the upstream resize contract changes "
                "(resolution_chain.verify_against_upstream fails first)"
            ),
            measurement_axis=_ADVISORY,
            hardware_substrate="macos_arm64",
        ),
    )

    anchor_measured = EmpiricalAnchor(
        anchor_id="blind_coordinate_bit_identity_n600_byte_delta_20260711",
        measurement_utc=_UTC,
        inputs={
            "frames": "experiments/results/mlx_fleet_gt_cache/gt_n600.npz['gt_f0','gt_f1'] "
                      "(600 real camera pairs, 874x1164x3 uint8)",
            "preprocess": "REAL torch DistortionNet preprocess (PoseNet both frames -> bilinear -> "
                          "rgb_to_yuv6; SegNet frame1 -> bilinear); NO scorer forward needed",
            "fill": "blind pixels of BOTH frames overwritten with ARBITRARY random uint8 (adversarial)",
            "byte_codec": "brotli-q11 lossless byte-close of full frame vs 768x1024 retained sub-grid",
            "proof_artifact": _PROOF,
        },
        predicted_output={
            "hypothesis": (
                "filling blind pixels with arbitrary content changes NO scorer input -> d_seg + d_pose "
                "provably unchanged; the encoder can drop the blind residuals for a rate cut"
            ),
        },
        empirical_output={
            "n_pairs": _N_PAIRS,
            "all_bit_identical": True,
            "max_abs_diff_pose": _MAX_ABS_DIFF_POSE,            # 0.0
            "max_abs_diff_seg": _MAX_ABS_DIFF_SEG,              # 0.0
            "n_failures": _N_FAILURES,                          # 0 / 600
            "byte_delta_frames": _BYTE_DELTA_FRAMES,
            "bytes_full_mean": _BYTES_FULL_MEAN,                # 1,438,886 B
            "bytes_retained_mean": _BYTES_RETAINED_MEAN,        # 1,142,811 B
            "byte_delta_mean": _BYTE_DELTA_MEAN,                # 296,075 B/frame
            "delta_fraction_mean": _DELTA_FRACTION_MEAN,        # 20.55% (< 22.70% blind: brotli redundancy)
            "byte_closed_archive_context_bytes": _ARCHIVE_CONTEXT_BYTES,  # 69,984 (generator; direct saving 0)
            "verdict": (
                "PROVEN over n600: arbitrary blind-pixel content is bit-identical through R -> zero "
                "Δd_seg/Δd_pose. Real byte delta 20.55% of stored camera-res content. SCOPE: the direct "
                "saving applies to a camera-res-storing section; a pure-generator witness archive stores "
                "no camera pixels (direct saving 0) until it carries such a section."
            ),
            "verdict_scope": (
                "FORMULATION — the bit-identity is UNIVERSAL (any frame, proven n600); the 20.55% byte "
                "delta is INSTANCE (real gt frames, brotli-q11) and representation-dependent"
            ),
        },
        residual=0.0,
        source_artifact=_PROOF,
        measurement_method="n600_bit_identity_through_real_torch_preprocess_plus_brotli_byte_delta",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria=(
                "wire the retained-sub-grid store + generic fill into a byte-closed chain that carries "
                "camera-res content, then exact-eval (advisory -> contest authority); re-measure the byte "
                "delta on the actual sidecar/residual codec (not brotli-on-raw) for the adopting chain"
            ),
            measurement_axis=_ADVISORY,
            hardware_substrate="macos_arm64",
        ),
    )

    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=(
            "blind-coordinate rate lever: camera pixels read by NO scorer resize (230,904/frame = "
            "22.70%; retained = a regular 768x1024 sub-grid) filled by a rule-118 generic decode-time "
            "rule -> encoder stores only the retained sub-grid; PROVEN n600 bit-identical through R "
            "(max|Δpose|=max|Δseg|=0.0) at a real brotli byte delta of 20.55%"
        ),
        one_line_summary=(
            "230,904 camera px/frame are blind to BOTH scorers (bilinear-resize zero-weight set); fill "
            "them generically -> PROVEN zero Δd_seg/Δd_pose n600, 20.55% real byte delta. Pointer UNMOVED."
        ),
        latex_form=(
            r"\text{blind}(i,l) \iff \big(\textstyle\max_j D^{col}_{j,i}=0\big)\ \vee\ "
            r"\big(\textstyle\max_k D^{row}_{k,l}=0\big);\quad "
            r"R(\text{fill blind})\equiv R(\text{orig}) \Rightarrow \Delta d_{seg}=\Delta d_{pose}=0"
        ),
        python_callable_module_path="tac.through_r.blind_coordinate:build_blind_mask",
        domain_of_validity={
            "vehicle": ["byte_close_receiver", "camera_res_residual_sidecar", "level_set_witness"],
            "labels": "n600 real gt camera frames (gt_n600.npz gt_f0/gt_f1)",
            "verdict_scope": (
                "FORMULATION — blind-mask geometry + bit-identity are UNIVERSAL/EXACT (any frame, proven "
                "n600); the 20.55% byte delta is INSTANCE (brotli-q11 on real gt frames), the saving "
                "available to a camera-res-storing representation, NOT a witness-generator direct saving"
            ),
            "scope": "receiver/byte-close RATE lever; NOT a byte-closed exact-eval score",
            "measurement_axis": ["macOS-CPU advisory"],
            "promotion_eligible": False,
            "note": (
                "means != ends: the blind set + bit-identity are exact PROOFs; the rate cut is real but "
                "only realized when an adopting chain stores camera-res content and byte-closes + exact-"
                "evals the retained sub-grid. A pure generator saves 0 directly."
            ),
        },
        units_in={
            "resize_kernel": "bilinear_down_weights",
            "camera_frame": "camera_uint8_rgb_874x1164",
        },
        units_out={
            "blind_fraction": "fraction",
            "byte_delta": "bytes_per_frame",
            "delta_d_seg_d_pose": "exactly_zero (proven)",
        },
        empirical_anchors=(anchor_derived, anchor_measured),
        predicted_vs_empirical_residual={
            "derived_geometry_source_inspection": 0.0,
            "measured_bit_identity_n600": 0.0,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools/witness_apply_pass.py",                       # #406 apply-pass stage 6 (blind_coord_401)
            "tools/blind_coordinate_proof.py",                   # the n600 proof artifact tool
            "tools/levelset_byte_close_and_eval.py",             # future: retained-subgrid store in the receiver
            _MEMO,
        ),
        canonical_producers=(
            "tac.through_r.blind_coordinate",                    # the mask/fill/proof/byte-delta module
            "tac.through_r.flip_inverse",                        # the exact resize kernel it reuses (#391)
            "tac.through_r.resolution_chain",                    # the pinned resize contract
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria=(
                "adopt the retained-sub-grid store + generic fill in a camera-res-storing chain, byte-close "
                "+ exact-eval (advisory -> contest authority); re-measure the byte delta on that chain's real "
                "codec"
            ),
            measurement_axis=_ADVISORY,
            hardware_substrate="macos_arm64",
        ),
    )


def populate_blind_coordinate_rate_lever_equation(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration of the blind-coordinate rate-lever law (#401).

    EQUATIONS leg of FEED-blindcoord-401; DSL leg = N/A (a receiver/byte-close rate lever, wired
    into the #406 apply-pass as stage 6). Mechanism = tac.through_r.blind_coordinate."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_blind_coordinate_rate_lever_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes="blind_coordinate_rate_lever_20260711 (equations leg of #401; blind-mask geometry "
              "230,904 px/frame DERIVED from the resize kernel; n600 bit-identity-through-R PROVEN "
              "(max diff 0.0) + 20.55% real byte delta MEASURED; DSL leg N/A; advisory NON-PROMOTABLE)",
    )
    return eq


__all__ = [
    "EQUATION_ID",
    "build_blind_coordinate_rate_lever_v1",
    "populate_blind_coordinate_rate_lever_equation",
]
