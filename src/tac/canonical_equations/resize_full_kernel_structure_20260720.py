# SPDX-License-Identifier: MIT
"""Canonical direct-sum law for the full shared-resize kernel.

This is a structural equation, not a score equation. It extends the #49/S12
mask-only surface with the complete separable real-linear kernel and keeps the
#532 bounded-uint8 intersection as a separately measured lower-bound field.
"""

from __future__ import annotations

from typing import Any

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.optimization.resize_full_kernel import FullResizeKernel
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "separable_resize_full_kernel_direct_sum_v1"
SOURCE_MEMO = ".omx/research/null_compiler_full_kernel_20260720T163500Z.md"
SOURCE_RECEIPT = ".omx/research/null_compiler_full_kernel_20260720T163500Z.json"
CALIBRATION_UTC = "2026-07-20T16:54:10Z"


def full_resize_kernel_direct_sum(
    *,
    camera_h: int = 874,
    camera_w: int = 1164,
    scorer_h: int = 384,
    scorer_w: int = 512,
) -> dict[str, Any]:
    """Return the exact implicit-kernel dimension/decomposition law."""

    compiler = FullResizeKernel.build(
        camera_h=camera_h,
        camera_w=camera_w,
        scorer_h=scorer_h,
        scorer_w=scorer_w,
    )
    return {
        "equation_id": EQUATION_ID,
        "projector": "P_ker(X)=X-Q_h X Q_w=P_h X+Q_h X P_w",
        "parameterization": "K(U,V)=N_h U+A_h^T V N_w^T",
        "orthogonal_direct_sum": True,
        "exact_rational_support_authority": (
            "DisjointResizeOperator AxisSupport integer numerators"
        ),
        "float_projection_authority": "fp32_or_fp64_explicit_dtype",
        "uint8_authority": "exact_integer_numerator_equality",
        **compiler.coverage().to_dict(),
        "score_claim": False,
        "promotion_eligible": False,
    }


def build_separable_resize_full_kernel_direct_sum_v1() -> CanonicalEquation:
    memo_provenance = build_provenance_for_research_sidecar(
        sidecar_path=SOURCE_MEMO,
        reactivation_criteria=(
            "MAIN adoption plus receiver-closed counted-byte replay on exact archive "
            "bytes; the structural law alone carries no score or promotion claim"
        ),
        measurement_axis="[DERIVED exact structure; macOS-CPU advisory fixture anchor]",
        hardware_substrate="macos_arm64_cpu",
        captured_at_utc=CALIBRATION_UTC,
    )
    receipt_provenance = build_provenance_for_research_sidecar(
        sidecar_path=SOURCE_RECEIPT,
        reactivation_criteria=(
            "MAIN adoption plus a receiver-closed, exact counted-byte win; this "
            "one-frame advisory anchor carries no score or promotion claim"
        ),
        measurement_axis="[Darwin-arm64 CPU advisory]",
        hardware_substrate="macOS-26.4-arm64-arm-64bit-Mach-O",
        captured_at_utc=CALIBRATION_UTC,
    )
    anchor = EmpiricalAnchor(
        anchor_id="resize_full_kernel_fixture49_exactness_and_mdl_20260720",
        measurement_utc=CALIBRATION_UTC,
        inputs={
            "fixture_sha256": (
                "2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9"
            ),
            "decoded_frame_sha256": (
                "47492a5864f0521f0ab6b129e4b172717139ba202fc3265a5220b6d1b15c24ed"
            ),
            "frames": 1,
            "preference": "constant",
            "max_nodes_per_block": 128,
        },
        predicted_output={
            "full_nullity_per_channel": 820_728,
            "exact_resize_numerator_equality_required": True,
            "coder_admission_must_not_regress_old_mask": True,
        },
        empirical_output={
            "full_nullity_per_channel": 820_728,
            "primitive_basis_uint8_reachability_lower_bound_fraction": (
                0.34193139099271214
            ),
            "exact_resize_numerator_equal": True,
            "constant_candidate_delta_vs_old_mask_brotli_bytes": 512_550,
            "constant_candidate_delta_vs_old_mask_lzma_bytes": 546_524,
            "selected_name": "old_zero_weight_mask",
            "score_claim": False,
            "pointer_moved": False,
        },
        residual=0.0,
        source_artifact=SOURCE_RECEIPT,
        measurement_method=(
            "SHA-pinned frame-0 decode; exact integer resize-numerator verification; "
            "canonical primitive-basis signed-unit uint8 reachability; Brotli-q11 and "
            "LZMA raw-frame coder admission"
        ),
        provenance=receipt_provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Separable shared-resize full-kernel direct sum",
        one_line_summary=(
            "ker(A_h(.)A_w^T)=ker(A_h) tensor R^W direct-sum "
            "row(A_h) tensor ker(A_w), with an implicit exact-rational basis and "
            "a separately bounded uint8-lattice intersection."
        ),
        latex_form=(
            r"P_{\ker A}(X)=X-Q_hXQ_w=P_hX+Q_hXP_w,\quad "
            r"K(U,V)=N_hU+A_h^{\mathsf T}VN_w^{\mathsf T},\quad "
            r"\dim\ker A=(H-h)W+h(W-w)=HW-hw"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.resize_full_kernel_structure_20260720:"
            "full_resize_kernel_direct_sum"
        ),
        domain_of_validity={
            "operator": "separable align_corners=false bilinear downsample",
            "required_geometry_property": "full-row-rank disjoint two-tap axis supports",
            "camera_hw": [874, 1164],
            "scorer_hw": [384, 512],
            "real_linear_ceiling": "exact",
            "uint8_intersection": (
                "fixture-dependent; canonical primitive-basis reachability is a "
                "lower bound, never silently equated to the full bounded lattice"
            ),
            "sisters": [
                "resize_exploit_flip_fix_frontier_v1",
                "bounded_uint8_resize_preimage_cell_feasibility_v1",
                "evaluator_resize_blind_coordinate_law_v1",
            ],
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "verdict_scope": "structural real-linear kernel law",
        },
        units_in={
            "X": "camera-resolution real or uint8 channel plane",
            "A_h,A_w": "exact half-pixel resize axis operators",
        },
        units_out={
            "P_ker(X)": "camera-resolution real null component",
            "K(U,V)": "implicit full-kernel camera perturbation",
            "coverage": "dimensions and fractions per channel",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "full_nullity": 0.0,
            "exact_resize_numerator_equality": 0.0,
        },
        last_calibration_utc=CALIBRATION_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "r2b sparse target-selection free-derivation",
            "R1 d_B preimage-cell compiler",
            "#401 blind-coordinate full affine fill",
        ),
        canonical_producers=(
            "tac.optimization.resize_full_kernel.FullResizeKernel",
            "tac.optimization.uint8_lattice_feasibility.DisjointResizeOperator",
        ),
        provenance=memo_provenance,
    )


def populate_separable_resize_full_kernel_direct_sum_equation(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_separable_resize_full_kernel_direct_sum_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
    )
    return equation


__all__ = [
    "EQUATION_ID",
    "SOURCE_RECEIPT",
    "build_separable_resize_full_kernel_direct_sum_v1",
    "full_resize_kernel_direct_sum",
    "populate_separable_resize_full_kernel_direct_sum_equation",
]
