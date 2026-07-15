# SPDX-License-Identifier: MIT
"""Canonical advisory law for AMC per-row tiered code allocation.

The callable is deliberately limited to the exact pair-local SegNet
composition.  It says nothing about PoseNet additivity or Brotli rate.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Iterable
from numbers import Real
from typing import Any

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "amc_perrow_tiered_code_bitalloc_v1"
AXIS = "[macOS-CPU advisory; NumPy-fp32 receiver; CPU frozen scorers]"
MEASUREMENT_UTC = "2026-07-14T00:00:00Z"
POINTER_STATUS = "UNCHANGED"
SCORE_CLAIM = False
PROMOTION_ELIGIBLE = False
RAW_RESULT_DIRS_STATUS = "ABSENT_IN_THIS_WORKTREE"
FRESH_JOINT_N600_STATUS = "OWED"
EXACT_CONTEST_CPU_TRANSFER_STATUS = "OWED"
CHECKPOINT_SHA256 = "2599ad8b396af2af220a3bdbeee2ade92f194771ae6ef01a6faa15d39333484c"
GT_SHA256 = "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"
BASELINE_ARCHIVE_BYTES = 63664
BASELINE_D_SEG = 0.03365824381510417
BASELINE_D_POSE = 151.79642088984443
BASELINE_SOURCE_ARTIFACT = ".omx/research/witness_sensitivity_bitalloc_336_20260713.md"
TIERED_SOURCE_ARTIFACT = ".omx/research/fable_amc_saliency_codex.md"
CUSTODY_SOURCE_ARTIFACT = (
    ".omx/research/sub015_DAG_cheapen_real95_tilehalo_fp16_20260713.md"
)

ARCHIVE_BYTES_LABEL = "MEASURED"
D_SEG_LABEL = "DERIVED_EXACT_FROM_MEASURED_PER_PAIR_ROWS"
TIERED_ARCHIVE_BYTES = {
    "role_f0int3_f1int8": 57960,
    "role_f0int2_f1int8": 55203,
    "role_f0int3_f1int4": 51953,
    "amc3_salient": 52762,
    "amc3_random": 52992,
    "pairkkt_f0int3": 52981,
}
TIERED_D_SEG = {
    "role_f0int3_f1int8": 0.03365824,
    "role_f0int2_f1int8": 0.03365824,
    "role_f0int3_f1int4": 0.03380370,
    "amc3_salient": 0.03401882,
    "amc3_random": 0.03392613,
    "pairkkt_f0int3": 0.03152334,
}

TOOL_MODULE = "tools.apply_amc_saliency_tiered_bitalloc_witness"
REVERSE_WATERFILL_CONSUMER = (
    "tac.canonical_equations.witness_measured_reverse_waterfill_20260713"
)
BYTE_ALLOCATION_CONSUMER = "tac.frontier_exact_bitalloc"

OWED_NOT_BUILT = "OWED_NOT_BUILT"
OWED_TRAIN_TIME_FLAGS = ("--code-row-bits-map", "--code-qat-tiered")
TIERED_CODE_QAT_REACTIVATION_CRITERION = (
    "competitive witness checkpoint plus operator GO to add both train-time flags "
    "through the typed DSL"
)


def _validated_count(value: Any, name: str) -> int:
    is_numpy_bool = type(value).__module__ == "numpy" and type(value).__name__ == "bool_"
    if isinstance(value, bool) or is_numpy_bool:
        raise ValueError(f"{name} must be finite and integral")
    try:
        integer = operator.index(value)
    except TypeError:
        if not isinstance(value, Real):
            raise ValueError(f"{name} must be finite and integral") from None
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{name} must be finite and integral") from None
        integer = int(numeric)
    if integer < 0:
        raise ValueError(f"{name} must be non-negative")
    if not isinstance(integer, int):
        raise ValueError(f"{name} must be finite and integral")
    return integer


def amc_perrow_tiered_code_bitalloc_law(
    mismatch_counts: Iterable[Any], pixel_counts: Iterable[Any]
) -> float:
    """Return exact pair-local SegNet ``d_seg`` from mismatch/pixel rows."""

    mismatches = list(mismatch_counts)
    pixels = list(pixel_counts)
    if not mismatches or not pixels:
        raise ValueError("mismatch_counts and pixel_counts must be non-empty")
    if len(mismatches) != len(pixels):
        raise ValueError("mismatch_counts and pixel_counts must have matching lengths")

    validated_mismatches = [_validated_count(v, "mismatch count") for v in mismatches]
    validated_pixels = [_validated_count(v, "pixel count") for v in pixels]
    if any(value == 0 for value in validated_pixels):
        raise ValueError("pixel counts must be greater than zero")
    if any(
        mismatch > pixels
        for mismatch, pixels in zip(validated_mismatches, validated_pixels, strict=True)
    ):
        raise ValueError("mismatch counts must not exceed pixel counts")
    return sum(validated_mismatches) / sum(validated_pixels)


# Short alias retained as the equation's natural callable name.
perrow_segnet_dseg = amc_perrow_tiered_code_bitalloc_law


def build_amc_perrow_tiered_code_bitalloc_v1() -> CanonicalEquation:
    """Build the advisory equation and its two custody-scoped anchors."""

    baseline_provenance = build_provenance_for_research_sidecar(
        sidecar_path=BASELINE_SOURCE_ARTIFACT,
        reactivation_criteria=(
            "fresh joint n600 d_pose/d_seg and exact contest-CPU transfer remain OWED"
        ),
        measurement_axis=AXIS,
        hardware_substrate="macOS arm64 local CPU; NumPy-fp32 receiver; CPU frozen scorers",
        captured_at_utc="2026-07-13T00:00:00Z",
    )
    tiered_provenance = build_provenance_for_research_sidecar(
        sidecar_path=TIERED_SOURCE_ARTIFACT,
        reactivation_criteria=TIERED_CODE_QAT_REACTIVATION_CRITERION,
        measurement_axis=AXIS,
        hardware_substrate="macOS arm64 local CPU; NumPy-fp32 receiver; CPU frozen scorers",
        captured_at_utc=MEASUREMENT_UTC,
    )
    baseline = EmpiricalAnchor(
        anchor_id="amc_perrow_n600_baseline_20260713",
        measurement_utc="2026-07-13T00:00:00Z",
        inputs={
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "gt_sha256": GT_SHA256,
            "population": 600,
            "archive_bytes": BASELINE_ARCHIVE_BYTES,
            "axis": AXIS,
            "raw_result_dirs": RAW_RESULT_DIRS_STATUS,
            "fresh_joint_n600_d_pose_d_seg": FRESH_JOINT_N600_STATUS,
            "exact_contest_cpu_transfer": EXACT_CONTEST_CPU_TRANSFER_STATUS,
            "pointer": POINTER_STATUS,
            "score_claim": SCORE_CLAIM,
            "promotion_eligible": PROMOTION_ELIGIBLE,
        },
        predicted_output={"d_seg": BASELINE_D_SEG, "law": "pair_local_segnet_exact"},
        empirical_output={
            "archive_bytes": BASELINE_ARCHIVE_BYTES,
            "d_seg": BASELINE_D_SEG,
            "d_pose": BASELINE_D_POSE,
            "measurement_label": "MEASURED",
        },
        residual=0.0,
        source_artifact=BASELINE_SOURCE_ARTIFACT,
        measurement_method="measured n600 baseline with frozen checkpoint and GT custody",
        provenance=baseline_provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )
    tiered = EmpiricalAnchor(
        anchor_id="amc_perrow_tiered_response_20260714",
        measurement_utc=MEASUREMENT_UTC,
        inputs={
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "gt_sha256": GT_SHA256,
            "population": 600,
            "axis": AXIS,
            "baseline_uniform_custody": "byte-identical 6/6",
            "baseline_uniform_custody_source_artifact": CUSTODY_SOURCE_ARTIFACT,
            "raw_result_dirs": RAW_RESULT_DIRS_STATUS,
            "fresh_joint_n600_d_pose_d_seg": FRESH_JOINT_N600_STATUS,
            "exact_contest_cpu_transfer": EXACT_CONTEST_CPU_TRANSFER_STATUS,
            "pointer": POINTER_STATUS,
            "score_claim": SCORE_CLAIM,
            "promotion_eligible": PROMOTION_ELIGIBLE,
        },
        predicted_output={"law": "pair_local_segnet_exact"},
        empirical_output={
            "archive_bytes": TIERED_ARCHIVE_BYTES,
            "archive_bytes_label": ARCHIVE_BYTES_LABEL,
            "d_seg": TIERED_D_SEG,
            "d_seg_label": D_SEG_LABEL,
            "d_pose": dict.fromkeys(TIERED_ARCHIVE_BYTES, "OWED"),
            "scoped_empirical_conclusion": (
                "response allocation dominates proxy saliency on this INSTANCE x FORMULATION; "
                "not a family theorem"
            ),
        },
        residual=0.0,
        source_artifact=TIERED_SOURCE_ARTIFACT,
        measurement_method="existing measured AMC tiered response and exact per-pair row derivation",
        provenance=tiered_provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="AMC per-row tiered code bit allocation",
        one_line_summary=(
            "Exact pair-local SegNet mismatch composition for advisory per-row code allocation."
        ),
        latex_form=r"d_{seg}(q)=\frac{\sum_i mismatch_i(q_i)}{\sum_i pixel\_count_i}",
        python_callable_module_path=(
            "tac.canonical_equations.amc_perrow_tiered_code_bitalloc_20260714:"
            "amc_perrow_tiered_code_bitalloc_law"
        ),
        domain_of_validity={
            "included": "non-empty pair-local SegNet rows with finite integral counts",
            "exact_scope": "each pair frame-1 code row affects that pair's SegNet row",
            "excluded": "PoseNet additivity and Brotli-rate additivity",
            "axis": AXIS,
            "pointer": POINTER_STATUS,
            "score_claim": SCORE_CLAIM,
            "promotion_eligible": PROMOTION_ELIGIBLE,
            "raw_result_dirs": RAW_RESULT_DIRS_STATUS,
            "fresh_joint_n600_d_pose_d_seg": FRESH_JOINT_N600_STATUS,
            "exact_contest_cpu_transfer": EXACT_CONTEST_CPU_TRANSFER_STATUS,
            "research_only": True,
            "TieredCodeQATLever": OWED_NOT_BUILT,
            "owed_design_flags": OWED_TRAIN_TIME_FLAGS,
            "TieredCodeQATLever_reactivation_criterion": (
                TIERED_CODE_QAT_REACTIVATION_CRITERION
            ),
            "verdict_scope": "INSTANCE x FORMULATION only; not a family theorem",
        },
        units_in={"mismatch_counts": "SegNet pixel mismatches", "pixel_counts": "SegNet pixels"},
        units_out={"d_seg": "normalized SegNet mismatch fraction"},
        empirical_anchors=(baseline, tiered),
        predicted_vs_empirical_residual={"d_seg": 0.0},
        last_calibration_utc=MEASUREMENT_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(REVERSE_WATERFILL_CONSUMER, BYTE_ALLOCATION_CONSUMER),
        canonical_producers=(TOOL_MODULE,),
        provenance=tiered_provenance,
    )


def populate_amc_perrow_tiered_code_bitalloc_v1(
    *, path=None, lock_path=None, agent=None, subagent_id=None
) -> CanonicalEquation:
    """Explicitly register the equation; importing this module is side-effect free."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_amc_perrow_tiered_code_bitalloc_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes="advisory AMC per-row SegNet law; no train-time lever built",
    )
    return equation


__all__ = [
    "AXIS",
    "BASELINE_ARCHIVE_BYTES",
    "BASELINE_D_POSE",
    "BASELINE_D_SEG",
    "BASELINE_SOURCE_ARTIFACT",
    "BYTE_ALLOCATION_CONSUMER",
    "CHECKPOINT_SHA256",
    "CUSTODY_SOURCE_ARTIFACT",
    "D_SEG_LABEL",
    "EQUATION_ID",
    "EXACT_CONTEST_CPU_TRANSFER_STATUS",
    "FRESH_JOINT_N600_STATUS",
    "GT_SHA256",
    "OWED_NOT_BUILT",
    "OWED_TRAIN_TIME_FLAGS",
    "PROMOTION_ELIGIBLE",
    "RAW_RESULT_DIRS_STATUS",
    "REVERSE_WATERFILL_CONSUMER",
    "SCORE_CLAIM",
    "TIERED_ARCHIVE_BYTES",
    "TIERED_CODE_QAT_REACTIVATION_CRITERION",
    "TIERED_D_SEG",
    "TIERED_SOURCE_ARTIFACT",
    "amc_perrow_tiered_code_bitalloc_law",
    "build_amc_perrow_tiered_code_bitalloc_v1",
    "perrow_segnet_dseg",
    "populate_amc_perrow_tiered_code_bitalloc_v1",
]
