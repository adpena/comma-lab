# SPDX-License-Identifier: MIT
"""Canonical laws for the int8 teacher and witness-QAT rungs.

The teacher law is an admission predicate, not a performance prediction.  The
witness law defines the receiver-closed post-hoc quantization gap and loads its
empirical anchor only from a terminal n600 receipt.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

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

A3_EQUATION_ID = "int8_teacher_w8a8_admission_v1"
B_EQUATION_ID = "int8_witness_posthoc_gap_v1"
REQUIRED_PAIRS = 600
MINIMUM_SPEEDUP = 1.5
MINIMUM_GLOBAL_GRADIENT_COSINE = 0.99
MINIMUM_PAIR_GRADIENT_COSINE = 0.99
EXPECTED_CHECKPOINT_SHA256 = "ef2c097f98f74dbd16e77c6f7b60f05e0a630b6bd65ee55bf334336c4549c965"
DEFAULT_B_RECEIPT = Path("experiments/results/int8_training_rungs_local_20260713/b_posthoc_gap_n600.json")
_UTC = "2026-07-13T00:00:00Z"


def int8_teacher_admission(
    *,
    speedup_x: float | None,
    global_gradient_cosine: float | None,
    minimum_pair_gradient_cosine: float | None,
    quality_pairs: int,
) -> dict[str, Any]:
    """Conjunctive A3 gate; missing measurements yield no verdict."""

    values = (speedup_x, global_gradient_cosine, minimum_pair_gradient_cosine)
    if any(value is None for value in values):
        return {
            "verdict": "NO_VERDICT_BLOCKED",
            "tests": None,
            "required_pairs": REQUIRED_PAIRS,
        }
    parsed = tuple(float(value) for value in values if value is not None)
    if not all(math.isfinite(value) for value in parsed):
        raise ValueError("teacher admission inputs must be finite")
    tests = {
        "speed": parsed[0] >= MINIMUM_SPEEDUP,
        "global_gradient_cosine": parsed[1] >= MINIMUM_GLOBAL_GRADIENT_COSINE,
        "minimum_pair_gradient_cosine": parsed[2] >= MINIMUM_PAIR_GRADIENT_COSINE,
        "n600_quality_coverage": quality_pairs >= REQUIRED_PAIRS,
    }
    return {
        "verdict": "GO" if all(tests.values()) else "NO_GO",
        "tests": tests,
        "required_pairs": REQUIRED_PAIRS,
    }


def int8_witness_posthoc_gap(*, d_seg_fp32: float, d_seg_int8: float) -> dict[str, float]:
    """Return the signed QAT prize ceiling from receiver-realized d_seg arms."""

    if not all(math.isfinite(float(value)) for value in (d_seg_fp32, d_seg_int8)):
        raise ValueError("d_seg inputs must be finite")
    gap = float(d_seg_int8) - float(d_seg_fp32)
    return {
        "d_seg_gap_int8_minus_fp32": gap,
        "seg_score_unit_gap_100x": 100.0 * gap,
        "positive_recovery_prize_ceiling": max(0.0, gap),
    }


def heterogeneous_overlap_seconds(
    *, gpu_witness_seconds: float, ane_forward_seconds: float, synchronization_seconds: float = 0.0
) -> float:
    """Ideal independent-forward overlap law; not valid for a missing scorer VJP."""

    values = (gpu_witness_seconds, ane_forward_seconds, synchronization_seconds)
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in values):
        raise ValueError("overlap times must be finite and non-negative")
    return max(float(gpu_witness_seconds), float(ane_forward_seconds)) + float(synchronization_seconds)


def build_int8_teacher_admission_v1() -> CanonicalEquation:
    return CanonicalEquation(
        equation_id=A3_EQUATION_ID,
        name="W8A8 frozen-teacher training-path admission predicate",
        one_line_summary="Admit W8A8 only at n600 when global/min-pair gradient cosine >=0.99 and measured step speedup >=1.5x.",
        latex_form=r"GO_A=1[n=600]\,1[C_g\ge0.99]\,1[\min_i C_i\ge0.99]\,1[T_{32}/T_8\ge1.5]",
        python_callable_module_path=("tac.canonical_equations.int8_training_rungs_20260713:int8_teacher_admission"),
        domain_of_validity={
            "included": ["exact v7.5.2 EMA", "real n600 scorer states", "declared W8A8 QDQ/VJP policy"],
            "excluded": ["proxy loss only", "n<600", "unmeasured native int8/ANE transfer"],
        },
        units_in={
            "speedup_x": "ratio",
            "global_gradient_cosine": "dimensionless",
            "minimum_pair_gradient_cosine": "dimensionless",
            "quality_pairs": "pairs",
        },
        units_out={"verdict": "categorical"},
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("tac.witness_dsl.int8_training_rungs_policy",),
        canonical_producers=("tools.probe_mlx_real_n600_int8",),
        provenance=build_provenance_for_predicted(
            model_id=A3_EQUATION_ID,
            inputs_sha256="0" * 64,
            measurement_axis="[predicted]",
            hardware_substrate="macos_arm64_mlx",
            captured_at_utc=_UTC,
        ),
    )


def _load_terminal_b_receipt(path: str | Path) -> dict[str, Any]:
    receipt_path = Path(path)
    receipt = json.loads(receipt_path.read_text())
    measurement = receipt.get("measurement", {})
    packet = receipt.get("provenance", {}).get("packet", {})
    if receipt.get("status") != "MEASURED":
        raise ValueError("B receipt must be terminal MEASURED")
    if measurement.get("n_pairs") != REQUIRED_PAIRS or measurement.get("n600_evidence") is not True:
        raise ValueError("B receipt must contain exact n600 evidence")
    if receipt.get("provenance", {}).get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("B receipt checkpoint SHA mismatch")
    if packet.get("parse_back_equals_direct_int8_dequant") is not True:
        raise ValueError("B receipt lacks canonical parse-back equality")
    derived = int8_witness_posthoc_gap(
        d_seg_fp32=float(measurement["d_seg_fp32_ema"]),
        d_seg_int8=float(measurement["d_seg_parsed_int8"]),
    )
    for key in ("d_seg_gap_int8_minus_fp32", "seg_score_unit_gap_100x"):
        value = derived[key]
        if not math.isclose(float(measurement[key]), value, rel_tol=0.0, abs_tol=1.0e-15):
            raise ValueError(f"B receipt {key} does not re-derive")
    return receipt


def build_int8_witness_posthoc_gap_v1(
    receipt_path: str | Path = DEFAULT_B_RECEIPT,
) -> CanonicalEquation:
    receipt = _load_terminal_b_receipt(receipt_path)
    measurement = receipt["measurement"]
    anchor = EmpiricalAnchor(
        anchor_id="int8_witness_posthoc_gap_v752_n600_20260713",
        measurement_utc=receipt["completed_at_utc"],
        inputs={
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "n_pairs": REQUIRED_PAIRS,
            "quantizer": "LVLS1 per-tensor symmetric absmax/127",
        },
        predicted_output={"sign": "unknown_before_measurement"},
        empirical_output={
            "d_seg_fp32_ema": measurement["d_seg_fp32_ema"],
            "d_seg_parsed_int8": measurement["d_seg_parsed_int8"],
            "d_seg_gap_int8_minus_fp32": measurement["d_seg_gap_int8_minus_fp32"],
            "seg_score_unit_gap_100x": measurement["seg_score_unit_gap_100x"],
        },
        residual=0.0,
        source_artifact=str(receipt_path),
        measurement_method="canonical LVLS1 parse-back -> NumPy receiver -> real R -> frozen CPU SegNet, n600",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=receipt_path,
            reactivation_criteria="repeat as a converged QAT-vs-control n600 A/B and exact contest-axis replay",
            measurement_axis="[macOS-CPU advisory]",
            hardware_substrate="macos_arm64",
            captured_at_utc=receipt["completed_at_utc"],
        ),
    )
    return CanonicalEquation(
        equation_id=B_EQUATION_ID,
        name="Receiver-closed post-hoc int8 gap and QAT recovery ceiling",
        one_line_summary="The signed QAT prize is d_seg(parsed LVLS1 int8) minus d_seg(fp32 EMA), measured through the same receiver/R/SegNet at n600.",
        latex_form=r"\Delta d_{seg}^{post8}=d_{seg}(R(Q_8(W)))-d_{seg}(R(W));\quad P_{QAT}^{max}=\max(0,\Delta d_{seg}^{post8})",
        python_callable_module_path=("tac.canonical_equations.int8_training_rungs_20260713:int8_witness_posthoc_gap"),
        domain_of_validity={
            "included": ["exact v7.5.2 EMA", "LVLS1 parser", "first 600 cached real pairs"],
            "excluded": ["d_pose", "QAT outcome", "other checkpoints", "contest score"],
        },
        units_in={"d_seg_fp32": "fraction", "d_seg_int8": "fraction"},
        units_out={"d_seg_gap": "fraction", "seg_score_unit_gap": "score_units"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"receiver_closed_n600": 0.0},
        last_calibration_utc=receipt["completed_at_utc"],
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("tac.witness_dsl.int8_training_rungs_policy",),
        canonical_producers=("tools.probe_int8_witness_byteclose_gap",),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=receipt_path,
            reactivation_criteria="same as empirical anchor; advisory equation is not promotion authority",
            measurement_axis="[macOS-CPU advisory]",
            hardware_substrate="macos_arm64",
            captured_at_utc=receipt["completed_at_utc"],
        ),
    )


__all__ = [
    "A3_EQUATION_ID",
    "B_EQUATION_ID",
    "build_int8_teacher_admission_v1",
    "build_int8_witness_posthoc_gap_v1",
    "heterogeneous_overlap_seconds",
    "int8_teacher_admission",
    "int8_witness_posthoc_gap",
]
