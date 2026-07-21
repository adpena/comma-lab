# SPDX-License-Identifier: MIT
"""Fail-closed admission law for cells -> RGB -> lattice realization.

Exact factor-2 projection is only one conjunct.  A realization is admissible
only when the receiver itself derives the RGB planes from the counted seed,
all described semantic cells survive, double decode is identical, the pose
tube holds, and the added seed cost is zero.  The n600 source-RGB control is a
measured negative anchor: it validates projection/lattice/scorer mechanics but
cannot stand in for the absent cells-to-RGB receiver.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.canonical_equations.evaluators import register_evaluator
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "predict_project_realization_admissibility_v1"
BLOCKER_ID = "D2_ZERO_BYTE_SEMANTIC_CELLS_TO_RGB_ADMISSION_FALSE"
SOURCE_RECEIPT = ".omx/research/realization_g2b_supportfill_receipt_20260721.json"
SOURCE_RECEIPT_SHA256 = "daac2782ed724c9696fc49e6b968c40f29df8b0a31d0c210deb718d130fadea6"
UTC = "2026-07-21T09:37:19Z"
PAIR_COUNT = 600


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _unit_fraction(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def predict_project_realization_certificate(
    *,
    pair_count: int,
    uint8_factor2_exact_fraction: float,
    double_decode_identical_pair_count: int,
    semantic_cells_to_rgb_exact_pair_count: int,
    pose_within_declared_tube_pair_count: int,
    additional_seed_bytes: int,
    receiver_derived_rgb: bool,
) -> dict[str, Any]:
    """Evaluate the complete realization conjunction without score authority."""

    pairs = _strict_int(pair_count, "pair_count", minimum=1)
    factor2_fraction = _unit_fraction(
        uint8_factor2_exact_fraction, "uint8_factor2_exact_fraction"
    )
    double_decode = _strict_int(
        double_decode_identical_pair_count,
        "double_decode_identical_pair_count",
    )
    semantic_exact = _strict_int(
        semantic_cells_to_rgb_exact_pair_count,
        "semantic_cells_to_rgb_exact_pair_count",
    )
    pose_within = _strict_int(
        pose_within_declared_tube_pair_count,
        "pose_within_declared_tube_pair_count",
    )
    added_bytes = _strict_int(additional_seed_bytes, "additional_seed_bytes")
    if double_decode > pairs or semantic_exact > pairs or pose_within > pairs:
        raise ValueError("pair counts must not exceed pair_count")

    predicates = {
        "n600": pairs == PAIR_COUNT,
        "factor2_uint8_exact": factor2_fraction == 1.0,
        "double_decode_identical": double_decode == pairs,
        "semantic_cells_to_rgb_exact": semantic_exact == pairs,
        "pose_within_declared_tube": pose_within == pairs,
        "zero_added_seed_bytes": added_bytes == 0,
        "receiver_derived_rgb": _strict_bool(receiver_derived_rgb, "receiver_derived_rgb"),
    }
    accepted = all(predicates.values())
    return {
        "accepted": accepted,
        "status": "ADMISSIBLE" if accepted else BLOCKER_ID,
        "predicates": predicates,
        "failed_predicates": tuple(name for name, passed in predicates.items() if not passed),
        "pair_count": pairs,
        "additional_seed_bytes": added_bytes,
        "score_claim": False,
        "promotion_eligible": False,
    }


def predict_project_realization_admissibility(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """LawRef evaluator for :func:`predict_project_realization_certificate`."""

    return predict_project_realization_certificate(
        pair_count=inputs["pair_count"],
        uint8_factor2_exact_fraction=inputs["uint8_factor2_exact_fraction"],
        double_decode_identical_pair_count=inputs["double_decode_identical_pair_count"],
        semantic_cells_to_rgb_exact_pair_count=inputs[
            "semantic_cells_to_rgb_exact_pair_count"
        ],
        pose_within_declared_tube_pair_count=inputs[
            "pose_within_declared_tube_pair_count"
        ],
        additional_seed_bytes=inputs["additional_seed_bytes"],
        receiver_derived_rgb=inputs["receiver_derived_rgb"],
    )


register_evaluator(EQUATION_ID, predict_project_realization_admissibility)


def build_predict_project_realization_admissibility_v1() -> CanonicalEquation:
    """Build the n600-measured realization admission law."""

    provenance = build_provenance_for_research_sidecar(
        sidecar_path=SOURCE_RECEIPT,
        reactivation_criteria=(
            "Supply a deterministic receiver-side cells-to-RGB decoder from counted seed bytes, "
            "then rerun n16/n64/n600 through the same factor-2, hard SegNet, and PoseNet path."
        ),
        measurement_axis="[macOS-CPU advisory]",
        hardware_substrate="macos_arm64_cpu",
        captured_at_utc=UTC,
    )
    measured_inputs = {
        "pair_count": PAIR_COUNT,
        "uint8_factor2_exact_fraction": 1.0,
        "double_decode_identical_pair_count": PAIR_COUNT,
        "semantic_cells_to_rgb_exact_pair_count": 0,
        "pose_within_declared_tube_pair_count": PAIR_COUNT,
        "additional_seed_bytes": 707_788_800,
        "receiver_derived_rgb": False,
    }
    measured = predict_project_realization_certificate(**measured_inputs)
    failed_fraction = len(measured["failed_predicates"]) / len(measured["predicates"])
    anchor = EmpiricalAnchor(
        anchor_id="realization_g2b_source_rgb_control_n600_20260721",
        measurement_utc=UTC,
        inputs={
            **measured_inputs,
            "source_rgb_custody": "encoder_supplied_counted",
            "source_receipt_sha256": SOURCE_RECEIPT_SHA256,
        },
        predicted_output={
            "accepted_iff_all_predicates_hold": True,
            "required_additional_seed_bytes": 0,
            "required_semantic_exact_pairs": PAIR_COUNT,
        },
        empirical_output={
            **measured,
            "d_seg_realized_vs_frozen_target": 0.0001518673366970486,
            "d_seg_description_vs_frozen_target": 0.3434977213541667,
            "d_seg_realized_argmax_vs_description": 0.3435325537787543,
            "d_pose_realized_vs_frozen_target": 0.0001016086067031589,
        },
        residual=failed_fraction,
        source_artifact=SOURCE_RECEIPT,
        measurement_method=(
            "n600 exact source-derived RGB planes; deterministic double decode through "
            "realize_projected_rgb_plane_camera_uint8; native CPU-Torch frozen SegNet/PoseNet"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Predict-project realization hard admission",
        one_line_summary=(
            "Exact lattice projection is admissible only when receiver-derived RGB also preserves "
            "all described cells at zero added seed bytes; the n600 counted source-RGB control fails."
        ),
        latex_form=(
            r"A=\mathbf 1[n=600]\mathbf 1[R_{2}=1]\mathbf 1[D_{2}=n]"
            r"\mathbf 1[C_{rgb}=n]\mathbf 1[P_{tube}=n]\mathbf 1[B_{add}=0]"
            r"\mathbf 1[RGB_{receiver}]"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.predict_project_realization_admissibility_20260721:"
            "predict_project_realization_admissibility"
        ),
        domain_of_validity={
            "vehicle": "G2b seed_compose_b2 support-fill realization",
            "pair_count": PAIR_COUNT,
            "projection": "factor-2 fp32/rational lattice to uint8 camera",
            "authority": "admission gate only; no score authority",
            "verdict_scope": (
                "falsifies the asserted zero-byte semantic cells-to-RGB link for this seed; "
                "does not kill predict-project or support-fill families"
            ),
            "blocker_id": BLOCKER_ID,
            "score_claim": False,
            "promotion_eligible": False,
        },
        units_in={
            "pair_count": "pairs",
            "uint8_factor2_exact_fraction": "fraction",
            "pair_predicate_counts": "pairs",
            "additional_seed_bytes": "bytes",
            "receiver_derived_rgb": "bool",
        },
        units_out={"accepted": "bool", "status": "categorical"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"failed_predicate_fraction_n600": failed_fraction},
        last_calibration_utc=UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools/measure_realization_g2_lattice.py",
            "G2b successor admission",
            "predict-project bit allocator",
        ),
        canonical_producers=(
            "tools/measure_realization_g2_lattice.py",
            SOURCE_RECEIPT,
        ),
        provenance=provenance,
    )


def populate_predict_project_realization_admissibility_v1(
    *, path=None, lock_path=None, agent=None, subagent_id=None
) -> CanonicalEquation:
    """Append the measured law through the locked canonical registry helper."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_predict_project_realization_admissibility_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "Task #578 G2b source-RGB control: exact lattice but zero-byte semantic "
            "cells-to-RGB admission false; pointer unmoved"
        ),
    )
    return equation


__all__ = [
    "BLOCKER_ID",
    "EQUATION_ID",
    "build_predict_project_realization_admissibility_v1",
    "populate_predict_project_realization_admissibility_v1",
    "predict_project_realization_admissibility",
    "predict_project_realization_certificate",
]
