# SPDX-License-Identifier: MIT
"""Canonical min-S row law for the DDM MS2R tolerance-waterfilled solve.

The equation is executable but grants no measurement authority.  A row is
admissible only after BUNDLE-COMPLETE, exact parse-back, and uint8 revalidation.
The global solve minimizes this row value while waterfilling the error
allowance over typed blocks and racing real coders.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
)
from tac.canonical_equations.evaluators import register_evaluator
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID: Final = "ddm_tolerance_capped_min_score_waterfill_v1"
RATE_DENOMINATOR_BYTES: Final = 37_545_489
REPO: Final = Path(__file__).resolve().parents[3]
RECEIPT: Final = REPO / (
    ".omx/research/ddm_ms2r_tolerance_capped_solve_20260724T152730Z/"
    "00_preflight_receipt.json"
)


def _exact_nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be an exact nonnegative integer")
    return value


def tolerance_capped_rung_score(
    *,
    seg_errors: int,
    scored_pixels: int,
    d_pose: float,
    raw_compact_bytes: int,
    best_coded_bytes: int,
    allowed_errors: int,
    bundle_complete: bool,
    parseback_exact: bool,
    uint8_reverified: bool,
) -> dict[str, Any]:
    """Evaluate one same-object rung after all scientific admission gates.

    ``best_coded_bytes`` is the minimum exact parse-back size from the mandated
    coder race.  ``raw_compact_bytes`` is retained as a separate compactness
    column and may not be silently substituted for coded bytes.
    """

    errors = _exact_nonnegative_int(seg_errors, "seg_errors")
    pixels = _exact_nonnegative_int(scored_pixels, "scored_pixels")
    raw_bytes = _exact_nonnegative_int(raw_compact_bytes, "raw_compact_bytes")
    coded_bytes = _exact_nonnegative_int(best_coded_bytes, "best_coded_bytes")
    allowance = _exact_nonnegative_int(allowed_errors, "allowed_errors")
    if pixels == 0:
        raise ValueError("scored_pixels must be positive")
    if coded_bytes > raw_bytes:
        raise ValueError("best_coded_bytes cannot exceed the raw-compact control")
    if any(
        not isinstance(value, bool)
        for value in (bundle_complete, parseback_exact, uint8_reverified)
    ):
        raise ValueError("admission gates must be exact booleans")
    if not (bundle_complete and parseback_exact and uint8_reverified):
        raise ValueError(
            "rung score requires BUNDLE-COMPLETE, exact parse-back, and uint8 revalidation"
        )
    pose = float(d_pose)
    if not math.isfinite(pose) or pose < 0.0:
        raise ValueError("d_pose must be finite and nonnegative")
    d_seg = errors / pixels
    rate = 25.0 * coded_bytes / RATE_DENOMINATOR_BYTES
    score = 100.0 * d_seg + math.sqrt(10.0 * pose) + rate
    return {
        "admissible_inside_error_cap": errors <= allowance,
        "d_seg": d_seg,
        "d_pose": pose,
        "raw_compact_bytes": raw_bytes,
        "best_coded_bytes": coded_bytes,
        "coder_gain_bytes": raw_bytes - coded_bytes,
        "rate_term": rate,
        "joint_S": score,
    }


def _evaluate(inputs: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "seg_errors",
        "scored_pixels",
        "d_pose",
        "raw_compact_bytes",
        "best_coded_bytes",
        "allowed_errors",
        "bundle_complete",
        "parseback_exact",
        "uint8_reverified",
    }
    if set(inputs) != required:
        raise ValueError("MS2R rung inputs differ from the canonical callable contract")
    return tolerance_capped_rung_score(**dict(inputs))


register_evaluator(EQUATION_ID, _evaluate)


def build_ddm_ms2r_tolerance_capped_solve(
    *,
    source_receipt: Path = RECEIPT,
) -> CanonicalEquation:
    """Build the research-only tolerance-waterfill law."""

    provenance = build_provenance_for_research_sidecar(
        source_receipt,
        reactivation_criteria=(
            "Close all RG3 assignment obligations, materialize an MS3 "
            "BUNDLE-COMPLETE, then run the resumable n600 waterfilled homotopy "
            "with real coder races and exact uint8/parse-back remeasurement."
        ),
        measurement_axis="[macOS-CPU frozen-scorer advisory]",
        hardware_substrate="darwin_arm64_cpu_torch_threads4_batch32",
        captured_at_utc="2026-07-24T15:40:00Z",
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="DDM tolerance-capped min-S scorer-waterfill",
        one_line_summary=(
            "Minimize exact same-object score over uint8 lattice members while "
            "waterfilling a global error allowance and racing real coders."
        ),
        latex_form=(
            r"x^\star=\arg\min_{x\in\mathbb Z_{256}^{n},\,"
            r"\sum_b e_b(x)\le E}\left[100\frac{\sum_b e_b(x)}{N}"
            r"+\sqrt{10D_{\rm pose}(x)}+\frac{25}{37545489}"
            r"\min_{c\in\mathcal C}B_c(x)\right],\quad"
            r"\lambda_b=-\partial B^\star/\partial e_b"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.ddm_ms2r_tolerance_capped_solve_20260724:"
            "tolerance_capped_rung_score"
        ),
        domain_of_validity={
            "required_bundle": "MS3 BUNDLE-COMPLETE",
            "lattice": "uint8 realized through exact R and parse-back",
            "waterfill_axes": [
                "stratum",
                "scorer_visibility",
                "g4_temporal_class",
            ],
            "visibility_partition": [
                "both-blind gauge",
                "seg-only fine chroma",
                "frame0 pose-only",
                "joint",
            ],
            "coder_race": [
                "RAW_COMPACT",
                "ZLIB9",
                "RAW_LZMA1",
                "ORDER1_CONTEXT_ARITHMETIC",
                "E4_BROTLI_Q11",
                "G4_FREE_DECODER_DERIVED_SPATIAL_CONTEXT",
            ],
            "current_status": (
                "BLOCKED_MS3_BUNDLE_PARTIAL_PF2_BUCKET_INPUT_ASSIGNMENT_ABSENT"
            ),
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
            "verdict_scope": (
                "INSTANCE current bundle x FORMULATION tolerance-waterfill preflight"
            ),
        },
        units_in={
            "seg_errors": "argmax pixel errors",
            "scored_pixels": "argmax pixels",
            "d_pose": "official PoseNet MSE",
            "raw_compact_bytes": "bytes",
            "best_coded_bytes": "bytes",
            "allowed_errors": "argmax pixel errors",
            "admission_gates": "booleans",
        },
        units_out={
            "d_seg": "fraction",
            "joint_S": "contest score-formula units on advisory components",
            "bytes": "bytes",
        },
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc="2026-07-24T15:40:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools.preflight_ddm_ms2r_tolerance_capped_solve",
            "train-decision-table SOLVE column",
        ),
        canonical_producers=(
            "tac.optimization.ddm_typed_quotient_solve",
            "tac.optimization.ddm_metric_custody_bundle",
        ),
        provenance=provenance,
    )


def populate_ddm_ms2r_tolerance_capped_solve(
    *,
    source_receipt: Path = RECEIPT,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Append the law through the locked canonical registry helper."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_ddm_ms2r_tolerance_capped_solve(
        source_receipt=source_receipt,
    )
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "MS2R tolerance-waterfill law; current MS3 bundle PARTIAL; "
            "162 duals remain NULL; score_claim=false; MAIN review required"
        ),
    )
    return equation


__all__ = [
    "EQUATION_ID",
    "RATE_DENOMINATOR_BYTES",
    "build_ddm_ms2r_tolerance_capped_solve",
    "populate_ddm_ms2r_tolerance_capped_solve",
    "tolerance_capped_rung_score",
]
