# SPDX-License-Identifier: MIT
"""Canonical conditional-admission law for zero-byte receiver transforms."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
)
from tac.canonical_equations.evaluators import register_evaluator
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID: Final = "ddm_pa2_zero_byte_conditional_greedy_v1"
RATE_DENOMINATOR_BYTES: Final = 37_545_489
REPO: Final = Path(__file__).resolve().parents[3]
RECEIPT: Final = (
    REPO
    / ".omx/research/ddm_pa2_zero_byte_decode_family_20260724T194836Z/receipt.json"
)


def zero_byte_conditional_score(
    *,
    seg_errors: int,
    scored_pixels: int,
    d_pose: float,
    archive_bytes: int,
    decoded_frame_only: bool,
    archive_identity_exact: bool,
    n600_batch32_complete: bool,
) -> float:
    """Evaluate one receiver output only after its zero-byte gates pass."""

    for name, value in (
        ("seg_errors", seg_errors),
        ("scored_pixels", scored_pixels),
        ("archive_bytes", archive_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be an exact nonnegative integer")
    if scored_pixels == 0:
        raise ValueError("scored_pixels must be positive")
    if any(
        not isinstance(value, bool)
        for value in (
            decoded_frame_only,
            archive_identity_exact,
            n600_batch32_complete,
        )
    ):
        raise ValueError("zero-byte admission gates must be exact booleans")
    if not (
        decoded_frame_only
        and archive_identity_exact
        and n600_batch32_complete
    ):
        raise ValueError(
            "score requires decoded-frame-only derivation, exact archive identity, "
            "and complete n600 batch32 measurement"
        )
    pose = float(d_pose)
    if not math.isfinite(pose) or pose < 0.0:
        raise ValueError("d_pose must be finite and nonnegative")
    return (
        100.0 * seg_errors / scored_pixels
        + math.sqrt(10.0 * pose)
        + 25.0 * archive_bytes / RATE_DENOMINATOR_BYTES
    )


def select_strict_conditional(
    *,
    current_score: float,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Choose the lowest freshly measured conditional iff it is strict."""

    current = float(current_score)
    if not math.isfinite(current):
        raise ValueError("current_score must be finite")
    if not candidates:
        raise ValueError("at least one conditional candidate is required")
    normalized: list[dict[str, Any]] = []
    for candidate in candidates:
        if set(candidate) != {"member", "score", "archive_byte_delta"}:
            raise ValueError("conditional candidate keys differ")
        member = candidate["member"]
        score = float(candidate["score"])
        byte_delta = candidate["archive_byte_delta"]
        if not isinstance(member, str) or not member:
            raise ValueError("member must be a nonempty string")
        if not math.isfinite(score):
            raise ValueError("candidate score must be finite")
        if isinstance(byte_delta, bool) or not isinstance(byte_delta, int):
            raise ValueError("archive_byte_delta must be an exact integer")
        if byte_delta != 0:
            raise ValueError("zero-byte family candidate changed archive bytes")
        normalized.append(
            {
                "member": member,
                "score": score,
                "archive_byte_delta": byte_delta,
            }
        )
    best = min(normalized, key=lambda row: (row["score"], row["member"]))
    admitted = best["score"] < current
    return {
        "selected_member": best["member"] if admitted else None,
        "strict_improvement": admitted,
        "current_score": current,
        "selected_score": best["score"] if admitted else current,
        "conditional_delta_score": best["score"] - current,
    }


def _evaluate(inputs: Mapping[str, Any]) -> float:
    required = {
        "seg_errors",
        "scored_pixels",
        "d_pose",
        "archive_bytes",
        "decoded_frame_only",
        "archive_identity_exact",
        "n600_batch32_complete",
    }
    if set(inputs) != required:
        raise ValueError("PA2 score inputs differ from the canonical contract")
    return zero_byte_conditional_score(**dict(inputs))


register_evaluator(EQUATION_ID, _evaluate)


def build_ddm_pa2_zero_byte_decode_family(
    *,
    source_receipt: Path = RECEIPT,
) -> CanonicalEquation:
    """Build the research-only PA2 conditional selection law."""

    provenance = build_provenance_for_research_sidecar(
        source_receipt,
        reactivation_criteria=(
            "Re-run n600 batch32 on any new decoded-frame-only receiver member "
            "or any newly sealed base; admit only after exact archive identity "
            "and a freshly measured strict conditional improvement."
        ),
        measurement_axis="[macOS-CPU frozen-scorer advisory]",
        hardware_substrate="darwin_arm64_cpu_torch2.11_threads4_batch32",
        captured_at_utc="2026-07-24T21:20:00Z",
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="DDM PA2 zero-byte conditional greedy law",
        one_line_summary=(
            "Greedily compose decoded-frame-only transforms using freshly "
            "remeasured conditional score while exact archive bytes remain fixed."
        ),
        latex_form=(
            r"S_B(T\mid A)=100d_{\rm seg}(T\circ A(B))"
            r"+\sqrt{10d_{\rm pose}(T\circ A(B))}"
            r"+25|B|/37545489,\quad"
            r"T^\star=\arg\min_T S_B(T\mid A),\quad"
            r"A\leftarrow T^\star\circ A\ \mathrm{iff}\ "
            r"S_B(T^\star\mid A)<S_B(A)"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.ddm_pa2_zero_byte_decode_family_20260724:"
            "zero_byte_conditional_score"
        ),
        domain_of_validity={
            "transform_inputs": "decoded frames plus frozen generic geometry only",
            "forbidden_free_inputs": (
                "per-pair tables, target labels, video-derived coefficients, "
                "selected gauge positions, residual payloads"
            ),
            "measurement": "n600 batch32 frozen scorers",
            "archive_gate": "same bytes and SHA-256 before/after",
            "composition": "fresh non-telescoping joint remeasure",
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        },
        units_in={
            "seg_errors": "argmax pixel errors",
            "scored_pixels": "argmax pixels",
            "d_pose": "official PoseNet MSE",
            "archive_bytes": "exact counted bytes",
            "admission_gates": "booleans",
        },
        units_out={"score": "contest-formula units on advisory components"},
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc="2026-07-24T21:20:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools.measure_ddm_pa2_zero_byte_decode_family",
            "DDM receiver composition planner",
        ),
        canonical_producers=(
            "tac.optimization.ddm_pa2_zero_byte_decode_family",
        ),
        provenance=provenance,
    )


def populate_ddm_pa2_zero_byte_decode_family(
    *,
    source_receipt: Path = RECEIPT,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Append the law through the locked canonical registry helper."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_ddm_pa2_zero_byte_decode_family(
        source_receipt=source_receipt,
    )
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "PA2 n600 batch32 three-base conditional solve; IC2 xi-hat only; "
            "research_only=true; score_claim=false; MAIN review required"
        ),
    )
    return equation


__all__ = [
    "EQUATION_ID",
    "RATE_DENOMINATOR_BYTES",
    "build_ddm_pa2_zero_byte_decode_family",
    "populate_ddm_pa2_zero_byte_decode_family",
    "select_strict_conditional",
    "zero_byte_conditional_score",
]
