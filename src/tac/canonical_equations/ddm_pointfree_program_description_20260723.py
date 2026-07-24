# SPDX-License-Identifier: MIT
"""Canonical two-typed rate law for DDM PF1 discrete skeletons.

The law exposes only event topology, template reuse, and repetition-with-
variation to PF1. Continuous values remain opaque typed fiber slots encoded by
their native analog coders. The generic interpreter is rule-118 receiver code,
while both skeleton and fiber payloads are counted. This module therefore
grants no license to tokenize continuous fibers, add PF1 beside G1/DV2, or hide
a learned constant in receiver code.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

REPO: Final = Path(__file__).resolve().parents[3]
EQUATION_ID: Final = "ddm_pf1_two_typed_discrete_skeleton_rate_delta_v1"
AXIS: Final = "[macOS-CPU frozen-scorer advisory]"
MEASUREMENT_UTC: Final = "2026-07-24T01:26:42Z"
POINTER: Final = "0.1910828242 [contest-CPU]"
RECEIPT: Final = (
    ".omx/research/ddm_pf1_pointfree_program_description_n600_20260723T235900Z/"
    "ddm_pf1_pointfree_program_description_n600_scope_corrected_receipt.json"
)
RECEIPT_SHA256: Final = "afe14bf62d3ff18e56a0e669f96f5adfb97c5eaa068ed4c1cf3c35d837d398e0"
FORMULATIONS: Final = frozenset({"LITERAL", "SHARED_LIBRARY", "STRUCTURAL"})
SCOPE_ELIGIBLE_FORMULATIONS: Final = frozenset({"STRUCTURAL"})

RateDisposition = Literal["ADMIT_SUBSTITUTIVE_PROGRAM", "KEEP_FLAT_CONTROL"]


@dataclass(frozen=True, slots=True)
class ProgramRateEvaluation:
    """One apples-to-apples real-coder comparison."""

    delta_program_minus_flat_bytes: int
    delta_skeleton_bytes: int
    delta_fiber_bytes: int
    disposition: RateDisposition
    semantic_parseback_exact: Literal[True]
    same_description_content: Literal[True]
    opaque_native_fibers_counted_separately: Literal[True]

    @property
    def program_wins(self) -> bool:
        return self.delta_program_minus_flat_bytes < 0


def _positive_counted_bytes(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def evaluate_two_typed_program_vs_flat_bytes(
    *,
    program_skeleton_counted_bytes: int,
    program_fiber_counted_bytes: int,
    flat_skeleton_counted_bytes: int,
    flat_fiber_counted_bytes: int,
    semantic_parseback_exact: bool,
    same_description_content: bool,
) -> ProgramRateEvaluation:
    """Evaluate the exact two-typed skeleton-plus-fiber rate delta.

    Continuous fibers remain opaque and native-coded; only the discrete
    skeleton is exposed to the PF1 real coder. The comparison fails closed
    unless both operands encode the same content and public parse-back is exact.
    """

    program_skeleton = _positive_counted_bytes(
        program_skeleton_counted_bytes, "program_skeleton_counted_bytes"
    )
    flat_skeleton = _positive_counted_bytes(
        flat_skeleton_counted_bytes, "flat_skeleton_counted_bytes"
    )
    for value, name in (
        (program_fiber_counted_bytes, "program_fiber_counted_bytes"),
        (flat_fiber_counted_bytes, "flat_fiber_counted_bytes"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if not isinstance(semantic_parseback_exact, bool):
        raise ValueError("semantic_parseback_exact must be bool")
    if not isinstance(same_description_content, bool):
        raise ValueError("same_description_content must be bool")
    if not semantic_parseback_exact or not same_description_content:
        raise ValueError("program-vs-flat rate requires exact same-description replay")
    delta_skeleton = program_skeleton - flat_skeleton
    delta_fiber = program_fiber_counted_bytes - flat_fiber_counted_bytes
    delta = delta_skeleton + delta_fiber
    disposition: RateDisposition = (
        "ADMIT_SUBSTITUTIVE_PROGRAM" if delta < 0 else "KEEP_FLAT_CONTROL"
    )
    return ProgramRateEvaluation(
        delta_program_minus_flat_bytes=delta,
        delta_skeleton_bytes=delta_skeleton,
        delta_fiber_bytes=delta_fiber,
        disposition=disposition,
        semantic_parseback_exact=True,
        same_description_content=True,
        opaque_native_fibers_counted_separately=True,
    )


def discrete_skeleton_formulation_closed(
    deltas_by_formulation: Mapping[str, int],
) -> bool:
    """Apply the preregistered scope-eligible formulation closure rule.

    Only formulations explicitly authorized in
    ``SCOPE_ELIGIBLE_FORMULATIONS`` may enter this predicate. The current
    equation authorizes only STRUCTURAL, so it cannot close; adding successor
    formulations requires recalibrating the equation and its allowlist.
    """

    if not isinstance(deltas_by_formulation, Mapping):
        raise ValueError("deltas_by_formulation must be a mapping")
    if not set(deltas_by_formulation).issubset(SCOPE_ELIGIBLE_FORMULATIONS):
        raise ValueError("formulation is not scope-eligible")
    deltas: list[int] = []
    for name in sorted(deltas_by_formulation):
        value = deltas_by_formulation[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} delta must be an integer")
        deltas.append(value)
    return len(deltas) >= 3 and all(delta >= 0 for delta in deltas)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_receipt(receipt_path: Path | None = None) -> tuple[dict[str, Any], str]:
    source = receipt_path or REPO / RECEIPT
    if not source.is_file():
        raise ValueError("DDM PF1 receipt is unavailable")
    if receipt_path is None and _sha256_path(source) != RECEIPT_SHA256:
        raise ValueError("DDM PF1 receipt SHA-256 drifted")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != "ddm_pf1_pointfree_program_description_receipt.v2":
        raise ValueError("DDM PF1 receipt schema drifted")
    if (
        payload.get("evidence_axis") != AXIS
        or payload.get("score_claim") is not False
        or payload.get("pointer_moved") is not False
        or payload.get("promotion_eligible") is not False
    ):
        raise ValueError("DDM PF1 receipt authority firewall drifted")
    rate = payload.get("rate_matrix", {})
    if rate.get("semantic_parseback_exact_all_rows") is not True:
        raise ValueError("DDM PF1 receipt lost exact parse-back custody")
    receiver = payload.get("receiver_measurement", {})
    if receiver.get("source_program_archive_byte_identical") is not True:
        raise ValueError("DDM PF1 receiver is not byte-identical to its source archive")
    scope = payload.get("scope_correction", {})
    if (
        scope.get("scope") != "DISCRETE_SKELETON_ONLY"
        or scope.get("continuous_fibers_not_tokenized") is not True
    ):
        raise ValueError("DDM PF1 discrete-skeleton scope correction drifted")
    source_name = str(source if receipt_path is not None else RECEIPT)
    return payload, source_name


def _rate_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rate = payload["rate_matrix"]
    rows = [*rate["component_rows"], *rate["bundle_rows"]]
    if len(rows) != 18:
        raise ValueError("DDM PF1 receipt must contain 18 preregistered rate rows")
    return rows


def build_ddm_pf1_two_typed_discrete_skeleton_rate_delta_v1(
    receipt_path: Path | None = None,
) -> CanonicalEquation:
    """Build the scope-corrected PF1 rate equation from its immutable receipt."""

    payload, source = _load_receipt(receipt_path)
    rows = _rate_rows(payload)
    eligible_rows = [
        row for row in rows if row["discrete_skeleton_scope_eligible"]
    ]
    if len(eligible_rows) != 6 or any(
        row["formulation"] not in SCOPE_ELIGIBLE_FORMULATIONS
        or row["two_typed_split_status"] != "MEASURED_EXACT"
        for row in eligible_rows
    ):
        raise ValueError("DDM PF1 scope-eligible two-typed row set drifted")
    residual = 0
    split_summary: dict[str, dict[str, int]] = {}
    for row in eligible_rows:
        evaluated = evaluate_two_typed_program_vs_flat_bytes(
            program_skeleton_counted_bytes=row["program_skeleton_counted_bytes"],
            program_fiber_counted_bytes=row["program_fiber_counted_bytes"],
            flat_skeleton_counted_bytes=row["flat_skeleton_counted_bytes"],
            flat_fiber_counted_bytes=row["flat_fiber_counted_bytes"],
            semantic_parseback_exact=row["semantic_parseback_exact"],
            same_description_content=True,
        )
        residual = max(
            residual,
            abs(evaluated.delta_program_minus_flat_bytes - row["delta_program_minus_flat_bytes"]),
            abs(evaluated.delta_skeleton_bytes - row["delta_skeleton_bytes"]),
            abs(evaluated.delta_fiber_bytes - row["delta_fiber_bytes"]),
        )
        split_summary[row["description"]] = {
            "delta_skeleton_bytes": evaluated.delta_skeleton_bytes,
            "delta_fiber_bytes": evaluated.delta_fiber_bytes,
            "delta_total_bytes": evaluated.delta_program_minus_flat_bytes,
        }

    stored_falsifier = payload["rate_matrix"]["formulation_falsifier"]
    recomputed_closed = {
        name: discrete_skeleton_formulation_closed(
            {
                formulation: row["delta_program_minus_flat_by_formulation"][formulation]
                for formulation in row["scope_eligible_formulations"]
            }
        )
        for name, row in stored_falsifier.items()
    }
    if any(recomputed_closed.values()):
        raise ValueError("DDM PF1 measured descriptions unexpectedly formulation-closed")

    provenance = build_provenance_for_research_sidecar(
        sidecar_path=RECEIPT,
        reactivation_criteria=(
            "Recalibrate after two additional scope-eligible skeleton formulations, a "
            "changed native fiber coder, or receiver/contest-axis replay; never infer "
            "continuous-fiber sufficiency or additivity."
        ),
        measurement_axis=AXIS,
        hardware_substrate="darwin_arm64_cpu_torch",
        captured_at_utc=MEASUREMENT_UTC,
    )
    anchor = EmpiricalAnchor(
        anchor_id="ddm_pf1_n600_two_typed_discrete_skeleton_rate_matrix_20260724",
        measurement_utc=MEASUREMENT_UTC,
        inputs={
            "description_count": 6,
            "formulations": sorted(FORMULATIONS),
            "scope_eligible_formulations": sorted(SCOPE_ELIGIBLE_FORMULATIONS),
            "scope_eligible_rate_row_count": len(eligible_rows),
            "rate_row_count": len(rows),
            "typed_config_sha256": payload["typed_config_sha256"],
            "receipt_sha256": RECEIPT_SHA256,
            "source_program_archive_byte_identical": True,
        },
        predicted_output={
            "law": "delta_B = delta_B_discrete_skeleton + delta_B_opaque_native_fiber",
            "admission": "delta_B < 0 under exact two-typed same-description replay",
            "closure": ">=3 scope-eligible formulations all have delta_B >= 0",
        },
        empirical_output={
            "structural_two_typed_split": split_summary,
            "formulation_closed": recomputed_closed,
            "verdict": payload["falsifier"]["verdict"],
            "verdict_scope": payload["falsifier"]["verdict_scope"],
            "pointer": POINTER,
            "pointer_moved": False,
            "score_claim": False,
        },
        residual=float(residual),
        source_artifact=source,
        measurement_method=(
            "mechanical compile of settled G1/V15/DV2 discrete event/template skeletons; "
            "G1 centroid/shape, V15 RGB amplitude, and DV2 arithmetic numeric streams "
            "remain opaque native-coded typed fibers; public byte-identical replay; exact "
            "two-part real-coder accounting against identical-content flat control"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        noise_floor=0.0,
        noise_floor_provenance="integer byte counts under deterministic real-coder replay",
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="DDM PF1 two-typed discrete-skeleton rate delta",
        one_line_summary=(
            "PF1 replaces a flat skeleton only when exact replay has fewer total bytes "
            "after opaque native-coded continuous fibers are counted separately."
        ),
        latex_form=(
            r"\Delta B=\Delta B_{\mathrm{skeleton}}+\Delta B_{\mathrm{fiber}},\quad "
            r"\mathrm{admit}\iff\mathrm{ReplayExact}\land\Delta B<0"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.ddm_pointfree_program_description_20260723:"
            "evaluate_two_typed_program_vs_flat_bytes"
        ),
        domain_of_validity={
            "vehicle": "DDM PF1 discrete skeleton over settled G1, V15, and DV2",
            "comparison": "substitutive within pool program_description",
            "counting_boundary": {
                "generic_interpreter": "FREE_rule118",
                "discrete_skeleton": "COUNTED_real_coder",
                "continuous_fiber": "COUNTED_opaque_native_coder",
            },
            "required": [
                "same description content",
                "semantic parse-back exact",
                "skeleton-versus-fiber split",
            ],
            "excluded": [
                "tokenizing amplitudes, statistics, phases, spectral coefficients, xi, or margins",
                "additive use beside G1 or DV2",
                "general program synthesis",
                "hidden video constants in receiver code",
                "contest score or promotion claim",
                "family closure from one formulation",
            ],
            "verdict_scope": payload["falsifier"]["verdict_scope"],
            "score_claim": False,
        },
        units_in={
            "program_skeleton_counted_bytes": "bytes",
            "program_fiber_counted_bytes": "bytes",
            "flat_skeleton_counted_bytes": "bytes",
            "flat_fiber_counted_bytes": "bytes",
            "semantic_parseback_exact": "boolean",
            "same_description_content": "boolean",
        },
        units_out={
            "delta_program_minus_flat_bytes": "bytes",
            "delta_skeleton_bytes": "bytes",
            "delta_fiber_bytes": "bytes",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"stored_integer_byte_delta_replay": float(residual)},
        last_calibration_utc=MEASUREMENT_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.optimization.ddm_pointfree_program.rate_row",
            "menu1.c1.program_description",
        ),
        canonical_producers=(
            "tools/measure_ddm_pf1_pointfree_program.py",
            RECEIPT,
        ),
        provenance=provenance,
    )


def populate_ddm_pf1_two_typed_discrete_skeleton_rate_delta_v1(
    *,
    receipt_path: Path | None = None,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Append the PF1 law through the locked canonical registry helper."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_ddm_pf1_two_typed_discrete_skeleton_rate_delta_v1(receipt_path)
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "DDM PF1 scope-corrected exact two-typed n600 rate matrix; six discrete-"
            "skeleton wins; opaque continuous fibers open; pointer unmoved; MAIN review"
        ),
    )
    return equation


__all__ = [
    "AXIS",
    "EQUATION_ID",
    "FORMULATIONS",
    "POINTER",
    "RECEIPT",
    "RECEIPT_SHA256",
    "SCOPE_ELIGIBLE_FORMULATIONS",
    "ProgramRateEvaluation",
    "build_ddm_pf1_two_typed_discrete_skeleton_rate_delta_v1",
    "discrete_skeleton_formulation_closed",
    "evaluate_two_typed_program_vs_flat_bytes",
    "populate_ddm_pf1_two_typed_discrete_skeleton_rate_delta_v1",
]
