# SPDX-License-Identifier: MIT
"""Canonical fail-closed admission law for a counted shared receiver.

The predicate conjunction is necessary but not sufficient.  This v1 surface
cannot confer authority until a canonical contest-CPU evaluator and trusted
production-receiver parser derive every predicate from replayed bytes.
"""

from __future__ import annotations

import math
from typing import Any

from tac.boundary_math.shared_receiver_admission import (
    BLOCKER_ID,
    MAX_ARCHIVE_BYTES,
    MAX_BYTES_PER_PAIR,
    MAX_D_SEG,
    PAIR_COUNT,
)
from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "shared_receiver_counted_spatial_hard_oracle_admission_v1"
SOURCE_RECEIPT = ".omx/research/shared_receiver_r1_20260720.json"
SOURCE_DENSE_RECEIPT = ".omx/research/shared_receiver_r1_dense_quotient_field_receipt_20260720.json"
UTC = "2026-07-20T15:32:08Z"


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def shared_receiver_hard_admission_certificate(
    *,
    n_pairs: int,
    archive_bytes: int,
    d_seg: float,
    d_pose: float,
    exact_archive: bool,
    archive_parseback_identical: bool,
    production_receiver: bool,
    through_r_authority: bool,
    hard_cpu_torch_oracle: bool,
    packet_mutation_changes_decoded: bool,
    scorer_free_spatial_rgb_pullback: bool,
    content_hashes_bound: bool,
) -> dict[str, Any]:
    """Report structural predicates without granting self-asserted authority."""

    if isinstance(n_pairs, bool) or not isinstance(n_pairs, int) or n_pairs <= 0:
        raise ValueError("n_pairs must be a positive integer")
    if isinstance(archive_bytes, bool) or not isinstance(archive_bytes, int) or archive_bytes <= 0:
        raise ValueError("archive_bytes must be a positive integer")
    d_seg_value = _finite_nonnegative(d_seg, "d_seg")
    d_pose_value = _finite_nonnegative(d_pose, "d_pose")
    predicates = {
        "n600": n_pairs == PAIR_COUNT,
        "archive_in_box": archive_bytes <= MAX_ARCHIVE_BYTES,
        "bytes_per_pair_in_box": archive_bytes / PAIR_COUNT <= MAX_BYTES_PER_PAIR,
        "d_seg_in_box": d_seg_value <= MAX_D_SEG,
        "exact_archive": _strict_bool(exact_archive, "exact_archive"),
        "archive_parseback_identical": _strict_bool(archive_parseback_identical, "archive_parseback_identical"),
        "production_receiver": _strict_bool(production_receiver, "production_receiver"),
        "through_r_authority": _strict_bool(through_r_authority, "through_r_authority"),
        "hard_cpu_torch_oracle": _strict_bool(hard_cpu_torch_oracle, "hard_cpu_torch_oracle"),
        "packet_mutation_changes_decoded": _strict_bool(
            packet_mutation_changes_decoded, "packet_mutation_changes_decoded"
        ),
        "scorer_free_spatial_rgb_pullback": _strict_bool(
            scorer_free_spatial_rgb_pullback, "scorer_free_spatial_rgb_pullback"
        ),
        "content_hashes_bound": _strict_bool(content_hashes_bound, "content_hashes_bound"),
    }
    structural_conjunction = all(predicates.values())
    return {
        "accepted": False,
        "status": BLOCKER_ID,
        "predicates": predicates,
        "structural_conjunction": structural_conjunction,
        "trusted_contest_cpu_verifier_wired": False,
        "n_pairs": n_pairs,
        "archive_bytes": archive_bytes,
        "bytes_per_pair": archive_bytes / PAIR_COUNT,
        "d_seg": d_seg_value,
        "d_pose": d_pose_value,
        "score_claim": False,
        "promotion_eligible": False,
    }


def build_shared_receiver_counted_spatial_hard_oracle_admission_v1() -> CanonicalEquation:
    """Build the measured hard-admission law and its dense-form anchor."""

    provenance = build_provenance_for_research_sidecar(
        sidecar_path=SOURCE_RECEIPT,
        reactivation_criteria=(
            "Wire the canonical contest-CPU evaluator and trusted production-receiver "
            "parser first; then re-evaluate one content-addressed exact n600 archive "
            "that binds the counted generator, parse-back, causality canary, and oracle."
        ),
        measurement_axis="[macOS-CPU advisory]",
        hardware_substrate="macos_arm64",
        captured_at_utc=UTC,
    )
    anchor = EmpiricalAnchor(
        anchor_id="shared_receiver_dense_float32_section_n600_20260720",
        measurement_utc=UTC,
        inputs={
            "n_pairs": PAIR_COUNT,
            "source_field_bytes": 1_887_436_928,
            "source_field_sha256": ("59e96781aa1bac153bc8bb277cecdbd4b4e98fdfd41f50aa2294537b90390944"),
            "compression": "zip_deflate9_deterministic_metadata_zip64",
        },
        predicted_output={
            "max_section_bytes_if_archive_admissible": MAX_ARCHIVE_BYTES,
        },
        empirical_output={
            "section_zip_bytes": 561_502_227,
            "section_zip_sha256": ("d594fd6194d1482cb641cb640dd236b31d962420183811646d01dd6927276132"),
            "bytes_per_pair": 935_837.045,
            "through_r_authority": False,
            "d_seg": None,
            "d_pose": None,
        },
        residual=float(561_502_227 - MAX_ARCHIVE_BYTES),
        source_artifact=SOURCE_DENSE_RECEIPT,
        measurement_method=(
            "exact source SHA-256 plus deterministic single-member ZIP64/deflate9, "
            "fixed member metadata, CRC replay, and exact output-byte hash"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Shared counted receiver hard admission",
        one_line_summary=(
            "The exact n600 receiver conjunction is necessary, but v1 always refuses "
            "authority until canonical contest-CPU replay derives every predicate."
        ),
        latex_form=(
            r"A=V_{trusted}\mathbf 1[n=600]\mathbf 1[B\le286680]"
            r"\mathbf 1[d_{seg}\le3.39\times10^{-4}]"
            r"\prod_{q\in\{parse,R,CPU,causal,pullback,hash\}}\mathbf 1[q],\quad V_{trusted}=0"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.shared_receiver_hard_admission_20260720:shared_receiver_hard_admission_certificate"
        ),
        domain_of_validity={
            "n_pairs": PAIR_COUNT,
            "max_archive_bytes": MAX_ARCHIVE_BYTES,
            "max_bytes_per_pair": MAX_BYTES_PER_PAIR,
            "max_d_seg": MAX_D_SEG,
            "authority": "hard CPU-Torch through actual R on exact archive bytes",
            "authority_available": False,
            "trusted_verifier_debt": (
                "canonical contest-CPU evaluator invocation plus trusted production-receiver parser"
            ),
            "verdict_scope": (
                "conjunctive admission only; the measured negative covers dense-f32, "
                "PDW1 label/fill, and the measured sparse-repair prefix, not the family"
            ),
            "blocker_id": BLOCKER_ID,
            "score_claim": False,
            "promotion_eligible": False,
        },
        units_in={
            "archive_bytes": "bytes",
            "d_seg": "fraction",
            "d_pose": "official PoseNet MSE",
            "receiver_predicates": "bool",
        },
        units_out={
            "accepted": "bool",
            "status": "categorical",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"dense_section_excess_bytes": float(561_502_227 - MAX_ARCHIVE_BYTES)},
        last_calibration_utc=UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.boundary_math.shared_receiver_admission.evaluate_shared_receiver_admission",
            "tac.boundary_math.integer_plane_emitter_byte_close.build_counted_archive",
            "tools.measure_shared_receiver_admission",
        ),
        canonical_producers=(SOURCE_RECEIPT, SOURCE_DENSE_RECEIPT),
        provenance=provenance,
    )


def populate_shared_receiver_counted_spatial_hard_oracle_admission_v1(
    *, path=None, lock_path=None, agent=None, subagent_id=None
) -> CanonicalEquation:
    """Append the measured admission law through the locked registry helper."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_shared_receiver_counted_spatial_hard_oracle_admission_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "Supersedes the self-authored candidate-admission interpretation: R1 measured "
            "dense-section rate; trusted contest-CPU verifier remains explicitly unwired"
        ),
    )
    return equation


__all__ = [
    "EQUATION_ID",
    "build_shared_receiver_counted_spatial_hard_oracle_admission_v1",
    "populate_shared_receiver_counted_spatial_hard_oracle_admission_v1",
    "shared_receiver_hard_admission_certificate",
]
