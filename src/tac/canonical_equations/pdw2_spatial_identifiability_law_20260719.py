# SPDX-License-Identifier: MIT
"""Canonical law for coefficient-only PDW2 spatial non-identifiability.

The blocker is explicit and exact for the current contract:
`PDW2_COEFFICIENT_ONLY_SPATIAL_NONIDENTIFIABILITY`.
"""

from __future__ import annotations

from typing import Any

from tac.boundary_math.pdw2_spatial_receiver import (
    PDW2_COEFFICIENT_ONLY_SPATIAL_NONIDENTIFIABILITY,
)
from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "pdw2_coefficient_only_spatial_nonidentifiability_v1"
SOURCE_MEMO = ".omx/research/pdw2_spatial_receiver_576_implementation_spec_20260719.md"
SOURCE_RECEIPT = ".omx/research/pdw2_spatial_receiver_576_blocker_receipt_20260719.json"
UTC = "2026-07-19T20:54:44Z"


def pdw2_spatial_nonidentifiability_admissibility(
    *,
    packet_to_partition_claim: bool,
    through_r_field_present: bool,
) -> dict[str, Any]:
    """Evaluate whether a segment partition claim is admissible.

    A packet-only claim is FORMULATION-SCOPED INADMISSIBLE for the sealed
    138-byte packet: the executable anchor supplies two finite fields that
    produce different partitions under one fixed arithmetic. Through-R input
    is required before a partition claim can be entertained as non-null.
    """

    if packet_to_partition_claim and not through_r_field_present:
        admissible = False
        basis = "blocked"
    elif packet_to_partition_claim and through_r_field_present:
        admissible = False
        basis = "insufficient_input_metadata"
    else:
        admissible = False
        basis = "not_applicable"
    return {
        "admissible": admissible,
        "basis": basis,
        "blocker_id": PDW2_COEFFICIENT_ONLY_SPATIAL_NONIDENTIFIABILITY,
        "packet_to_partition_claim": bool(packet_to_partition_claim),
        "through_r_field_present": bool(through_r_field_present),
        "score_claim": False,
        "promotion_eligible": False,
    }


def build_pdw2_spatial_identifiability_law_v1() -> CanonicalEquation:
    def _prov(path: str, utc: str):
        return build_provenance_for_research_sidecar(
            sidecar_path=path,
            reactivation_criteria=(
                "PDW2 spatial receiver probe + law gate is scoped to packet-only "
                "claims. A non-identity patching of the same packet+constant field "
                "must preserve a packet-level blocker until through-R metadata and "
                "pullback path are added. This is a direct proof built from the "
                "declared target arithmetic and deterministic partition witnesses; "
                "no imported theorem or external paper is needed."
            ),
            measurement_axis="[macOS-CPU advisory]",
            hardware_substrate="macos_arm64",
            captured_at_utc=utc,
        )

    receipt_provenance = _prov(SOURCE_RECEIPT, UTC)
    anchor = EmpiricalAnchor(
        anchor_id="pdw2_packet_only_spatial_nonidentifiability_n600_20260719",
        measurement_utc=UTC,
        inputs={
            "packet_bytes": 138,
            "packet_sha256": "93c0d3320e6673aed1975426a6c8c1bbc41475f295ea62b357ad7a6bf9427568",
            "quotient_shape": [600, 384, 512, 4],
            "receiver_arithmetic": "native float32 zero-sum gauge plus first-max argmax",
        },
        predicted_output={
            "packet_only_unique_partition": False,
            "coefficient_mutation_must_change_at_least_one_real_field_cell": True,
        },
        empirical_output={
            "constant_field_a_class": 1,
            "constant_field_b_class": 2,
            "same_packet_two_partitions": True,
            "n600_partition_sha256": (
                "7b4558bccf58194e2274ee70e0086a48bd925685963b363db0bc6df13863d870"
            ),
            "mutated_coefficient_mismatch_pixels_n24": 108,
            "deterministic_n600_replay": True,
            "through_r_authority": False,
            "d_seg": None,
            "d_pose": None,
        },
        residual=0.0,
        source_artifact=SOURCE_RECEIPT,
        measurement_method=(
            "strict 138-byte packet parse-back; executable two-field witness; read-only "
            "n24 then n600 quotient memmap receiver; coefficient mutation canary; "
            "independent deterministic n600 replay"
        ),
        provenance=receipt_provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )

    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="PDW2 packet-only non-identifiability",
        one_line_summary=(
            "A single #553 PDW2 packet admits multiple spatial partitions for"
            " different quotient fields; packet-only through-R claims are blocked."
        ),
        latex_form=(
            r"\mathrm{admissible}(\hat y_{\mathrm{spatial}})\Longleftarrow"
            r"\neg(\text{packet-only claim})\wedge\text{through-R metadata/pullback}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.pdw2_spatial_identifiability_law_20260719:"
            "pdw2_spatial_nonidentifiability_admissibility"
        ),
        domain_of_validity={
            "receiver_contract": "scorer-free #553 PDW2 + explicit quotient field",
            "packet_sha256": "93c0d3320e6673aed1975426a6c8c1bbc41475f295ea62b357ad7a6bf9427568",
            "claim_type": "packet-only spatial-partition admissibility",
            "instances": ["n24", "n600"],
            "verdict_scope": (
                "packet-only formulation scope: same packet admits distinct"
                " partitions on finite fields unless through-R pullback is added"
            ),
            "blocker": PDW2_COEFFICIENT_ONLY_SPATIAL_NONIDENTIFIABILITY,
            "score_claim": False,
        },
        units_in={
            "packet_to_partition_claim": "bool",
            "through_r_field_present": "bool",
        },
        units_out={
            "admissible": "bool",
            "basis": "categorical",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "packet_only_blocker": 0.0,
        },
        last_calibration_utc=UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "src.tac.boundary_math.pdw2_spatial_receiver",
            "tools.probe_pdw2_spatial_receiver",
        ),
        canonical_producers=(SOURCE_MEMO, SOURCE_RECEIPT),
        provenance=receipt_provenance,
    )


def populate_pdw2_spatial_identifiability_law_v1(
    *, path=None, lock_path=None, agent=None, subagent_id=None
) -> CanonicalEquation:
    """Append the measured law through the locked canonical registry helper."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_pdw2_spatial_identifiability_law_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes="Task #576 measured coefficient consumption; packet-only spatial claim blocked",
    )
    return equation


__all__ = [
    "EQUATION_ID",
    "PDW2_COEFFICIENT_ONLY_SPATIAL_NONIDENTIFIABILITY",
    "build_pdw2_spatial_identifiability_law_v1",
    "pdw2_spatial_nonidentifiability_admissibility",
    "populate_pdw2_spatial_identifiability_law_v1",
]
