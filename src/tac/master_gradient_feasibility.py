# SPDX-License-Identifier: MIT
"""Feasibility contract for score-response "master gradient" probes.

The contest archive is a structured packet: usually a ZIP container, often with
an entropy-coded monolithic member inside it. A raw byte or bit flip on that
outer archive is therefore not a local derivative of the contest score. It is a
grammar-violating operation unless it is lowered through a packet-aware mutation
operator that rebuilds container metadata and proves inflate success.

This module keeps that distinction explicit for campaign planners:

* raw archive-byte finite differences are blocked;
* grammar-aware mutation-operator response matrices are allowed as probes;
* neither path is a score claim or promotion artifact by itself.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

MutationGrain = Literal[
    "raw_archive_bit",
    "raw_archive_byte",
    "zip_member_payload_byte",
    "logical_section_parameter",
    "grammar_aware_operator",
    "repacked_archive_candidate",
]

AxisLabel = Literal[
    "contest_cpu",
    "contest_cuda",
    "paired_contest_cpu_cuda",
    "diagnostic",
]

RAW_ARCHIVE_GRAINS: frozenset[str] = frozenset({"raw_archive_bit", "raw_archive_byte"})
OPERATOR_GRAINS: frozenset[str] = frozenset(
    {
        "logical_section_parameter",
        "grammar_aware_operator",
        "repacked_archive_candidate",
    }
)

VALID_AXIS_LABELS: frozenset[str] = frozenset(
    {"contest_cpu", "contest_cuda", "paired_contest_cpu_cuda", "diagnostic"}
)


@dataclass(frozen=True)
class MasterGradientFeasibility:
    """False-authority guard for planned score-response probes."""

    mutation_grain: str
    axis_label: str | None
    raw_byte_gradient_valid: bool
    operator_response_valid: bool
    requires_valid_packet_mutations: bool
    ready_for_operator_probe: bool
    ready_for_provider_dispatch: bool
    score_claim: bool
    promotion_eligible: bool
    rank_or_kill_eligible: bool
    ready_for_exact_eval_dispatch: bool
    evidence_grade: str
    verdict: str
    recommended_probe_grain: str
    blockers: tuple[str, ...]
    required_proofs: tuple[str, ...]
    rationale: tuple[str, ...]

    def to_manifest(self) -> dict[str, object]:
        return asdict(self)


def audit_master_gradient_probe_plan(
    *,
    mutation_grain: MutationGrain | str,
    archive_is_zip: bool = True,
    payload_is_entropy_coded: bool = True,
    updates_zip_headers: bool = False,
    updates_crc: bool = False,
    repacks_archive: bool = False,
    proves_inflate_success: bool = False,
    axis_label: AxisLabel | str | None = None,
) -> MasterGradientFeasibility:
    """Classify a planned score-response probe before dispatch.

    ``ready_for_operator_probe`` means the plan is coherent enough to run as a
    research/probe job. It still cannot claim a score, promote a lane, rank a
    candidate, or kill a method family without a byte-closed exact-eval result.
    """

    grain = str(mutation_grain)
    axis = None if axis_label is None else str(axis_label)
    blockers: list[str] = []
    required: list[str] = []
    rationale: list[str] = []

    if axis is None:
        blockers.append("missing_axis_label")
        required.append("axis_label")
    elif axis not in VALID_AXIS_LABELS:
        blockers.append("unknown_axis_label")
        required.append("axis_label_in_contest_cpu_contest_cuda_paired_or_diagnostic")

    if grain in RAW_ARCHIVE_GRAINS:
        if archive_is_zip:
            blockers.append("raw_archive_byte_flip_breaks_zip_or_central_directory_grammar")
        if payload_is_entropy_coded:
            blockers.append("raw_archive_byte_flip_is_not_local_in_entropy_coded_payload")
        if not updates_zip_headers:
            blockers.append("zip_headers_not_rebuilt")
        if not updates_crc:
            blockers.append("zip_crc_not_rebuilt")
        blockers.append("raw_byte_score_derivative_not_well_defined_for_discrete_packet")
        rationale.extend(
            (
                "archive bytes are container or compressed-stream symbols, not smooth model parameters",
                "finite differences must preserve packet grammar before a score response is meaningful",
            )
        )
        return _result(
            mutation_grain=grain,
            axis_label=axis,
            raw_byte_gradient_valid=False,
            operator_response_valid=False,
            ready_for_operator_probe=False,
            verdict="blocked_raw_archive_gradient",
            recommended_probe_grain="grammar_aware_operator",
            blockers=blockers,
            required_proofs=[
                *required,
                "packet_aware_mutation_operator",
                "repacked_archive_with_fresh_crc",
                "inflate_success_proof",
            ],
            rationale=rationale,
        )

    if grain == "zip_member_payload_byte" and payload_is_entropy_coded:
        blockers.append("zip_member_payload_byte_flip_is_not_local_in_entropy_coded_stream")
        rationale.append(
            "updating ZIP metadata can make the container valid while the inner codec semantics are still invalid"
        )
        return _result(
            mutation_grain=grain,
            axis_label=axis,
            raw_byte_gradient_valid=False,
            operator_response_valid=False,
            ready_for_operator_probe=False,
            verdict="blocked_entropy_stream_payload_gradient",
            recommended_probe_grain="logical_section_parameter",
            blockers=blockers,
            required_proofs=[
                *required,
                "logical_section_parser",
                "codec_valid_mutation_operator",
                "inflate_success_proof",
            ],
            rationale=rationale,
        )

    if grain not in OPERATOR_GRAINS and grain != "zip_member_payload_byte":
        blockers.append("unknown_mutation_grain")
        required.append("known_mutation_grain")

    if grain in OPERATOR_GRAINS:
        if not repacks_archive:
            blockers.append("archive_not_repacked_after_mutation")
            required.append("repacked_archive")
        if archive_is_zip and not updates_zip_headers:
            blockers.append("zip_headers_not_rebuilt")
            required.append("updated_zip_headers")
        if archive_is_zip and not updates_crc:
            blockers.append("zip_crc_not_rebuilt")
            required.append("updated_zip_crc")
        if not proves_inflate_success:
            blockers.append("inflate_success_not_proven")
            required.append("inflate_success_proof")
        rationale.extend(
            (
                "operator-grain responses are discrete score-response measurements, not raw byte derivatives",
                "the measured object should be an N_operator x 3 response matrix over valid packet mutations",
            )
        )

    ready = not blockers and grain in OPERATOR_GRAINS
    return _result(
        mutation_grain=grain,
        axis_label=axis,
        raw_byte_gradient_valid=False,
        operator_response_valid=ready,
        ready_for_operator_probe=ready,
        verdict="operator_response_probe_ready" if ready else "blocked_until_packet_proofs_land",
        recommended_probe_grain=grain if ready else "grammar_aware_operator",
        blockers=blockers,
        required_proofs=tuple(dict.fromkeys(required)),
        rationale=rationale,
    )


def _result(
    *,
    mutation_grain: str,
    axis_label: str | None,
    raw_byte_gradient_valid: bool,
    operator_response_valid: bool,
    ready_for_operator_probe: bool,
    verdict: str,
    recommended_probe_grain: str,
    blockers: list[str] | tuple[str, ...],
    required_proofs: list[str] | tuple[str, ...],
    rationale: list[str] | tuple[str, ...],
) -> MasterGradientFeasibility:
    return MasterGradientFeasibility(
        mutation_grain=mutation_grain,
        axis_label=axis_label,
        raw_byte_gradient_valid=raw_byte_gradient_valid,
        operator_response_valid=operator_response_valid,
        requires_valid_packet_mutations=True,
        ready_for_operator_probe=ready_for_operator_probe,
        ready_for_provider_dispatch=False,
        score_claim=False,
        promotion_eligible=False,
        rank_or_kill_eligible=False,
        ready_for_exact_eval_dispatch=False,
        evidence_grade="[planning; master-gradient feasibility audit]",
        verdict=verdict,
        recommended_probe_grain=recommended_probe_grain,
        blockers=tuple(dict.fromkeys(blockers)),
        required_proofs=tuple(dict.fromkeys(required_proofs)),
        rationale=tuple(rationale),
    )
