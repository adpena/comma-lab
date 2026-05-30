# SPDX-License-Identifier: MIT
"""tac.provenance — canonical Provenance contract for all score-claiming surfaces.

Per operator NON-NEGOTIABLE 2026-05-17 verbatim: *"We need to fix the
provenance issue for all and fix it permanently and canonically and make
it easy"*.

The package extincts 5 phantom-score class instances in one session at the
META layer (a single canonical contract that subsumes 5 sister gates rather
than per-instance fixes). See ``tac.provenance.contract`` docstring for the
full enumeration.

Quick start:

    from tac.provenance import (
        Provenance,
        ScoreClaim,
        ProvenanceEvidenceGrade,
        ProvenanceKind,
        build_provenance_for_archive_member,
        build_provenance_for_research_sidecar,
        validate_provenance,
        validate_score_claim,
        audit_score_claim_dict,
    )

    # Promotable archive-member score
    prov = build_provenance_for_archive_member(
        archive_zip_path="submissions/a1/archive.zip",
        member_name="0.bin",
        measurement_axis="[contest-CUDA]",
        hardware_substrate="linux_x86_64_t4",
        evidence_grade=ProvenanceEvidenceGrade.PROMOTABLE_EXACT_CONTEST_CUDA,
    )
    claim = ScoreClaim(score_value=0.192, provenance=prov, rationale="PR101 GOLD")
    assert claim.contest_compliant  # auto-derived from prov

    # Non-promotable research sidecar (Catalog #321 anchor)
    sidecar_prov = build_provenance_for_research_sidecar(
        sidecar_path="experiments/results/pr101_state_dict/state_dict.pt",
        reactivation_criteria="awaiting archive member byte verification",
    )
    # sidecar_prov.score_claim_valid is False by construction.

See ``docs/provenance_canonical_usage.md`` for the full developer guide.

Cross-references:
  * Catalog #323 STRICT preflight gate (canonical umbrella).
  * Sister gates #287 (empirical-claim-tag) / #249 (filename-vs-content) /
    #319 (autopilot venn reweight) / #321 (research-sidecar phantom score)
    / #185 (live-count drift) — all preserved as defense-in-depth.
  * Audit tool: ``tools/audit_provenance_compliance.py``.
"""

from __future__ import annotations

from tac.provenance.adapters import (
    contest_result_to_provenance,
    cost_band_anchor_to_provenance,
    council_record_to_provenance,
    deliverability_proof_to_provenance,
    master_gradient_plan_to_provenance,
    modal_call_id_ledger_event_to_provenance,
    substrate_composition_row_to_provenance,
    wyner_ziv_layer_result_to_provenance,
)
from tac.provenance.builders import (
    build_provenance_aggregate,
    build_provenance_for_archive_member,
    build_provenance_for_archive_seed_procedural_generation,
    build_provenance_for_forbidden_out_of_archive_payload,
    build_provenance_for_macos_cpu_advisory,
    build_provenance_for_macos_mlx_research_signal,
    build_provenance_for_mps_proxy,
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
    build_provenance_for_weight_derived_codebook,
    build_provenance_invalid_byte_identity_artifact,
    register_forbidden_out_of_archive_payload_probe_outcome,
    requires_canonical_provenance,
)
from tac.provenance.contract import (
    CANONICAL_HARDWARE_SUBSTRATES,
    CANONICAL_MEASUREMENT_AXES,
    NULL_NOT_A_SCORE_CLAIM,
    PROVENANCE_SCHEMA_VERSION,
    InvalidProvenanceError,
    MissingProvenanceError,
    Provenance,
    ProvenanceEvidenceGrade,
    ProvenanceKind,
    ScoreClaim,
)
from tac.provenance.validator import (
    DEFAULT_PROVENANCE_STALE_DAYS,
    audit_score_claim_dict,
    provenance_to_dict,
    validate_provenance,
    validate_score_claim,
)

__all__ = [
    # Contract
    "PROVENANCE_SCHEMA_VERSION",
    "CANONICAL_MEASUREMENT_AXES",
    "CANONICAL_HARDWARE_SUBSTRATES",
    "ProvenanceKind",
    "ProvenanceEvidenceGrade",
    "MissingProvenanceError",
    "InvalidProvenanceError",
    "Provenance",
    "ScoreClaim",
    "NULL_NOT_A_SCORE_CLAIM",
    # Builders
    "build_provenance_for_archive_member",
    "build_provenance_for_research_sidecar",
    "build_provenance_for_predicted",
    "build_provenance_for_macos_cpu_advisory",
    "build_provenance_for_macos_mlx_research_signal",
    "build_provenance_for_mps_proxy",
    "build_provenance_aggregate",
    "build_provenance_invalid_byte_identity_artifact",
    "build_provenance_for_archive_seed_procedural_generation",
    "build_provenance_for_weight_derived_codebook",
    "build_provenance_for_forbidden_out_of_archive_payload",
    "register_forbidden_out_of_archive_payload_probe_outcome",
    "requires_canonical_provenance",
    # Validator
    "DEFAULT_PROVENANCE_STALE_DAYS",
    "validate_provenance",
    "validate_score_claim",
    "audit_score_claim_dict",
    "provenance_to_dict",
    # Adapters
    "contest_result_to_provenance",
    "cost_band_anchor_to_provenance",
    "council_record_to_provenance",
    "deliverability_proof_to_provenance",
    "master_gradient_plan_to_provenance",
    "modal_call_id_ledger_event_to_provenance",
    "substrate_composition_row_to_provenance",
    "wyner_ziv_layer_result_to_provenance",
]
