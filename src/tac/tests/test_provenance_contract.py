# SPDX-License-Identifier: MIT
"""Tests for tac.provenance.contract — canonical Provenance + ScoreClaim dataclasses.

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against"
+ "Subagent coherence-by-default": tests pin the contract invariants AND
the 5 phantom-score class anchors that motivated the package.
"""

from __future__ import annotations

import pytest

from tac.provenance.contract import (
    CANONICAL_HARDWARE_SUBSTRATES,
    CANONICAL_MEASUREMENT_AXES,
    NULL_NOT_A_SCORE_CLAIM,
    PROVENANCE_SCHEMA_VERSION,
    InvalidProvenanceError,
    Provenance,
    ProvenanceEvidenceGrade,
    ProvenanceKind,
    ScoreClaim,
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------


def test_schema_version_pinned():
    assert PROVENANCE_SCHEMA_VERSION == "provenance_v1_20260517"


def test_canonical_measurement_axes_includes_all_lane_tags():
    """Per CLAUDE.md FORBIDDEN PATTERNS: every score must carry an axis tag."""
    assert "[contest-CUDA]" in CANONICAL_MEASUREMENT_AXES
    assert "[contest-CPU]" in CANONICAL_MEASUREMENT_AXES
    assert "[macOS-CPU advisory]" in CANONICAL_MEASUREMENT_AXES
    assert "[macOS-MLX research-signal]" in CANONICAL_MEASUREMENT_AXES
    assert "[MPS-PROXY]" in CANONICAL_MEASUREMENT_AXES


def test_canonical_hardware_substrates_includes_modal_and_macos():
    assert "linux_x86_64_t4" in CANONICAL_HARDWARE_SUBSTRATES
    assert "linux_x86_64_modal_cpu" in CANONICAL_HARDWARE_SUBSTRATES
    assert "macos_arm64" in CANONICAL_HARDWARE_SUBSTRATES
    assert "macos_arm64_mlx" in CANONICAL_HARDWARE_SUBSTRATES


# -----------------------------------------------------------------------------
# Provenance: basic shape validation
# -----------------------------------------------------------------------------


def _valid_contest_archive_member_kwargs() -> dict:
    return {
        "artifact_kind": ProvenanceKind.CONTEST_ARCHIVE_MEMBER,
        "source_path": "submissions/a1/archive.zip:0.bin",
        "source_sha256": "a" * 64,
        "measurement_axis": "[contest-CUDA]",
        "hardware_substrate": "linux_x86_64_t4",
        "evidence_grade": ProvenanceEvidenceGrade.PROMOTABLE_EXACT_CONTEST_CUDA,
        "promotion_eligible": True,
        "score_claim_valid": True,
        "captured_at_utc": "2026-05-17T22:00:00Z",
        "canonical_helper_invocation": "tac.provenance.builders.build_provenance_for_archive_member",
        "contest_archive_zip_path": "submissions/a1/archive.zip",
        "contest_archive_member_name": "0.bin",
    }


def test_canonical_contest_archive_member_constructs():
    prov = Provenance(**_valid_contest_archive_member_kwargs())
    assert prov.promotion_eligible
    assert prov.score_claim_valid


def test_empty_source_path_rejected():
    kwargs = _valid_contest_archive_member_kwargs()
    kwargs["source_path"] = ""
    with pytest.raises(InvalidProvenanceError) as exc_info:
        Provenance(**kwargs)
    assert "source_path" in str(exc_info.value)


def test_invalid_sha256_rejected():
    kwargs = _valid_contest_archive_member_kwargs()
    kwargs["source_sha256"] = "not_a_sha"
    with pytest.raises(InvalidProvenanceError):
        Provenance(**kwargs)


def test_uppercase_sha256_rejected():
    """sha256 must be lowercase hex per canonical convention."""
    kwargs = _valid_contest_archive_member_kwargs()
    kwargs["source_sha256"] = "A" * 64
    with pytest.raises(InvalidProvenanceError):
        Provenance(**kwargs)


def test_short_sha256_rejected():
    kwargs = _valid_contest_archive_member_kwargs()
    kwargs["source_sha256"] = "a" * 63
    with pytest.raises(InvalidProvenanceError):
        Provenance(**kwargs)


def test_canonical_helper_invocation_non_dotted_rejected():
    kwargs = _valid_contest_archive_member_kwargs()
    kwargs["canonical_helper_invocation"] = "not-a-function-name"
    with pytest.raises(InvalidProvenanceError):
        Provenance(**kwargs)


# -----------------------------------------------------------------------------
# Provenance: kind-specific invariants
# -----------------------------------------------------------------------------


def test_contest_archive_member_requires_zip_path():
    kwargs = _valid_contest_archive_member_kwargs()
    kwargs["contest_archive_zip_path"] = ""
    with pytest.raises(InvalidProvenanceError):
        Provenance(**kwargs)


def test_contest_archive_member_requires_member_name():
    kwargs = _valid_contest_archive_member_kwargs()
    kwargs["contest_archive_member_name"] = ""
    with pytest.raises(InvalidProvenanceError):
        Provenance(**kwargs)


def test_research_sidecar_cannot_be_promotable():
    """Catalog #321 anchor: research sidecar cannot claim score."""
    with pytest.raises(InvalidProvenanceError) as exc_info:
        Provenance(
            artifact_kind=ProvenanceKind.RESEARCH_SIDECAR,
            source_path="experiments/results/pr101_state_dict/state_dict.pt",
            source_sha256="b" * 64,
            measurement_axis="[research-signal]",
            hardware_substrate="unknown",
            evidence_grade=ProvenanceEvidenceGrade.RESEARCH_ONLY,
            promotion_eligible=True,  # FORBIDDEN
            score_claim_valid=False,
            captured_at_utc="2026-05-17T22:00:00Z",
            canonical_helper_invocation="tac.provenance.builders.build_provenance_for_research_sidecar",
        )
    assert "RESEARCH_SIDECAR" in str(exc_info.value)


def test_research_sidecar_cannot_have_valid_score_claim():
    with pytest.raises(InvalidProvenanceError):
        Provenance(
            artifact_kind=ProvenanceKind.RESEARCH_SIDECAR,
            source_path="experiments/results/pr101_state_dict/state_dict.pt",
            source_sha256="b" * 64,
            measurement_axis="[research-signal]",
            hardware_substrate="unknown",
            evidence_grade=ProvenanceEvidenceGrade.RESEARCH_ONLY,
            promotion_eligible=False,
            score_claim_valid=True,  # FORBIDDEN
            captured_at_utc="2026-05-17T22:00:00Z",
            canonical_helper_invocation="tac.provenance.builders.build_provenance_for_research_sidecar",
        )


def test_aggregate_requires_non_empty_composed_from():
    with pytest.raises(InvalidProvenanceError) as exc_info:
        Provenance(
            artifact_kind=ProvenanceKind.AGGREGATE_OF_PROVENANCES,
            source_path="<aggregate>",
            source_sha256="c" * 64,
            measurement_axis="[empirical]",
            hardware_substrate="linux_x86_64_t4",
            evidence_grade=ProvenanceEvidenceGrade.RESEARCH_ONLY,
            promotion_eligible=False,
            score_claim_valid=False,
            captured_at_utc="2026-05-17T22:00:00Z",
            canonical_helper_invocation="tac.provenance.builders.build_provenance_aggregate",
            composed_from=(),
        )
    assert "AGGREGATE" in str(exc_info.value)


def test_aggregate_byte_identity_detector_catalog_823():
    """Catalog #823 anchor: 2+ composed_from sharing sha → must be INVALID_BYTE_IDENTITY_ARTIFACT."""
    part_a = Provenance(
        artifact_kind=ProvenanceKind.RESEARCH_SIDECAR,
        source_path="path_a",
        source_sha256="08f12d722dd33f9061deee72f49d782035597f78cd65ed1463a241ab430a7529",
        measurement_axis="[research-signal]",
        hardware_substrate="unknown",
        evidence_grade=ProvenanceEvidenceGrade.RESEARCH_ONLY,
        promotion_eligible=False,
        score_claim_valid=False,
        captured_at_utc="2026-05-17T22:00:00Z",
        canonical_helper_invocation="tac.tests.test",
        rejection_reason="test fixture",
    )
    part_b = Provenance(
        artifact_kind=ProvenanceKind.RESEARCH_SIDECAR,
        source_path="path_b",
        source_sha256="08f12d722dd33f9061deee72f49d782035597f78cd65ed1463a241ab430a7529",  # SAME
        measurement_axis="[research-signal]",
        hardware_substrate="unknown",
        evidence_grade=ProvenanceEvidenceGrade.RESEARCH_ONLY,
        promotion_eligible=False,
        score_claim_valid=False,
        captured_at_utc="2026-05-17T22:00:00Z",
        canonical_helper_invocation="tac.tests.test",
        rejection_reason="test fixture",
    )

    # Constructing AGGREGATE without INVALID_BYTE_IDENTITY_ARTIFACT grade
    # MUST raise because part_a + part_b share sha256
    with pytest.raises(InvalidProvenanceError) as exc_info:
        Provenance(
            artifact_kind=ProvenanceKind.AGGREGATE_OF_PROVENANCES,
            source_path="<aggregate>",
            source_sha256="d" * 64,
            measurement_axis="[empirical]",
            hardware_substrate="linux_x86_64_t4",
            evidence_grade=ProvenanceEvidenceGrade.RESEARCH_ONLY,
            promotion_eligible=False,
            score_claim_valid=False,
            captured_at_utc="2026-05-17T22:00:00Z",
            canonical_helper_invocation="tac.tests.test",
            composed_from=(part_a, part_b),
        )
    assert "byte-identical" in str(exc_info.value).lower()
    assert "Catalog #823" in str(exc_info.value)


def test_aggregate_byte_identity_with_invalid_grade_constructs():
    """Catalog #823: AGGREGATE with byte-identity OK if explicitly INVALID_BYTE_IDENTITY_ARTIFACT."""
    part_a = Provenance(
        artifact_kind=ProvenanceKind.RESEARCH_SIDECAR,
        source_path="path_a",
        source_sha256="0" * 64,
        measurement_axis="[research-signal]",
        hardware_substrate="unknown",
        evidence_grade=ProvenanceEvidenceGrade.RESEARCH_ONLY,
        promotion_eligible=False,
        score_claim_valid=False,
        captured_at_utc="2026-05-17T22:00:00Z",
        canonical_helper_invocation="tac.tests.test",
        rejection_reason="test fixture",
    )
    part_b = part_a  # same instance OK (test simplification)

    # OK if explicit INVALID sentinel
    prov = Provenance(
        artifact_kind=ProvenanceKind.AGGREGATE_OF_PROVENANCES,
        source_path="<aggregate>",
        source_sha256="d" * 64,
        measurement_axis="[empirical]",
        hardware_substrate="linux_x86_64_t4",
        evidence_grade=ProvenanceEvidenceGrade.INVALID_BYTE_IDENTITY_ARTIFACT,
        promotion_eligible=False,
        score_claim_valid=False,
        captured_at_utc="2026-05-17T22:00:00Z",
        canonical_helper_invocation="tac.tests.test",
        composed_from=(part_a, part_b),
        rejection_reason="byte-identity detected",
    )
    assert prov.evidence_grade == ProvenanceEvidenceGrade.INVALID_BYTE_IDENTITY_ARTIFACT


def test_advisory_non_promotable_cannot_be_promotable():
    """Catalog #192 anchor: macOS-CPU is NEVER 1:1 contest-compliant."""
    with pytest.raises(InvalidProvenanceError):
        Provenance(
            artifact_kind=ProvenanceKind.ADVISORY_NON_PROMOTABLE,
            source_path="x",
            source_sha256="e" * 64,
            measurement_axis="[macOS-CPU advisory]",
            hardware_substrate="macos_arm64",
            evidence_grade=ProvenanceEvidenceGrade.MACOS_CPU_ADVISORY,
            promotion_eligible=True,  # FORBIDDEN
            score_claim_valid=False,
            captured_at_utc="2026-05-17T22:00:00Z",
            canonical_helper_invocation="tac.tests.test",
        )


def test_predicted_cannot_be_promotable():
    with pytest.raises(InvalidProvenanceError):
        Provenance(
            artifact_kind=ProvenanceKind.PREDICTED_FROM_MODEL,
            source_path="<predictor:m1>",
            source_sha256="f" * 64,
            measurement_axis="[predicted]",
            hardware_substrate="unknown",
            evidence_grade=ProvenanceEvidenceGrade.PREDICTED,
            promotion_eligible=True,  # FORBIDDEN
            score_claim_valid=False,
            captured_at_utc="2026-05-17T22:00:00Z",
            canonical_helper_invocation="tac.tests.test",
        )


def test_archive_seed_procedural_generation_is_non_promotable_supporting_provenance():
    prov = Provenance(
        artifact_kind=ProvenanceKind.PROCEDURAL_GENERATION_FROM_ARCHIVE_SEED,
        source_path="submissions/hash_seed/archive.zip:seed.bin",
        source_sha256="1" * 64,
        measurement_axis="[research-signal]",
        hardware_substrate="unknown",
        evidence_grade=ProvenanceEvidenceGrade.RESEARCH_ONLY,
        promotion_eligible=False,
        score_claim_valid=False,
        captured_at_utc="2026-05-18T16:00:00Z",
        canonical_helper_invocation=("tac.provenance.builders.build_provenance_for_archive_seed_procedural_generation"),
        rejection_reason="seed bytes are charged inside archive.zip",
    )
    assert prov.artifact_kind == ProvenanceKind.PROCEDURAL_GENERATION_FROM_ARCHIVE_SEED


def test_archive_seed_procedural_generation_requires_rationale():
    with pytest.raises(InvalidProvenanceError, match="requires non-empty rationale"):
        Provenance(
            artifact_kind=ProvenanceKind.PROCEDURAL_GENERATION_FROM_ARCHIVE_SEED,
            source_path="submissions/hash_seed/archive.zip:seed.bin",
            source_sha256="1" * 64,
            measurement_axis="[research-signal]",
            hardware_substrate="unknown",
            evidence_grade=ProvenanceEvidenceGrade.RESEARCH_ONLY,
            promotion_eligible=False,
            score_claim_valid=False,
            captured_at_utc="2026-05-18T16:00:00Z",
            canonical_helper_invocation=(
                "tac.provenance.builders.build_provenance_for_archive_seed_procedural_generation"
            ),
        )


def test_weight_derived_codebook_cannot_have_valid_score_claim():
    with pytest.raises(InvalidProvenanceError, match="score_claim_valid"):
        Provenance(
            artifact_kind=ProvenanceKind.WEIGHT_DERIVED_CODEBOOK,
            source_path="submissions/weight_codebook/archive.zip:renderer.bin",
            source_sha256="2" * 64,
            measurement_axis="[research-signal]",
            hardware_substrate="unknown",
            evidence_grade=ProvenanceEvidenceGrade.RESEARCH_ONLY,
            promotion_eligible=False,
            score_claim_valid=True,
            captured_at_utc="2026-05-18T16:00:00Z",
            canonical_helper_invocation=("tac.provenance.builders.build_provenance_for_weight_derived_codebook"),
            rejection_reason="codebook derived from shipped renderer weights",
        )


def test_forbidden_out_of_archive_payload_requires_rejection_reason():
    with pytest.raises(InvalidProvenanceError, match="FORBIDDEN_OUT_OF_ARCHIVE"):
        Provenance(
            artifact_kind=ProvenanceKind.FORBIDDEN_OUT_OF_ARCHIVE_PAYLOAD,
            source_path="/external/payload.bin",
            source_sha256="3" * 64,
            measurement_axis="[research-signal]",
            hardware_substrate="unknown",
            evidence_grade=ProvenanceEvidenceGrade.RESEARCH_ONLY,
            promotion_eligible=False,
            score_claim_valid=False,
            captured_at_utc="2026-05-18T16:00:00Z",
            canonical_helper_invocation=(
                "tac.provenance.builders.build_provenance_for_forbidden_out_of_archive_payload"
            ),
        )


# -----------------------------------------------------------------------------
# Provenance: grade × axis × hardware cross-checks
# -----------------------------------------------------------------------------


def test_cuda_promotable_requires_cuda_axis():
    kwargs = _valid_contest_archive_member_kwargs()
    kwargs["measurement_axis"] = "[contest-CPU]"  # wrong axis
    with pytest.raises(InvalidProvenanceError) as exc_info:
        Provenance(**kwargs)
    assert "contest-CUDA" in str(exc_info.value)


def test_cuda_promotable_requires_cuda_hardware():
    kwargs = _valid_contest_archive_member_kwargs()
    kwargs["hardware_substrate"] = "macos_arm64"  # wrong hardware
    with pytest.raises(InvalidProvenanceError) as exc_info:
        Provenance(**kwargs)
    assert "CUDA hardware" in str(exc_info.value)


def test_cpu_promotable_requires_cpu_hardware():
    kwargs = _valid_contest_archive_member_kwargs()
    kwargs["evidence_grade"] = ProvenanceEvidenceGrade.PROMOTABLE_EXACT_CONTEST_CPU
    kwargs["measurement_axis"] = "[contest-CPU]"
    # leave hardware as linux_x86_64_t4 → invalid
    with pytest.raises(InvalidProvenanceError) as exc_info:
        Provenance(**kwargs)
    assert "linux_x86_64_cpu" in str(exc_info.value)


def test_macos_advisory_requires_macos_axis():
    with pytest.raises(InvalidProvenanceError):
        Provenance(
            artifact_kind=ProvenanceKind.ADVISORY_NON_PROMOTABLE,
            source_path="x",
            source_sha256="0" * 64,
            measurement_axis="[contest-CPU]",  # wrong; should be [macOS-CPU advisory]
            hardware_substrate="macos_arm64",
            evidence_grade=ProvenanceEvidenceGrade.MACOS_CPU_ADVISORY,
            promotion_eligible=False,
            score_claim_valid=False,
            captured_at_utc="2026-05-17T22:00:00Z",
            canonical_helper_invocation="tac.tests.test",
        )


def test_macos_mlx_research_signal_constructs_non_promotable():
    prov = Provenance(
        artifact_kind=ProvenanceKind.ADVISORY_NON_PROMOTABLE,
        source_path="experiments/results/mlx_probe/replay.json",
        source_sha256="4" * 64,
        measurement_axis="[macOS-MLX research-signal]",
        hardware_substrate="macos_arm64_mlx",
        evidence_grade=ProvenanceEvidenceGrade.MACOS_MLX_RESEARCH_SIGNAL,
        promotion_eligible=False,
        score_claim_valid=False,
        captured_at_utc="2026-05-30T22:00:00Z",
        canonical_helper_invocation="tac.tests.test",
    )
    assert not prov.promotion_eligible
    assert not prov.score_claim_valid


def test_macos_mlx_research_signal_requires_mlx_axis():
    with pytest.raises(InvalidProvenanceError, match="macOS-MLX"):
        Provenance(
            artifact_kind=ProvenanceKind.ADVISORY_NON_PROMOTABLE,
            source_path="experiments/results/mlx_probe/replay.json",
            source_sha256="4" * 64,
            measurement_axis="[macOS-CPU advisory]",
            hardware_substrate="macos_arm64_mlx",
            evidence_grade=ProvenanceEvidenceGrade.MACOS_MLX_RESEARCH_SIGNAL,
            promotion_eligible=False,
            score_claim_valid=False,
            captured_at_utc="2026-05-30T22:00:00Z",
            canonical_helper_invocation="tac.tests.test",
        )


def test_macos_mlx_research_signal_requires_mlx_hardware():
    with pytest.raises(InvalidProvenanceError, match="macos_arm64_mlx"):
        Provenance(
            artifact_kind=ProvenanceKind.ADVISORY_NON_PROMOTABLE,
            source_path="experiments/results/mlx_probe/replay.json",
            source_sha256="4" * 64,
            measurement_axis="[macOS-MLX research-signal]",
            hardware_substrate="macos_arm64",
            evidence_grade=ProvenanceEvidenceGrade.MACOS_MLX_RESEARCH_SIGNAL,
            promotion_eligible=False,
            score_claim_valid=False,
            captured_at_utc="2026-05-30T22:00:00Z",
            canonical_helper_invocation="tac.tests.test",
        )


def test_mps_proxy_requires_macos_arm64_hardware():
    """Per CLAUDE.md MPS auth eval is NOISE."""
    with pytest.raises(InvalidProvenanceError):
        Provenance(
            artifact_kind=ProvenanceKind.ADVISORY_NON_PROMOTABLE,
            source_path="x",
            source_sha256="0" * 64,
            measurement_axis="[MPS-PROXY]",
            hardware_substrate="linux_x86_64_cpu",  # wrong hardware
            evidence_grade=ProvenanceEvidenceGrade.MPS_PROXY,
            promotion_eligible=False,
            score_claim_valid=False,
            captured_at_utc="2026-05-17T22:00:00Z",
            canonical_helper_invocation="tac.tests.test",
        )


def test_invalid_byte_identity_requires_rejection_reason():
    """Catalog #823 anchor structural requirement."""
    with pytest.raises(InvalidProvenanceError):
        Provenance(
            artifact_kind=ProvenanceKind.RESEARCH_SIDECAR,
            source_path="x",
            source_sha256="0" * 64,
            measurement_axis="[research-signal]",
            hardware_substrate="unknown",
            evidence_grade=ProvenanceEvidenceGrade.INVALID_BYTE_IDENTITY_ARTIFACT,
            promotion_eligible=False,
            score_claim_valid=False,
            captured_at_utc="2026-05-17T22:00:00Z",
            canonical_helper_invocation="tac.tests.test",
            rejection_reason="",  # EMPTY = FORBIDDEN
        )


# -----------------------------------------------------------------------------
# ScoreClaim invariants
# -----------------------------------------------------------------------------


def test_score_claim_auto_derives_contest_compliant():
    prov = Provenance(**_valid_contest_archive_member_kwargs())
    claim = ScoreClaim(score_value=0.192, provenance=prov, rationale="test")
    assert claim.contest_compliant


def test_score_claim_non_promotable_provenance_forces_non_compliant():
    """Even if caller sets contest_compliant=True, derived value wins."""
    prov = Provenance(
        artifact_kind=ProvenanceKind.RESEARCH_SIDECAR,
        source_path="x",
        source_sha256="0" * 64,
        measurement_axis="[research-signal]",
        hardware_substrate="unknown",
        evidence_grade=ProvenanceEvidenceGrade.RESEARCH_ONLY,
        promotion_eligible=False,
        score_claim_valid=False,
        captured_at_utc="2026-05-17T22:00:00Z",
        canonical_helper_invocation="tac.tests.test",
        rejection_reason="test",
    )
    # Caller sets contest_compliant=True → REJECTED
    with pytest.raises(InvalidProvenanceError) as exc_info:
        ScoreClaim(
            score_value=0.477,  # the pr101_state_dict phantom score
            provenance=prov,
            contest_compliant=True,
        )
    assert "promotable" in str(exc_info.value).lower() or "score_claim_valid" in str(exc_info.value)


def test_score_claim_requires_provenance_instance():
    with pytest.raises(InvalidProvenanceError):
        ScoreClaim(
            score_value=0.5,
            provenance={"fake": "dict"},  # not a Provenance
        )


def test_score_claim_default_contest_compliant_false():
    prov = Provenance(
        artifact_kind=ProvenanceKind.RESEARCH_SIDECAR,
        source_path="x",
        source_sha256="0" * 64,
        measurement_axis="[research-signal]",
        hardware_substrate="unknown",
        evidence_grade=ProvenanceEvidenceGrade.RESEARCH_ONLY,
        promotion_eligible=False,
        score_claim_valid=False,
        captured_at_utc="2026-05-17T22:00:00Z",
        canonical_helper_invocation="tac.tests.test",
        rejection_reason="test",
    )
    claim = ScoreClaim(score_value=0.5, provenance=prov)
    assert not claim.contest_compliant


# -----------------------------------------------------------------------------
# Sentinel
# -----------------------------------------------------------------------------


def test_null_sentinel_is_research_sidecar():
    assert NULL_NOT_A_SCORE_CLAIM.artifact_kind == ProvenanceKind.RESEARCH_SIDECAR
    assert not NULL_NOT_A_SCORE_CLAIM.promotion_eligible
    assert not NULL_NOT_A_SCORE_CLAIM.score_claim_valid
    assert "sentinel" in NULL_NOT_A_SCORE_CLAIM.source_path


def test_null_sentinel_identity():
    """Sentinel is a module-level constant; identity should hold."""
    from tac.provenance import NULL_NOT_A_SCORE_CLAIM as imported_again

    assert NULL_NOT_A_SCORE_CLAIM is imported_again


# -----------------------------------------------------------------------------
# Frozen invariant
# -----------------------------------------------------------------------------


def test_provenance_is_frozen():
    """Provenance is frozen per HISTORICAL_PROVENANCE Catalog #110/#113."""
    prov = Provenance(**_valid_contest_archive_member_kwargs())
    with pytest.raises((AttributeError, Exception)):
        prov.source_sha256 = "different"
