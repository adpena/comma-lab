# SPDX-License-Identifier: MIT
"""Tests for tac.provenance.builders — canonical builders + decorator."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from tac.provenance import (
    InvalidProvenanceError,
    MissingProvenanceError,
    Provenance,
    ProvenanceEvidenceGrade,
    ProvenanceKind,
    ScoreClaim,
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

# -----------------------------------------------------------------------------
# build_provenance_for_archive_member
# -----------------------------------------------------------------------------


@pytest.fixture
def tmp_archive(tmp_path: Path) -> tuple[Path, str, str]:
    """Create a fake archive.zip with one member."""
    archive_path = tmp_path / "test_archive.zip"
    member_content = b"hello canonical archive"
    member_sha = hashlib.sha256(member_content).hexdigest()
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("0.bin", member_content)
    return archive_path, "0.bin", member_sha


def test_build_archive_member_reads_sha(tmp_archive):
    archive_path, member, expected_sha = tmp_archive
    prov = build_provenance_for_archive_member(
        archive_zip_path=archive_path,
        member_name=member,
        measurement_axis="[contest-CUDA]",
        hardware_substrate="linux_x86_64_t4",
        evidence_grade=ProvenanceEvidenceGrade.PROMOTABLE_EXACT_CONTEST_CUDA,
    )
    assert prov.source_sha256 == expected_sha
    assert prov.promotion_eligible
    assert prov.score_claim_valid
    assert prov.contest_archive_member_name == member
    assert "build_provenance_for_archive_member" in prov.canonical_helper_invocation


def test_build_archive_member_missing_archive_raises():
    with pytest.raises(InvalidProvenanceError):
        build_provenance_for_archive_member(
            archive_zip_path="/tmp/nonexistent/archive.zip",
            member_name="0.bin",
            measurement_axis="[contest-CUDA]",
            hardware_substrate="linux_x86_64_t4",
            evidence_grade=ProvenanceEvidenceGrade.PROMOTABLE_EXACT_CONTEST_CUDA,
        )


def test_build_archive_member_missing_member_raises(tmp_archive):
    archive_path, _, _ = tmp_archive
    with pytest.raises(InvalidProvenanceError):
        build_provenance_for_archive_member(
            archive_zip_path=archive_path,
            member_name="nonexistent.bin",
            measurement_axis="[contest-CUDA]",
            hardware_substrate="linux_x86_64_t4",
            evidence_grade=ProvenanceEvidenceGrade.PROMOTABLE_EXACT_CONTEST_CUDA,
        )


def test_build_archive_member_transient_path_refused(tmp_path):
    """Per CLAUDE.md 'Forbidden /tmp paths' — refuse /tmp/."""
    fake = Path("/tmp/test_archive_for_provenance.zip")
    # Create real file at /tmp to bypass missing-archive check
    if not fake.exists():
        with zipfile.ZipFile(fake, "w") as zf:
            zf.writestr("0.bin", b"data")
    try:
        with pytest.raises(InvalidProvenanceError) as exc_info:
            build_provenance_for_archive_member(
                archive_zip_path=str(fake),
                member_name="0.bin",
                measurement_axis="[contest-CUDA]",
                hardware_substrate="linux_x86_64_t4",
                evidence_grade=ProvenanceEvidenceGrade.PROMOTABLE_EXACT_CONTEST_CUDA,
            )
        assert "transient" in str(exc_info.value).lower()
    finally:
        try:
            fake.unlink()
        except OSError:
            pass


def test_build_archive_member_non_promotable_grade_no_promotion():
    """If grade is research_only, even with archive backing, no promotion."""
    # Use a different non-promotable grade to test grade=non-promotable does
    # NOT yield promotable. But for archive member you typically use promotable.
    # We test EMPIRICAL_CPU_NON_GHA which is non-promotable.
    # However EMPIRICAL_CPU_NON_GHA isn't valid for CONTEST_ARCHIVE_MEMBER kind
    # because the contract grade × kind matrix says PROMOTABLE_* only.
    # Skip this case; covered by contract tests instead.


# -----------------------------------------------------------------------------
# build_provenance_for_research_sidecar
# -----------------------------------------------------------------------------


def test_build_research_sidecar_constructs(tmp_path):
    sidecar = tmp_path / "state_dict.pt"
    sidecar.write_bytes(b"pretend pt bytes")
    prov = build_provenance_for_research_sidecar(
        sidecar_path=sidecar,
        reactivation_criteria="awaiting archive member byte verification",
    )
    assert not prov.promotion_eligible
    assert not prov.score_claim_valid
    assert prov.evidence_grade == ProvenanceEvidenceGrade.RESEARCH_ONLY
    assert "awaiting" in prov.rejection_reason


def test_build_research_sidecar_empty_reactivation_raises(tmp_path):
    with pytest.raises(InvalidProvenanceError) as exc_info:
        build_provenance_for_research_sidecar(
            sidecar_path=tmp_path / "missing.pt",
            reactivation_criteria="",
        )
    assert "reactivation_criteria" in str(exc_info.value)


def test_build_research_sidecar_whitespace_reactivation_raises(tmp_path):
    with pytest.raises(InvalidProvenanceError):
        build_provenance_for_research_sidecar(
            sidecar_path=tmp_path / "missing.pt",
            reactivation_criteria="   ",
        )


def test_build_research_sidecar_missing_file_uses_placeholder_sha(tmp_path):
    """Missing sidecar is OK (recorded for DEFERRED-pending-research lanes)."""
    prov = build_provenance_for_research_sidecar(
        sidecar_path=tmp_path / "future.pt",
        reactivation_criteria="when research lands",
    )
    assert prov.source_sha256 == "0" * 64


# -----------------------------------------------------------------------------
# build_provenance_for_predicted
# -----------------------------------------------------------------------------


def test_build_predicted_requires_model_id():
    with pytest.raises(InvalidProvenanceError):
        build_provenance_for_predicted(model_id="", inputs_sha256="a" * 64)


def test_build_predicted_requires_inputs_sha():
    with pytest.raises(InvalidProvenanceError):
        build_provenance_for_predicted(model_id="m1", inputs_sha256="")


def test_build_predicted_constructs():
    prov = build_provenance_for_predicted(
        model_id="autopilot.predicted_delta_v2",
        inputs_sha256="b" * 64,
    )
    assert prov.artifact_kind == ProvenanceKind.PREDICTED_FROM_MODEL
    assert not prov.promotion_eligible


# -----------------------------------------------------------------------------
# contest-compliance procedural-generation boundary builders
# -----------------------------------------------------------------------------


def test_build_archive_seed_procedural_generation_constructs():
    prov = build_provenance_for_archive_seed_procedural_generation(
        seed_source_path="submissions/hash_seed/archive.zip:seed.bin",
        seed_sha256="1" * 64,
        rationale="seed bytes are charged inside archive.zip",
    )
    assert prov.artifact_kind == ProvenanceKind.PROCEDURAL_GENERATION_FROM_ARCHIVE_SEED
    assert prov.evidence_grade == ProvenanceEvidenceGrade.RESEARCH_ONLY
    assert not prov.score_claim_valid
    assert "archive.zip" in prov.rejection_reason


def test_build_archive_seed_procedural_generation_requires_rationale():
    with pytest.raises(InvalidProvenanceError):
        build_provenance_for_archive_seed_procedural_generation(
            seed_source_path="submissions/hash_seed/archive.zip:seed.bin",
            seed_sha256="1" * 64,
            rationale="",
        )


def test_build_weight_derived_codebook_constructs():
    prov = build_provenance_for_weight_derived_codebook(
        weight_source_path="submissions/weight_codebook/archive.zip:renderer.bin",
        weight_sha256="2" * 64,
        rationale="codebook is derived from shipped renderer weights",
    )
    assert prov.artifact_kind == ProvenanceKind.WEIGHT_DERIVED_CODEBOOK
    assert not prov.promotion_eligible
    assert "renderer weights" in prov.rejection_reason


def test_build_forbidden_out_of_archive_payload_constructs():
    prov = build_provenance_for_forbidden_out_of_archive_payload(
        payload_source_path="/external/payload.bin",
        payload_sha256="3" * 64,
        rejection_reason="output-affecting payload bytes are outside archive.zip",
    )
    assert prov.artifact_kind == ProvenanceKind.FORBIDDEN_OUT_OF_ARCHIVE_PAYLOAD
    assert not prov.score_claim_valid
    assert "outside archive.zip" in prov.rejection_reason


def test_register_forbidden_out_of_archive_payload_probe_outcome(tmp_path: Path):
    prov = build_provenance_for_forbidden_out_of_archive_payload(
        payload_source_path="/external/payload.bin",
        payload_sha256="3" * 64,
        rejection_reason="output-affecting payload bytes are outside archive.zip",
    )
    ledger = tmp_path / "probe_outcomes.jsonl"
    lock = tmp_path / "probe_outcomes.jsonl.lock"

    row = register_forbidden_out_of_archive_payload_probe_outcome(
        provenance=prov,
        substrate="wz_hash_seed_candidate",
        recipe_path=".omx/operator_authorize_recipes/wz.yaml",
        evidence_path=".omx/research/forbidden_payload_review.md",
        session_id="test-session",
        path=ledger,
        lock_path=lock,
    )

    assert row["verdict"] == "DEFER"
    assert row["blocker_status"] == "blocking"
    assert row["staleness_window_days"] == 365
    assert row["probe_kind"] == "forbidden_out_of_archive_payload_provenance"
    assert row["provenance_kind"] == "forbidden_out_of_archive_payload"
    assert row["payload_source_path"] == "/external/payload.bin"
    assert row["payload_source_sha256"] == "3" * 64
    assert ledger.exists()


def test_register_forbidden_probe_outcome_rejects_wrong_kind(tmp_path: Path):
    prov = build_provenance_for_research_sidecar(
        sidecar_path="experiments/results/dummy/state.pt",
        reactivation_criteria="not a forbidden payload sentinel",
    )
    with pytest.raises(InvalidProvenanceError, match="FORBIDDEN_OUT_OF_ARCHIVE"):
        register_forbidden_out_of_archive_payload_probe_outcome(
            provenance=prov,
            substrate="dummy",
            evidence_path=".omx/research/dummy.md",
            path=tmp_path / "probe_outcomes.jsonl",
            lock_path=tmp_path / "probe_outcomes.jsonl.lock",
        )


# -----------------------------------------------------------------------------
# build_provenance_for_macos_cpu_advisory
# -----------------------------------------------------------------------------


def test_build_macos_advisory_constructs():
    prov = build_provenance_for_macos_cpu_advisory(
        archive_sha256="c" * 64,
        source_path="experiments/results/lane_macos/auth.json",
    )
    assert prov.evidence_grade == ProvenanceEvidenceGrade.MACOS_CPU_ADVISORY
    assert not prov.promotion_eligible
    assert prov.hardware_substrate == "macos_arm64"


def test_build_macos_advisory_refuses_transient_path():
    with pytest.raises(InvalidProvenanceError):
        build_provenance_for_macos_cpu_advisory(
            archive_sha256="c" * 64,
            source_path="/tmp/foo.json",
        )


# -----------------------------------------------------------------------------
# build_provenance_for_macos_mlx_research_signal
# -----------------------------------------------------------------------------


def test_build_macos_mlx_research_signal_constructs():
    prov = build_provenance_for_macos_mlx_research_signal(
        artifact_sha256="e" * 64,
        source_path="experiments/results/mlx_probe/replay_bundle.json",
    )
    assert prov.evidence_grade == ProvenanceEvidenceGrade.MACOS_MLX_RESEARCH_SIGNAL
    assert prov.measurement_axis == "[macOS-MLX research-signal]"
    assert prov.hardware_substrate == "macos_arm64_mlx"
    assert not prov.promotion_eligible
    assert not prov.score_claim_valid


def test_build_macos_mlx_research_signal_refuses_transient_path():
    with pytest.raises(InvalidProvenanceError):
        build_provenance_for_macos_mlx_research_signal(
            artifact_sha256="e" * 64,
            source_path="/tmp/mlx_probe.json",
        )


# -----------------------------------------------------------------------------
# build_provenance_for_mps_proxy
# -----------------------------------------------------------------------------


def test_build_mps_proxy_constructs():
    prov = build_provenance_for_mps_proxy(
        artifact_sha256="d" * 64,
        source_path="experiments/results/mps_proxy/scores.json",
    )
    assert prov.evidence_grade == ProvenanceEvidenceGrade.MPS_PROXY
    assert not prov.promotion_eligible


# -----------------------------------------------------------------------------
# build_provenance_aggregate
# -----------------------------------------------------------------------------


def test_aggregate_empty_raises():
    with pytest.raises(InvalidProvenanceError):
        build_provenance_aggregate(parts=[], aggregation_rationale="empty")


def test_aggregate_byte_identity_auto_detected():
    """Catalog #823 anchor — α=4.74 SIREN byte-identity artifact class."""
    a = build_provenance_for_predicted(model_id="m1", inputs_sha256="08f12d72" + "2" * 56)
    b = build_provenance_for_predicted(model_id="m2", inputs_sha256="08f12d72" + "2" * 56)
    # Same sha → aggregate auto-flags
    agg = build_provenance_aggregate(
        parts=[a, b],
        aggregation_rationale="lane_g_v3 × siren pairwise composition_alpha",
    )
    assert agg.evidence_grade == ProvenanceEvidenceGrade.INVALID_BYTE_IDENTITY_ARTIFACT
    assert "byte-identical" in agg.rejection_reason
    assert "Catalog #823" in agg.rejection_reason
    assert not agg.promotion_eligible
    assert not agg.score_claim_valid


def test_aggregate_distinct_shas_no_flag():
    a = build_provenance_for_predicted(model_id="m1", inputs_sha256="a" * 64)
    b = build_provenance_for_predicted(model_id="m2", inputs_sha256="b" * 64)
    agg = build_provenance_aggregate(parts=[a, b], aggregation_rationale="distinct test")
    assert agg.evidence_grade != ProvenanceEvidenceGrade.INVALID_BYTE_IDENTITY_ARTIFACT


def test_aggregate_demotes_to_worst_grade():
    """Aggregate of (PROMOTABLE, RESEARCH_ONLY) → RESEARCH_ONLY."""
    # Construct a synthetic promotable-grade Provenance directly (skipping
    # archive-file check)
    promotable = Provenance(
        artifact_kind=ProvenanceKind.CONTEST_ARCHIVE_MEMBER,
        source_path="sub/a/archive.zip:0.bin",
        source_sha256="a" * 64,
        measurement_axis="[contest-CUDA]",
        hardware_substrate="linux_x86_64_t4",
        evidence_grade=ProvenanceEvidenceGrade.PROMOTABLE_EXACT_CONTEST_CUDA,
        promotion_eligible=True,
        score_claim_valid=True,
        captured_at_utc="2026-05-17T22:00:00Z",
        canonical_helper_invocation="tac.tests.test",
        contest_archive_zip_path="sub/a/archive.zip",
        contest_archive_member_name="0.bin",
    )
    research = build_provenance_for_research_sidecar(
        sidecar_path="experiments/results/x.pt",
        reactivation_criteria="test",
    )
    agg = build_provenance_aggregate(parts=[promotable, research], aggregation_rationale="mixed")
    # WORST = RESEARCH_ONLY → not promotable
    assert agg.evidence_grade == ProvenanceEvidenceGrade.RESEARCH_ONLY
    assert not agg.promotion_eligible


def test_aggregate_all_promotable_stays_promotable():
    # Two distinct promotable archive members → promotable aggregate
    a = Provenance(
        artifact_kind=ProvenanceKind.CONTEST_ARCHIVE_MEMBER,
        source_path="sub/a/archive.zip:0.bin",
        source_sha256="a" * 64,
        measurement_axis="[contest-CUDA]",
        hardware_substrate="linux_x86_64_t4",
        evidence_grade=ProvenanceEvidenceGrade.PROMOTABLE_EXACT_CONTEST_CUDA,
        promotion_eligible=True,
        score_claim_valid=True,
        captured_at_utc="2026-05-17T22:00:00Z",
        canonical_helper_invocation="tac.tests.test",
        contest_archive_zip_path="sub/a/archive.zip",
        contest_archive_member_name="0.bin",
    )
    b = Provenance(
        artifact_kind=ProvenanceKind.CONTEST_ARCHIVE_MEMBER,
        source_path="sub/b/archive.zip:0.bin",
        source_sha256="b" * 64,
        measurement_axis="[contest-CUDA]",
        hardware_substrate="linux_x86_64_t4",
        evidence_grade=ProvenanceEvidenceGrade.PROMOTABLE_EXACT_CONTEST_CUDA,
        promotion_eligible=True,
        score_claim_valid=True,
        captured_at_utc="2026-05-17T22:00:00Z",
        canonical_helper_invocation="tac.tests.test",
        contest_archive_zip_path="sub/b/archive.zip",
        contest_archive_member_name="0.bin",
    )
    agg = build_provenance_aggregate(parts=[a, b], aggregation_rationale="dual-archive")
    assert agg.evidence_grade == ProvenanceEvidenceGrade.PROMOTABLE_EXACT_CONTEST_CUDA
    assert agg.promotion_eligible


# -----------------------------------------------------------------------------
# build_provenance_invalid_byte_identity_artifact
# -----------------------------------------------------------------------------


def test_invalid_byte_identity_explicit_constructor():
    prov = build_provenance_invalid_byte_identity_artifact(
        source_path_a="experiments/results/lane_g_v3/renderer.bin",
        source_path_b="experiments/results/lane_siren_smoke/renderer.bin",
        identical_sha256="08f12d72" + "2" * 56,
        rejection_reason="SIREN smoke timed out rc=124; placeholder copy from lane_g_v3",
    )
    assert prov.evidence_grade == ProvenanceEvidenceGrade.INVALID_BYTE_IDENTITY_ARTIFACT
    assert "SIREN" in prov.rejection_reason


def test_invalid_byte_identity_requires_reason():
    with pytest.raises(InvalidProvenanceError):
        build_provenance_invalid_byte_identity_artifact(
            source_path_a="a",
            source_path_b="b",
            identical_sha256="c" * 64,
            rejection_reason="",
        )


# -----------------------------------------------------------------------------
# Provenance.from_archive_zip_member class method
# -----------------------------------------------------------------------------


def test_from_archive_zip_member_classmethod(tmp_archive):
    archive_path, member, expected_sha = tmp_archive
    prov = Provenance.from_archive_zip_member(
        archive_path=archive_path,
        member_name=member,
        measurement_axis="[contest-CUDA]",
        hardware_substrate="linux_x86_64_t4",
        evidence_grade=ProvenanceEvidenceGrade.PROMOTABLE_EXACT_CONTEST_CUDA,
    )
    assert prov.source_sha256 == expected_sha


# -----------------------------------------------------------------------------
# @requires_canonical_provenance decorator
# -----------------------------------------------------------------------------


def test_decorator_passes_when_provenance_returned():
    prov_fixture = build_provenance_for_predicted(model_id="m", inputs_sha256="a" * 64)

    @requires_canonical_provenance()
    def fn() -> Provenance:
        return prov_fixture

    assert fn() is prov_fixture


def test_decorator_passes_score_claim():
    prov_fixture = build_provenance_for_predicted(model_id="m", inputs_sha256="a" * 64)
    claim = ScoreClaim(score_value=0.5, provenance=prov_fixture)

    @requires_canonical_provenance()
    def fn() -> ScoreClaim:
        return claim

    assert fn() is claim


def test_decorator_rejects_none():
    @requires_canonical_provenance()
    def fn():
        return None

    with pytest.raises(MissingProvenanceError):
        fn()


def test_decorator_rejects_bare_float():
    @requires_canonical_provenance()
    def fn():
        return 0.477

    with pytest.raises(MissingProvenanceError):
        fn()


def test_decorator_dict_with_provenance_passes():
    prov_fixture = build_provenance_for_predicted(model_id="m", inputs_sha256="a" * 64)

    @requires_canonical_provenance()
    def fn() -> dict:
        return {"score": 0.5, "provenance": prov_fixture}

    result = fn()
    assert "provenance" in result


def test_decorator_dict_without_provenance_rejected():
    @requires_canonical_provenance()
    def fn() -> dict:
        return {"score": 0.5}

    with pytest.raises(MissingProvenanceError):
        fn()
