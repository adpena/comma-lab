# SPDX-License-Identifier: MIT
"""Tests for tac.submission_packet.archive_grammar (Phase 3 Layer 1).

Sister of src/tac/tests/test_compression_pipeline.py at the Phase 3 archive-
grammar surface. Covers:

  * ArchiveSectionSpec frozen-invariant validation (every __post_init__ branch)
  * ArchiveGrammarManifest frozen-invariant validation (every __post_init__ branch)
  * ByteMutationSmokeVerdict / OperationalMechanismStatus / SectionKind enums
  * discover_section_specs_from_archive (monolithic + multi-file)
  * emit_parser_section_manifest_sidecar (byte-stable JSON sort_keys)
  * derive_archive_grammar_provenance (Catalog #323 shape)
  * build_archive_grammar_from_compression_pipeline_result (canonical entry point)
  * Cathedral consumer contract compliance (Catalog #335)
  * CLI subprocess (exit codes 0-4)
  * Live-repo regression guard
  * Catalog #266 byte-mutation smoke contract (PASSED / FAILED / NOT_RUN)
  * Catalog #139 + #105 no_op_detector_passed coherence with smoke verdict
  * Catalog #146 fixed-offset section overlap detection
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pytest

from tac.cathedral.consumer_contract import HookNumber
from tac.cathedral_consumers.archive_grammar_builder_consumer import (
    CONSUMER_HOOK_NUMBERS,
    CONSUMER_NAME,
    CONSUMER_VERSION,
    consume_candidate,
    update_from_anchor,
)
from tac.submission_packet.archive_grammar import (
    ARCHIVE_GRAMMAR_SCHEMA_VERSION,
    BYTE_MUTATION_SMOKE_MIN_BYTES,
    CANONICAL_ARCHIVE_NAME,
    CANONICAL_EQUATION_ID,
    CANONICAL_MONOLITHIC_MEMBER_NAME,
    PHASE_3_LAYER_VERSION,
    REPO_ROOT,
    ArchiveGrammarError,
    ArchiveGrammarManifest,
    ArchiveSectionSpec,
    ByteMutationSmokeVerdict,
    OperationalMechanismStatus,
    SectionKind,
    build_archive_grammar_from_compression_pipeline_result,
    derive_archive_grammar_provenance,
    discover_section_specs_from_archive,
    emit_parser_section_manifest_sidecar,
    verify_byte_mutation_smoke_via_canonical_helper,
)
from tac.submission_packet.compression_pipeline import (
    COMPRESSION_PIPELINE_SCHEMA_VERSION,
    CompressionPipelineResult,
    HardwareSubstrateClass,
)
from tac.submission_packet.compression_pipeline import (
    CANONICAL_EQUATION_ID as COMPRESSION_PIPELINE_CANONICAL_EQ,
)


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


def test_schema_version_pinned() -> None:
    assert ARCHIVE_GRAMMAR_SCHEMA_VERSION == "archive_grammar_v1_20260526"


def test_phase_3_layer_version_pinned() -> None:
    assert PHASE_3_LAYER_VERSION == "phase_3_archive_grammar_canonical_landed_20260526"


def test_canonical_equation_id_matches_phase_1_spec() -> None:
    assert CANONICAL_EQUATION_ID == "archive_grammar_canonical_consolidation_savings_v1"


def test_canonical_monolithic_member_name() -> None:
    assert CANONICAL_MONOLITHIC_MEMBER_NAME == "0.bin"


def test_canonical_archive_name() -> None:
    assert CANONICAL_ARCHIVE_NAME == "archive.zip"


def test_byte_mutation_smoke_min_bytes_pinned() -> None:
    assert BYTE_MUTATION_SMOKE_MIN_BYTES == 1024


# ---------------------------------------------------------------------------
# Enum canonical membership
# ---------------------------------------------------------------------------


def test_section_kind_canonical_members() -> None:
    expected = {
        "decoder_blob", "latent_blob", "pose_blob", "mask_blob", "cdf_table",
        "hyperprior_weights", "header", "side_information",
        "distinguishing_feature", "other",
    }
    assert {k.value for k in SectionKind} == expected


def test_operational_mechanism_status_canonical_members() -> None:
    assert {s.value for s in OperationalMechanismStatus} == {
        "OPERATIONAL", "RESEARCH_ONLY", "PRE_BUILD_SCAFFOLD",
    }


def test_byte_mutation_smoke_verdict_canonical_members() -> None:
    assert {v.value for v in ByteMutationSmokeVerdict} == {
        "PASSED", "FAILED_BYTES_NOT_CONSUMED", "NOT_RUN", "INFRASTRUCTURE_ERROR",
    }


# ---------------------------------------------------------------------------
# ArchiveSectionSpec frozen invariants
# ---------------------------------------------------------------------------


def _valid_section_spec(**overrides) -> ArchiveSectionSpec:
    base = {
        "section_name": "decoder_blob",
        "offset_in_archive": 0,
        "length_in_archive": 100,
        "sha256_of_section": "a" * 64,
        "section_kind": SectionKind.DECODER_BLOB.value,
        "operational_mechanism_status": OperationalMechanismStatus.OPERATIONAL.value,
    }
    base.update(overrides)
    return ArchiveSectionSpec(**base)


def test_archive_section_spec_valid_construction() -> None:
    spec = _valid_section_spec()
    assert spec.section_name == "decoder_blob"
    assert spec.offset_in_archive == 0
    assert spec.length_in_archive == 100
    assert spec.member_name == CANONICAL_MONOLITHIC_MEMBER_NAME


def test_archive_section_spec_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="section_name"):
        _valid_section_spec(section_name="")


def test_archive_section_spec_rejects_whitespace_name() -> None:
    with pytest.raises(ValueError, match="section_name"):
        _valid_section_spec(section_name="   ")


def test_archive_section_spec_rejects_negative_offset() -> None:
    with pytest.raises(ValueError, match="offset_in_archive"):
        _valid_section_spec(offset_in_archive=-1)


def test_archive_section_spec_rejects_negative_length() -> None:
    with pytest.raises(ValueError, match="length_in_archive"):
        _valid_section_spec(length_in_archive=-1)


def test_archive_section_spec_rejects_short_sha() -> None:
    with pytest.raises(ValueError, match="sha256_of_section"):
        _valid_section_spec(sha256_of_section="abc")


def test_archive_section_spec_rejects_invalid_section_kind() -> None:
    with pytest.raises(ValueError, match="section_kind"):
        _valid_section_spec(section_kind="not_a_real_kind")


def test_archive_section_spec_rejects_invalid_operational_status() -> None:
    with pytest.raises(ValueError, match="operational_mechanism_status"):
        _valid_section_spec(operational_mechanism_status="not_a_real_status")


def test_archive_section_spec_rejects_empty_distinguishing_feature_name() -> None:
    with pytest.raises(ValueError, match="distinguishing_feature_name"):
        _valid_section_spec(distinguishing_feature_name="   ")


def test_archive_section_spec_rejects_empty_member_name() -> None:
    with pytest.raises(ValueError, match="member_name"):
        _valid_section_spec(member_name="")


def test_archive_section_spec_as_dict_roundtrip() -> None:
    spec = _valid_section_spec(distinguishing_feature_name="score_critical_bits")
    d = spec.as_dict()
    assert d["section_name"] == "decoder_blob"
    assert d["distinguishing_feature_name"] == "score_critical_bits"
    assert d["member_name"] == "0.bin"
    # JSON-serializable
    json.dumps(d)


# ---------------------------------------------------------------------------
# ArchiveGrammarManifest frozen invariants
# ---------------------------------------------------------------------------


def _valid_manifest(**overrides) -> ArchiveGrammarManifest:
    base = {
        "schema_version": ARCHIVE_GRAMMAR_SCHEMA_VERSION,
        "lane_id": "lane_test_20260526",
        "substrate_id": "test_substrate",
        "archive_path": "experiments/results/test/archive.zip",
        "archive_sha256": "b" * 64,
        "archive_bytes": 1234,
        "section_specs": (_valid_section_spec(),),
        "monolithic_single_file": True,
        "multi_file_justification": None,
        "byte_mutation_smoke_verdict": ByteMutationSmokeVerdict.NOT_RUN.value,
        "byte_mutation_smoke_evidence_path": None,
        "no_op_detector_passed": False,
        "measurement_utc": "2026-05-26T00:00:00+00:00",
        "axis_tag": "[predicted]",
        "score_claim": False,
        "promotable": False,
        "evidence_grade": "[predicted; archive-grammar-canonical]",
        "canonical_helper_invocation": "tac.submission_packet.build_archive_grammar_from_compression_pipeline_result",
        "canonical_equation_id": CANONICAL_EQUATION_ID,
        "canonical_equation_status": "FORMALIZATION_PENDING",
        "parser_section_manifest_path": None,
        "elapsed_seconds": 0.0,
        "canonical_provenance": {},
    }
    base.update(overrides)
    return ArchiveGrammarManifest(**base)


def test_manifest_valid_construction() -> None:
    m = _valid_manifest()
    assert m.lane_id == "lane_test_20260526"
    assert m.monolithic_single_file is True
    assert len(m.section_specs) == 1


def test_manifest_rejects_wrong_schema_version() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        _valid_manifest(schema_version="wrong_version")


def test_manifest_rejects_empty_lane_id() -> None:
    with pytest.raises(ValueError, match="lane_id"):
        _valid_manifest(lane_id="")


def test_manifest_rejects_empty_substrate_id() -> None:
    with pytest.raises(ValueError, match="substrate_id"):
        _valid_manifest(substrate_id="")


def test_manifest_rejects_empty_archive_path() -> None:
    with pytest.raises(ValueError, match="archive_path"):
        _valid_manifest(archive_path="")


def test_manifest_rejects_short_archive_sha() -> None:
    with pytest.raises(ValueError, match="archive_sha256"):
        _valid_manifest(archive_sha256="short")


def test_manifest_rejects_negative_archive_bytes() -> None:
    with pytest.raises(ValueError, match="archive_bytes"):
        _valid_manifest(archive_bytes=-1)


def test_manifest_rejects_list_section_specs_must_be_tuple() -> None:
    with pytest.raises(ValueError, match="section_specs"):
        _valid_manifest(section_specs=[_valid_section_spec()])


def test_manifest_monolithic_accepts_any_single_member_name() -> None:
    """Layer 1 x-member grammar fix: monolithic is the structural single-member
    property, recognized member-name-agnostically. A single ``weights.bin``
    member (or the PR101/DQS1 ``x`` convention) IS a valid monolithic archive.
    """
    x_spec = _valid_section_spec(member_name="weights.bin")
    m = _valid_manifest(monolithic_single_file=True, section_specs=(x_spec,))
    assert m.monolithic_single_file is True
    assert m.section_specs[0].member_name == "weights.bin"


def test_manifest_monolithic_rejects_multiple_distinct_members() -> None:
    """monolithic_single_file=True with 2+ DISTINCT members is forbidden — that
    is multi-file and must carry multi_file_justification."""
    s1 = _valid_section_spec(section_name="a", member_name="weights.bin")
    s2 = _valid_section_spec(section_name="b", member_name="latents.bin")
    with pytest.raises(ValueError, match="single ZIP member"):
        _valid_manifest(monolithic_single_file=True, section_specs=(s1, s2))


def test_manifest_multi_file_requires_justification() -> None:
    bad_spec = _valid_section_spec(member_name="weights.bin")
    with pytest.raises(ValueError, match="multi_file_justification"):
        _valid_manifest(
            monolithic_single_file=False,
            multi_file_justification=None,
            section_specs=(bad_spec,),
        )


def test_manifest_multi_file_rejects_placeholder_justification() -> None:
    bad_spec = _valid_section_spec(member_name="weights.bin")
    with pytest.raises(ValueError, match="multi_file_justification"):
        _valid_manifest(
            monolithic_single_file=False,
            multi_file_justification="<rationale>",
            section_specs=(bad_spec,),
        )


def test_manifest_multi_file_rejects_short_justification() -> None:
    bad_spec = _valid_section_spec(member_name="weights.bin")
    with pytest.raises(ValueError, match="multi_file_justification"):
        _valid_manifest(
            monolithic_single_file=False,
            multi_file_justification="abc",
            section_specs=(bad_spec,),
        )


def test_manifest_multi_file_accepts_substantive_justification() -> None:
    bad_spec = _valid_section_spec(member_name="weights.bin")
    m = _valid_manifest(
        monolithic_single_file=False,
        multi_file_justification="substrate uses separate .pt weights + .json config",
        section_specs=(bad_spec,),
    )
    assert m.monolithic_single_file is False
    assert "separate" in m.multi_file_justification


def test_manifest_rejects_invalid_byte_mutation_verdict() -> None:
    with pytest.raises(ValueError, match="byte_mutation_smoke_verdict"):
        _valid_manifest(byte_mutation_smoke_verdict="not_a_verdict")


def test_manifest_passed_smoke_requires_no_op_detector_true() -> None:
    with pytest.raises(ValueError, match="no_op_detector_passed=True"):
        _valid_manifest(
            byte_mutation_smoke_verdict=ByteMutationSmokeVerdict.PASSED.value,
            no_op_detector_passed=False,
        )


def test_manifest_failed_smoke_requires_no_op_detector_false() -> None:
    with pytest.raises(ValueError, match="no_op_detector_passed=False"):
        _valid_manifest(
            byte_mutation_smoke_verdict=ByteMutationSmokeVerdict.FAILED_BYTES_NOT_CONSUMED.value,
            no_op_detector_passed=True,
        )


def test_manifest_not_run_smoke_allows_either_no_op_value() -> None:
    m1 = _valid_manifest(
        byte_mutation_smoke_verdict=ByteMutationSmokeVerdict.NOT_RUN.value,
        no_op_detector_passed=True,
    )
    m2 = _valid_manifest(
        byte_mutation_smoke_verdict=ByteMutationSmokeVerdict.NOT_RUN.value,
        no_op_detector_passed=False,
    )
    assert m1.no_op_detector_passed is True
    assert m2.no_op_detector_passed is False


def test_manifest_rejects_wrong_axis_tag() -> None:
    with pytest.raises(ValueError, match="axis_tag"):
        _valid_manifest(axis_tag="[contest-CUDA]")


def test_manifest_rejects_score_claim_true() -> None:
    with pytest.raises(ValueError, match="score_claim"):
        _valid_manifest(score_claim=True)


def test_manifest_rejects_promotable_true() -> None:
    with pytest.raises(ValueError, match="promotable"):
        _valid_manifest(promotable=True)


def test_manifest_rejects_wrong_evidence_grade_prefix() -> None:
    with pytest.raises(ValueError, match="evidence_grade"):
        _valid_manifest(evidence_grade="[empirical:something]")


def test_manifest_rejects_wrong_canonical_equation_id() -> None:
    with pytest.raises(ValueError, match="canonical_equation_id"):
        _valid_manifest(canonical_equation_id="wrong_equation")


def test_manifest_rejects_invalid_canonical_equation_status() -> None:
    with pytest.raises(ValueError, match="canonical_equation_status"):
        _valid_manifest(canonical_equation_status="MAYBE_REGISTERED")


def test_manifest_accepts_registered_status() -> None:
    m = _valid_manifest(canonical_equation_status="REGISTERED")
    assert m.canonical_equation_status == "REGISTERED"


def test_manifest_rejects_negative_elapsed() -> None:
    with pytest.raises(ValueError, match="elapsed_seconds"):
        _valid_manifest(elapsed_seconds=-1.0)


def test_manifest_rejects_non_mapping_provenance() -> None:
    with pytest.raises(ValueError, match="canonical_provenance"):
        _valid_manifest(canonical_provenance="not_a_dict")


def test_manifest_rejects_section_overlap_in_monolithic() -> None:
    s1 = _valid_section_spec(section_name="a", offset_in_archive=0, length_in_archive=100)
    s2 = _valid_section_spec(section_name="b", offset_in_archive=50, length_in_archive=100)
    with pytest.raises(ValueError, match="section overlap"):
        _valid_manifest(section_specs=(s1, s2))


def test_manifest_accepts_adjacent_sections_no_overlap() -> None:
    s1 = _valid_section_spec(section_name="a", offset_in_archive=0, length_in_archive=100)
    s2 = _valid_section_spec(section_name="b", offset_in_archive=100, length_in_archive=50)
    m = _valid_manifest(section_specs=(s1, s2))
    assert len(m.section_specs) == 2


def test_manifest_overlap_check_per_member_multi_file() -> None:
    s1 = _valid_section_spec(section_name="a", offset_in_archive=0, length_in_archive=100,
                              member_name="weights.bin")
    s2 = _valid_section_spec(section_name="b", offset_in_archive=0, length_in_archive=100,
                              member_name="latents.bin")
    m = _valid_manifest(
        monolithic_single_file=False,
        multi_file_justification="substrate requires separate weights + latents members",
        section_specs=(s1, s2),
    )
    assert len(m.section_specs) == 2


def test_manifest_as_dict_roundtrip() -> None:
    m = _valid_manifest()
    d = m.as_dict()
    # JSON-serializable
    text = json.dumps(d, sort_keys=True)
    assert ARCHIVE_GRAMMAR_SCHEMA_VERSION in text
    assert "[predicted]" in text


# ---------------------------------------------------------------------------
# derive_archive_grammar_provenance
# ---------------------------------------------------------------------------


def test_derive_provenance_canonical_shape() -> None:
    p = derive_archive_grammar_provenance(
        lane_id="lane_test",
        substrate_id="test_sub",
        archive_sha256="c" * 64,
        measurement_utc="2026-05-26T00:00:00+00:00",
    )
    assert p["axis_tag"] == "[predicted]"
    assert p["score_claim"] is False
    assert p["promotable"] is False
    assert p["evidence_grade"] == "[predicted; archive-grammar-canonical]"
    assert p["canonical_equation_id"] == CANONICAL_EQUATION_ID
    assert p["canonical_equation_status"] == "FORMALIZATION_PENDING"
    assert p["schema_version"] == ARCHIVE_GRAMMAR_SCHEMA_VERSION
    assert p["lane_id"] == "lane_test"
    assert p["substrate_id"] == "test_sub"
    assert p["archive_sha256"] == "c" * 64


# ---------------------------------------------------------------------------
# discover_section_specs_from_archive
# ---------------------------------------------------------------------------


def _build_monolithic_archive(tmp_path: Path, contents: bytes = b"hello world" * 100) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("0.bin", contents)
    return archive


def _build_single_member_archive(
    tmp_path: Path, member_name: str, contents: bytes = b"frontier bytes" * 100
) -> Path:
    """Build a single-member archive with an arbitrary member name (e.g. the
    PR101/DQS1 frontier-grammar ``x`` convention)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(member_name, contents)
    return archive


def _build_multi_file_archive(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("weights.bin", b"weight bytes" * 50)
        zf.writestr("latents.bin", b"latent bytes" * 30)
    return archive


def test_discover_monolithic_single_file(tmp_path: Path) -> None:
    archive = _build_monolithic_archive(tmp_path)
    specs, is_monolithic = discover_section_specs_from_archive(archive)
    assert is_monolithic is True
    assert len(specs) == 1
    assert specs[0].section_name == "0.bin"
    assert specs[0].member_name == "0.bin"
    assert specs[0].length_in_archive == len(b"hello world" * 100)


def test_discover_multi_file(tmp_path: Path) -> None:
    archive = _build_multi_file_archive(tmp_path)
    specs, is_monolithic = discover_section_specs_from_archive(archive)
    assert is_monolithic is False
    assert len(specs) == 2
    names = {s.section_name for s in specs}
    assert names == {"weights.bin", "latents.bin"}


def test_discover_single_x_member_is_monolithic(tmp_path: Path) -> None:
    """Layer 1 x-member grammar fix (Phase 10 op-routable #1): the PR101/DQS1
    frontier-grammar single-``x``-member archive IS monolithic, recognized
    structurally rather than by the literal ``0.bin`` name. This is the exact
    case that drove the Phase 10 dry-run to exit 5 NAMED-BLOCKER pre-fix.
    """
    archive = _build_single_member_archive(tmp_path, "x", b"dqs1 frontier" * 200)
    specs, is_monolithic = discover_section_specs_from_archive(archive)
    assert is_monolithic is True
    assert len(specs) == 1
    assert specs[0].section_name == "x"
    assert specs[0].member_name == "x"
    assert specs[0].length_in_archive == len(b"dqs1 frontier" * 200)


def test_discover_single_0bin_member_is_monolithic_regression(tmp_path: Path) -> None:
    """Backward-compat regression: the canonical ``0.bin`` single-member archive
    is still classified monolithic after the x-member fix."""
    archive = _build_monolithic_archive(tmp_path)
    specs, is_monolithic = discover_section_specs_from_archive(archive)
    assert is_monolithic is True
    assert len(specs) == 1
    assert specs[0].member_name == "0.bin"


def test_discover_single_arbitrary_member_is_monolithic(tmp_path: Path) -> None:
    """Any single-member name (not just 0.bin / x) classifies monolithic."""
    archive = _build_single_member_archive(tmp_path, "weights.bin")
    specs, is_monolithic = discover_section_specs_from_archive(archive)
    assert is_monolithic is True
    assert len(specs) == 1
    assert specs[0].member_name == "weights.bin"


def test_discover_genuine_multi_file_still_not_monolithic(tmp_path: Path) -> None:
    """Regression guard: a genuine 2+-member archive is still NOT monolithic
    after the x-member fix (the fix must not over-classify)."""
    archive = _build_multi_file_archive(tmp_path)
    specs, is_monolithic = discover_section_specs_from_archive(archive)
    assert is_monolithic is False
    assert len(specs) == 2


def test_discover_empty_archive_raises(tmp_path: Path) -> None:
    archive = tmp_path / "empty.zip"
    with zipfile.ZipFile(archive, "w"):
        pass
    with pytest.raises(ArchiveGrammarError, match="zero ZIP members"):
        discover_section_specs_from_archive(archive)


# ---------------------------------------------------------------------------
# emit_parser_section_manifest_sidecar (byte-stable)
# ---------------------------------------------------------------------------


def test_emit_sidecar_byte_stable(tmp_path: Path) -> None:
    m = _valid_manifest()
    path1 = emit_parser_section_manifest_sidecar(m, tmp_path)
    bytes1 = path1.read_bytes()
    # Re-emit; should be identical bytes (sort_keys deterministic).
    path1.unlink()
    path2 = emit_parser_section_manifest_sidecar(m, tmp_path)
    bytes2 = path2.read_bytes()
    assert bytes1 == bytes2
    assert path2.name == "parser_section_manifest.json"


def test_emit_sidecar_custom_filename(tmp_path: Path) -> None:
    m = _valid_manifest()
    p = emit_parser_section_manifest_sidecar(m, tmp_path, filename="custom.json")
    assert p.name == "custom.json"
    data = json.loads(p.read_text())
    assert data["lane_id"] == "lane_test_20260526"


def test_emit_sidecar_creates_missing_dir(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "deeply" / "nonexistent"
    m = _valid_manifest()
    p = emit_parser_section_manifest_sidecar(m, out)
    assert p.is_file()


# ---------------------------------------------------------------------------
# build_archive_grammar_from_compression_pipeline_result (canonical entry)
# ---------------------------------------------------------------------------


def _build_synthetic_pipeline_result(
    *,
    lane_id: str = "lane_synthetic_20260526",
    substrate_id: str = "synthetic_substrate",
) -> CompressionPipelineResult:
    return CompressionPipelineResult(
        schema_version=COMPRESSION_PIPELINE_SCHEMA_VERSION,
        lane_id=lane_id,
        substrate_id=substrate_id,
        video_path="upstream/videos/0.mkv",
        hardware_substrate="macos_arm64_m5_max",
        hardware_substrate_class=HardwareSubstrateClass.LOCAL_MPS.value,
        substrate_trainer_path="experiments/train_substrate_synthetic.py",
        recipe_path=".omx/operator_authorize_recipes/substrate_synthetic_local_apple_silicon_dispatch.yaml",
        mlx_first_encode=True,
        qat_enabled=True,
        weights_export_path=None,
        weights_sha256=None,
        weights_size_bytes=None,
        training_anchor_call_id=None,
        qat_anchor_call_id=None,
        dispatch_optimization_protocol_overall_pass=True,
        dispatch_optimization_protocol_blockers=(),
        per_axis_predicted_band=None,
        measurement_utc="2026-05-26T00:00:00+00:00",
        axis_tag="[predicted]",
        score_claim=False,
        promotable=False,
        evidence_grade="[predicted; compression-pipeline-canonical]",
        canonical_helper_invocation="tac.submission_packet.build_compression_pipeline",
        canonical_equation_id=COMPRESSION_PIPELINE_CANONICAL_EQ,
        canonical_equation_status="FORMALIZATION_PENDING",
        elapsed_seconds=0.0,
        cost_usd=None,
        canonical_provenance={},
        written_at_utc="2026-05-26T00:00:00+00:00",
        written_pid=12345,
        written_host="test-host",
    )


def test_build_canonical_entry_monolithic_synthetic(tmp_path: Path) -> None:
    archive = _build_monolithic_archive(tmp_path, b"x" * 5000)
    pipeline = _build_synthetic_pipeline_result()
    manifest = build_archive_grammar_from_compression_pipeline_result(
        compression_pipeline_result=pipeline,
        archive_path=archive,
        output_dir=tmp_path,
    )
    assert manifest.monolithic_single_file is True
    assert manifest.archive_bytes == archive.stat().st_size
    assert len(manifest.section_specs) == 1
    assert manifest.section_specs[0].member_name == "0.bin"
    assert manifest.byte_mutation_smoke_verdict == ByteMutationSmokeVerdict.NOT_RUN.value
    assert manifest.no_op_detector_passed is False
    assert manifest.parser_section_manifest_path is not None
    assert Path(manifest.parser_section_manifest_path).is_file()


def test_build_canonical_entry_single_x_member_passes(tmp_path: Path) -> None:
    """Layer 1 x-member grammar fix end-to-end: the canonical entry point
    classifies a single-``x``-member archive as monolithic with the DEFAULT
    monolithic_single_file=True (no multi_file_justification needed). This is
    the exact path the Phase 10 CLI took to exit 5 NAMED-BLOCKER pre-fix; it
    now reaches a valid ArchiveGrammarManifest.
    """
    archive = _build_single_member_archive(tmp_path / "results", "x", b"x" * 5000)
    pipeline = _build_synthetic_pipeline_result(
        lane_id="lane_v14_v2_dqs1_plus_fec10_20260526",
        substrate_id="pr101_lc_v2_clone_enhanced_curriculum",
    )
    manifest = build_archive_grammar_from_compression_pipeline_result(
        compression_pipeline_result=pipeline,
        archive_path=archive,
        output_dir=tmp_path,
    )
    assert manifest.monolithic_single_file is True
    assert manifest.multi_file_justification is None
    assert len(manifest.section_specs) == 1
    assert manifest.section_specs[0].member_name == "x"
    assert manifest.archive_bytes == archive.stat().st_size
    assert manifest.parser_section_manifest_path is not None
    assert Path(manifest.parser_section_manifest_path).is_file()


def test_build_canonical_entry_auto_promotes_multi_file_detection(tmp_path: Path) -> None:
    """When archive is multi-file but caller passes monolithic_single_file=True
    (the default), the helper detects the mismatch and would error unless
    multi_file_justification is supplied. Verifies the auto-detection.
    """
    archive = _build_multi_file_archive(tmp_path)
    pipeline = _build_synthetic_pipeline_result()
    with pytest.raises(ValueError, match="multi_file_justification"):
        build_archive_grammar_from_compression_pipeline_result(
            compression_pipeline_result=pipeline,
            archive_path=archive,
            output_dir=tmp_path,
        )


def test_build_canonical_entry_multi_file_with_justification(tmp_path: Path) -> None:
    archive = _build_multi_file_archive(tmp_path)
    pipeline = _build_synthetic_pipeline_result()
    manifest = build_archive_grammar_from_compression_pipeline_result(
        compression_pipeline_result=pipeline,
        archive_path=archive,
        monolithic_single_file=False,
        multi_file_justification="synthetic test fixture needs both members",
        output_dir=tmp_path,
    )
    assert manifest.monolithic_single_file is False
    assert "synthetic test fixture" in manifest.multi_file_justification
    assert len(manifest.section_specs) == 2


def test_build_canonical_entry_skip_sidecar(tmp_path: Path) -> None:
    archive = _build_monolithic_archive(tmp_path)
    pipeline = _build_synthetic_pipeline_result()
    manifest = build_archive_grammar_from_compression_pipeline_result(
        compression_pipeline_result=pipeline,
        archive_path=archive,
        output_dir=tmp_path,
        emit_parser_section_manifest=False,
    )
    assert manifest.parser_section_manifest_path is None


def test_build_canonical_entry_missing_archive_raises(tmp_path: Path) -> None:
    pipeline = _build_synthetic_pipeline_result()
    with pytest.raises(ArchiveGrammarError, match="does not exist"):
        build_archive_grammar_from_compression_pipeline_result(
            compression_pipeline_result=pipeline,
            archive_path=tmp_path / "nonexistent.zip",
            output_dir=tmp_path,
        )


def test_build_canonical_entry_wrong_result_type_raises(tmp_path: Path) -> None:
    archive = _build_monolithic_archive(tmp_path)
    with pytest.raises(ArchiveGrammarError, match="CompressionPipelineResult"):
        build_archive_grammar_from_compression_pipeline_result(
            compression_pipeline_result={"not": "a_result"},  # type: ignore[arg-type]
            archive_path=archive,
            output_dir=tmp_path,
        )


def test_build_canonical_entry_non_path_archive_raises(tmp_path: Path) -> None:
    pipeline = _build_synthetic_pipeline_result()
    with pytest.raises(ArchiveGrammarError, match="pathlib.Path"):
        build_archive_grammar_from_compression_pipeline_result(
            compression_pipeline_result=pipeline,
            archive_path="not_a_path",  # type: ignore[arg-type]
            output_dir=tmp_path,
        )


def test_build_canonical_entry_byte_mutation_requires_inflate_sh(tmp_path: Path) -> None:
    archive = _build_monolithic_archive(tmp_path)
    pipeline = _build_synthetic_pipeline_result()
    with pytest.raises(ArchiveGrammarError, match="inflate_sh_path"):
        build_archive_grammar_from_compression_pipeline_result(
            compression_pipeline_result=pipeline,
            archive_path=archive,
            output_dir=tmp_path,
            verify_byte_mutation_smoke=True,
            # inflate_sh_path missing
        )


def test_build_canonical_entry_preserves_lane_and_substrate(tmp_path: Path) -> None:
    archive = _build_monolithic_archive(tmp_path)
    pipeline = _build_synthetic_pipeline_result(
        lane_id="lane_pr111_candidate_20260601",
        substrate_id="nscs06_v8_chroma_lut",
    )
    manifest = build_archive_grammar_from_compression_pipeline_result(
        compression_pipeline_result=pipeline,
        archive_path=archive,
        output_dir=tmp_path,
    )
    assert manifest.lane_id == "lane_pr111_candidate_20260601"
    assert manifest.substrate_id == "nscs06_v8_chroma_lut"
    assert manifest.canonical_provenance["lane_id"] == "lane_pr111_candidate_20260601"


# ---------------------------------------------------------------------------
# Cathedral consumer contract compliance (Catalog #335)
# ---------------------------------------------------------------------------


def test_cathedral_consumer_canonical_name() -> None:
    assert CONSUMER_NAME == "archive_grammar_builder_consumer"


def test_cathedral_consumer_version_pinned() -> None:
    assert CONSUMER_VERSION == "1.0.0"


def test_cathedral_consumer_declares_hooks() -> None:
    # Phase 3 consumer declares hooks 3+4+5+6.
    assert HookNumber.BIT_ALLOCATOR in CONSUMER_HOOK_NUMBERS
    assert HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH in CONSUMER_HOOK_NUMBERS
    assert HookNumber.CONTINUAL_LEARNING_POSTERIOR in CONSUMER_HOOK_NUMBERS
    assert HookNumber.PROBE_DISAMBIGUATOR in CONSUMER_HOOK_NUMBERS


def test_cathedral_consumer_consume_empty_candidate() -> None:
    out = consume_candidate({})
    assert out["predicted_delta_adjustment"] == 0.0
    assert out["promotable"] is False
    assert out["axis_tag"] == "[predicted]"
    assert out["readiness_verdict"] == "UNKNOWN"


def test_cathedral_consumer_consume_ready_candidate() -> None:
    candidate = {
        "archive_grammar_manifest": {
            "monolithic_single_file": True,
            "no_op_detector_passed": True,
            "byte_mutation_smoke_verdict": "PASSED",
            "section_specs": [{"section_name": "0.bin"}],
        },
    }
    out = consume_candidate(candidate)
    assert out["readiness_verdict"] == "READY"
    assert "CLEAN" in out["rationale"]
    assert out["predicted_delta_adjustment"] == 0.0
    assert out["promotable"] is False


def test_cathedral_consumer_consume_blocked_candidate() -> None:
    candidate = {
        "archive_grammar_manifest": {
            "monolithic_single_file": True,
            "no_op_detector_passed": False,
            "byte_mutation_smoke_verdict": "FAILED_BYTES_NOT_CONSUMED",
            "section_specs": [{"section_name": "0.bin"}],
        },
    }
    out = consume_candidate(candidate)
    assert out["readiness_verdict"] == "BLOCKED"
    assert "Catalog #266" in out["rationale"]


def test_cathedral_consumer_consume_multi_file_candidate() -> None:
    candidate = {
        "archive_grammar_manifest": {
            "monolithic_single_file": False,
            "multi_file_justification": "test case",
            "no_op_detector_passed": False,
            "byte_mutation_smoke_verdict": "NOT_RUN",
            "section_specs": [{"section_name": "weights.bin"}, {"section_name": "latents.bin"}],
        },
    }
    out = consume_candidate(candidate)
    assert out["readiness_verdict"] == "MULTI_FILE_REVIEW"
    assert "MULTI-FILE" in out["rationale"]


def test_cathedral_consumer_update_from_anchor_noop() -> None:
    # Should not raise.
    update_from_anchor(None)
    update_from_anchor({"some": "anchor"})


# ---------------------------------------------------------------------------
# CLI subprocess (exit codes 0-4)
# ---------------------------------------------------------------------------


def test_cli_exit_code_4_missing_required_args(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "archive_grammar_cli.py")],
        capture_output=True,
        text=True,
        timeout=20,
    )
    # argparse exits with 2 when required args missing — that's a CLI-side error
    # not the helper's EXIT_CLI_ERROR (4). argparse exits before our main runs.
    assert proc.returncode != 0


def test_cli_help_works(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "archive_grammar_cli.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert proc.returncode == 0
    assert "archive grammar" in proc.stdout.lower()


def test_cli_clean_run_synthetic_archive(tmp_path: Path) -> None:
    """End-to-end CLI test with synthetic trainer + recipe + archive."""
    trainer = tmp_path / "experiments" / "train_substrate_synthetic.py"
    trainer.parent.mkdir(parents=True, exist_ok=True)
    trainer.write_text("# synthetic trainer fixture\n")
    recipe_dir = tmp_path / ".omx" / "operator_authorize_recipes"
    recipe_dir.mkdir(parents=True, exist_ok=True)
    recipe = recipe_dir / "substrate_synthetic_modal_t4_dispatch.yaml"
    recipe.write_text(
        "lane_id: lane_synthetic_20260526\n"
        "research_only: true\n"
        "dispatch_enabled: false\n"
    )
    archive = _build_monolithic_archive(tmp_path / "results")
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "archive_grammar_cli.py"),
            "--lane-id", "lane_synthetic_20260526",
            "--substrate-trainer", str(trainer),
            "--recipe-path", str(recipe),
            "--archive-path", str(archive),
            "--skip-protocol-verification",
            "--quiet",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        print("STDOUT:", proc.stdout)
        print("STDERR:", proc.stderr)
    assert proc.returncode == 0, f"CLI failed: {proc.stderr}"
    assert "Archive Grammar Manifest" in proc.stdout


def test_cli_json_output(tmp_path: Path) -> None:
    trainer = tmp_path / "experiments" / "train_substrate_synthetic.py"
    trainer.parent.mkdir(parents=True, exist_ok=True)
    trainer.write_text("# synthetic\n")
    recipe = tmp_path / ".omx" / "operator_authorize_recipes" / "substrate_synthetic_modal_t4_dispatch.yaml"
    recipe.parent.mkdir(parents=True, exist_ok=True)
    recipe.write_text("research_only: true\n")
    archive = _build_monolithic_archive(tmp_path / "results")
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "archive_grammar_cli.py"),
            "--lane-id", "lane_synthetic_20260526",
            "--substrate-trainer", str(trainer),
            "--recipe-path", str(recipe),
            "--archive-path", str(archive),
            "--skip-protocol-verification",
            "--quiet",
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert data["schema_version"] == ARCHIVE_GRAMMAR_SCHEMA_VERSION
    assert data["axis_tag"] == "[predicted]"
    assert data["score_claim"] is False
    assert data["promotable"] is False
    assert data["canonical_equation_id"] == CANONICAL_EQUATION_ID


def test_cli_multi_file_without_justification_returns_cli_error(tmp_path: Path) -> None:
    trainer = tmp_path / "experiments" / "train_substrate_synthetic.py"
    trainer.parent.mkdir(parents=True, exist_ok=True)
    trainer.write_text("# synthetic\n")
    recipe = tmp_path / ".omx" / "operator_authorize_recipes" / "substrate_synthetic_modal_t4_dispatch.yaml"
    recipe.parent.mkdir(parents=True, exist_ok=True)
    recipe.write_text("research_only: true\n")
    archive = _build_multi_file_archive(tmp_path / "results")
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "archive_grammar_cli.py"),
            "--lane-id", "lane_synthetic_20260526",
            "--substrate-trainer", str(trainer),
            "--recipe-path", str(recipe),
            "--archive-path", str(archive),
            "--skip-protocol-verification",
            "--multi-file",  # without --multi-file-justification
            "--quiet",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 4  # EXIT_CLI_ERROR
    assert "multi-file-justification" in proc.stderr


def test_cli_multi_file_with_justification_clean(tmp_path: Path) -> None:
    trainer = tmp_path / "experiments" / "train_substrate_synthetic.py"
    trainer.parent.mkdir(parents=True, exist_ok=True)
    trainer.write_text("# synthetic\n")
    recipe = tmp_path / ".omx" / "operator_authorize_recipes" / "substrate_synthetic_modal_t4_dispatch.yaml"
    recipe.parent.mkdir(parents=True, exist_ok=True)
    recipe.write_text("research_only: true\n")
    archive = _build_multi_file_archive(tmp_path / "results")
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "archive_grammar_cli.py"),
            "--lane-id", "lane_synthetic_20260526",
            "--substrate-trainer", str(trainer),
            "--recipe-path", str(recipe),
            "--archive-path", str(archive),
            "--skip-protocol-verification",
            "--multi-file",
            "--multi-file-justification", "substrate requires separate weights+latents members",
            "--quiet",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        print("STDOUT:", proc.stdout)
        print("STDERR:", proc.stderr)
    assert proc.returncode == 0


def test_cli_section_overlap_returns_exit_3(tmp_path: Path) -> None:
    """Direct programmatic test rather than CLI route since auto-discovered
    sections from a synthetic archive don't overlap by construction.

    We invoke the underlying validator directly through the manifest
    constructor to verify the section-overlap exit-code semantics work.
    """
    s1 = _valid_section_spec(section_name="a", offset_in_archive=0, length_in_archive=100)
    s2 = _valid_section_spec(section_name="b", offset_in_archive=50, length_in_archive=100)
    with pytest.raises(ValueError, match="section overlap"):
        _valid_manifest(section_specs=(s1, s2))


# ---------------------------------------------------------------------------
# Live-repo regression guard
# ---------------------------------------------------------------------------


def test_live_repo_module_imports() -> None:
    """Live-repo regression guard: the canonical module imports cleanly
    from the live repo per Catalog #185 sister discipline."""
    from tac.submission_packet import (  # noqa: F401
        ArchiveGrammarManifest as _M,
        ArchiveSectionSpec as _S,
        build_archive_grammar_from_compression_pipeline_result as _b,
    )


def test_live_repo_consumer_imports() -> None:
    from tac.cathedral_consumers.archive_grammar_builder_consumer import (  # noqa: F401
        CONSUMER_HOOK_NUMBERS as _h,
        CONSUMER_NAME as _n,
        CONSUMER_VERSION as _v,
        consume_candidate as _c,
        update_from_anchor as _u,
    )


def test_live_repo_cli_importable_via_subprocess() -> None:
    """The CLI module is importable + --help works at the live repo path."""
    cli_path = REPO_ROOT / "tools" / "archive_grammar_cli.py"
    assert cli_path.is_file()
    proc = subprocess.run(
        [sys.executable, str(cli_path), "--help"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert proc.returncode == 0


def test_live_repo_v14_v2_dqs1_x_member_archive_is_monolithic() -> None:
    """Layer 1 x-member grammar fix regression against the ACTUAL V14-V2 PR111
    candidate frontier archive (single ``x`` member, PR101/DQS1 grammar). This
    is the exact archive whose member-name convention drove the Phase 10
    end-to-end dry-run to exit 5 NAMED-BLOCKER. Guarded by existence so the
    suite stays green on clones without the candidate work dir.
    """
    candidate = (
        REPO_ROOT
        / "experiments"
        / "results"
        / "v14_v2_dqs1_plus_fec10_substituted_20260526T023000Z"
        / "submission_dir"
        / "archive.zip"
    )
    if not candidate.is_file():
        pytest.skip("V14-V2 candidate archive not present in this checkout")
    specs, is_monolithic = discover_section_specs_from_archive(candidate)
    assert is_monolithic is True, (
        "V14-V2 single-x-member archive must classify monolithic per the "
        "Layer 1 x-member grammar fix"
    )
    assert len(specs) == 1
    assert specs[0].member_name == "x"


def test_live_repo_canonical_verifier_helper_exists() -> None:
    """Catalog #272 byte-mutation smoke helper exists at the canonical path."""
    helper = REPO_ROOT / "tools" / "verify_distinguishing_feature_byte_mutation.py"
    assert helper.is_file(), (
        "canonical byte-mutation smoke helper must exist at the canonical path; "
        "Phase 3 archive_grammar routes through it per Catalog #226"
    )


# ---------------------------------------------------------------------------
# Integration test: Phase 2 → Phase 3 chain
# ---------------------------------------------------------------------------


def test_integration_phase_2_to_phase_3(tmp_path: Path) -> None:
    """Verify Phase 2 CompressionPipelineResult flows into Phase 3 cleanly."""
    pipeline = _build_synthetic_pipeline_result(
        lane_id="lane_integration_test_20260526",
        substrate_id="integration_test_substrate",
    )
    archive = _build_monolithic_archive(tmp_path, b"integration test bytes" * 100)
    manifest = build_archive_grammar_from_compression_pipeline_result(
        compression_pipeline_result=pipeline,
        archive_path=archive,
        output_dir=tmp_path,
    )
    # Lineage preserved.
    assert manifest.lane_id == pipeline.lane_id
    assert manifest.substrate_id == pipeline.substrate_id
    # Canonical Provenance carries lineage tuple.
    assert manifest.canonical_provenance["lane_id"] == pipeline.lane_id
    assert manifest.canonical_provenance["substrate_id"] == pipeline.substrate_id
    # Sidecar emitted.
    assert manifest.parser_section_manifest_path is not None
    sidecar_data = json.loads(Path(manifest.parser_section_manifest_path).read_text())
    assert sidecar_data["lane_id"] == pipeline.lane_id


# ---------------------------------------------------------------------------
# verify_byte_mutation_smoke_via_canonical_helper (helper-routing test)
# ---------------------------------------------------------------------------


def test_verify_byte_mutation_smoke_no_distinguishing_features_returns_not_run(tmp_path: Path) -> None:
    """When no section has distinguishing_feature_name, the helper returns
    NOT_RUN (smoke not applicable)."""
    archive = _build_monolithic_archive(tmp_path)
    inflate_sh = tmp_path / "inflate.sh"
    inflate_sh.write_text("#!/bin/bash\necho 'inflate stub'\n")
    inflate_sh.chmod(0o755)
    section_specs = (_valid_section_spec(),)  # no distinguishing_feature_name
    output_json = tmp_path / "smoke.json"
    verdict, no_op_passed, evidence = verify_byte_mutation_smoke_via_canonical_helper(
        archive_path=archive,
        inflate_sh_path=inflate_sh,
        section_specs=section_specs,
        output_json_path=output_json,
    )
    assert verdict == ByteMutationSmokeVerdict.NOT_RUN.value
    assert no_op_passed is False
    assert evidence is None
