from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import brotli
import pytest

from tac.hnerv_decoder_recode import (
    PACKED_STATE_SCHEMA,
    encode_global_prev_symbol_context_range_fixture,
    encode_global_prev_symbol_mixed_context_fixture,
    encode_hdm3_q_brotli_split_fixture,
    parse_packed_decoder_brotli,
)
from tac.hnerv_entropy_candidate_packet import (
    CANDIDATE_STREAM_REQUIREMENT_ID,
    DECODED_OUTPUT_EQUIVALENCE_REQUIREMENT_ID,
    HDC2_ARCHIVE_CANDIDATE_CONTRACT_REQUIREMENT_ID,
    HDC2_DIRECT_ARCHIVE_RUNTIME_REQUIREMENTS,
    HDC2_RUNTIME_DECODER_CONTRACT_REQUIREMENT_ID,
    HDC2_STREAM_ARTIFACT_REQUIREMENTS,
    ROUNDTRIP_REQUIREMENT_ID,
    SOURCE_ARCHIVE_REQUIREMENT_ID,
    SOURCE_STREAM_REQUIREMENT_ID,
    HnervEntropyCandidatePacketError,
    build_candidate_packet_manifest,
    build_hdc2_stream_byte_equivalence_work_product,
    discover_candidate_audit_inputs,
    discovery_report_input_paths,
)
from tac.hnerv_lowlevel_packer import PackedHnervPayload, write_stored_single_member_zip
from tac.optimization.entropy_codec_gap_audit import build_entropy_codec_gap_audit
from tac.repo_io import json_text, sha256_bytes, sha256_file

REPO = Path(__file__).resolve().parents[3]


def test_candidate_packet_records_selected_target_and_missing_artifacts(tmp_path: Path) -> None:
    audit_path = tmp_path / "entropy_audit.json"
    audit_path.write_text(json_text(_audit()), encoding="utf-8")

    manifest = build_candidate_packet_manifest(audit_path, repo_root=REPO)

    assert manifest["tool"] == "tac.hnerv_entropy_candidate_packet"
    assert manifest["score_claim"] is False
    assert manifest["dispatch_attempted"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["ready_for_byte_closed_candidate_build"] is False
    assert manifest["selected_target"]["rank"] == 1
    assert manifest["selected_target"]["label"] == "hnerv_decoder_weights"
    assert manifest["selected_target"]["target_kind"] == "known_payload_entropy_gap"
    assert manifest["audit_source"]["sha256"] == sha256_file(audit_path)
    assert manifest["available_inputs"] == [manifest["audit_source"]]
    assert "source_archive_manifest_with_archive_sha256_bytes_and_runtime_tree_sha256" in manifest[
        "source_artifact_requirements"
    ]
    assert "old_new_decoded_output_sha256_equality_report" in manifest["byte_equivalence_requirements"]
    assert "runtime_tree_parity_manifest" in manifest["runtime_parity_requirements"]
    assert "candidate_archive_manifest_with_member_sha256s" in manifest["archive_manifest_requirements"]
    assert "runtime_tree_parity_manifest" in manifest["missing_artifacts"]
    assert "missing_artifact:runtime_tree_parity_manifest" in manifest["dispatch_blockers"]
    assert "requires_lane_dispatch_claim_before_gpu" in manifest["dispatch_blockers"]
    assert "requires_exact_cuda_auth_eval" in manifest["dispatch_blockers"]


def test_candidate_packet_rejects_placeholder_requirement_artifacts(
    tmp_path: Path,
) -> None:
    audit_path = tmp_path / "entropy_audit.json"
    audit = _audit()
    audit_path.write_text(json_text(audit), encoding="utf-8")
    target = audit["entropy_overhead_target_ranking"][0]
    requirement_ids = list(target["exact_next_artifact_requirements"])
    requirement_ids.append("runtime_tree_parity_manifest")
    artifacts = {}
    for requirement_id in dict.fromkeys(requirement_ids):
        artifact = tmp_path / f"{requirement_id}.json"
        artifact.write_text(json.dumps({"requirement_id": requirement_id}, sort_keys=True), encoding="utf-8")
        artifacts[requirement_id] = artifact

    manifest = build_candidate_packet_manifest(
        audit_path,
        artifact_paths=artifacts,
        repo_root=REPO,
    )

    assert SOURCE_ARCHIVE_REQUIREMENT_ID in manifest["invalid_requirement_artifacts"]
    assert SOURCE_STREAM_REQUIREMENT_ID in manifest["invalid_requirement_artifacts"]
    assert CANDIDATE_STREAM_REQUIREMENT_ID in manifest["invalid_requirement_artifacts"]
    assert DECODED_OUTPUT_EQUIVALENCE_REQUIREMENT_ID in manifest["invalid_requirement_artifacts"]
    assert ROUNDTRIP_REQUIREMENT_ID in manifest["invalid_requirement_artifacts"]
    assert SOURCE_ARCHIVE_REQUIREMENT_ID in manifest["missing_artifacts"]
    assert f"invalid_requirement_artifact:{SOURCE_ARCHIVE_REQUIREMENT_ID}" in manifest[
        "readiness_blockers"
    ]
    assert manifest["ready_for_local_packet_review"] is False
    assert manifest["ready_for_archive_preflight"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    records = {row["id"]: row for row in manifest["packet_requirements"]}
    assert records[SOURCE_ARCHIVE_REQUIREMENT_ID]["available"] is False
    assert records[SOURCE_ARCHIVE_REQUIREMENT_ID]["missing_reason"] == "validation_blockers"
    assert records[SOURCE_ARCHIVE_REQUIREMENT_ID]["sha256"] == sha256_file(
        artifacts[SOURCE_ARCHIVE_REQUIREMENT_ID]
    )
    assert {row["id"] for row in manifest["available_inputs"]} == {"entropy_audit_json"}


def test_hdc2_stream_work_product_closes_stream_requirements_but_not_archive(
    tmp_path: Path,
) -> None:
    fixture = _write_hdc2_fixture(tmp_path)
    work_product = build_hdc2_stream_byte_equivalence_work_product(
        fixture["profile_path"],
        fixture["archive_path"],
        source_exact_eval_json_path=fixture["exact_eval_path"],
        candidate_stream_path=tmp_path / "candidate_hdc2.bin",
        repo_root=tmp_path,
    )
    artifacts = _write_hdc2_requirement_artifacts(tmp_path, work_product)

    manifest = build_candidate_packet_manifest(
        fixture["profile_path"],
        artifact_paths=artifacts,
        repo_root=tmp_path,
    )

    assert work_product["score_claim"] is False
    assert work_product["dispatch_attempted"] is False
    assert work_product["ready_for_exact_eval_dispatch"] is False
    candidate_stream = work_product["candidate_stream_file"]
    assert candidate_stream["path"] == "candidate_hdc2.bin"
    assert candidate_stream["bytes"] > 0
    assert candidate_stream["sha256"] == sha256_file(tmp_path / "candidate_hdc2.bin")
    assert work_product["decoded_output_equivalence_report"]["old_new_sha256_equal"] is True
    assert work_product["roundtrip_decode_validation_manifest"]["roundtrip_valid"] is True
    assert work_product["byte_accounted_model_overhead_reduction_manifest"][
        "accounting_closed"
    ] is True
    assert work_product["byte_accounted_model_overhead_reduction_manifest"][
        "target_bytes"
    ] == work_product["candidate_stream_section_manifest"]["stream"]["header_bytes"]
    assert work_product["byte_accounted_static_model_context_reduction_manifest"][
        "accounting_closed"
    ] is True
    assert work_product["byte_accounted_static_model_context_reduction_manifest"][
        "static_context_header_reduction_bytes"
    ] > 0
    assert work_product["old_new_model_context_table_diff"]["raw_equal"] is True
    assert work_product["old_new_model_context_table_diff"]["header_bytes_delta"] < 0
    assert work_product["remaining_blockers"][:2] == [
        "hdc2_runtime_decoder_contract_with_inflate_consumer_missing",
        "hdc2_archive_candidate_manifest_with_decoder_stream_consumed_missing",
    ]
    bounded = work_product["bounded_hdc2_recode_variants"]
    assert len(bounded) == 2
    hdm2 = next(
        row
        for row in bounded
        if row["variant"]
        == "mixed_range_raw_global_prev_symbol_schema_indexed_q_streams_plus_raw_scales"
    )
    hdm3 = next(
        row
        for row in bounded
        if row["variant"] == "hdm3_q_brotli_split_fixed_schema_q_stream_plus_raw_scales"
    )
    assert hdm2["variant"] == (
        "mixed_range_raw_global_prev_symbol_schema_indexed_q_streams_plus_raw_scales"
    )
    assert hdm2["raw_equal"] is True
    assert hdm2["q_roundtrip_equal"] is True
    assert hdm2["scale_roundtrip_equal"] is True
    assert hdm2["archive_ready"] is False
    assert hdm3["codec"] == "HDM3_fixed_schema_q_brotli_raw_scales"
    assert hdm3["raw_equal"] is True
    assert hdm3["q_roundtrip_equal"] is True
    assert hdm3["scale_roundtrip_equal"] is True
    assert hdm3["header_bytes"] == 7
    assert hdm3["q_brotli_bytes"] > 0
    assert hdm3["candidate_stream_file"]["bytes"] == hdm3["bytes"]

    assert manifest["invalid_requirement_artifacts"] == []
    assert SOURCE_ARCHIVE_REQUIREMENT_ID not in manifest["missing_artifacts"]
    assert SOURCE_STREAM_REQUIREMENT_ID not in manifest["missing_artifacts"]
    assert CANDIDATE_STREAM_REQUIREMENT_ID not in manifest["missing_artifacts"]
    assert DECODED_OUTPUT_EQUIVALENCE_REQUIREMENT_ID not in manifest["missing_artifacts"]
    assert ROUNDTRIP_REQUIREMENT_ID not in manifest["missing_artifacts"]
    assert "byte_accounted_model_overhead_reduction_manifest" not in manifest[
        "missing_artifacts"
    ]
    assert "byte_accounted_static_model_context_reduction_manifest" not in manifest[
        "missing_artifacts"
    ]
    assert "old_new_model_context_table_diff" not in manifest["missing_artifacts"]
    assert "candidate_archive_manifest_with_member_sha256s" in manifest["missing_artifacts"]
    assert "runtime_tree_parity_manifest" in manifest["missing_artifacts"]
    assert HDC2_RUNTIME_DECODER_CONTRACT_REQUIREMENT_ID in manifest["missing_artifacts"]
    assert HDC2_ARCHIVE_CANDIDATE_CONTRACT_REQUIREMENT_ID in manifest["missing_artifacts"]
    assert (
        f"missing_artifact:{HDC2_RUNTIME_DECODER_CONTRACT_REQUIREMENT_ID}"
        in manifest["readiness_blockers"]
    )
    assert (
        f"missing_artifact:{HDC2_ARCHIVE_CANDIDATE_CONTRACT_REQUIREMENT_ID}"
        in manifest["readiness_blockers"]
    )
    assert "missing_candidate_archive_manifest" in manifest["readiness_blockers"]
    assert "missing_runtime_tree_parity_manifest" in manifest["readiness_blockers"]
    assert manifest["ready_for_local_packet_review"] is False
    assert manifest["ready_for_archive_preflight"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    available_ids = {row["id"] for row in manifest["available_inputs"]}
    assert SOURCE_ARCHIVE_REQUIREMENT_ID in available_ids
    assert SOURCE_STREAM_REQUIREMENT_ID in available_ids
    assert CANDIDATE_STREAM_REQUIREMENT_ID in available_ids
    assert DECODED_OUTPUT_EQUIVALENCE_REQUIREMENT_ID in available_ids
    assert ROUNDTRIP_REQUIREMENT_ID in available_ids
    assert "byte_accounted_model_overhead_reduction_manifest" in available_ids
    assert "byte_accounted_static_model_context_reduction_manifest" in available_ids
    assert "old_new_model_context_table_diff" in available_ids


def test_candidate_packet_can_build_audit_from_stream_profile(tmp_path: Path) -> None:
    profile_path = tmp_path / "streams.json"
    profile_path.write_text(
        json.dumps(
            {
                "source_label": "pr106x_entropy_profile",
                "streams": _streams(),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    manifest = build_candidate_packet_manifest(profile_path, repo_root=REPO)

    assert manifest["audit_source_kind"] == "stream_profile_built_entropy_codec_gap_audit"
    assert manifest["audit_summary"]["source_label"] == "pr106x_entropy_profile"
    assert manifest["selected_target"]["rank"] == 1
    assert manifest["selected_target"]["label"] == "hnerv_decoder_weights"


def test_candidate_packet_can_adapt_hnerv_structural_recode_profile(tmp_path: Path) -> None:
    profile_path = tmp_path / "hnerv_decoder_recode_profile.json"
    profile_path.write_text(json_text(_structural_recode_profile()), encoding="utf-8")

    manifest = build_candidate_packet_manifest(profile_path, repo_root=REPO)

    assert manifest["audit_source_kind"] == "hnerv_structural_recode_profile_adapted_entropy_overhead_audit"
    assert manifest["audit_summary"]["tool"] == (
        "tac.hnerv_entropy_candidate_packet.hnerv_profile_entropy_overhead_adapter"
    )
    assert manifest["audit_summary"]["target_count"] == 2
    assert manifest["selected_target"]["rank"] == 1
    assert manifest["selected_target"]["target_kind"] == "known_model_overhead"
    assert manifest["selected_target"]["target_bytes"] == 40
    assert set(HDC2_DIRECT_ARCHIVE_RUNTIME_REQUIREMENTS).issubset(
        set(manifest["selected_target"]["row"]["exact_next_artifact_requirements"])
    )
    assert manifest["selected_target"]["row"]["source_variant"]["variant"] == (
        "range_prev_symbol_global_q_streams_plus_raw_scales"
    )
    assert "byte_accounted_model_overhead_reduction_manifest" in manifest["missing_artifacts"]
    assert manifest["ready_for_exact_eval_dispatch"] is False


def test_discovery_rejects_hnerv_profiles_without_entropy_stream_counts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    candidate_dir = root / "experiments" / "results" / "public_pr95_hnerv"
    candidate_dir.mkdir(parents=True)
    self_output_dir = root / "experiments" / "results" / "hnerv_entropy_packet_discovery"
    self_output_dir.mkdir(parents=True)
    (self_output_dir / "discovery_report.json").write_text(
        json_text(
            {
                "tool": "tac.hnerv_entropy_candidate_packet.discover_candidate_audit_inputs",
                "candidate_inputs": [],
            }
        ),
        encoding="utf-8",
    )
    profile_path = candidate_dir / "pr95_hnerv_muon_packing_profile.json"
    profile_path.write_text(
        json_text(
            {
                "archive_sha256": "a" * 64,
                "archive_bytes": 178417,
                "score_claim": False,
                "sections": [
                    {
                        "name": "decoder_brotli",
                        "raw_bytes": 230048,
                        "compressed_bytes": 162349,
                        "entropy_bits_per_byte": 7.998084,
                        "sha256": "b" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = discover_candidate_audit_inputs(repo_root=root)

    assert report["tool"] == "tac.hnerv_entropy_candidate_packet.discover_candidate_audit_inputs"
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["ready_for_packet_materialization"] is False
    assert report["candidate_input_count"] == 1
    assert report["valid_input_count"] == 0
    assert report["selected_entropy_audit"] is None
    assert "hnerv_entropy_codec_gap_audit_json_with_entropy_overhead_target_ranking" in report[
        "missing_source_artifacts"
    ]
    candidate = report["candidate_inputs"][0]
    assert candidate["valid"] is False
    assert candidate["source_json"]["bytes"] == profile_path.stat().st_size
    assert candidate["source_json"]["sha256"] == sha256_file(profile_path)
    assert candidate["missing_required_fields"] == [
        "entropy_overhead_target_ranking",
        "streams",
    ]
    assert candidate["missing_data"]["classification"] == "hnerv_section_profile_summary_only"
    assert "streams[*].symbol_counts_full_histogram" in candidate["missing_data"]["required_inputs"]
    assert "sections[*].entropy_bits_per_byte is a lossy summary" in candidate["missing_data"]["notes"][0]
    assert report["missing_data_report"]["valid_entropy_audit_available"] is False
    assert profile_path.as_posix().endswith(
        report["missing_data_report"]["candidate_source_files"][0]["path"]
    )
    assert "entropy audit/profile JSON must contain" in candidate["rejection_reason"]
    assert discovery_report_input_paths(report, root) == [profile_path]


def test_discovery_selects_adapted_hnerv_structural_recode_profile(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    profile_dir = root / "experiments" / "results" / "hnerv_decoder_recode"
    profile_dir.mkdir(parents=True)
    profile_path = profile_dir / "profile.json"
    profile_path.write_text(json_text(_structural_recode_profile()), encoding="utf-8")

    report = discover_candidate_audit_inputs(repo_root=root)

    assert report["candidate_input_count"] == 1
    assert report["valid_input_count"] == 1
    assert report["ready_for_packet_materialization"] is True
    assert report["missing_source_artifacts"] == []
    assert report["missing_data_report"]["valid_entropy_audit_available"] is True
    assert report["selected_entropy_audit"]["path"] == "experiments/results/hnerv_decoder_recode/profile.json"
    candidate = report["candidate_inputs"][0]
    assert candidate["audit_source_kind"] == "hnerv_structural_recode_profile_adapted_entropy_overhead_audit"
    assert candidate["audit_summary"]["target_count"] == 2
    assert candidate["selected_target"]["target_kind"] == "known_model_overhead"
    assert candidate["selected_target"]["target_bytes"] == 40


def test_discovery_selects_valid_stream_profile_by_repo_relative_path(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    invalid_dir = root / "experiments" / "results" / "a_hnerv_profile"
    valid_dir = root / "experiments" / "results" / "b_hnerv_entropy"
    invalid_dir.mkdir(parents=True)
    valid_dir.mkdir(parents=True)
    (invalid_dir / "invalid_hnerv_profile.json").write_text(
        json_text({"sections": []}),
        encoding="utf-8",
    )
    valid_path = valid_dir / "valid_hnerv_stream_profile.json"
    valid_path.write_text(
        json_text({"source_label": "valid_hnerv", "streams": _streams()}),
        encoding="utf-8",
    )

    report = discover_candidate_audit_inputs(repo_root=root)

    assert report["candidate_input_count"] == 2
    assert report["valid_input_count"] == 1
    assert report["ready_for_packet_materialization"] is True
    assert report["missing_source_artifacts"] == []
    assert report["selected_entropy_audit"]["path"] == (
        "experiments/results/b_hnerv_entropy/valid_hnerv_stream_profile.json"
    )
    valid = next(row for row in report["candidate_inputs"] if row["valid"] is True)
    assert valid["audit_source_kind"] == "stream_profile_built_entropy_codec_gap_audit"
    assert valid["audit_summary"]["stream_count"] == 2
    assert valid["selected_target"]["label"] == "hnerv_decoder_weights"


def test_candidate_packet_rejects_unknown_artifact_for_selected_target(tmp_path: Path) -> None:
    audit_path = tmp_path / "entropy_audit.json"
    audit_path.write_text(json_text(_audit()), encoding="utf-8")

    with pytest.raises(HnervEntropyCandidatePacketError, match="unknown selected-target requirement"):
        build_candidate_packet_manifest(
            audit_path,
            artifact_paths={"not_a_requirement": tmp_path / "x.json"},
            repo_root=REPO,
        )


def test_build_hnerv_entropy_candidate_packet_cli_writes_manifest(tmp_path: Path) -> None:
    audit_path = tmp_path / "entropy_audit.json"
    json_out = tmp_path / "packet.json"
    audit_path.write_text(json_text(_audit()), encoding="utf-8")
    source_archive_manifest = tmp_path / "source_archive_manifest.json"
    source_archive_manifest.write_text('{"archive_sha256":"fixture"}\n', encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_hnerv_entropy_candidate_packet.py"),
            "--entropy-audit",
            str(audit_path),
            "--source-archive-manifest",
            str(source_archive_manifest),
            "--json-out",
            str(json_out),
            "--fail-if-missing",
        ],
        cwd=REPO,
        text=True,
        check=False,
        capture_output=True,
    )

    assert proc.returncode == 1
    assert "candidate packet missing required artifacts" in proc.stderr
    assert "source_archive_manifest_with_archive_sha256_bytes_and_runtime_tree_sha256" in proc.stderr
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["tool_run_manifest"]["tool"] == "tools/build_hnerv_entropy_candidate_packet.py"
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["packet_requirements"][0]["id"] == "roundtrip_payload_recode_manifest"
    assert "source_archive_manifest_with_archive_sha256_bytes_and_runtime_tree_sha256" in payload[
        "invalid_requirement_artifacts"
    ]
    source_record = next(
        row
        for row in payload["packet_requirements"]
        if row["id"] == "source_archive_manifest_with_archive_sha256_bytes_and_runtime_tree_sha256"
    )
    assert source_record["missing_reason"] == "validation_blockers"


def test_build_hnerv_entropy_candidate_packet_cli_materializes_hdc2_stream_work_product(
    tmp_path: Path,
) -> None:
    fixture = _write_hdc2_fixture(tmp_path)
    output_dir = tmp_path / "hdc2"
    packet_out = tmp_path / "packet.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_hnerv_entropy_candidate_packet.py"),
            "--entropy-audit",
            str(fixture["profile_path"]),
            "--hdc2-stream-work-product-profile",
            str(fixture["profile_path"]),
            "--hdc2-stream-work-product-source-archive",
            str(fixture["archive_path"]),
            "--hdc2-stream-work-product-source-exact-eval-json",
            str(fixture["exact_eval_path"]),
            "--hdc2-stream-work-product-dir",
            str(output_dir),
            "--json-out",
            str(packet_out),
            "--fail-if-missing",
        ],
        cwd=REPO,
        text=True,
        check=False,
        capture_output=True,
    )

    assert proc.returncode == 1
    assert "candidate packet missing required artifacts" in proc.stderr
    assert "candidate_archive_manifest_with_member_sha256s" in proc.stderr
    packet = json.loads(packet_out.read_text(encoding="utf-8"))
    assert (output_dir / "candidate_hdc2_global_prev_symbol_stream.bin").is_file()
    assert (output_dir / "candidate_hdm3_q_brotli_split_stream.bin").is_file()
    assert (
        output_dir / f"{SOURCE_STREAM_REQUIREMENT_ID}.json"
    ).is_file()
    assert (
        output_dir / "byte_accounted_model_overhead_reduction_manifest.json"
    ).is_file()
    assert (output_dir / "old_new_model_context_table_diff.json").is_file()
    assert SOURCE_STREAM_REQUIREMENT_ID not in packet["missing_artifacts"]
    assert CANDIDATE_STREAM_REQUIREMENT_ID not in packet["missing_artifacts"]
    assert ROUNDTRIP_REQUIREMENT_ID not in packet["missing_artifacts"]
    assert "byte_accounted_model_overhead_reduction_manifest" not in packet[
        "missing_artifacts"
    ]
    assert "old_new_model_context_table_diff" not in packet["missing_artifacts"]
    assert "candidate_archive_manifest_with_member_sha256s" in packet["missing_artifacts"]
    assert HDC2_RUNTIME_DECODER_CONTRACT_REQUIREMENT_ID in packet["missing_artifacts"]
    assert HDC2_ARCHIVE_CANDIDATE_CONTRACT_REQUIREMENT_ID in packet["missing_artifacts"]
    assert packet["ready_for_exact_eval_dispatch"] is False


def test_build_hnerv_entropy_candidate_packet_cli_requires_hdc2_exact_eval_json(
    tmp_path: Path,
) -> None:
    fixture = _write_hdc2_fixture(tmp_path)
    output_dir = tmp_path / "hdc2"
    packet_out = tmp_path / "packet.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_hnerv_entropy_candidate_packet.py"),
            "--entropy-audit",
            str(fixture["profile_path"]),
            "--hdc2-stream-work-product-profile",
            str(fixture["profile_path"]),
            "--hdc2-stream-work-product-source-archive",
            str(fixture["archive_path"]),
            "--hdc2-stream-work-product-dir",
            str(output_dir),
            "--json-out",
            str(packet_out),
        ],
        cwd=REPO,
        text=True,
        check=False,
        capture_output=True,
    )

    assert proc.returncode == 2
    assert "--hdc2-stream-work-product-source-exact-eval-json" in proc.stderr
    assert not packet_out.exists()


def test_build_hnerv_entropy_candidate_packet_cli_discovers_missing_source_inputs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "scan"
    root.mkdir()
    profile_path = root / "orphan_hnerv_profile.json"
    profile_path.write_text(json_text({"sections": []}), encoding="utf-8")
    json_out = tmp_path / "discovery.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_hnerv_entropy_candidate_packet.py"),
            "--search-root",
            str(root),
            "--json-out",
            str(json_out),
            "--fail-if-missing",
        ],
        cwd=REPO,
        text=True,
        check=False,
        capture_output=True,
    )

    assert proc.returncode == 1
    assert "discovery missing source artifacts" in proc.stderr
    assert "hnerv_entropy_codec_gap_audit_json_with_entropy_overhead_target_ranking" in proc.stderr
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["tool"] == "tac.hnerv_entropy_candidate_packet.discover_candidate_audit_inputs"
    assert payload["valid_input_count"] == 0
    assert payload["candidate_inputs"][0]["source_json"]["sha256"] == sha256_file(profile_path)
    assert "streams" in payload["candidate_inputs"][0]["missing_required_fields"]
    assert payload["tool_run_manifest"]["input_files"][0]["sha256"] == sha256_file(profile_path)


def test_build_hnerv_entropy_candidate_packet_cli_materializes_adapted_audit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "scan"
    root.mkdir()
    profile_path = root / "hnerv_decoder_recode_profile.json"
    profile_path.write_text(json_text(_structural_recode_profile()), encoding="utf-8")
    audit_out = tmp_path / "entropy_audit.json"
    packet_out = tmp_path / "packet.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_hnerv_entropy_candidate_packet.py"),
            "--search-root",
            str(root),
            "--entropy-audit-json-out",
            str(audit_out),
            "--json-out",
            str(packet_out),
            "--fail-if-missing",
        ],
        cwd=REPO,
        text=True,
        check=False,
        capture_output=True,
    )

    assert proc.returncode == 1
    assert "candidate packet missing required artifacts" in proc.stderr
    assert "candidate_archive_manifest_with_member_sha256s" in proc.stderr
    audit = json.loads(audit_out.read_text(encoding="utf-8"))
    packet = json.loads(packet_out.read_text(encoding="utf-8"))
    assert audit["tool"] == "tac.hnerv_entropy_candidate_packet.hnerv_profile_entropy_overhead_adapter"
    assert audit["audit_source_kind"] == "hnerv_structural_recode_profile_adapted_entropy_overhead_audit"
    assert audit["entropy_overhead_target_ranking"][0]["target_kind"] == "known_model_overhead"
    assert set(HDC2_DIRECT_ARCHIVE_RUNTIME_REQUIREMENTS).issubset(
        set(audit["entropy_overhead_target_ranking"][0]["exact_next_artifact_requirements"])
    )
    assert audit["ready_for_exact_eval_dispatch"] is False
    assert packet["audit_source_kind"] == "entropy_codec_gap_audit"
    assert packet["audit_source"]["path"].endswith("entropy_audit.json")
    assert packet["selected_target"]["target_kind"] == "known_model_overhead"
    assert packet["ready_for_exact_eval_dispatch"] is False


def _audit() -> dict:
    return build_entropy_codec_gap_audit(_streams(), source_label="hnerv_pr106x")


def _streams() -> list[dict]:
    return [
        {
            "label": "hnerv_decoder_weights",
            "actual_bytes": 512,
            "encoded_payload_bytes": 400,
            "model_overhead_bytes": 96,
            "container_overhead_bytes": 16,
            "symbol_counts": {"0": 300, "1": 40, "2": 8, "3": 4},
            "codec_surface": "src/tac/hnerv_decoder_recode.py",
        },
        {
            "label": "hnerv_latent_sidecar",
            "actual_bytes": 128,
            "symbol_counts": {"0": 90, "1": 6, "2": 1, "3": 1},
            "codec_surface": "src/tac/hnerv_decoder_recode.py",
        },
    ]


def _write_hdc2_fixture(tmp_path: Path) -> dict[str, Path]:
    source_raw = _packed_decoder_raw_fixture()
    source_brotli = brotli.compress(source_raw, quality=11)
    packed = PackedHnervPayload(
        header=b"",
        decoder_packed_brotli=source_brotli,
        latents_and_sidecar_brotli=b"latent-sidecar",
    )
    archive_path = tmp_path / "source.zip"
    write_stored_single_member_zip(archive_path, member_name="0.bin", payload=packed.to_bytes())
    parsed = parse_packed_decoder_brotli(source_brotli)
    hdc2_payload, stats = encode_global_prev_symbol_context_range_fixture(parsed)
    hdm2_payload, hdm2_stats = encode_global_prev_symbol_mixed_context_fixture(parsed)
    hdm3_payload, hdm3_stats = encode_hdm3_q_brotli_split_fixture(parsed)
    profile = {
        "schema_version": 1,
        "tool": "tac.hnerv_decoder_recode.build_structural_recode_profile",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_archive_preflight": False,
        "ready_for_exact_eval_dispatch": False,
        "source_label": "fixture_hnerv_archive",
        "source_archive_sha256": sha256_file(archive_path),
        "source_decoder_section_sha256": sha256_bytes(source_brotli),
        "source_decoder_section_bytes": len(source_brotli),
        "source_decoder_raw_sha256": sha256_bytes(source_raw),
        "source_decoder_raw_bytes": len(source_raw),
        "record_count": len(PACKED_STATE_SCHEMA),
        "q_stream_bytes": len(parsed.q_stream),
        "scale_stream_bytes": len(parsed.scale_stream),
        "entropy_summary": {
            "score_claim": False,
            "per_tensor_prev_symbol_entropy_floor_plus_raw_scales_bytes": max(
                1,
                len(hdc2_payload) - 7,
            ),
        },
        "variants": [
            {
                "variant": "range_prev_symbol_global_q_streams_plus_raw_scales",
                "codec": "HDC2_global_prev_symbol_range_uint8",
                "bytes": len(hdc2_payload),
                "header_bytes": int(stats["header_bytes"]),
                "range_payload_bytes": int(stats["range_payload_bytes"]),
                "raw_scale_bytes": len(parsed.scale_stream),
                "raw_equal": True,
                "q_roundtrip_equal": True,
                "scale_roundtrip_equal": True,
                "sha256": sha256_bytes(hdc2_payload),
                "context_count": int(stats["context_count"]),
                "context_token_count": int(stats["context_token_count"]),
            },
            {
                "variant": (
                    "mixed_range_raw_global_prev_symbol_schema_indexed_q_streams_plus_raw_scales"
                ),
                "codec": "HDM2_global_prev_symbol_mixed_range_raw_schema_indexed_uint8",
                "bytes": len(hdm2_payload),
                "header_bytes": int(hdm2_stats["header_bytes"]),
                "range_payload_bytes": int(hdm2_stats["range_payload_bytes"]),
                "raw_payload_bytes": int(hdm2_stats["raw_payload_bytes"]),
                "mixed_payload_bytes": int(hdm2_stats["mixed_payload_bytes"]),
                "raw_scale_bytes": len(parsed.scale_stream),
                "raw_equal": True,
                "q_roundtrip_equal": True,
                "scale_roundtrip_equal": True,
                "sha256": sha256_bytes(hdm2_payload),
                "context_count": int(hdm2_stats["context_count"]),
                "context_token_count": int(hdm2_stats["context_token_count"]),
                "raw_context_count": int(hdm2_stats["raw_context_count"]),
                "range_context_count": int(hdm2_stats["range_context_count"]),
                "schema_metadata_elided_vs_hdc2_bytes": int(
                    hdm2_stats["schema_metadata_elided_vs_hdc2_bytes"]
                ),
            },
            {
                "variant": "hdm3_q_brotli_split_fixed_schema_q_stream_plus_raw_scales",
                "codec": "HDM3_fixed_schema_q_brotli_raw_scales",
                "bytes": len(hdm3_payload),
                "header_bytes": int(hdm3_stats["header_bytes"]),
                "q_brotli_bytes": int(hdm3_stats["q_brotli_bytes"]),
                "q_stream_bytes": int(hdm3_stats["q_stream_bytes"]),
                "raw_scale_bytes": int(hdm3_stats["raw_scale_bytes"]),
                "raw_equal": True,
                "q_roundtrip_equal": True,
                "scale_roundtrip_equal": True,
                "sha256": sha256_bytes(hdm3_payload),
            }
        ],
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json_text(profile), encoding="utf-8")
    exact_eval_path = tmp_path / "contest_auth_eval.json"
    exact_eval_path.write_text(
        json_text(
            {
                "score_claim": False,
                "provenance": {
                    "archive_sha256": sha256_file(archive_path),
                    "archive_size_bytes": archive_path.stat().st_size,
                    "inflate_runtime_manifest": {
                        "runtime_tree_sha256": "e" * 64,
                        "files": [],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return {
        "archive_path": archive_path,
        "profile_path": profile_path,
        "exact_eval_path": exact_eval_path,
    }


def _packed_decoder_raw_fixture() -> bytes:
    q_parts = []
    scale_parts = []
    for index, (_name, shape) in enumerate(PACKED_STATE_SCHEMA):
        count = math.prod(shape)
        q_parts.append(bytes((index * 17 + offset) % 251 for offset in range(count)))
        scale_parts.append(bytes([index, index + 1, index + 2, index + 3]))
    return b"".join(q_parts) + b"".join(scale_parts)


def _write_hdc2_requirement_artifacts(
    tmp_path: Path,
    work_product: dict,
) -> dict[str, Path]:
    artifacts = {}
    for name, requirement_id in HDC2_STREAM_ARTIFACT_REQUIREMENTS.items():
        path = tmp_path / f"{requirement_id}.json"
        path.write_text(json_text(work_product[name]), encoding="utf-8")
        artifacts[requirement_id] = path
    return artifacts


def _structural_recode_profile() -> dict:
    return {
        "schema_version": 1,
        "tool": "tac.hnerv_decoder_recode.build_structural_recode_profile",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_archive_preflight": False,
        "ready_for_exact_eval_dispatch": False,
        "source_label": "fixture_hnerv_pr106",
        "source_archive_sha256": "a" * 64,
        "source_decoder_section_sha256": "b" * 64,
        "source_decoder_section_bytes": 90,
        "source_decoder_raw_sha256": "c" * 64,
        "source_decoder_raw_bytes": 120,
        "record_count": 2,
        "q_stream_bytes": 116,
        "scale_stream_bytes": 4,
        "entropy_summary": {
            "score_claim": False,
            "per_tensor_prev_symbol_entropy_floor_plus_raw_scales_bytes": 50,
        },
        "variants": [
            {
                "variant": "range_prev_symbol_global_q_streams_plus_raw_scales",
                "codec": "HDC2_global_prev_symbol_range_uint8",
                "bytes": 100,
                "header_bytes": 40,
                "range_payload_bytes": 55,
                "raw_scale_bytes": 5,
                "raw_equal": True,
                "q_roundtrip_equal": True,
                "scale_roundtrip_equal": True,
                "sha256": "d" * 64,
            }
        ],
    }
