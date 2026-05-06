from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.hnerv_entropy_candidate_packet import (
    HnervEntropyCandidatePacketError,
    build_candidate_packet_manifest,
    discover_candidate_audit_inputs,
    discovery_report_input_paths,
)
from tac.optimization.entropy_codec_gap_audit import build_entropy_codec_gap_audit
from tac.repo_io import json_text, sha256_file

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


def test_candidate_packet_hashes_all_available_requirements_but_stays_non_dispatchable(
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

    assert manifest["missing_artifacts"] == []
    assert manifest["readiness_blockers"] == []
    assert manifest["ready_for_local_packet_review"] is True
    assert manifest["ready_for_archive_preflight"] is True
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["dispatch_blockers"] == [
        "packet_manifest_is_not_dispatch_authorization",
        "requires_operator_review_of_byte_equivalence_runtime_parity_and_archive_manifest",
        "requires_lane_dispatch_claim_before_gpu",
        "requires_exact_cuda_auth_eval",
    ]
    records = {row["id"]: row for row in manifest["packet_requirements"]}
    for requirement_id, artifact in artifacts.items():
        assert records[requirement_id]["available"] is True
        assert records[requirement_id]["sha256"] == sha256_file(artifact)
    assert {row["id"] for row in manifest["available_inputs"]} == {"entropy_audit_json", *artifacts}


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
    assert "entropy audit/profile JSON must contain" in candidate["rejection_reason"]
    assert discovery_report_input_paths(report, root) == [profile_path]


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
    )

    assert proc.returncode == 1
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["tool_run_manifest"]["tool"] == "tools/build_hnerv_entropy_candidate_packet.py"
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["packet_requirements"][0]["id"] == "roundtrip_payload_recode_manifest"
    assert any(row["id"] == "source_archive_manifest_with_archive_sha256_bytes_and_runtime_tree_sha256" for row in payload["available_inputs"])


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
    )

    assert proc.returncode == 1
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["tool"] == "tac.hnerv_entropy_candidate_packet.discover_candidate_audit_inputs"
    assert payload["valid_input_count"] == 0
    assert payload["candidate_inputs"][0]["source_json"]["sha256"] == sha256_file(profile_path)
    assert "streams" in payload["candidate_inputs"][0]["missing_required_fields"]
    assert payload["tool_run_manifest"]["input_files"][0]["sha256"] == sha256_file(profile_path)


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
