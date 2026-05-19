# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tac.master_gradient_pr101_score_response_matrix import (
    build_pr101_pose_axis_score_response_matrix,
    render_pr101_pose_axis_score_response_matrix_markdown,
)
from tac.repo_io import sha256_file


def _runtime_manifest() -> dict:
    return {
        "schema": "contest_auth_eval_runtime_dependency_manifest_v1",
        "runtime_root": "/tmp/local/submission_dir",
        "runtime_file_count": 2,
        "runtime_tree_sha256": "local_tree",
        "runtime_content_tree_sha256": "content_tree",
        "files": [],
        "external_dependency_roots": [],
        "repo_local_tac_import_manifest": {
            "schema": "contest_auth_eval_repo_local_tac_import_manifest_v1",
            "runtime_root_name": "submission_dir",
            "file_count": 0,
            "files": [],
        },
        "upstream_evaluate_py": {"relative_path": "evaluate.py", "bytes": 3, "sha256": "c" * 64},
    }


def _inputs(tmp_path: Path) -> dict:
    source_archive = tmp_path / "source.zip"
    candidate_archive = tmp_path / "candidate.zip"
    source_archive.write_bytes(b"source archive bytes")
    candidate_archive.write_bytes(b"source archive bytes")
    submission_dir = tmp_path / "submission"
    submission_dir.mkdir()
    (submission_dir / "inflate.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    source_sha = sha256_file(source_archive)
    candidate_sha = sha256_file(candidate_archive)
    source_operator_payload = {
        "schema": "pose_byte_hoist_op7_manifest_v1",
        "blockers": ["packet_proofs_missing"],
        "source_anchor": {
            "score_axis_dominance_available": True,
            "scored_archive_sha256": source_sha,
            "scored_archive_bytes": source_archive.stat().st_size,
        },
    }
    source_operator_manifest = tmp_path / "source_operator_manifest.json"
    source_operator_manifest.write_text(
        json.dumps(source_operator_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    operator_manifest = {
        "schema": "tac_pr101_pose_axis_decoder_recompression_candidate_v1",
        "candidate_id": "unit-pr101-op7",
        "mutation_mode": "raw_byte_delta",
        "mutation_operator": "decoder_codec_coordinate_response",
        "component_moving_candidate": True,
        "semantic_equivalence_expected": False,
        "ready_for_score_response_probe": False,
        "source_archive": {
            "path": source_archive.as_posix(),
            "bytes": source_archive.stat().st_size,
            "sha256": source_sha,
        },
        "source_operator_manifest": {
            "path": source_operator_manifest.as_posix(),
            "sha256": sha256_file(source_operator_manifest),
            "schema": "pose_byte_hoist_op7_manifest_v1",
        },
        "candidate_archive": {
            "path": candidate_archive.as_posix(),
            "bytes": candidate_archive.stat().st_size,
            "sha256": candidate_sha,
        },
        "selected_pose_axis_candidate": {"rank": 1},
        "replacement_stream": {"raw_mutation": {"mutation_kind": "single_raw_byte_delta"}},
        "score_claim": False,
    }
    packet_manifest = {
        "schema": "tac_monolithic_packet_candidate_v1",
        "candidate_id": "unit-pr101-op7",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "candidate_archive": {
            "path": candidate_archive.as_posix(),
            "bytes": candidate_archive.stat().st_size,
            "sha256": candidate_sha,
        },
    }
    runtime_proof = {
        "schema": "tac_runtime_consumption_proof_v1",
        "candidate_id": "unit-pr101-op7",
        "candidate_archive_sha256": candidate_sha,
        "runtime_grammar": "pr101_fixed_offset_hnerv_microcodec",
        "ready_for_exact_eval_runtime": True,
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "blockers": [],
    }
    return {
        "source_archive": source_archive,
        "candidate_archive": candidate_archive,
        "submission_dir": submission_dir,
        "operator_manifest": operator_manifest,
        "packet_manifest": packet_manifest,
        "runtime_proof": runtime_proof,
        "source_operator_manifest": source_operator_manifest,
    }


def _write_source_operator_manifest(
    data: dict,
    payload: dict,
) -> None:
    path = data["source_operator_manifest"]
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    data["operator_manifest"]["source_operator_manifest"]["sha256"] = sha256_file(path)


def test_pr101_pose_axis_score_response_matrix_pairs_contest_axes(tmp_path: Path) -> None:
    data = _inputs(tmp_path)

    matrix = build_pr101_pose_axis_score_response_matrix(
        source_archive=data["source_archive"],
        source_submission_dir=data["submission_dir"],
        operator_candidate_manifest=data["operator_manifest"],
        packet_candidate_manifest=data["packet_manifest"],
        runtime_consumption_proof=data["runtime_proof"],
        runtime_manifest=_runtime_manifest(),
        repo_root=Path.cwd(),
        label="unit-op7",
        lane_id="unit_op7_lane",
        output_root="experiments/results/unit_score_response",
        include_diagnostics=False,
    )

    pairs = {pair["target_id"]: pair for pair in matrix["target_pairs"]}
    assert matrix["schema"] == "pr101_pose_axis_score_response_matrix_v1"
    assert matrix["score_claim"] is False
    assert matrix["score_claim_valid"] is False
    assert matrix["promotion_eligible"] is False
    assert matrix["ready_for_provider_dispatch"] is False
    assert matrix["ready_for_exact_eval_dispatch"] is False
    assert matrix["raw_archive_byte_coordinates_allowed"] is False
    assert matrix["candidate_specs_are_dispatchable"] is False
    assert matrix["ready_for_score_response_probe"] is False
    assert matrix["ready_for_score_response_probe_after_exact_eval"] is True
    assert matrix["candidate_archive"]["archive_byte_delta"] == 0
    assert set(pairs) == {
        "local_cpu_current_host_auto",
        "modal_contest_cpu_linux_x86_auto",
        "modal_contest_cuda_t4_auto",
    }
    cuda = pairs["modal_contest_cuda_t4_auto"]
    assert cuda["score_axis"] == "contest_cuda"
    assert cuda["contest_compliant"] is True
    assert "--axis" in cuda["score_response_probe_command"]
    assert "contest_cuda" in cuda["score_response_probe_command"]
    assert "modal_contest_cuda_t4_auto_baseline_exact_eval_json_missing" in matrix[
        "score_response_blockers"
    ]
    assert "modal_contest_cpu_linux_x86_auto_active_lane_claim_missing" in matrix[
        "dispatch_blockers"
    ]
    assert "full_inflate_output_manifest_missing_until_auth_eval" in matrix[
        "dispatch_blockers"
    ]
    assert "paired_contest_cuda_cpu_exact_eval_missing" in matrix["dispatch_blockers"]


def test_pr101_pose_axis_score_response_matrix_blocks_non_component_moving(
    tmp_path: Path,
) -> None:
    data = _inputs(tmp_path)
    data["operator_manifest"]["component_moving_candidate"] = False

    matrix = build_pr101_pose_axis_score_response_matrix(
        source_archive=data["source_archive"],
        source_submission_dir=data["submission_dir"],
        operator_candidate_manifest=data["operator_manifest"],
        packet_candidate_manifest=data["packet_manifest"],
        runtime_consumption_proof=data["runtime_proof"],
        runtime_manifest=_runtime_manifest(),
        repo_root=Path.cwd(),
        label="unit-op7",
        lane_id="unit_op7_lane",
        output_root="experiments/results/unit_score_response",
        include_diagnostics=False,
    )

    assert "operator_candidate_not_component_moving" in matrix["authority_blockers"]
    assert matrix["ready_for_score_response_probe_after_exact_eval"] is False


def test_pr101_pose_axis_score_response_matrix_blocks_missing_source_operator_manifest(
    tmp_path: Path,
) -> None:
    data = _inputs(tmp_path)
    del data["operator_manifest"]["source_operator_manifest"]

    matrix = build_pr101_pose_axis_score_response_matrix(
        source_archive=data["source_archive"],
        source_submission_dir=data["submission_dir"],
        operator_candidate_manifest=data["operator_manifest"],
        packet_candidate_manifest=data["packet_manifest"],
        runtime_consumption_proof=data["runtime_proof"],
        runtime_manifest=_runtime_manifest(),
        repo_root=Path.cwd(),
        label="unit-op7",
        lane_id="unit_op7_lane",
        output_root="experiments/results/unit_score_response",
        include_diagnostics=False,
    )

    assert "source_operator_manifest_missing" in matrix["authority_blockers"]
    assert matrix["ready_for_score_response_probe_after_exact_eval"] is False


def test_pr101_pose_axis_score_response_matrix_blocks_source_dominance_gaps(
    tmp_path: Path,
) -> None:
    data = _inputs(tmp_path)
    data["operator_manifest"]["source_operator_manifest"]["sha256"] = "0" * 64
    _write_source_operator_manifest(
        data,
        {
            "schema": "pose_byte_hoist_op7_manifest_v1",
            "blockers": ["anchor_score_axis_dominance_not_persisted"],
            "source_anchor": {
                "score_axis_dominance_available": False,
            },
        },
    )
    data["operator_manifest"]["source_operator_manifest"]["sha256"] = "0" * 64

    matrix = build_pr101_pose_axis_score_response_matrix(
        source_archive=data["source_archive"],
        source_submission_dir=data["submission_dir"],
        operator_candidate_manifest=data["operator_manifest"],
        packet_candidate_manifest=data["packet_manifest"],
        runtime_consumption_proof=data["runtime_proof"],
        runtime_manifest=_runtime_manifest(),
        repo_root=Path.cwd(),
        label="unit-op7",
        lane_id="unit_op7_lane",
        output_root="experiments/results/unit_score_response",
        include_diagnostics=False,
    )

    blockers = matrix["authority_blockers"]
    assert "source_operator_manifest_sha256_mismatch" in blockers
    assert "anchor_score_axis_dominance_not_persisted" in blockers
    assert "source_anchor_score_axis_dominance_available_not_true" in blockers
    assert "source_anchor_scored_archive_sha256_missing" in blockers
    assert "source_anchor_scored_archive_bytes_missing" in blockers
    assert matrix["ready_for_score_response_probe_after_exact_eval"] is False


def test_pr101_pose_axis_score_response_matrix_blocks_scored_archive_custody_mismatch(
    tmp_path: Path,
) -> None:
    data = _inputs(tmp_path)
    _write_source_operator_manifest(
        data,
        {
            "schema": "pose_byte_hoist_op7_manifest_v1",
            "blockers": [],
            "source_anchor": {
                "score_axis_dominance_available": True,
                "scored_archive_sha256": "a" * 64,
                "scored_archive_bytes": data["source_archive"].stat().st_size + 1,
            },
        },
    )

    matrix = build_pr101_pose_axis_score_response_matrix(
        source_archive=data["source_archive"],
        source_submission_dir=data["submission_dir"],
        operator_candidate_manifest=data["operator_manifest"],
        packet_candidate_manifest=data["packet_manifest"],
        runtime_consumption_proof=data["runtime_proof"],
        runtime_manifest=_runtime_manifest(),
        repo_root=Path.cwd(),
        label="unit-op7",
        lane_id="unit_op7_lane",
        output_root="experiments/results/unit_score_response",
        include_diagnostics=False,
    )

    assert "source_anchor_scored_archive_sha256_mismatch" in matrix["authority_blockers"]
    assert "source_anchor_scored_archive_bytes_mismatch" in matrix["authority_blockers"]
    assert matrix["ready_for_score_response_probe_after_exact_eval"] is False


def test_pr101_pose_axis_score_response_matrix_blocks_authority_leaks(
    tmp_path: Path,
) -> None:
    data = _inputs(tmp_path)
    data["operator_manifest"]["score_claim"] = True
    data["operator_manifest"]["promotion_eligible"] = True
    data["operator_manifest"]["raw_archive_byte_coordinates_allowed"] = True
    data["operator_manifest"]["candidate_specs_are_dispatchable"] = True
    data["packet_manifest"]["rank_or_kill_eligible"] = True
    data["runtime_proof"]["promotion_eligible"] = True

    matrix = build_pr101_pose_axis_score_response_matrix(
        source_archive=data["source_archive"],
        source_submission_dir=data["submission_dir"],
        operator_candidate_manifest=data["operator_manifest"],
        packet_candidate_manifest=data["packet_manifest"],
        runtime_consumption_proof=data["runtime_proof"],
        runtime_manifest=_runtime_manifest(),
        repo_root=Path.cwd(),
        label="unit-op7",
        lane_id="unit_op7_lane",
        output_root="experiments/results/unit_score_response",
        include_diagnostics=False,
    )

    assert "operator_candidate_score_claim_not_false" in matrix["authority_blockers"]
    assert "operator_candidate_promotion_eligible_true" in matrix["authority_blockers"]
    assert (
        "operator_candidate_raw_archive_byte_coordinates_allowed_true"
        in matrix["authority_blockers"]
    )
    assert "operator_candidate_specs_are_dispatchable_true" in matrix["authority_blockers"]
    assert "packet_candidate_rank_or_kill_eligible_true" in matrix["authority_blockers"]
    assert "runtime_consumption_proof_promotion_eligible_true" in matrix[
        "authority_blockers"
    ]
    assert matrix["score_claim"] is False


def test_pr101_pose_axis_score_response_matrix_markdown_is_authority_labeled(
    tmp_path: Path,
) -> None:
    data = _inputs(tmp_path)
    matrix = build_pr101_pose_axis_score_response_matrix(
        source_archive=data["source_archive"],
        source_submission_dir=data["submission_dir"],
        operator_candidate_manifest=data["operator_manifest"],
        packet_candidate_manifest=data["packet_manifest"],
        runtime_consumption_proof=data["runtime_proof"],
        runtime_manifest=_runtime_manifest(),
        repo_root=Path.cwd(),
        label="unit-op7",
        lane_id="unit_op7_lane",
        output_root="experiments/results/unit_score_response",
        include_diagnostics=False,
    )

    text = render_pr101_pose_axis_score_response_matrix_markdown(matrix)

    assert "score_claim: false" in text
    assert "promotion_eligible: false" in text
    assert "modal_contest_cuda_t4_auto" in text


def test_pr101_pose_axis_score_response_matrix_cli_writes_json_and_md(
    tmp_path: Path,
) -> None:
    data = _inputs(tmp_path)
    for name, payload in (
        ("operator.json", data["operator_manifest"]),
        ("candidate.json", data["packet_manifest"]),
        ("runtime.json", data["runtime_proof"]),
    ):
        (tmp_path / name).write_text(
            json.dumps(payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    out = tmp_path / "matrix.json"
    md = tmp_path / "matrix.md"

    proc = subprocess.run(
        [
            sys.executable,
            str(Path.cwd() / "tools/build_pr101_pose_axis_score_response_matrix.py"),
            "--source-archive",
            data["source_archive"].as_posix(),
            "--source-submission-dir",
            data["submission_dir"].as_posix(),
            "--operator-manifest",
            (tmp_path / "operator.json").as_posix(),
            "--candidate-manifest",
            (tmp_path / "candidate.json").as_posix(),
            "--runtime-proof",
            (tmp_path / "runtime.json").as_posix(),
            "--json-out",
            out.as_posix(),
            "--md-out",
            md.as_posix(),
            "--no-diagnostics",
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert out.is_file()
    assert md.is_file()
    assert "score_claim=false" in proc.stdout
