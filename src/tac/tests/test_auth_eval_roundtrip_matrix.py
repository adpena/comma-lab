# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tac.auth_eval_roundtrip_matrix import (
    AuthEvalRoundtripInput,
    build_auth_eval_roundtrip_matrix,
    collect_auth_eval_roundtrip_results,
)


REPO = Path(__file__).resolve().parents[3]


def _runtime_manifest() -> dict:
    return {
        "schema": "contest_auth_eval_runtime_dependency_manifest_v1",
        "runtime_root": "/tmp/local/submission_dir",
        "runtime_file_count": 2,
        "runtime_tree_sha256": "local_tree",
        "runtime_content_tree_sha256": "content_tree",
        "files": [
            {
                "relative_path": "inflate.sh",
                "repo_relative_path": "/tmp/local/submission_dir/inflate.sh",
                "bytes": 9,
                "sha256": "a" * 64,
            },
            {
                "relative_path": "inflate.py",
                "repo_relative_path": "/tmp/local/submission_dir/inflate.py",
                "bytes": 9,
                "sha256": "b" * 64,
            },
        ],
        "external_dependency_roots": [],
        "repo_local_tac_import_manifest": {
            "schema": "contest_auth_eval_repo_local_tac_import_manifest_v1",
            "runtime_root_name": "submission_dir",
            "file_count": 0,
            "files": [],
        },
        "upstream_evaluate_py": {"relative_path": "evaluate.py", "bytes": 3, "sha256": "c" * 64},
    }


def test_roundtrip_matrix_separates_contest_axes_from_diagnostics() -> None:
    candidate = AuthEvalRoundtripInput(
        archive="experiments/results/candidate/archive.zip",
        submission_dir="experiments/results/candidate/submission_dir",
        label="fixture",
        lane_id="fixture_lane",
    )

    matrix = build_auth_eval_roundtrip_matrix(
        candidate=candidate,
        runtime_manifest=_runtime_manifest(),
        repo_root=REPO,
        host_system="Darwin",
        host_machine="arm64",
    )

    targets = {row["target_id"]: row for row in matrix["targets"]}
    assert targets["modal_contest_cuda_t4_auto"]["contest_compliant"] is True
    assert targets["modal_contest_cuda_t4_auto"]["score_axis"] == "contest_cuda"
    assert targets["modal_contest_cpu_linux_x86_auto"]["contest_compliant"] is True
    assert targets["modal_contest_cpu_linux_x86_auto"]["score_axis"] == "contest_cpu"
    assert targets["local_cpu_current_host_auto"]["contest_compliant"] is False
    assert targets["local_cpu_current_host_auto"]["score_axis"] == "macos_cpu_advisory"
    assert "local_host_not_linux_x86_64" in targets["local_cpu_current_host_auto"]["diagnostic_blockers"]
    assert targets["modal_cuda_scorer_force_inflate_cpu_diagnostic"]["contest_compliant"] is False
    assert "--inflate-device" in targets["modal_cuda_scorer_force_inflate_cpu_diagnostic"]["command"]
    assert set(matrix["contest_compliant_target_ids"]) == {
        "modal_contest_cuda_t4_auto",
        "modal_contest_cpu_linux_x86_auto",
    }


def test_local_linux_x86_cpu_is_marked_contest_cpu() -> None:
    candidate = AuthEvalRoundtripInput(
        archive="archive.zip",
        submission_dir="submission_dir",
        label="fixture",
    )
    matrix = build_auth_eval_roundtrip_matrix(
        candidate=candidate,
        runtime_manifest=_runtime_manifest(),
        repo_root=REPO,
        include_diagnostics=False,
        host_system="Linux",
        host_machine="x86_64",
    )
    local = next(row for row in matrix["targets"] if row["target_id"] == "local_cpu_current_host_auto")
    assert local["contest_compliant"] is True
    assert local["score_axis"] == "contest_cpu"
    assert local["score_claim_possible_after_recovery"] is True


def test_roundtrip_matrix_cli_from_packet_manifest(tmp_path: Path) -> None:
    submission = tmp_path / "submission_dir"
    submission.mkdir()
    (submission / "inflate.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    manifest = tmp_path / "packet_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "archive": {"path": "archive.zip"},
                "runtime": {"path": submission.as_posix()},
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "matrix.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools/build_auth_eval_roundtrip_matrix.py"),
            "--packet-manifest",
            str(manifest),
            "--json-out",
            str(out),
            "--label",
            "fixture",
            "--no-diagnostics",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "auth_eval_roundtrip_matrix_v1"
    assert len(payload["targets"]) == 3


def test_collect_roundtrip_results_preserves_partial_axis_signal(tmp_path: Path) -> None:
    candidate = AuthEvalRoundtripInput(
        archive="archive.zip",
        submission_dir="submission_dir",
        label="fixture",
    )
    matrix = build_auth_eval_roundtrip_matrix(
        candidate=candidate,
        runtime_manifest=_runtime_manifest(),
        repo_root=REPO,
        include_diagnostics=True,
        host_system="Darwin",
        host_machine="arm64",
    )
    cuda_dir = tmp_path / "cuda"
    cuda_dir.mkdir()
    (cuda_dir / "contest_auth_eval.json").write_text(
        json.dumps(
            {
                "canonical_score": 0.226,
                "avg_posenet_dist": 0.000138,
                "avg_segnet_dist": 0.000642,
                "archive_size_bytes": 187209,
                "n_samples": 600,
                "evidence_grade": "B",
                "score_axis": "diagnostic_leaky_payload_label",
                "provenance": {"archive_sha256": "a" * 64},
            }
        ),
        encoding="utf-8",
    )
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    (pending_dir / "modal_auth_eval_recover_summary.json").write_text(
        json.dumps({"status": "pending"}),
        encoding="utf-8",
    )

    payload = collect_auth_eval_roundtrip_results(
        matrix,
        target_result_dirs={
            "modal_contest_cuda_t4_auto": cuda_dir.as_posix(),
            "modal_cuda_scorer_force_inflate_cpu_diagnostic": pending_dir.as_posix(),
        },
    )

    rows = {row["target_id"]: row for row in payload["rows"]}
    assert payload["recovered_count"] == 1
    assert payload["pending_count"] == 1
    assert rows["modal_contest_cuda_t4_auto"]["axis_result_class"] == "contest_cuda_anchor"
    assert rows["modal_contest_cuda_t4_auto"]["evidence_grade"] == "contest-CUDA"
    assert rows["modal_contest_cuda_t4_auto"]["score_axis"] == "contest_cuda"
    assert rows["modal_contest_cuda_t4_auto"]["payload_evidence_grade"] == "B"
    assert (
        rows["modal_contest_cuda_t4_auto"]["payload_score_axis"]
        == "diagnostic_leaky_payload_label"
    )
    assert rows["modal_contest_cuda_t4_auto"]["contest_axis_anchor"] is True
    assert (
        rows["modal_contest_cuda_t4_auto"]["score_claim_possible_after_result_review"]
        is True
    )
    assert rows["modal_contest_cuda_t4_auto"]["score_claim"] is False
    assert rows["modal_contest_cuda_t4_auto"]["rank_or_kill_eligible"] is False
    assert (
        "roundtrip_matrix_is_command_planner_not_claim_surface"
        in rows["modal_contest_cuda_t4_auto"]["result_review_blockers"]
    )
    assert rows["modal_cuda_scorer_force_inflate_cpu_diagnostic"]["status"] == "pending"
    assert (
        rows["modal_cuda_scorer_force_inflate_cpu_diagnostic"]["axis_result_class"]
        == "not_recovered"
    )
