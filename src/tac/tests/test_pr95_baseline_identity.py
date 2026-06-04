# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

from tac.analysis.pr95_baseline_identity import build_pr95_baseline_identity

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_pr95_baseline_identity_selects_exact_eval_blocked_archive_over_smaller_smoke(
    tmp_path: Path,
) -> None:
    source_archive = _write_bytes(tmp_path / "source_archive.zip", b"source-pr95")
    smoke_archive = _write_bytes(tmp_path / "smoke_archive.zip", b"a" * 7)
    bridge_archive = _write_bytes(tmp_path / "bridge_archive.zip", b"b" * 8)
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    (runtime_root / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    smoke_report = _write_stage8_report(
        tmp_path / "smoke_report.json",
        source_archive=source_archive,
        candidate_archive=smoke_archive,
        blockers=[
            "stage8_best_checkpoint_missing_packaged_final_checkpoint",
            "contest_cpu_cuda_exact_eval_missing",
        ],
        runtime_root=runtime_root,
    )
    bridge_report = _write_stage8_report(
        tmp_path / "bridge_report.json",
        source_archive=source_archive,
        candidate_archive=bridge_archive,
        blockers=["contest_cpu_cuda_exact_eval_missing"],
        runtime_root=runtime_root,
    )
    receiver_smoke = _write_receiver_proof(
        tmp_path / "receiver_proof.json",
        archive=_write_bytes(tmp_path / "receiver_smoke_archive.zip", b"c" * 3),
    )

    report = build_pr95_baseline_identity(
        source_artifacts=(smoke_report, bridge_report, receiver_smoke),
        output_root=tmp_path / "exact_eval",
    )

    selected = report["selected_reusable_candidate_archive"]
    assert report["schema"] == "pr95_baseline_identity.v1"
    assert report["baseline_identity_reusable"] is True
    assert selected["path"] == bridge_archive.as_posix()
    assert selected["bytes"] == bridge_archive.stat().st_size
    assert selected["sha256"] == _sha(bridge_archive)
    assert selected["reusable_identity"] is True
    assert report["source_archive"]["sha256"] == _sha(source_archive)
    assert report["exact_axis_status"]["contest_cpu"]["present"] is False
    assert "pr95_contest_cpu_exact_eval_missing" in report["blockers"]
    assert "pr95_contest_cuda_exact_eval_missing" in report["blockers"]
    local_work_order = report["local_cpu_mlx_work_order"]
    assert local_work_order["schema"] == "pr95_baseline_local_cpu_mlx_work_order.v1"
    assert local_work_order["ready"] is True
    assert local_work_order["local_cpu_axis_tag"] == "[macOS-CPU advisory]"
    assert local_work_order["mlx_axis_tag"] == "[macOS-MLX research-signal]"
    assert "--archive" in local_work_order["local_cpu_command_argv"]
    assert bridge_archive.as_posix() in local_work_order["local_cpu_command_argv"]
    assert "--inflate-sh" in local_work_order["local_cpu_command_argv"]
    assert (runtime_root / "inflate.sh").as_posix() in local_work_order[
        "local_cpu_command_argv"
    ]
    modal_policy = report["modal_dispatch_policy"]
    assert modal_policy["modal_dispatch_allowed"] is False
    assert modal_policy["reason"] == "non_frontier_control_arm_modal_dispatch_forbidden"
    work_order = report["paired_exact_eval_work_order"]
    assert work_order["ready"] is False
    assert work_order["modal_dispatch_allowed"] is False
    assert "modal_reserved_for_frontier_candidates" in work_order["blockers"]
    receiver_row = next(
        row
        for row in report["candidate_archives"]
        if row["role"] == "receiver_runtime_consumption_identity"
    )
    assert "receiver_proof_archive_not_stage8_public_archive_candidate" in receiver_row[
        "blockers"
    ]
    assert receiver_row["reusable_identity"] is False
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False


def test_pr95_baseline_identity_records_existing_cpu_exact_anchor_without_promotion(
    tmp_path: Path,
) -> None:
    candidate_archive = _write_bytes(tmp_path / "candidate_archive.zip", b"candidate")
    stage8_report = _write_stage8_report(
        tmp_path / "stage8_report.json",
        source_archive=_write_bytes(tmp_path / "source_archive.zip", b"source"),
        candidate_archive=candidate_archive,
        blockers=["contest_cpu_cuda_exact_eval_missing"],
        runtime_root=None,
    )
    cpu_eval = tmp_path / "contest_cpu.json"
    cpu_eval.write_text(
        json.dumps(
            {
                "score": 0.2,
                "score_axis": "contest_cpu",
                "score_claim": True,
                "score_claim_valid": True,
                "archive_sha256": _sha(candidate_archive),
                "evidence_grade": "contest-cpu",
                "lane_tag": "pr95",
            }
        ),
        encoding="utf-8",
    )

    report = build_pr95_baseline_identity(
        source_artifacts=(stage8_report, cpu_eval),
        output_root=tmp_path / "exact_eval",
    )

    assert report["exact_axis_status"]["contest_cpu"]["present"] is True
    assert report["exact_axis_status"]["contest_cpu"]["score"] == 0.2
    assert report["exact_axis_status"]["contest_cuda"]["present"] is False
    assert "pr95_contest_cpu_exact_eval_missing" not in report["blockers"]
    assert "pr95_contest_cuda_exact_eval_missing" in report["blockers"]
    assert report["promotion_eligible"] is False


def test_build_pr95_baseline_identity_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    source_archive = _write_bytes(tmp_path / "source_archive.zip", b"source")
    candidate_archive = _write_bytes(tmp_path / "candidate_archive.zip", b"candidate")
    stage8_report = _write_stage8_report(
        tmp_path / "stage8_report.json",
        source_archive=source_archive,
        candidate_archive=candidate_archive,
        blockers=["contest_cpu_cuda_exact_eval_missing"],
        runtime_root=None,
    )
    output_json = tmp_path / "identity.json"
    output_md = tmp_path / "identity.md"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "build_pr95_baseline_identity.py"),
            "--source-artifact",
            stage8_report.as_posix(),
            "--output-json",
            output_json.as_posix(),
            "--output-md",
            output_md.as_posix(),
            "--output-root",
            (tmp_path / "exact_eval").as_posix(),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert summary["schema"] == "pr95_baseline_identity.v1"
    assert summary["baseline_identity_reusable"] is True
    assert summary["local_cpu_mlx_ready"] is False
    assert summary["modal_dispatch_allowed"] is False
    assert "pr95_runtime_root_missing_for_local_cpu_replay" in payload[
        "local_cpu_mlx_work_order"
    ]["blockers"]
    assert "modal_reserved_for_frontier_candidates" in payload[
        "paired_exact_eval_work_order"
    ]["blockers"]
    assert output_md.read_text(encoding="utf-8").startswith("# PR95 Baseline Identity")


def _write_stage8_report(
    path: Path,
    *,
    source_archive: Path,
    candidate_archive: Path,
    blockers: list[str],
    runtime_root: Path | None,
) -> Path:
    argv = ["tools/run_pr95_stage8_from_public_archive.py"]
    if runtime_root is not None:
        argv.extend(["--public-submission-root", runtime_root.as_posix()])
    path.write_text(
        json.dumps(
            {
                "schema": "pr95_stage8_from_public_archive_lane.v1",
                "source_archive_zip": source_archive.as_posix(),
                "source_archive_zip_sha256": _sha(source_archive),
                "candidate_archive_zip_path": candidate_archive.as_posix(),
                "candidate_archive_zip_bytes": candidate_archive.stat().st_size,
                "candidate_archive_zip_sha256": _sha(candidate_archive),
                "score_axis": "[macOS-CPU advisory]",
                "score_authority": "none_until_contest_cpu_cuda_exact_eval",
                "exact_gate": {
                    "schema": "exact_gate_blocker.v1",
                    "ready_for_exact_eval_dispatch": False,
                    "blockers": blockers,
                },
                "reproducibility": {"argv_template": argv},
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_receiver_proof(path: Path, *, archive: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "pr95_hnerv_receiver_proof.v1",
                "archive_zip_path": archive.as_posix(),
                "archive_sha256": _sha(archive),
                "blockers": [],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_bytes(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()
