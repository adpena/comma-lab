# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

from tac.local_acceleration.mlx_replay_bundle import (
    MLX_LOCAL_REPLAY_BUNDLE_SCHEMA,
    build_mlx_local_replay_bundle,
)
from tac.repo_io import write_json


def test_mlx_replay_bundle_hashes_artifacts_and_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "0.mkv"
    source.write_bytes(b"video")
    schedule = tmp_path / "schedule.json"
    write_json(schedule, {"schema": "unit_schedule.v1", "score_claim": False})
    report = tmp_path / "report.json"
    write_json(
        report,
        {
            "schema": "unit_report.v1",
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )

    bundle = build_mlx_local_replay_bundle(
        repo_root=tmp_path,
        bundle_id="unit",
        axis="[macOS-MLX research-signal]",
        commands=[["python", "run.py", "--smoke"]],
        input_artifact_paths=[source, schedule],
        artifact_paths=[report],
        metadata={"lane": "unit"},
        argv=["build_mlx_local_replay_bundle.py"],
    )

    assert bundle["schema"] == MLX_LOCAL_REPLAY_BUNDLE_SCHEMA
    assert bundle["score_claim"] is False
    assert bundle["promotion_eligible"] is False
    assert bundle["ready_for_exact_eval_dispatch"] is False
    assert bundle["replay_readiness"]["local_replay_ready"] is True
    assert bundle["input_artifacts"][0]["sha256"]
    assert bundle["input_artifacts"][1]["schema"] == "unit_schedule.v1"
    assert bundle["output_artifacts"][0]["schema"] == "unit_report.v1"
    assert bundle["replay_commands"] == [["python", "run.py", "--smoke"]]
    assert bundle["environment"]["full_env_sha256"]


def test_mlx_replay_bundle_records_missing_artifact_blocker(tmp_path: Path) -> None:
    bundle = build_mlx_local_replay_bundle(
        repo_root=tmp_path,
        bundle_id="unit_missing",
        axis="[macOS-MLX research-signal]",
        commands=[["python", "run.py"]],
        artifact_paths=[tmp_path / "missing.json"],
    )

    assert bundle["replay_readiness"]["local_replay_ready"] is False
    assert bundle["missing_artifacts"] == ["missing.json"]
    assert "missing_replay_artifacts" in bundle["replay_readiness"]["exact_eval_blockers"]


def test_mlx_replay_bundle_does_not_drop_byte_closed_receiver_proof_signal(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "byte_closed_candidate.json"
    write_json(
        candidate,
        {
            "schema": "pact_nerv_ia3_byte_closed_candidate.v1",
            "byte_closed_candidate_emitted": True,
            "runtime_adapter_ready": True,
            "candidate_runtime_adapter_blocker_cleared": True,
            "candidate_runtime_tree_sha256": "runtime123",
            "runtime_consumption_proof_sha256": "proof123",
            "receiver_contract_satisfied": True,
            "full_frame_inflate_parity_satisfied": True,
            "receiver_verification": {"proof_sha256": "abc123"},
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )

    bundle = build_mlx_local_replay_bundle(
        repo_root=tmp_path,
        bundle_id="unit_byte_closed",
        axis="[macOS-MLX research-signal]",
        commands=[["python", "materialize.py"]],
        artifact_paths=[candidate],
    )

    readiness = bundle["replay_readiness"]
    assert readiness["byte_closed_receiver_proof_present"] is True
    assert readiness["runtime_custody_present"] is True
    assert (
        "byte_closed_archive_and_receiver_runtime_proof_required_before_dispatch"
        not in readiness["exact_eval_blockers"]
    )
    assert "runtime_adapter_custody_required_before_dispatch" not in readiness["exact_eval_blockers"]
    assert "contest_cpu_or_cuda_exact_eval_required_before_promotion" in readiness["exact_eval_blockers"]


def test_mlx_replay_bundle_keeps_runtime_custody_blocker_until_tree_hash_present(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "byte_closed_candidate.json"
    write_json(
        candidate,
        {
            "schema": "pact_nerv_ia3_byte_closed_candidate.v1",
            "byte_closed_candidate_emitted": True,
            "receiver_contract_satisfied": True,
            "full_frame_inflate_parity_satisfied": True,
            "receiver_verification": {"proof_sha256": "abc123"},
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )

    bundle = build_mlx_local_replay_bundle(
        repo_root=tmp_path,
        bundle_id="unit_runtime_custody_missing",
        axis="[macOS-MLX research-signal]",
        commands=[["python", "materialize.py"]],
        artifact_paths=[candidate],
    )

    readiness = bundle["replay_readiness"]
    assert readiness["byte_closed_receiver_proof_present"] is True
    assert readiness["runtime_custody_present"] is False
    assert (
        "byte_closed_archive_and_receiver_runtime_proof_required_before_dispatch"
        not in readiness["exact_eval_blockers"]
    )
    assert "runtime_adapter_custody_required_before_dispatch" in readiness["exact_eval_blockers"]
