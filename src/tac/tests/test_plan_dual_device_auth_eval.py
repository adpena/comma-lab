# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest


def _load_tool():
    repo_root = Path(__file__).resolve().parents[3]
    tool_path = repo_root / "tools" / "plan_dual_device_auth_eval.py"
    spec = importlib.util.spec_from_file_location("plan_dual_device_auth_eval", tool_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {tool_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_score_artifact(
    path: Path,
    *,
    device: str,
    archive_sha256: str,
    archive_bytes: int,
    runtime_tree_sha256: str = "runtime-tree-fixture",
    inflated_output_aggregate_sha256: str | None = None,
) -> None:
    payload = {
        "canonical_score": 0.2 if device == "cpu" else 0.23,
        "avg_segnet_dist": 0.0005,
        "avg_posenet_dist": 0.00004,
        "archive_sha256": archive_sha256,
        "archive_size_bytes": archive_bytes,
        "device": device,
        "n_samples": 600,
        "runtime_tree_sha256": runtime_tree_sha256,
    }
    if inflated_output_aggregate_sha256 is not None:
        payload["provenance"] = {
            "device": device,
            "gpu_t4_match": device == "cuda",
            "inflated_output_manifest": {
                "path": f"{device}/inflated_outputs_manifest.json",
                "sha256": f"{device}-manifest-sha",
                "payload": {
                    "aggregate_sha256": inflated_output_aggregate_sha256,
                    "raw_file_count": 1,
                    "total_bytes": 603_979_776,
                },
            }
        }
    if device == "cpu":
        payload.update(
            {
                "hardware": "github-actions-ubuntu-latest-x86_64",
                "evidence_grade": "contest-CPU-1to1",
                "promotion_eligible": False,
                "score_claim_valid": False,
                "rank_or_kill_eligible": False,
            }
        )
    else:
        payload.update(
            {
                "evidence_grade": "A++",
                "gpu_t4_match": True,
                "promotion_eligible": True,
                "score_claim_valid": True,
            }
        )
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_dual_plan_emits_cpu_and_cuda_commands_for_same_archive(tmp_path: Path) -> None:
    mod = _load_tool()
    archive = tmp_path / "archive.zip"
    with ZipFile(archive, "w") as zf:
        zf.writestr("x", b"payload")
    inflate = tmp_path / "inflate.sh"
    inflate.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    plan = mod.build_plan(
        archive=archive,
        inflate_sh=inflate,
        label="fixture",
        repo_root=Path("."),
        run_id="fixture-run",
        output_root=Path("experiments/results/dual_device_auth_eval"),
        upstream_dir=Path("upstream"),
        video_names_file=Path("upstream/public_test_video_names.txt"),
        inflate_timeout=1800,
        evaluate_timeout=1800,
    )

    assert set(plan["evals"]) == {"cuda", "cpu"}
    assert plan["archive"]["path"] == str(archive)
    assert plan["archive"]["sha256"] == mod._sha256(archive)
    assert plan["input_closure"]["ready_to_execute"] is True
    assert plan["input_closure"]["missing_inputs"] == []
    for device in ("cuda", "cpu"):
        command = plan["evals"][device]["command"]
        assert command[command.index("--archive") + 1] == str(archive)
        assert command[command.index("--inflate-sh") + 1] == str(inflate)
        assert command[command.index("--device") + 1] == device
        assert command[command.index("--json-out") + 1] == (
            f"experiments/results/dual_device_auth_eval/fixture-run/{device}/contest_auth_eval.json"
        )
        assert plan["evals"][device]["json_out"] == (
            f"experiments/results/dual_device_auth_eval/fixture-run/{device}/contest_auth_eval.json"
        )
        assert "--keep-work-dir" in command
    assert plan["evals"]["cuda"]["promotion_eligible_from_this_axis"] is True
    assert plan["evals"]["cpu"]["promotion_eligible_from_this_axis"] is False
    matrix = plan["device_axis_matrix"]
    assert matrix["schema"] == "device_axis_auth_eval_matrix_plan.v1"
    assert set(matrix["entries"]) == {
        "scorer_cuda__inflate_auto",
        "scorer_cuda__inflate_cpu",
        "scorer_cuda__inflate_cuda",
        "scorer_cpu__inflate_auto",
        "scorer_cpu__inflate_cpu",
        "scorer_cpu__inflate_cuda",
    }
    cuda_cpu_inflate = matrix["entries"]["scorer_cuda__inflate_cpu"]
    assert cuda_cpu_inflate["score_axis"] == "diagnostic_cuda"
    assert cuda_cpu_inflate["diagnostic_only"] is True
    assert cuda_cpu_inflate["promotion_eligible_from_this_axis"] is False
    assert cuda_cpu_inflate["command"][
        cuda_cpu_inflate["command"].index("--inflate-device") + 1
    ] == "cpu"
    cuda_auto = matrix["entries"]["scorer_cuda__inflate_auto"]
    assert cuda_auto["score_axis"] == "contest_cuda"
    assert cuda_auto["promotion_eligible_from_this_axis"] is True
    assert cuda_auto["command"][cuda_auto["command"].index("--inflate-device") + 1] == "auto"
    completion = plan["dual_axis_completion"]
    assert completion["paired_score_artifacts_complete"] is False
    assert completion["global_priority_eligible"] is False
    assert set(completion["blockers"]) == {
        "missing_contest_cuda_score_artifact",
        "missing_contest_cpu_score_artifact",
    }


def test_dual_plan_single_axis_artifact_stays_incomplete(tmp_path: Path) -> None:
    mod = _load_tool()
    archive = tmp_path / "archive.zip"
    with ZipFile(archive, "w") as zf:
        zf.writestr("x", b"payload")
    archive_sha = mod._sha256(archive)
    cpu_json = tmp_path / "cpu.json"
    _write_score_artifact(
        cpu_json,
        device="cpu",
        archive_sha256=archive_sha,
        archive_bytes=archive.stat().st_size,
    )

    plan = mod.build_plan(
        archive=archive,
        inflate_sh=tmp_path / "inflate.sh",
        label="fixture",
        repo_root=Path("."),
        run_id="fixture-run",
        output_root=Path("experiments/results/dual_device_auth_eval"),
        upstream_dir=Path("upstream"),
        video_names_file=Path("upstream/public_test_video_names.txt"),
        inflate_timeout=1800,
        evaluate_timeout=1800,
        cpu_artifact_json=cpu_json,
    )

    completion = plan["dual_axis_completion"]
    assert completion["artifacts"]["cpu"]["valid_for_axis"] is True
    assert completion["paired_score_artifacts_complete"] is False
    assert completion["blockers"] == [
        "missing_contest_cuda_score_artifact",
    ]
    assert completion["frontier_or_medal_band_complete"] is False


def test_dual_plan_marks_pair_complete_only_with_matching_cpu_and_cuda_artifacts(
    tmp_path: Path,
) -> None:
    mod = _load_tool()
    archive = tmp_path / "archive.zip"
    with ZipFile(archive, "w") as zf:
        zf.writestr("x", b"payload")
    archive_sha = mod._sha256(archive)
    archive_bytes = archive.stat().st_size
    cpu_json = tmp_path / "cpu.json"
    cuda_json = tmp_path / "cuda.json"
    _write_score_artifact(
        cpu_json,
        device="cpu",
        archive_sha256=archive_sha,
        archive_bytes=archive_bytes,
    )
    _write_score_artifact(
        cuda_json,
        device="cuda",
        archive_sha256=archive_sha,
        archive_bytes=archive_bytes,
    )

    plan = mod.build_plan(
        archive=archive,
        inflate_sh=tmp_path / "inflate.sh",
        label="fixture",
        repo_root=Path("."),
        run_id="fixture-run",
        output_root=Path("experiments/results/dual_device_auth_eval"),
        upstream_dir=Path("upstream"),
        video_names_file=Path("upstream/public_test_video_names.txt"),
        inflate_timeout=1800,
        evaluate_timeout=1800,
        cpu_artifact_json=cpu_json,
        cuda_artifact_json=cuda_json,
    )

    completion = plan["dual_axis_completion"]
    assert completion["blockers"] == []
    assert completion["paired_score_artifacts_complete"] is True
    assert completion["drift_mechanism_complete"] is False
    assert completion["mechanism_blockers"] == ["raw_output_manifest_missing"]
    assert completion["global_priority_eligible"] is False
    assert completion["frontier_or_medal_band_complete"] is True
    assert completion["rank_or_kill_eligible"] is False
    assert completion["rank_or_kill_blockers"] == [
        "dual_axis_pair_completeness_is_not_adjudicated_rank_or_kill_authority"
    ]
    assert completion["same_archive_sha256"] is True
    assert completion["same_archive_bytes"] is True
    assert completion["same_runtime_tree_sha256"] is True
    assert completion["same_inflated_output_aggregate_sha256"] is None
    assert completion["raw_output_pairing_status"] == "raw_output_manifest_missing"
    assert plan["promotion_eligible"] is False


def test_dual_plan_compares_raw_output_hashes_without_blocking_score_pair(
    tmp_path: Path,
) -> None:
    mod = _load_tool()
    archive = tmp_path / "archive.zip"
    with ZipFile(archive, "w") as zf:
        zf.writestr("x", b"payload")
    archive_sha = mod._sha256(archive)
    archive_bytes = archive.stat().st_size
    cpu_json = tmp_path / "cpu.json"
    cuda_json = tmp_path / "cuda.json"
    _write_score_artifact(
        cpu_json,
        device="cpu",
        archive_sha256=archive_sha,
        archive_bytes=archive_bytes,
        inflated_output_aggregate_sha256="1" * 64,
    )
    _write_score_artifact(
        cuda_json,
        device="cuda",
        archive_sha256=archive_sha,
        archive_bytes=archive_bytes,
        inflated_output_aggregate_sha256="2" * 64,
    )

    plan = mod.build_plan(
        archive=archive,
        inflate_sh=tmp_path / "inflate.sh",
        label="fixture",
        repo_root=Path("."),
        run_id="fixture-run",
        output_root=Path("experiments/results/dual_device_auth_eval"),
        upstream_dir=Path("upstream"),
        video_names_file=Path("upstream/public_test_video_names.txt"),
        inflate_timeout=1800,
        evaluate_timeout=1800,
        cpu_artifact_json=cpu_json,
        cuda_artifact_json=cuda_json,
    )

    completion = plan["dual_axis_completion"]
    assert completion["blockers"] == []
    assert completion["paired_score_artifacts_complete"] is True
    assert completion["drift_mechanism_complete"] is True
    assert completion["mechanism_blockers"] == []
    assert completion["global_priority_eligible"] is True
    assert completion["rank_or_kill_eligible"] is False
    assert completion["same_inflated_output_aggregate_sha256"] is False
    assert completion["raw_output_pairing_status"] == "different_inflated_outputs"
    assert (
        completion["artifacts"]["cpu"]["inflated_output_manifest"]["aggregate_sha256"]
        == "1" * 64
    )


def test_dual_plan_rejects_paired_artifacts_for_different_archive_sha(
    tmp_path: Path,
) -> None:
    mod = _load_tool()
    archive = tmp_path / "archive.zip"
    with ZipFile(archive, "w") as zf:
        zf.writestr("x", b"payload")
    archive_sha = mod._sha256(archive)
    archive_bytes = archive.stat().st_size
    cpu_json = tmp_path / "cpu.json"
    cuda_json = tmp_path / "cuda.json"
    _write_score_artifact(
        cpu_json,
        device="cpu",
        archive_sha256=archive_sha,
        archive_bytes=archive_bytes,
    )
    _write_score_artifact(
        cuda_json,
        device="cuda",
        archive_sha256="0" * 64,
        archive_bytes=archive_bytes,
    )

    plan = mod.build_plan(
        archive=archive,
        inflate_sh=tmp_path / "inflate.sh",
        label="fixture",
        repo_root=Path("."),
        run_id="fixture-run",
        output_root=Path("experiments/results/dual_device_auth_eval"),
        upstream_dir=Path("upstream"),
        video_names_file=Path("upstream/public_test_video_names.txt"),
        inflate_timeout=1800,
        evaluate_timeout=1800,
        cpu_artifact_json=cpu_json,
        cuda_artifact_json=cuda_json,
    )

    completion = plan["dual_axis_completion"]
    assert completion["paired_score_artifacts_complete"] is False
    assert "cpu_cuda_archive_sha256_mismatch" in completion["blockers"]
    assert "cuda_artifact_archive_sha256_mismatch_with_plan" in completion["blockers"]


def test_dual_plan_requires_runtime_tree_hashes_for_paired_completion(
    tmp_path: Path,
) -> None:
    mod = _load_tool()
    archive = tmp_path / "archive.zip"
    with ZipFile(archive, "w") as zf:
        zf.writestr("x", b"payload")
    archive_sha = mod._sha256(archive)
    archive_bytes = archive.stat().st_size
    cpu_json = tmp_path / "cpu.json"
    cuda_json = tmp_path / "cuda.json"
    _write_score_artifact(
        cpu_json,
        device="cpu",
        archive_sha256=archive_sha,
        archive_bytes=archive_bytes,
    )
    _write_score_artifact(
        cuda_json,
        device="cuda",
        archive_sha256=archive_sha,
        archive_bytes=archive_bytes,
    )
    cuda_payload = json.loads(cuda_json.read_text(encoding="utf-8"))
    cuda_payload.pop("runtime_tree_sha256")
    cuda_json.write_text(json.dumps(cuda_payload), encoding="utf-8")

    plan = mod.build_plan(
        archive=archive,
        inflate_sh=tmp_path / "inflate.sh",
        label="fixture",
        repo_root=Path("."),
        run_id="fixture-run",
        output_root=Path("experiments/results/dual_device_auth_eval"),
        upstream_dir=Path("upstream"),
        video_names_file=Path("upstream/public_test_video_names.txt"),
        inflate_timeout=1800,
        evaluate_timeout=1800,
        cpu_artifact_json=cpu_json,
        cuda_artifact_json=cuda_json,
    )

    completion = plan["dual_axis_completion"]
    assert completion["paired_score_artifacts_complete"] is False
    assert "cuda_artifact_runtime_tree_sha256_missing" in completion["blockers"]
    assert completion["same_runtime_tree_sha256"] is None


def test_runtime_tree_hash_can_come_from_provenance_manifest(tmp_path: Path) -> None:
    mod = _load_tool()
    path = tmp_path / "artifact.json"
    path.write_text(
        json.dumps(
            {
                "provenance": {
                    "inflate_runtime_manifest": {
                        "runtime_tree_sha256": "nested-runtime-tree"
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert mod._runtime_tree_sha256(payload) == "nested-runtime-tree"


def test_public_pr_inputs_resolve_from_wrapped_ledger(tmp_path: Path) -> None:
    mod = _load_tool()
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "pr": 102,
                        "leaderboard_name": "hnerv",
                        "archive": {"path": "a.zip"},
                        "source": {"key_files": {"inflate_sh": {"path": "inflate.sh"}}},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    archive, inflate, label, row = mod._public_pr_inputs(ledger, 102)

    assert archive == Path("a.zip")
    assert inflate == Path("inflate.sh")
    assert label.startswith("public-pr102-hnerv")
    assert row["pr"] == 102


def test_dual_plan_records_missing_input_closure(tmp_path: Path) -> None:
    mod = _load_tool()

    plan = mod.build_plan(
        archive=tmp_path / "missing.zip",
        inflate_sh=tmp_path / "missing-inflate.sh",
        label="fixture",
        repo_root=Path("."),
        run_id="fixture-run",
        output_root=Path("experiments/results/dual_device_auth_eval"),
        upstream_dir=Path("upstream"),
        video_names_file=Path("upstream/public_test_video_names.txt"),
        inflate_timeout=1800,
        evaluate_timeout=1800,
    )

    assert plan["input_closure"]["ready_to_execute"] is False
    assert set(plan["input_closure"]["missing_inputs"]) == {"archive", "inflate_sh"}
    assert plan["score_claim"] is False
    assert plan["promotion_eligible"] is False


def test_execute_refuses_missing_input_closure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_tool()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plan_dual_device_auth_eval.py",
            "--archive",
            str(tmp_path / "missing.zip"),
            "--inflate-sh",
            str(tmp_path / "missing-inflate.sh"),
            "--execute",
            "cpu",
        ],
    )

    with pytest.raises(SystemExit, match="missing inputs: archive, inflate_sh"):
        mod.main()


def test_execute_refuses_without_dispatch_claim_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_tool()
    archive = tmp_path / "archive.zip"
    with ZipFile(archive, "w") as zf:
        zf.writestr("x", b"payload")
    inflate = tmp_path / "inflate.sh"
    inflate.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plan_dual_device_auth_eval.py",
            "--archive",
            str(archive),
            "--inflate-sh",
            str(inflate),
            "--execute",
            "cpu",
        ],
    )

    with pytest.raises(SystemExit, match="requires --lane-id and --instance-job-id"):
        mod.main()


def test_execute_claims_and_closes_terminally_before_and_after_eval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_tool()
    archive = tmp_path / "archive.zip"
    with ZipFile(archive, "w") as zf:
        zf.writestr("x", b"payload")
    inflate = tmp_path / "inflate.sh"
    inflate.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append([str(part) for part in command])
        return mod.subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plan_dual_device_auth_eval.py",
            "--archive",
            str(archive),
            "--inflate-sh",
            str(inflate),
            "--run-id",
            "fixture-run",
            "--execute",
            "cpu",
            "--lane-id",
            "dual_eval_fixture",
            "--instance-job-id",
            "fixture-job",
        ],
    )

    assert mod.main() == 0

    assert len(calls) == 3
    assert calls[0][:3] == [".venv/bin/python", "tools/claim_lane_dispatch.py", "claim"]
    assert "--status" in calls[0]
    assert calls[0][calls[0].index("--status") + 1] == "active_eval_running"
    assert calls[1][0:2] == [".venv/bin/python", "experiments/contest_auth_eval.py"]
    assert calls[1][calls[1].index("--device") + 1] == "cpu"
    assert calls[2][:3] == [".venv/bin/python", "tools/claim_lane_dispatch.py", "claim"]
    assert calls[2][calls[2].index("--status") + 1] == "completed_auth_eval_plan_execute"
    assert "--force" in calls[2]
