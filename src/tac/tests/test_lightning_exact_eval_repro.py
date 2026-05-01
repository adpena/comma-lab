"""Tests for credential-safe Lightning exact-eval orchestration."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "lightning_exact_eval_repro.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "lightning_exact_eval_repro_under_test",
        str(SCRIPT),
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_repo(tmp_path: Path) -> tuple[Path, Path]:
    archive = tmp_path / "experiments/results/candidate/archive.zip"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"candidate archive bytes")
    baseline = tmp_path / "experiments/results/frontier/contest_auth_eval.json"
    baseline.parent.mkdir(parents=True)
    baseline.write_text(
        json.dumps(
            {
                "score_recomputed_from_components": 1.043987524793892,
                "archive_size_bytes": 686635,
                "avg_posenet_dist": 0.00346442,
                "avg_segnet_dist": 0.00400656,
                "n_samples": 600,
                "provenance": {
                    "device": "cuda",
                    "archive_sha256": "0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f",
                    "gpu_t4_match": True,
                },
            },
            indent=2,
        )
        + "\n"
    )
    return archive, baseline


def _base_args(archive: Path, baseline: Path) -> list[str]:
    return [
        "--job-name",
        "owv3_exact_eval_test",
        "--archive",
        str(archive),
        "--baseline-json",
        str(baseline),
        "--predicted-band",
        "1.0",
        "1.1",
        "--regression-threshold",
        "1.2",
        "--max-posenet-relative",
        "1.05",
        "--max-segnet-relative",
        "1.002",
        "--studio",
        "pact",
    ]


def _flag_value(cmd: list[str], flag: str) -> str:
    idx = cmd.index(flag)
    return cmd[idx + 1]


def test_stage_command_uses_operator_supplied_ssh_alias_without_key_material(tmp_path: Path) -> None:
    mod = _load_module()
    archive, baseline = _fixture_repo(tmp_path)
    args = mod.build_parser().parse_args(
        _base_args(archive, baseline)
        + [
            "--stage-workspace",
            "--remote",
            "lightning-pact",
            "--extra-artifact",
            str(tmp_path / "experiments/results/candidate/archive.zip"),
        ]
    )

    plan = mod.build_plan(args, repo_root=tmp_path)
    stage_cmd = plan["commands"]["stage_workspace"]
    stage_string = plan["command_strings"]["stage_workspace"]

    assert stage_cmd is not None
    assert _flag_value(stage_cmd, "--remote") == "lightning-pact"
    assert "StrictHostKeyChecking" not in stage_string
    assert " -i " not in f" {stage_string} "
    assert "experiments/results/candidate/archive.zip" in plan["artifacts"]
    assert "experiments/results/frontier/contest_auth_eval.json" in plan["artifacts"]


def test_queue_command_is_dry_run_and_uses_writable_remote_workspace_path(tmp_path: Path) -> None:
    mod = _load_module()
    archive, baseline = _fixture_repo(tmp_path)
    args = mod.build_parser().parse_args(_base_args(archive, baseline))

    plan = mod.build_plan(args, repo_root=tmp_path)
    queue_cmd = plan["commands"]["queue_exact_eval"]
    queue_string = plan["command_strings"]["queue_exact_eval"]

    assert queue_cmd is not None
    assert "--dry-run" in queue_cmd
    assert _flag_value(queue_cmd, "--archive") == (
        "/teamspace/studios/this_studio/pact/experiments/results/candidate/archive.zip"
    )
    assert _flag_value(queue_cmd, "--expected-archive-sha256") == hashlib.sha256(
        b"candidate archive bytes"
    ).hexdigest()
    assert _flag_value(queue_cmd, "--expected-archive-size-bytes") == str(len(b"candidate archive bytes"))
    assert "/teamspace/jobs/" not in queue_string
    assert "--adjudicate" in queue_cmd
    assert "--device cuda" not in queue_cmd


def test_baseline_json_populates_adjudication_flags(tmp_path: Path) -> None:
    mod = _load_module()
    archive, baseline = _fixture_repo(tmp_path)
    args = mod.build_parser().parse_args(_base_args(archive, baseline))

    plan = mod.build_plan(args, repo_root=tmp_path)
    queue_cmd = plan["commands"]["queue_exact_eval"]
    assert queue_cmd is not None

    assert _flag_value(queue_cmd, "--baseline-score") == "1.0439875247938919"
    assert _flag_value(queue_cmd, "--baseline-archive-bytes") == "686635"
    assert _flag_value(queue_cmd, "--baseline-posenet-dist") == "0.00346442"
    assert _flag_value(queue_cmd, "--baseline-segnet-dist") == "0.00400656"
    assert _flag_value(queue_cmd, "--component-reference-label") == (
        "experiments/results/frontier/contest_auth_eval.json"
    )
    assert plan["queue_metadata"]["baseline_json"] == "experiments/results/frontier/contest_auth_eval.json"


def test_submit_requires_staging_or_explicit_remote_custody_and_target_backend(tmp_path: Path) -> None:
    mod = _load_module()
    archive, baseline = _fixture_repo(tmp_path)

    no_stage = mod.build_parser().parse_args(_base_args(archive, baseline) + ["--submit"])
    with pytest.raises(ValueError, match="--stage-workspace or --allow-unstaged-submit"):
        mod.build_plan(no_stage, repo_root=tmp_path)

    no_backend_args = _base_args(archive, baseline)
    studio_idx = no_backend_args.index("--studio")
    del no_backend_args[studio_idx : studio_idx + 2]
    no_backend = mod.build_parser().parse_args(
        no_backend_args + ["--submit", "--allow-unstaged-submit"]
    )
    with pytest.raises(ValueError, match="--studio or --image"):
        mod.build_plan(no_backend, repo_root=tmp_path)


def test_read_only_sdk_artifact_view_is_rejected_as_output_dir(tmp_path: Path) -> None:
    mod = _load_module()
    archive, baseline = _fixture_repo(tmp_path)
    args = mod.build_parser().parse_args(
        _base_args(archive, baseline)
        + [
            "--output-dir",
            "/teamspace/jobs/owv3/artifacts",
        ]
    )
    with pytest.raises(ValueError, match="read-only artifact view"):
        mod.build_plan(args, repo_root=tmp_path)
