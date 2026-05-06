from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "lightning_dispatch_pr106_yshift_score_table.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "lightning_dispatch_pr106_yshift_score_table_test", TOOL_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(tmp_path: Path) -> SimpleNamespace:
    del tmp_path
    pr106 = REPO_ROOT / "pyproject.toml"
    return SimpleNamespace(
        job_name="lane_pr106_yshift_score_table_test",
        pr106_archive=pr106,
        ssh_target="lightning-pact",
        remote_pact="/teamspace/studios/this_studio/pact",
        python_bin=".venv/bin/python",
        ssh_connect_timeout=30,
        backend="batch",
        gpu_tier="T4",
        machine="g4dn.2xlarge",
        studio="lossy-compression-challenge",
        teamspace="comma-lab",
        user="adpena",
        cloud_account=None,
        allow_gpu_mismatch=False,
        candidate_radius=3,
        score_step=1.0,
        n_pairs=600,
        batch_pairs=8,
        candidate_batch_size=32,
        predicted_low=0.2065,
        predicted_high=0.208,
        estimated_cost=2.0,
        predicted_eta_hours=2.0,
        max_runtime_seconds=3 * 60 * 60,
        kill_criteria="kill on CUDA failure",
        agent="codex:test",
        force_claim=False,
        require_studio_cuda=False,
        dry_run_batch=True,
        skip_ssh_check=False,
        skip_stage=False,
        print_only=True,
    )


def _values_after(tokens: list[str], flag: str) -> list[str]:
    return [tokens[i + 1] for i, tok in enumerate(tokens) if tok == flag and i + 1 < len(tokens)]


def test_stage_command_stages_claim_ledger_after_local_claim(tmp_path: Path) -> None:
    tool = _load_tool()
    args = _args(tmp_path)
    cmd = tool.build_stage_command(args)

    artifacts = _values_after(cmd, "--artifact")
    assert "experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip" not in artifacts
    assert ".omx/state/active_lane_dispatch_claims.md" in artifacts
    assert str(args.pr106_archive.resolve().relative_to(REPO_ROOT)) in artifacts
    assert "--require-cuda" not in cmd
    assert _values_after(cmd, "--requirements-mode") == ["verify-only"]

    args.require_studio_cuda = True
    assert "--require-cuda" in tool.build_stage_command(args)


def test_claim_command_uses_stable_score_table_lane_and_job(tmp_path: Path) -> None:
    tool = _load_tool()
    args = _args(tmp_path)
    cmd = tool.build_claim_command(args, status="active_dispatching", notes="test")

    assert _values_after(cmd, "--lane-id") == ["lane_pr106_yshift_score_table"]
    assert _values_after(cmd, "--platform") == ["lightning"]
    assert _values_after(cmd, "--instance-job-id") == [args.job_name]
    assert _values_after(cmd, "--status") == ["active_dispatching"]
    assert "--force" not in cmd

    args.force_claim = True
    assert "--force" in tool.build_claim_command(args, status="active_dispatching", notes="test")


def test_dispatch_command_exports_score_table_contract(tmp_path: Path) -> None:
    tool = _load_tool()
    args = _args(tmp_path)
    cmd = tool.build_dispatch_command(args)

    assert "scripts/launch_lane_lightning.py" in cmd[1]
    assert _values_after(cmd, "--lane-script") == ["scripts/remote_lane_pr106_yshift_sidechannel.sh"]
    assert _values_after(cmd, "--label") == [args.job_name]
    assert _values_after(cmd, "--remote-workspace") == ["/teamspace/studios/this_studio/pact"]

    envs = _values_after(cmd, "--env")
    assert "PR106_YSHIFT_MODE=score_table" in envs
    assert f"PR106_YSHIFT_SCORE_TABLE_INSTANCE_JOB_ID={args.job_name}" in envs
    assert "PR106_YSHIFT_SCORE_TABLE_LANE_ID=lane_pr106_yshift_score_table" in envs
    assert "PR106_YSHIFT_N_PAIRS=600" in envs


def test_batch_command_runs_remote_script_and_copies_score_json(tmp_path: Path) -> None:
    tool = _load_tool()
    args = _args(tmp_path)
    command = tool.build_batch_command(args)

    assert "LIGHTNING_RUNNER_CUDA_PREFLIGHT_OK" in command
    assert "bash scripts/remote_lane_pr106_yshift_sidechannel.sh" in command
    assert "contest_auth_eval.json" in command
    assert "PR106_YSHIFT_LOG_DIR=" in command
    assert "PR106_YSHIFT_SCORE_TABLE_INSTANCE_JOB_ID=lane_pr106_yshift_score_table_test" in command


def test_batch_spec_records_generic_cuda_profile_role(tmp_path: Path) -> None:
    tool = _load_tool()
    args = _args(tmp_path)
    spec = tool.build_batch_spec(args)

    assert spec.role == "pr106_yshift_score_table_cuda"
    assert spec.machine == "g4dn.2xlarge"
    assert spec.studio == "lossy-compression-challenge"
    assert spec.teamspace == "comma-lab"
    assert spec.local_artifact_dir == "experiments/results/lightning_batch/lane_pr106_yshift_score_table_test"
    assert spec.queue_metadata["lane"] == "lane_pr106_yshift_score_table"


def test_dispatch_fails_before_claim_when_ssh_target_missing(tmp_path: Path) -> None:
    tool = _load_tool()
    args = _args(tmp_path)
    args.ssh_target = ""

    with pytest.raises(SystemExit, match="--ssh-target or LIGHTNING_SSH_TARGET"):
        tool.dispatch(args)


def test_batch_print_only_can_skip_ssh_and_stage_when_prestaged(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = _load_tool()
    args = _args(tmp_path)
    args.ssh_target = ""
    args.skip_ssh_check = True
    args.skip_stage = True
    args.backend = "batch"
    args.print_only = True

    assert tool.dispatch(args) == 0
    output = capsys.readouterr().out

    assert "=== ssh preflight ===" not in output
    assert "=== stage ===" not in output
    assert "=== claim ===" in output
    assert "=== batch command ===" in output
