from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "tools" / "dispatch_t1_balle_endtoend.py"
REMOTE_SCRIPT = REPO / "scripts" / "remote_lane_t1_balle_endtoend.sh"
SPEC = importlib.util.spec_from_file_location("dispatch_t1_balle_endtoend", MODULE_PATH)
assert SPEC is not None
t1 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = t1
SPEC.loader.exec_module(t1)


def test_vastai_default_gpu_tier_is_4090() -> None:
    assert t1.DEFAULT_GPU_TIER_BY_PROVIDER["vastai"] == "4090"
    assert t1.hourly_rate("vastai", "4090") == pytest.approx(0.42)


def test_unknown_provider_tier_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported provider/gpu-tier"):
        t1.hourly_rate("vastai", "h100")


def test_build_remote_command_enforces_main_before_remote_script(tmp_path: Path) -> None:
    plan = t1.DispatchPlan(
        provider="vastai",
        gpu_tier="4090",
        estimated_hours=24.0,
        cost_cap_usd=80.0,
        estimated_cost_usd=10.08,
        output_dir=tmp_path,
        epochs=3000,
        vastai_disk_gb=60,
    )
    command = t1.build_remote_command(plan)
    shell = command[-1]

    assert 'test "$(git branch --show-current)" = main' in shell
    assert "git pull --ff-only origin main" in shell
    assert "T1_RUN_CONTEST_CUDA_AUTH_EVAL=1" in shell
    assert "LOCAL_CUDA_WORKER=1" in shell
    assert 'T1_MOUNTED_CODE_GIT_BRANCH="$(git branch --show-current)"' in shell
    assert 'T1_MOUNTED_CODE_GIT_HEAD="$(git rev-parse HEAD)"' in shell
    assert "SINKHORN_MAX_POSITIONS_PER_CHUNK=2048" in shell
    assert "bash scripts/remote_lane_t1_balle_endtoend.sh" in shell


def test_non_dry_run_refuses_before_metadata(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            ".venv/bin/python",
            str(MODULE_PATH),
            "--provider",
            "vastai",
            "--skip-claim",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "provider launch is not implemented" in result.stdout
    assert "No lane claim or provider job was created" in result.stdout
    assert not (tmp_path / "vastai_scaffold_plan.json").exists()


def test_dry_run_vastai_writes_metadata_default_4090(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            ".venv/bin/python",
            str(MODULE_PATH),
            "--provider",
            "vastai",
            "--dry-run",
            "--skip-claim",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    metadata = json.loads((tmp_path / "vastai_scaffold_plan.json").read_text())
    assert metadata["provider"] == "vastai"
    assert metadata["gpu_tier"] == "4090"
    assert metadata["estimated_cost_usd"] == pytest.approx(10.08)
    assert metadata["dispatch_status"] == t1.SCAFFOLD_PLAN_STATUS
    assert metadata["provider_job_created"] is False
    assert metadata["gpu_work_created"] is False
    assert metadata["auth_eval_created"] is False
    assert metadata["score_claim"] is False
    assert metadata["dispatched_at_utc"] is None
    assert metadata["instance_id"] is None
    assert "bash scripts/remote_lane_t1_balle_endtoend.sh" in metadata["vastai_remote_command_template_only"]
    assert "LOCAL_CUDA_WORKER=1" in metadata["vastai_remote_command_template_only"]
    assert "T1_DISPATCH_CLAIMS_PATH" in metadata["vastai_remote_command_template_only"]
    assert "T1_MOUNTED_CODE_GIT_HEAD" in metadata["vastai_remote_command_template_only"]
    assert "SINKHORN_MAX_POSITIONS_PER_CHUNK=2048" in metadata["vastai_remote_command_template_only"]
    assert not (tmp_path / "vastai_metadata.json").exists()


def test_unsupported_vastai_h100_writes_no_metadata(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            ".venv/bin/python",
            str(MODULE_PATH),
            "--provider",
            "vastai",
            "--gpu-tier",
            "h100",
            "--dry-run",
            "--skip-claim",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "unsupported provider/gpu-tier" in result.stdout
    assert not (tmp_path / "vastai_scaffold_plan.json").exists()


def test_vastai_disk_below_scaffold_minimum_writes_no_plan(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            ".venv/bin/python",
            str(MODULE_PATH),
            "--provider",
            "vastai",
            "--vastai-disk-gb",
            "20",
            "--dry-run",
            "--skip-claim",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "--vastai-disk-gb >= 60" in result.stdout
    assert not (tmp_path / "vastai_scaffold_plan.json").exists()


def test_modal_dry_run_writes_scaffold_plan_not_harvest_metadata(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            ".venv/bin/python",
            str(MODULE_PATH),
            "--provider",
            "modal",
            "--dry-run",
            "--skip-claim",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    metadata = json.loads((tmp_path / "modal_scaffold_plan.json").read_text())
    assert metadata["modal_spawn_status"] == t1.SCAFFOLD_PLAN_STATUS
    assert metadata["modal_actuator"] == "experiments/modal_t1_balle_endtoend.py"
    assert "experiments/modal_t1_balle_endtoend.py --execute" in metadata[
        "modal_execute_command_template_only"
    ]
    assert metadata["modal_call_id"] is None
    assert metadata["harvester_invocation"] is None
    assert metadata["provider_job_created"] is False
    assert not (tmp_path / "modal_metadata.json").exists()


def test_remote_scaffold_script_refuses_by_default(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            "bash",
            str(REMOTE_SCRIPT),
        ],
        cwd=REPO,
        env={
            "WORKSPACE": str(tmp_path / "missing-workspace"),
            "LOG_DIR": str(tmp_path / "logs"),
            "OUTPUT_DIR": str(tmp_path / "out"),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 20
    assert "remote T1 refused by default" in result.stderr
    assert "T1_ALLOW_SCORE_DOMAIN_TRAINING=1" in result.stderr
    assert not (tmp_path / "logs").exists()


def test_remote_score_domain_requires_dispatch_instance_job_id(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            "bash",
            str(REMOTE_SCRIPT),
        ],
        cwd=REPO,
        env={
            "WORKSPACE": str(REPO),
            "LOG_DIR": str(tmp_path / "logs"),
            "OUTPUT_DIR": str(tmp_path / "out"),
            "T1_ALLOW_SCORE_DOMAIN_TRAINING": "1",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 21
    assert "requires T1_DISPATCH_INSTANCE_JOB_ID" in result.stderr
    assert not (tmp_path / "logs").exists()


def test_remote_score_domain_requires_active_matching_claim(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            "bash",
            str(REMOTE_SCRIPT),
        ],
        cwd=REPO,
        env={
            "WORKSPACE": str(REPO),
            "LOG_DIR": str(tmp_path / "logs"),
            "OUTPUT_DIR": str(tmp_path / "out"),
            "T1_ALLOW_SCORE_DOMAIN_TRAINING": "1",
            "T1_DISPATCH_INSTANCE_JOB_ID": "fake-t1-job-for-test",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 21
    assert "no active dispatch claim" in result.stderr
    assert "fake-t1-job-for-test" in result.stderr
    assert not (tmp_path / "logs").exists()


def test_remote_score_domain_refuses_env_only_claim_summary_without_remote_state_ledger(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "clean-remote"
    workspace.mkdir()

    result = subprocess.run(
        [
            "bash",
            str(REMOTE_SCRIPT),
        ],
        cwd=REPO,
        env={
            "WORKSPACE": str(workspace),
            "LOG_DIR": str(tmp_path / "logs"),
            "OUTPUT_DIR": str(tmp_path / "out"),
            "T1_ALLOW_SCORE_DOMAIN_TRAINING": "1",
            "T1_DISPATCH_INSTANCE_JOB_ID": "claimed-job-123",
            "T1_DISPATCH_CLAIM_SUMMARY_B64": "ignored-not-a-trusted-claim",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 21
    assert "claim helper missing" in combined
    assert "no active dispatch claim" not in combined
    assert not (tmp_path / "logs").exists()


def test_remote_train_pipeline_disables_pipefail_around_rc_capture() -> None:
    text = REMOTE_SCRIPT.read_text()

    for command_var, log_name, rc_var in (
        ("TRAIN_CMD", "train.log", "TRAIN_RC"),
        ("PACKET_CMD", "packet_compiler.log", "PACKET_RC"),
        ("AUTH_EVAL_CMD", "contest_auth_eval.log", "AUTH_EVAL_RC"),
    ):
        assert re.search(
            rf"set \+e\s+"
            rf'"\${{{command_var}\[@\]}}" 2>&1 \| tee "\$LOG_DIR/{log_name}"\s+'
            rf"{rc_var}=\${{PIPESTATUS\[0\]}}\s+"
            rf"set -e",
            text,
        )


def test_remote_t1_script_wires_packet_compile_and_contest_cuda_auth_eval() -> None:
    text = REMOTE_SCRIPT.read_text()

    assert "T1_RUN_CONTEST_CUDA_AUTH_EVAL" in text
    assert "tools/build_phase1_packet_compiler.py" in text
    assert "--input-packet \"$SUBMISSION_DIR\"" in text
    assert "--mode optimize" in text
    assert "--runtime-dep-closure torch brotli compressai" in text
    assert "--export-format phase1_three_member_x_decoder_bin_balle_bin" in text
    assert "--score-affecting-payload-changed" in text
    assert "experiments/contest_auth_eval.py" in text
    assert "--archive \"$PACKET_DIR/archive.zip\"" in text
    assert "--inflate-sh \"$PACKET_DIR/inflate.sh\"" in text
    assert "--device cuda" in text
    assert "--work-dir \"$AUTH_EVAL_WORK_DIR\"" in text
    assert "--json-out \"$AUTH_EVAL_JSON\"" in text
    assert "PACKET_COMPILER_RUNTIME_TREE_SHA=" in text
    assert "AUTH_EVAL_EXPECTED_RUNTIME_TREE_SHA=" in text
    assert "from experiments.contest_auth_eval import _runtime_dependency_manifest" in text
    assert "--expected-runtime-tree-sha256 \"$AUTH_EVAL_EXPECTED_RUNTIME_TREE_SHA\"" in text
    assert "--expected-runtime-tree-sha256 \"$RUNTIME_TREE_SHA\"" not in text
    assert "runtime_tree_sha256 = runtime_manifest.get(\"runtime_tree_sha256\")" in text
    assert "contest_auth_eval_runtime_tree_sha256_missing_or_invalid" in text
    assert "contest_auth_eval_runtime_tree_sha256_mismatch_expected" in text
    assert "\"runtime_tree_sha256\": runtime_tree_sha256" in text
    assert "\"packet_pre_manifest_runtime_tree_sha256\"" in text
    assert "\"runtime_tree_sha256\": manifest.get(\"runtime_tree_sha256\")" not in text
    assert "required_contest_cuda_evidence_blockers" in text
    assert "device mps" not in text.lower()


def test_remote_t1_script_closes_dispatch_claim_terminally() -> None:
    text = REMOTE_SCRIPT.read_text()

    assert "close_dispatch_claim()" in text
    assert "trap cleanup EXIT" in text
    assert "failed_remote_script_rc_${rc}" in text
    assert "completed_t1_score_domain_training_no_score_claim" in text
    assert "failed_t1_packet_compile_rc_${PACKET_RC}" in text
    assert "failed_t1_contest_auth_eval_rc_${AUTH_EVAL_RC}" in text
    assert "failed_t1_auth_eval_adjudication" in text
    assert "completed_t1_contest_cuda_auth_eval" in text
    assert "score_claim=false" in text
    assert '"promotion_eligible": False' in text
    assert "paired_contest_cpu_reproduction_required" in text
    assert "registry_level_promotion_required" in text
    assert "lane_registry_level_promotion_required" not in text
    contest_cuda_lines = [
        line
        for line in text.splitlines()
        if "[contest-CUDA]" in line and ("echo " in line or "log " in line)
    ]
    assert contest_cuda_lines == [
        '        echo "[lane-t1] LANE_T1_CONTEST_CUDA_AUTH_EVAL_DONE [contest-CUDA]"'
    ]
