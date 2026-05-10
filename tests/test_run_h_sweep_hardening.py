from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "experiments" / "run_h_sweep.py"
CLAIM_TOOL = REPO / "tools" / "claim_lane_dispatch.py"
SPEC = importlib.util.spec_from_file_location("run_h_sweep", MODULE_PATH)
assert SPEC is not None
run_h_sweep = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = run_h_sweep
SPEC.loader.exec_module(run_h_sweep)


def _assert_false_authority(payload: dict, *, dispatch_attempted: bool = False) -> None:
    assert payload["score_claim"] is False
    assert payload["score_claim_valid"] is False
    assert payload["proxy_only"] is True
    assert payload["promotion_eligible"] is False
    assert payload["rank_or_kill_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["dispatch_attempted"] is dispatch_attempted


def _write_active_claim(
    tmp_path: Path,
    *,
    lane_id: str = "h_sweep_modal",
    instance_job_id: str = "h_sweep_tiny_20260510T000000Z",
    notes: str = "H sweep tiny Modal launch; score_claim=false; cost=$0.50",
) -> Path:
    claims_path = tmp_path / "claims.md"
    subprocess.run(
        [
            ".venv/bin/python",
            str(CLAIM_TOOL),
            "claim",
            "--claims-path",
            str(claims_path),
            "--lane-id",
            lane_id,
            "--platform",
            "modal",
            "--instance-job-id",
            instance_job_id,
            "--agent",
            "pytest:h_sweep",
            "--status",
            "active_dispatching",
            "--notes",
            notes,
        ],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return claims_path


def test_default_modal_path_writes_plan_only_and_does_not_launch(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            ".venv/bin/python",
            str(MODULE_PATH),
            "--config",
            "tiny",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "plan-only default" in result.stdout
    assert "score_claim=false proxy_only=true" in result.stdout
    plan = json.loads((tmp_path / "h_sweep_modal_plan.json").read_text())
    assert plan["plan_status"] == run_h_sweep.PLAN_STATUS
    assert plan["provider_job_created"] is False
    assert plan["gpu_work_created"] is False
    assert plan["auth_eval_created"] is False
    assert plan["execution_required_flag"] == "--execute-modal"
    _assert_false_authority(plan)
    assert len(plan["configs"]) == 1
    entry = plan["configs"][0]
    _assert_false_authority(entry)
    assert entry["name"] == "tiny"
    assert entry["modal_actuator"] == "src/tac/deploy/modal/modal_asymmetric_warp_deploy.py"
    assert "modal_asymmetric_warp_deploy.py" in entry["modal_command_template_only"]
    assert "--tag h_sweep_tiny" in entry["modal_command_template_only"]
    assert not (tmp_path / "sweep_runs.json").exists()


def test_dry_run_is_compatibility_alias_for_plan_only(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            ".venv/bin/python",
            str(MODULE_PATH),
            "--config",
            "tiny",
            "--dry-run",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    plan = json.loads((tmp_path / "h_sweep_modal_plan.json").read_text())
    assert plan["plan_status"] == run_h_sweep.PLAN_STATUS
    _assert_false_authority(plan)


def test_execute_modal_refuses_without_claim_cost_metadata(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            ".venv/bin/python",
            str(MODULE_PATH),
            "--config",
            "tiny",
            "--execute-modal",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Modal execute refused before provider launch" in result.stdout
    assert "--execute-modal requires --lane-id" in result.stdout
    assert "--execute-modal requires --estimated-cost-usd" in result.stdout
    assert not (tmp_path / run_h_sweep.MODAL_EXECUTE_PREFLIGHT).exists()
    assert not (tmp_path / "sweep_runs.json").exists()


def test_execute_modal_refuses_cost_above_cap(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            ".venv/bin/python",
            str(MODULE_PATH),
            "--config",
            "tiny",
            "--execute-modal",
            "--lane-id",
            "h_sweep_modal",
            "--instance-job-id",
            "h_sweep_tiny_20260510T000000Z",
            "--estimated-cost-usd",
            "0.75",
            "--cost-cap-usd",
            "0.50",
            "--dispatch-claims-path",
            str(tmp_path / "claims.md"),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "estimated cost $0.75 exceeds cap $0.50" in result.stdout
    assert not (tmp_path / run_h_sweep.MODAL_EXECUTE_PREFLIGHT).exists()


def test_execute_modal_refuses_claim_without_cost_metadata(tmp_path: Path) -> None:
    claims_path = _write_active_claim(tmp_path, notes="H sweep tiny Modal launch; score_claim=false")

    result = subprocess.run(
        [
            ".venv/bin/python",
            str(MODULE_PATH),
            "--config",
            "tiny",
            "--execute-modal",
            "--lane-id",
            "h_sweep_modal",
            "--instance-job-id",
            "h_sweep_tiny_20260510T000000Z",
            "--estimated-cost-usd",
            "0.50",
            "--cost-cap-usd",
            "0.75",
            "--dispatch-claims-path",
            str(claims_path),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "active dispatch claim notes must include cost=$<usd>" in result.stdout
    assert not (tmp_path / run_h_sweep.MODAL_EXECUTE_PREFLIGHT).exists()


def test_execute_modal_refuses_all_configs_with_single_claim(tmp_path: Path) -> None:
    claims_path = _write_active_claim(tmp_path)

    result = subprocess.run(
        [
            ".venv/bin/python",
            str(MODULE_PATH),
            "--execute-modal",
            "--lane-id",
            "h_sweep_modal",
            "--instance-job-id",
            "h_sweep_tiny_20260510T000000Z",
            "--estimated-cost-usd",
            "0.50",
            "--cost-cap-usd",
            "0.75",
            "--dispatch-claims-path",
            str(claims_path),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--execute-modal requires exactly one --config" in result.stdout
    assert not (tmp_path / run_h_sweep.MODAL_EXECUTE_PREFLIGHT).exists()


def test_execute_modal_builds_canonical_modal_command_after_claim_and_cost_gate(
    monkeypatch, tmp_path: Path
) -> None:
    claims_path = _write_active_claim(tmp_path)
    calls: list[list[str]] = []
    envs: list[dict[str, str] | None] = []

    class Result:
        returncode = 0

    def fake_run_provider(cmd: list[str], env: dict[str, str] | None = None):  # type: ignore[no-untyped-def]
        calls.append(cmd)
        envs.append(env)
        return Result()

    monkeypatch.setattr(run_h_sweep, "run_provider_command", fake_run_provider)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_h_sweep.py",
            "--config",
            "tiny",
            "--execute-modal",
            "--lane-id",
            "h_sweep_modal",
            "--instance-job-id",
            "h_sweep_tiny_20260510T000000Z",
            "--estimated-cost-usd",
            "0.50",
            "--cost-cap-usd",
            "0.75",
            "--dispatch-claims-path",
            str(claims_path),
            "--output-dir",
            str(tmp_path),
        ],
    )

    run_h_sweep.main()

    assert len(calls) == 1
    command = calls[0]
    assert command[:4] == [sys.executable, "-m", "modal", "run"]
    assert command[4].endswith("src/tac/deploy/modal/modal_asymmetric_warp_deploy.py")
    assert "--tag" in command
    assert "h_sweep_tiny" in command
    assert envs[0] is not None
    assert envs[0]["PACT_DISPATCH_LANE_ID"] == "h_sweep_modal"
    assert envs[0]["PACT_DISPATCH_INSTANCE_JOB_ID"] == "h_sweep_tiny_20260510T000000Z"
    preflight = json.loads((tmp_path / run_h_sweep.MODAL_EXECUTE_PREFLIGHT).read_text())
    _assert_false_authority(preflight)
    assert preflight["provider_launch_allowed"] is True
    assert preflight["execute_gate"]["dispatch_claim_verified"] is True
    assert preflight["execute_gate"]["estimated_cost_usd"] == 0.50
    assert preflight["execute_gate"]["cost_cap_usd"] == 0.75
    assert preflight["execute_gate"]["claim_cost_usd"] == 0.50
    metadata = json.loads((tmp_path / "sweep_runs.json").read_text())
    _assert_false_authority(metadata, dispatch_attempted=True)
    assert metadata["execute_gate"]["dispatch_claim_verified"] is True
    _assert_false_authority(metadata["runs"][0], dispatch_attempted=True)
    assert metadata["runs"][0]["surface"] == "h_sweep_modal_execute"


def test_execute_modal_refuses_local_mode(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            ".venv/bin/python",
            str(MODULE_PATH),
            "--local",
            "--execute-modal",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--execute-modal cannot be combined with --local" in result.stdout
    assert not (tmp_path / "h_sweep_modal_plan.json").exists()


def test_collect_results_forces_proxy_false_authority(monkeypatch, tmp_path: Path) -> None:
    result_dir = tmp_path / "tiny"
    result_dir.mkdir()
    (result_dir / "training_summary.json").write_text(
        json.dumps({"best_score": 1.23, "epoch": 10}),
        encoding="utf-8",
    )
    monkeypatch.setattr(run_h_sweep, "RESULTS_DIR", tmp_path)
    sweep = {
        "configs": [
            {
                "name": "tiny",
                "base_ch": 16,
                "mid_ch": 24,
                "total_params": 89951,
                "fp4_kb": 43.9,
                "rate_term": 0.0003,
            }
        ]
    }

    rows = run_h_sweep.collect_results(sweep)

    assert rows[0]["proxy_score"] == 1.23
    _assert_false_authority(rows[0])
    assert rows[0]["surface"] == "h_sweep_collect_results"
