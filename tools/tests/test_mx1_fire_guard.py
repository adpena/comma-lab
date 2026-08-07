# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _host() -> dict[str, str]:
    return {
        "node": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "system": platform.system(),
    }


def _ticket_and_receipt(tmp_path: Path) -> tuple[Path, Path, Path]:
    input_cache = tmp_path / "gt_seg_cache.pt"
    target_cache = tmp_path / "gt_seg_cache.pt"
    init = tmp_path / "semantic_renderer.pt"
    for path in (input_cache, init):
        path.write_bytes(b"x")
    run_dir = tmp_path / "launch_arm_cap" / "n32_metal"
    verdict_path = run_dir / "fire_guard_verdict.json"
    receipt_path = tmp_path / "row1" / "mem_probe_receipt.json"
    ticket_path = tmp_path / "launch_ticket.json"
    ticket = {
        "schema": "ddm_mx1_row1_launch_ticket.v4_software_cap_fire_guarded",
        "mem_probe_receipt_required": True,
        "mem_probe_receipt_path": str(receipt_path),
        "argv_n32_arm_cap": [
            ".venv/bin/python",
            "tools/safe_run.py",
            "--rss-mb",
            "90000",
            "--",
            ".venv/bin/python",
            "experiments/ddm_mx1_pr130_semantic_renderer.py",
            "--mode",
            "mlx-train",
            "--device",
            "gpu",
            "--pairs",
            "32",
            "--lr",
            "2e-07",
            "--ce-fraction",
            "0.0",
            "--softplus-fraction",
            "-999.0",
            "--bits",
            "4",
            "--input-cache",
            str(input_cache),
            "--target-cache",
            str(target_cache),
            "--init",
            str(init),
            "--run-dir",
            str(run_dir),
            "--out",
            str(run_dir / "result.json"),
            "--fire-guard-verdict",
            str(verdict_path),
            "--launch-ticket-path",
            str(ticket_path),
            "--fire-argv-key",
            "argv_n32_arm_cap",
        ],
    }
    ticket_path.write_text(json.dumps(ticket), encoding="utf-8")
    receipt = {
        "schema": "ddm_mx1_load_phase_peak_receipt.v1",
        "status": "passed",
        "metal_fire_clearance": True,
        "host": _host(),
        "source_repo_head": "test",
        "device_request": "gpu",
        "pairs": 32,
        "mem_budget_gb_arg": None,
        "input_cache": str(input_cache),
        "target_cache": str(target_cache),
        "init_checkpoint": str(init),
        "argv_config": {
            "device": "gpu",
            "pairs": 32,
            "lr": 2e-7,
            "ce_fraction": 0.0,
            "softplus_fraction": -999.0,
            "bits": 4,
            "microbatch_pairs": 0,
            "mem_budget_gb": None,
            "allow_soft_mem_limit": False,
            "input_cache": str(input_cache),
            "target_cache": str(target_cache),
            "init": str(init),
        },
        "memory_limits": {
            "enforcement": "software_stage_step_cap",
            "software_cap_required": True,
            "software_cap_installed": True,
            "software_budget_bytes": 8_589_934_592,
            "hard_limit_required": False,
            "hard_limit_satisfied": False,
            "soft_limit_allowed_by_cli": False,
            "calls": [{"target": "set_memory_limit", "status": "applied", "hard_limit": False}],
        },
        "software_budget": {
            "enforcement": "software_stage_step_cap",
            "budget_bytes": 8_589_934_592,
            "check_count": 5,
            "last_check": {
                "stage": "after_train_step_000003",
                "within_budget": True,
                "combined_gib": 2.0,
                "budget_gib": 8.0,
            },
        },
        "train_result_summary": {
            "microbatch_plan": {
                "total_pairs": 32,
                "microbatch_pairs": 4,
                "chunk_count": 8,
                "mode": "serial_gradient_accumulation",
                "source": "gpu_default_4_pairs",
            },
        },
        "samples": [
            {
                "stage": "after_require_mlx_and_memory_limits",
                "mlx_active_gib": 1.0,
                "mlx_cache_gib": 0.1,
                "mlx_peak_gib": 1.1,
            },
            {
                "stage": "after_train_step_000003",
                "mlx_active_gib": 2.0,
                "mlx_cache_gib": 0.2,
                "mlx_peak_gib": 2.2,
            },
        ],
        "clearance_checks": {
            "required_stage": "after_train_step_000003",
            "has_required_stage_sample": True,
            "has_mlx_allocator_telemetry_at_required_stage": True,
        },
    }
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return ticket_path, receipt_path, verdict_path


def test_mx1_fire_guard_passes_matching_receipt(tmp_path: Path) -> None:
    ticket_path, _receipt_path, verdict_path = _ticket_and_receipt(tmp_path)

    result = subprocess.run(
        [
            ".venv/bin/python",
            "tools/mx1_fire_guard.py",
            "--ticket",
            str(ticket_path),
            "--argv-key",
            "argv_n32_arm_cap",
            "--out",
            str(verdict_path),
        ],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    verdict = json.loads(verdict_path.read_text())
    assert verdict["schema"] == "ddm_mx1_fire_guard_verdict.v1"
    assert verdict["status"] == "passed"
    assert verdict["reason_code"] == "fire_guard_passed"
    assert verdict["reason"] == "fire_guard_passed"
    assert verdict["checks"][0]["name"] == "receipt_freshness"


def test_mx1_fire_guard_refuses_missing_receipt(tmp_path: Path) -> None:
    ticket_path, receipt_path, verdict_path = _ticket_and_receipt(tmp_path)
    receipt_path.unlink()

    result = subprocess.run(
        [
            ".venv/bin/python",
            "tools/mx1_fire_guard.py",
            "--ticket",
            str(ticket_path),
            "--argv-key",
            "argv_n32_arm_cap",
            "--out",
            str(verdict_path),
        ],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 4
    verdict = json.loads(verdict_path.read_text())
    assert verdict["status"] == "failed"
    assert verdict["reason_code"] == "mem_probe_receipt_missing"
    assert verdict["reason"] == "mem_probe_receipt_missing"


def test_mx1_fire_guard_refuses_stale_receipt(tmp_path: Path) -> None:
    ticket_path, receipt_path, verdict_path = _ticket_and_receipt(tmp_path)
    stale = time.time() - (7 * 60 * 60)
    os.utime(receipt_path, (stale, stale))

    result = subprocess.run(
        [
            ".venv/bin/python",
            "tools/mx1_fire_guard.py",
            "--ticket",
            str(ticket_path),
            "--argv-key",
            "argv_n32_arm_cap",
            "--out",
            str(verdict_path),
        ],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 4
    verdict = json.loads(verdict_path.read_text())
    assert verdict["status"] == "failed"
    assert verdict["reason_code"] == "mem_probe_receipt_stale"


def test_mx1_fire_guard_refuses_microbatch_footprint_mismatch(tmp_path: Path) -> None:
    ticket_path, _receipt_path, verdict_path = _ticket_and_receipt(tmp_path)
    ticket = json.loads(ticket_path.read_text())
    argv = ticket["argv_n32_arm_cap"]
    insert_at = argv.index("--fire-guard-verdict")
    argv[insert_at:insert_at] = ["--microbatch-pairs", "32"]
    ticket_path.write_text(json.dumps(ticket), encoding="utf-8")

    result = subprocess.run(
        [
            ".venv/bin/python",
            "tools/mx1_fire_guard.py",
            "--ticket",
            str(ticket_path),
            "--argv-key",
            "argv_n32_arm_cap",
            "--out",
            str(verdict_path),
        ],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 4
    verdict = json.loads(verdict_path.read_text())
    assert verdict["status"] == "failed"
    assert verdict["reason_code"] == "receipt_config_mismatch"
    mismatches = verdict["checks"][-1]["detail"]["mismatches"]
    assert "microbatch_pairs" in mismatches


def test_mx1_fire_guard_passes_resume_key_with_matching_resume_receipt(tmp_path: Path) -> None:
    ticket_path, receipt_path, verdict_path = _ticket_and_receipt(tmp_path)
    ticket = json.loads(ticket_path.read_text())
    resume_key = "argv_n32_arm_cap_resume"
    resume_receipt_path = (
        tmp_path
        / "launch_arm_cap"
        / "n32_metal"
        / "mem_probe_resume"
        / "mem_probe_receipt.json"
    )
    resume_receipt_path.parent.mkdir(parents=True)
    resume_receipt_path.write_text(receipt_path.read_text(encoding="utf-8"), encoding="utf-8")
    resume_argv = list(ticket["argv_n32_arm_cap"])
    resume_argv[resume_argv.index("--fire-argv-key") + 1] = resume_key
    resume_argv.extend(["--resume-from", str(tmp_path / "launch_arm_cap" / "n32_metal" / "mlx.latest.npz")])
    ticket[resume_key] = resume_argv
    ticket["mem_probe_receipt_paths"] = {resume_key: str(resume_receipt_path)}
    ticket_path.write_text(json.dumps(ticket), encoding="utf-8")

    result = subprocess.run(
        [
            ".venv/bin/python",
            "tools/mx1_fire_guard.py",
            "--ticket",
            str(ticket_path),
            "--argv-key",
            resume_key,
            "--out",
            str(verdict_path),
        ],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    verdict = json.loads(verdict_path.read_text())
    assert verdict["status"] == "passed"
    assert verdict["reason_code"] == "fire_guard_passed"
    assert verdict["reason"] == "fire_guard_passed"
    assert verdict["receipt_path"] == str(resume_receipt_path)
    assert verdict["argv_key"] == resume_key
